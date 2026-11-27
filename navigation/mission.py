#!/usr/bin/env python3
"""Foxglove waypoint queue and the Go2 EDU Sport velocity gate for this Nav2 stack."""
from collections import deque
import json
import math
from pathlib import Path
import signal
import time

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Log
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Empty, String
from unitree_api.msg import Request
from visualization_msgs.msg import Marker, MarkerArray


class Mission(Node):
    def __init__(self):
        super().__init__('go2_nav_mission')
        self.route = deque()
        self.sequence = self.completed = 0
        self.armed = False
        self.active = self.goal_handle = None
        self.goal_sent_at = self.cancel_started = 0.
        self.nav_logs = {}
        self.action_fault = ''
        profile = yaml.safe_load(Path(__file__).with_name('nav2.yaml').read_text())['velocity_smoother']['ros__parameters']
        hi, lo = profile['max_velocity'], profile['min_velocity']
        if (len(lo) != 3 or len(hi) != 3 or not all(math.isfinite(v) for v in [*hi, *lo]) or
                not hi[0] > 0 or lo[:2] != [0., 0.] or hi[1] != 0. or not lo[2] < 0. < hi[2]):
            raise ValueError('Go2 Sport 运动桥需要前进/原地转向的速度配置')
        self.max_forward, self.min_yaw, self.max_yaw = hi[0], lo[2], hi[2]
        self.velocity_timeout = float(profile['velocity_timeout'])
        if not 0 < self.velocity_timeout <= 1.:
            raise ValueError('Nav2 速度命令超时必须大于 0 且不超过 1 秒')
        self.void_goal = False
        self.cancel_next = 0.
        self.last_valid = self.last_odom = self.last_scan = float('-inf')
        self.command = None
        self.moving = False
        self.owner_ok = False
        self.last_graph_check = self.last_report = float('-inf')
        self.reason = '等待在 map 中添加航点'
        retained = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.operator = self.create_publisher(String, '/operator/status', retained)
        self.control = self.create_publisher(String, '/control/status', retained)
        self.markers = self.create_publisher(MarkerArray, '/navigation/waypoints', retained)
        self.sport = self.create_publisher(Request, '/api/sport/request', 10)
        self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.create_subscription(PoseStamped, '/goal_pose', self.add_goal, 100)
        self.create_subscription(Bool, '/control/enable', self.enable, 4)
        self.create_subscription(Empty, '/navigation/resume', lambda _: self.arm(), 4)
        self.create_subscription(Empty, '/navigation/cancel', lambda _: self.cancel_current(), 4)
        self.create_subscription(Empty, '/navigation/clear_queue', lambda _: self.cancel(), 4)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose',
                                 lambda _: self.pause('重新设置初始位姿，待定位稳定再启用'), 1)
        self.create_subscription(Bool, '/localization/valid', self.validity, 1)
        self.create_subscription(Odometry, '/odom', self.odometry, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/navigation/scan', self.scan, qos_profile_sensor_data)
        self.create_subscription(Twist, '/cmd_vel_nav', self.velocity, 1)
        self.create_subscription(Log, '/rosout', self.on_nav_log, 20)
        self.create_timer(.05, self.tick)
        self.draw()

    def phase(self):
        if self.action_fault:
            return self.action_fault
        if self.void_goal:
            return '等待当前 Nav2 任务结束'
        if self.goal_handle is not None:
            return '执行中'
        if self.active is not None:
            return '等待 Nav2 接受'
        return '已启用' if self.armed else '待启用'

    def report(self):
        current = self.active[0] if self.active is not None else '-'
        self.operator.publish(String(data=f'{self.phase()} | 当前 #{current}，待完成 {len(self.route)}，'
                                          f'本次已完成 {self.completed} | {self.reason}'))
        self.control.publish(String(data=f'运动授权={self.armed} | {self.gate(time.monotonic()) or "输入就绪"}'
                                         f' | 已发送非零速度={self.moving}'))
        self.last_report = time.monotonic()

    def draw(self):
        markers = [Marker(action=Marker.DELETEALL)]
        for index, (identifier, goal) in enumerate(self.route):
            for text in (False, True):
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.ns = 'number' if text else 'waypoint'
                marker.id = identifier
                marker.type = Marker.TEXT_VIEW_FACING if text else Marker.SPHERE
                marker.pose.position.x = goal.pose.position.x
                marker.pose.position.y = goal.pose.position.y
                marker.pose.position.z = .32 if text else .10
                marker.pose.orientation.w = 1.
                marker.scale.x = marker.scale.y = marker.scale.z = .18
                marker.color.r = 1. if text or index == 0 else .15
                marker.color.g = 1. if text or index else .65
                marker.color.b = 1. if text else .2
                marker.color.a = 1.
                if text:
                    marker.text = str(identifier)
                markers.append(marker)
        self.markers.publish(MarkerArray(markers=markers))

    def add_goal(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        norm = math.hypot(q.x, q.y, q.z, q.w)
        if (msg.header.frame_id.lstrip('/') != 'map' or
                not all(math.isfinite(v) for v in (p.x, p.y, p.z, norm)) or norm < 1.e-6):
            self.reason = '航点无效，需要 map 坐标系的有限位置和有效朝向'
        else:
            # Nav2 operates on planar goals. Preserve the selected yaw.
            x, y, z, w = (v/norm for v in (q.x, q.y, q.z, q.w))
            yaw = math.atan2(2.*(w*z+x*y), 1.-2.*(y*y+z*z))
            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.pose.position.x, goal.pose.position.y = p.x, p.y
            goal.pose.orientation.z, goal.pose.orientation.w = math.sin(yaw*.5), math.cos(yaw*.5)
            self.sequence += 1
            self.route.append((self.sequence, goal))
            self.reason = f'航点 #{self.sequence} 已入队 ({p.x:.2f}, {p.y:.2f})'
            self.draw()
        self.report()

    def stamped(self, header):
        stamp = header.stamp.sec + header.stamp.nanosec * 1.e-9
        age = self.get_clock().now().nanoseconds * 1.e-9 - stamp
        return stamp > 0. and -.1 <= age <= .8

    def odometry(self, msg):
        if (msg.header.frame_id == 'odom' and msg.child_frame_id == 'base_link' and
                self.stamped(msg.header)):
            self.last_odom = time.monotonic()

    def scan(self, msg):
        if (msg.header.frame_id == 'go2_nav_scan' and self.stamped(msg.header) and
                0. < msg.range_min < msg.range_max and
                any(math.isfinite(r) and msg.range_min <= r <= msg.range_max for r in msg.ranges)):
            self.last_scan = time.monotonic()

    def validity(self, msg):
        self.last_valid = time.monotonic() if msg.data else float('-inf')
        if not msg.data and self.armed:
            self.pause('定位失效，队列已保留，定位恢复后请重新启用')

    def gate(self, now):
        if self.action_fault:
            return self.action_fault
        if not self.owner_ok:
            return '定位或速度来源未就绪，或发现重复导航运动桥'
        if now-self.last_valid >= .5:
            return '定位无效或心跳超时'
        if now-self.last_odom >= .8:
            return '本体里程计超时'
        if now-self.last_scan >= .8:
            return '实时扫描超时或没有有效障碍层回波'
        return ''

    def update_graph(self, now):
        if now-self.last_graph_check < .5:
            return
        self.last_graph_check = now
        writers = self.get_publishers_info_by_topic('/cmd_vel_nav')
        loc = self.get_publishers_info_by_topic('/localization/valid')
        actuators = self.get_publishers_info_by_topic('/api/sport/request')
        names = [ep.node_name for ep in actuators]
        self.owner_ok = (len(writers) == 1 and writers[0].node_name == 'velocity_smoother' and
                         len(loc) == 1 and loc[0].node_name == 'go2_gicp' and
                         names.count(self.get_name()) == 1 and 'go2_edu_dds_bridge' not in names)

    def arm(self):
        now = time.monotonic()
        error = self.gate(now)
        if self.armed and not error:
            return
        if self.action_fault:
            self.reason = self.action_fault
        elif self.active is not None:
            self.reason = '等待上一次 Nav2 任务结束，再启用'
        elif not self.route:
            self.reason = '请先发布 /goal_pose 添加航点'
        elif error or not self.nav.server_is_ready():
            self.reason = '暂不能启用：' + (error or 'Nav2 action 未就绪')
        else:
            self.armed = True
            self.reason = '已启用，按入队顺序导航'
        self.report()

    def enable(self, msg):
        if msg.data:
            self.arm()
        else:
            self.pause('已暂停，当前及后续航点保留；重新启用后重试')

    def cancel(self):
        self.route.clear()
        self.pause('已取消并清空队列')
        self.draw()

    def cancel_current(self):
        if self.active is None:
            self.pause('当前没有正在执行的目标，已暂停并保留待执行航点')
            return
        if self.route and self.route[0] is self.active:
            self.route.popleft()
        self.pause(f'已请求取消 #{self.active[0]}，其余航点保留；重新启用后继续')
        self.draw()

    def pause(self, reason):
        self.armed = False
        self.stop()
        if self.active is not None:
            if not self.void_goal:
                self.cancel_started = time.monotonic()
            self.void_goal = True
            self.request_cancel(time.monotonic())
        self.reason = reason
        self.report()

    def uncertain(self, reason):
        self.action_fault = reason + '；运动已锁定。若 Nav2 未确认任务结束，请重启整套导航'
        self.pause(self.action_fault)

    def request_cancel(self, now):
        if self.goal_handle is not None and now >= self.cancel_next:
            self.cancel_next = now + 1.
            try:
                self.goal_handle.cancel_goal_async()
            except Exception as exc:
                self.get_logger().warning('请求取消 Nav2 任务失败：' + str(exc))

    def dispatch(self):
        assert self.route and self.active is None
        self.active = self.route[0]
        self.goal_sent_at = time.monotonic()
        self.nav_logs.clear()
        self.void_goal = False
        self.cancel_next = 0.
        self.command = None
        target = NavigateToPose.Goal()
        target.pose = self.active[1]
        target.pose.header.stamp = self.get_clock().now().to_msg()
        try:
            self.nav.send_goal_async(target).add_done_callback(self.accepted)
        except Exception as exc:
            self.uncertain('目标发送状态不明：' + str(exc))

    def on_nav_log(self, msg):
        if self.active is None or self.void_goal or msg.level < 30:
            return
        # Only associate logs emitted after this particular goal was dispatched.
        start = self.active[1].header.stamp
        if (msg.stamp.sec, msg.stamp.nanosec) < (start.sec, start.nanosec):
            return
        logger = msg.name.split('.')[-1]
        detail = msg.msg.lower()
        if logger in ('planner_server', 'controller_server'):
            if 'aborting handle' in detail:
                return
            if msg.level < 40 and not any(word in detail for word in
                    ('failed', 'failure', 'unable', 'cannot', 'invalid', 'no valid', 'no path', 'timed out')):
                return
            priority = 2
        elif logger in ('local_costmap', 'global_costmap') and 'observation buffer' in detail:
            priority = 1
        else:
            return
        stamp = msg.stamp.sec * 10**9 + msg.stamp.nanosec
        self.nav_logs[priority] = (stamp, f'{logger}: {" ".join(msg.msg.split())[:240]}')

    def recent_nav_log(self):
        now = self.get_clock().now().nanoseconds
        for priority in sorted(self.nav_logs, reverse=True):
            stamp, detail = self.nav_logs[priority]
            if 0 <= now - stamp <= 12 * 10**9:
                return ' | 近期 Nav2 日志：' + detail
        return ''

    def accepted(self, future):
        try:
            handle = future.result()
        except Exception as exc:
            # An ambiguous transport failure must not authorize another action.
            self.uncertain('Nav2 目标响应异常：' + str(exc))
            return
        if not handle.accepted:
            self.action_fault = ''
            self.active, self.void_goal = None, False
            self.pause('Nav2 拒绝航点；检查地图、目标及 Nav2 状态后重新启用' + self.recent_nav_log())
            return
        self.goal_handle = handle
        try:
            handle.get_result_async().add_done_callback(self.result)
        except Exception as exc:
            self.uncertain('获取 Nav2 结果失败：' + str(exc))
            return
        if self.void_goal:
            self.request_cancel(time.monotonic())

    def result(self, future):
        try:
            status = future.result().status
        except Exception as exc:
            self.uncertain('Nav2 结果通信失败：' + str(exc))
            return
        if status not in (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED):
            self.uncertain('Nav2 未返回可确认的终态')
            return
        if self.action_fault:
            self.action_fault = ''
            self.reason = 'Nav2 已确认上一个任务结束，可重新启用'
        finished, invalidated = self.active, self.void_goal
        self.active, self.goal_handle, self.void_goal = None, None, False
        self.stop()
        if not invalidated:
            if status == GoalStatus.STATUS_SUCCEEDED and self.route and self.route[0] is finished:
                self.route.popleft()
                self.completed += 1
                self.reason = f'已到达 #{finished[0]}'
                if not self.route:
                    self.armed = False
                    self.reason += '，队列已完成'
            else:
                self.armed = False
                if status == GoalStatus.STATUS_ABORTED:
                    self.reason = 'Nav2 已中止（状态 6）；当前航点保留，已暂停' + self.recent_nav_log()
                else:
                    self.reason = f'Nav2 返回状态 {status}；当前航点保留，已暂停'
        self.draw()
        self.report()

    def velocity(self, msg):
        now = time.monotonic()
        if not self.armed or self.goal_handle is None or self.void_goal or self.gate(now):
            return
        v = [msg.linear.x, msg.linear.y, msg.angular.z]
        if not all(math.isfinite(i) for i in v):
            self.pause('Nav2 速度无效，已暂停')
            return
        # Same EDU profile as the original Go2 stack; never amplify low commands.
        vx = max(0., min(self.max_forward, v[0]))
        wz = max(self.min_yaw, min(self.max_yaw, v[2]))
        self.command = now, (vx, 0., wz)

    def request(self, api, parameters=''):
        msg = Request()
        msg.header.identity.id = time.time_ns()
        msg.header.identity.api_id = api
        msg.parameter = parameters
        self.sport.publish(msg)

    def stop(self):
        self.command = None
        if self.moving:
            self.request(1003)
            self.moving = False

    def tick(self):
        now = time.monotonic()
        self.update_graph(now)
        if self.active is not None and not self.action_fault:
            if self.goal_handle is None and now-self.goal_sent_at > 8.:
                self.uncertain('Nav2 超时未确认目标是否接受')
            elif self.void_goal and now-self.cancel_started > 8.:
                self.uncertain('Nav2 超时未确认取消结果')
        error = self.gate(now)
        if self.armed and error:
            self.pause(error + '；队列已保留')
        if self.void_goal:
            self.request_cancel(now)
        elif self.armed:
            if not self.nav.server_is_ready():
                self.pause('Nav2 action 已离线；队列已保留')
            elif self.active is None and self.route:
                self.dispatch()
        if self.command is not None:
            stamp, values = self.command
            if now-stamp > self.velocity_timeout or not self.armed or self.goal_handle is None or not any(values):
                self.stop()
            else:
                self.request(1008, json.dumps(dict(zip(('x', 'y', 'z'), values))))
                self.moving = True
        if now-self.last_report >= .5:
            self.report()


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    def terminate(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    node = Mission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        was_moving = node.moving
        node.pause('导航节点退出')
        if was_moving:
            # Keep the DDS writer alive briefly for the final StopMove sample.
            time.sleep(.10)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
