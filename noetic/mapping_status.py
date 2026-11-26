#!/usr/bin/env python3
"""Supply the three existing Foxglove status panels during a Go2 mapping run."""
import fcntl
import json
import math
import sys
from pathlib import Path
from threading import Lock
import time

VALID = '/localization/valid'
LOCALIZATION = '/localization/status'
OPERATOR = '/operator/status'
NATIVE = '/go2_native/odometry'
MAPPING = '/lio_sam/mapping/odometry'
SPORT = '/lf/sportmodestate'
INPUT = '/go2_visual/status_input'
FRESH = 2.0


class PanelState:
    def __init__(self):
        self.poses = {}
        self.reports = {}

    def pose(self, topic, msg, now, wall):
        stamp = msg.header.stamp.to_sec()
        position = msg.pose.pose.position
        if not (stamp > 0 and -0.25 <= wall - stamp <= FRESH and
                all(math.isfinite(v) for v in (position.x, position.y, position.z))):
            return
        previous = self.poses.get(topic)
        if previous and stamp <= previous[1] and now - previous[0] <= FRESH:
            return
        self.poses[topic] = (now, stamp, msg)

    def report(self, source, topic, data, received, now):
        if source not in ('dds', 'ros1_11311') or not (0 <= received <= now):
            return
        if topic == VALID:
            if type(data) is not bool:
                return
        elif topic in (LOCALIZATION, OPERATOR):
            if not isinstance(data, str) or not data.strip():
                return
        elif topic == SPORT:
            if not isinstance(data, dict) or not all(type(data.get(key)) is int for key in
                                                    ('mode', 'gait_type', 'error_code')):
                return
        else:
            return
        old = self.reports.get((topic, source))
        if old is None or received > old[0]:
            self.reports[topic, source] = received, data

    def recent_reports(self, topic, now, max_age):
        return [(stamp, source, data) for (name, source), (stamp, data) in self.reports.items()
                if name == topic and 0 <= now - stamp <= max_age]

    def current_pose(self, topic, now, wall):
        row = self.poses.get(topic)
        if row and 0 <= now - row[0] <= FRESH and -0.25 <= wall - row[1] <= FRESH:
            return row[2]
        return None

    def pose_text(self, topic, msg):
        if msg is None:
            return '数据超时' if topic in self.poses else '等待数据'
        p = msg.pose.pose.position
        return 'xyz=(%.2f, %.2f, %.2f) m [%s]' % (p.x, p.y, p.z, msg.header.frame_id)

    def panels(self, now, wall):
        # In the original Go2 stack this Bool gates navigation in a prior map.
        # Only a fresh, explicit upstream valid report can assert it.
        flags = self.recent_reports(VALID, now, .4)
        valid = bool(flags) and all(data for _, _, data in flags)
        native = self.current_pose(NATIVE, now, wall)
        mapping = self.current_pose(MAPPING, now, wall)
        loc = ['建图', '本体 ' + self.pose_text(NATIVE, native),
               'LIO ' + self.pose_text(MAPPING, mapping)]
        if flags:
            loc.append('先验地图定位报告=' + str(valid).lower())
        else:
            loc.append('未收到有效期内的先验地图定位报告')
        reports = self.recent_reports(LOCALIZATION, now, FRESH)
        if reports:
            _, source, message = max(reports)
            loc.append('%s: %s' % (source, message))

        operator = []
        reports = self.recent_reports(OPERATOR, now, float('inf'))
        if reports:
            stamp, source, message = max(reports)
            operator.append('最近操作报告 [%s, 收到于 %.1f 秒前] %s' % (source, now - stamp, message))
        else:
            operator.append('未收到操作节点报告')
        sports = self.recent_reports(SPORT, now, FRESH)
        if sports:
            _, _, sport = max(sports)
            operator.append('本体 mode={mode} gait_type={gait_type} error_code={error_code}'.format(**sport))
        if native is None:
            operator.append('本体速度' + self.pose_text(NATIVE, None))
        else:
            twist = native.twist.twist
            values = (twist.linear.x, twist.linear.y, twist.angular.z)
            if all(math.isfinite(v) for v in values):
                operator.append('本体里程计 vx=%.2f vy=%.2f m/s，wz=%.2f rad/s' % values)
            else:
                operator.append('本体速度无效')
        return {VALID: valid, LOCALIZATION: ' | '.join(loc), OPERATOR: ' | '.join(operator)}


def run(attached):
    import rospy
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Bool, String

    rospy.init_node('go2_mapping_status')
    state = PanelState()
    lock = Lock()

    def odom(msg, topic):
        with lock:
            state.pose(topic, msg, time.monotonic(), rospy.Time.now().to_sec())

    def remote(msg):
        try:
            entry = json.loads(msg.data)
            with lock:
                state.report(entry['source'], entry['topic'], entry['data'], entry['received'],
                             time.monotonic())
        except (KeyError, TypeError, ValueError) as exc:
            rospy.logwarn_throttle(10, '状态消息格式无效: %s', exc)

    subscriptions = [rospy.Subscriber(topic, Odometry, odom, callback_args=topic, queue_size=1)
                     for topic in (NATIVE, MAPPING)]
    subscriptions.append(rospy.Subscriber(INPUT, String, remote, queue_size=32))
    types = {VALID: Bool, LOCALIZATION: String, OPERATOR: String}
    publishers = {}
    master = rospy.get_master()
    if attached:
        code, message, original_bridge = master.lookupNode('/go2_native_odom_bridge')
        if code != 1:
            raise SystemExit('请先启动 --go2 建图。当前没有运行本体里程计桥。')
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        code, message, graph = master.getSystemState()
        if code != 1:
            raise RuntimeError(message)
        if attached:
            # A foreground attachment belongs only to this already-running mapping session.
            code, _, current_bridge = master.lookupNode('/go2_native_odom_bridge')
            if code != 1 or current_bridge != original_bridge:
                rospy.loginfo('本次建图的本体桥已退出，结束状态显示。')
                return
        registered = dict(graph[0])
        with lock:
            values = state.panels(time.monotonic(), rospy.Time.now().to_sec())
        for topic, msg_type in types.items():
            # Native ROS1 publishers own their topic, including after a late startup.
            if any(node != rospy.get_name() for node in registered.get(topic, ())):
                if topic in publishers:
                    publishers.pop(topic).unregister()
                continue
            if topic not in publishers:
                publishers[topic] = rospy.Publisher(topic, msg_type, queue_size=1)
            publishers[topic].publish(msg_type(data=values[topic]))
        rate.sleep()


if __name__ == '__main__':
    lock_path = Path.home()/'.ros/go2_mapping_status_11321.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as instance:
        try:
            fcntl.flock(instance, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('三个状态面板已由现有状态节点供应。')
        else:
            run(attached='--attach' in sys.argv[1:])
