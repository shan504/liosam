#!/usr/bin/env python3
"""GO2 native odometry only. The Humble reader and Noetic publisher use separate processes."""
from collections import defaultdict
import argparse
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import time

RAW = '/utlidar/robot_odom'
OUTPUT = '/go2_native/odometry'
WIRE = 'GO2ODOM1 '


def xyz(v):
    return [v.x, v.y, v.z]


def read_sample(msg, stamp, publisher):
    # Clock readings must be taken on the receiving host before stdout buffering.
    before = time.monotonic_ns()
    now = time.time_ns()
    after = time.monotonic_ns()
    pose, twist = msg.pose.pose, msg.twist.twist
    return dict(stamp=stamp, now=now, before=before, after=after, publisher=publisher,
                parent=msg.header.frame_id, child=msg.child_frame_id,
                p=xyz(pose.position), q=xyz(pose.orientation) + [pose.orientation.w],
                linear=xyz(twist.linear), angular=xyz(twist.angular),
                pcov=list(msg.pose.covariance), tcov=list(msg.twist.covariance))


def validated(sample):
    if (sample['parent'].lstrip('/'), sample['child'].lstrip('/')) != ('odom', 'base_link'):
        raise ValueError('需要 Go2 原生帧 odom -> base_link，收到 %s -> %s' %
                         (sample['parent'], sample['child']))
    for key, size in [('p',3),('q',4),('linear',3),('angular',3),('pcov',36),('tcov',36)]:
        data = sample[key]
        if len(data) != size or not all(math.isfinite(value) for value in data):
            raise ValueError('非法原生里程计字段 ' + key)
    norm = math.sqrt(sum(value*value for value in sample['q']))
    if abs(norm - 1.) > .01:
        raise ValueError('原生姿态四元数无效')
    return [v / norm for v in sample['q']]


def read_dds():
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    rclpy.init()
    node = rclpy.create_node('go2_native_odom_reader')

    def accept_dds(topic, msg):
        if topic == '/odom':
            infos = node.get_publishers_info_by_topic(topic)
            names = [i.node_namespace.rstrip('/') + '/' + i.node_name for i in infos]
            if names != ['/go2_edu_dds_bridge']:
                return
        elif node.count_publishers(topic) > 1:
            node.get_logger().error('原生话题有多个发布者，暂停接收', throttle_duration_sec=5.)
            return
        stamp = msg.header.stamp
        publisher = 'DDS ' + topic
        if topic == '/odom':
            publisher += ' /go2_edu_dds_bridge'
        sample = read_sample(msg, stamp.sec * 10**9 + stamp.nanosec, publisher)
        sample['mapped_by_native_bridge'] = topic == '/odom'
        print(WIRE + json.dumps(sample, separators=(',', ':')), flush=True)

    node.create_subscription(Odometry, RAW, lambda msg: accept_dds(RAW, msg), qos_profile_sensor_data)
    node.create_subscription(Odometry, '/odom', lambda msg: accept_dds('/odom', msg), qos_profile_sensor_data)
    try:
        from native_mount import install_dds_reader
        install_dds_reader(node)
    except ImportError as error:
        node.get_logger().warning('无法读取 DDS 静态外参: %s' % error)
    try:
        # The host owns this lease. A stopped supervisor can revoke it even
        # after its sudo credential has expired or the input is silent.
        while rclpy.ok() and Path('/run/go2_native/running').exists():
            rclpy.spin_once(node, timeout_sec=.5)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def subscribe_native_ros1(rospy, Odometry, label, deliver):
    def receive(msg, topic):
        source = msg._connection_header or {}
        name = source.get('callerid', '')
        if topic == '/odom' and name != '/go2_edu_dds_bridge':
            rospy.logwarn_throttle(10., '忽略非原 Go2 桥的 /odom，发布者 ' + name)
            return
        if not name or source.get('latching') == '1' or 'lio_sam' in name or name == rospy.get_name():
            rospy.logerr_throttle(5., '拒绝发布者 ' + repr(name))
            return
        sample = read_sample(msg, msg.header.stamp.to_nsec(), label + ' ' + topic + ' ' + name)
        sample['mapped_by_native_bridge'] = topic == '/odom'
        deliver(sample)

    return [rospy.Subscriber(topic, Odometry, receive, callback_args=topic,
                             queue_size=1, tcp_nodelay=True) for topic in (RAW, '/odom')]


def read_original_ros1():
    import rospy
    from nav_msgs.msg import Odometry
    from native_mount import wire_statics
    from tf2_msgs.msg import TFMessage
    from threading import Lock
    rospy.init_node('go2_native_source_reader', anonymous=True)
    if rospy.get_param('/use_sim_time', False):
        rospy.logwarn('跳过使用仿真时钟的 ROS1 master 11311')
        return
    static_edges = {}
    write_lock = Lock()
    def send(sample):
        with write_lock:
            print(WIRE + json.dumps(sample, separators=(',', ':')), flush=True)
    def receive_static(msg):
        with write_lock:
            line = wire_statics(msg, 'ros1_11311', static_edges)
            if line is not None:
                print(line, flush=True)
    subs = subscribe_native_ros1(rospy, Odometry, 'ROS1:11311', send)
    subs.append(rospy.Subscriber('/tf_static', TFMessage, receive_static, queue_size=16))
    rospy.spin()


def publish_ros1(mode):
    import rospy
    from nav_msgs.msg import Odometry
    from native_sensor_clock import SensorClock
    import threading
    rospy.init_node('go2_native_odom_bridge')
    if rospy.get_param('/use_sim_time', False):
        raise RuntimeError('需要实机 ROS 时钟')
    clocks = defaultdict(SensorClock)
    pub = rospy.Publisher(OUTPUT, Odometry, queue_size=4, tcp_nodelay=True)
    active_publisher = None
    last_received = None
    lock = threading.Lock()

    def convert(sample):
        nonlocal active_publisher, last_received
        with lock:
            try:
                q = validated(sample)
            except (ValueError, TypeError, KeyError) as e:
                rospy.logwarn_throttle(5., '忽略无效样本: %s' % e)
                return
            if active_publisher is not None and active_publisher != sample['publisher']:
                # The same robot input can also be present via its other bridge.
                return
            clock = clocks[sample['publisher']]
            if sample.get('mapped_by_native_bridge'):
                # The original Go2 bridge already mapped these stamps to host time.
                mapped = sample['stamp']
                if mapped <= 0 or (clock.last_odom_ns is not None and mapped <= clock.last_odom_ns):
                    return
            else:
                if not clock.observe_odom_sample(sample['stamp'], sample['now'], sample['before'], sample['after']):
                    status = clock.status()
                    rospy.logwarn_throttle(5., 'Go2 原生时钟: %s' % status['last_rejection_reason'])
                    return
                mapped = clock.convert(sample['stamp'])
            age = time.time_ns() - mapped
            if age < -clock.MAX_FUTURE_NS or age > clock.MAX_AGE_NS:
                rospy.logwarn_throttle(5., '丢弃传输过期的 Go2 位姿')
                return
            if sample.get('mapped_by_native_bridge'):
                clock.last_odom_ns = mapped
            m = Odometry()
            m.header.stamp = rospy.Time(*divmod(mapped, 10**9))
            m.header.frame_id = 'go2_native/odom'
            m.child_frame_id = 'go2_native/base_link'
            p, r = m.pose.pose.position, m.pose.pose.orientation
            p.x, p.y, p.z = sample['p']
            r.x, r.y, r.z, r.w = q
            linear, angular = m.twist.twist.linear, m.twist.twist.angular
            linear.x, linear.y, linear.z = sample['linear']
            angular.x, angular.y, angular.z = sample['angular']
            m.pose.covariance = sample['pcov']
            m.twist.covariance = sample['tcov']
            rospy.loginfo_once('已接通 Go2 原生里程计 (%s) -> %s', sample['publisher'], OUTPUT)
            active_publisher = sample['publisher']
            last_received = time.monotonic()
            pub.publish(m)

    from native_mount import PREFIX as MOUNT_LINE, relay as relay_mount
    mount_publishers = {}
    def pipe_reader(stream, label):
        try:
            for line in stream:
                if rospy.is_shutdown():
                    return
                if line.startswith(MOUNT_LINE):
                    relay_mount(line, mount_publishers)
                elif line.startswith(WIRE):
                    convert(json.loads(line[len(WIRE):]))
                else:
                    print(line.rstrip(), file=sys.stderr, flush=True)
        finally:
            if not rospy.is_shutdown():
                rospy.logwarn('%s 原生输入已结束', label)

    def report_wait(_event):
        with lock:
            if last_received is None:
                rospy.logwarn('正在等待 Go2 原生里程计有效样本，输出 %s 暂无数据', OUTPUT)
            elif time.monotonic() - last_received > 2.:
                rospy.logwarn('Go2 原生里程计中断 %.1f 秒，等待来源 %s',
                              time.monotonic() - last_received, active_publisher)

    rospy.loginfo('等待原生里程计 %s，输入模式 %s', RAW, mode)
    rospy.Timer(rospy.Duration(10.), report_wait)
    subscriptions = []
    if mode != 'pipe':
        subscriptions = subscribe_native_ros1(rospy, Odometry, 'ROS1:11321', convert)
        # The original Go2 bridge may still use the default Noetic master.
        # It must be read in a distinct process; rospy supports one master per process.
        def original_reader():
            import socket
            while not rospy.is_shutdown():
                try:
                    with socket.create_connection(('127.0.0.1', 11311), timeout=1.):
                        break
                except OSError:
                    time.sleep(2.)
            if rospy.is_shutdown():
                return
            env = dict(os.environ, ROS_MASTER_URI='http://127.0.0.1:11311')
            with subprocess.Popen([sys.executable, '-u', __file__, 'read-ros1'],
                                  env=env, stdout=subprocess.PIPE, text=True) as proc:
                def stop():
                    if proc.poll() is None:
                        proc.terminate()
                rospy.on_shutdown(stop)
                pipe_reader(proc.stdout, 'ROS1:11311')
        threading.Thread(target=original_reader, daemon=True).start()
    if mode != 'ros1':
        threading.Thread(target=pipe_reader, args=(sys.stdin, 'ROS2 DDS'), daemon=True).start()
    rospy.spin()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('dds', 'pipe', 'ros1', 'auto', 'read-ros1'))
    args = parser.parse_args()
    if args.mode == 'dds':
        read_dds()
    elif args.mode == 'read-ros1':
        read_original_ros1()
    else:
        publish_ros1(args.mode)
