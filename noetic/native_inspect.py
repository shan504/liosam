#!/usr/bin/env python3
"""Inspect one live ROS graph without injecting poses, frames or control."""
from collections import defaultdict
from functools import partial
import os
import struct
import sys
import threading
import time
from native_inspect_tf import StaticTF

CANDIDATES = ('/utlidar/robot_odom', '/odom')
WINDOW = 7.


class Observations:
    def __init__(self):
        self.rows = defaultdict(lambda: dict(n=0, times=set(), frames=set(), source=set(),
                                             current=False))
        self.lock = threading.Lock()
        self.tf = StaticTF()
        self.cameras = defaultdict(set)
        self.bodies = set()
        self.livox_frames = set()

    def observe(self, topic, kind, msg):
        header = msg.header
        if kind == 'camera':
            with self.lock:
                self.cameras[topic].add(header.frame_id)
            return
        stamp = header.stamp
        with self.lock:
            row = self.rows[topic]
            row['n'] += 1
            row['times'].add(stamp.to_nsec() if hasattr(stamp, 'to_nsec')
                             else stamp.sec * 10**9 + stamp.nanosec)
            row['frames'].add((header.frame_id, msg.child_frame_id))
            connection = getattr(msg, '_connection_header', {})
            row['current'] |= connection.get('latching', '0') != '1'
            if connection.get('callerid'):
                row['source'].add(connection['callerid'])
            if topic in CANDIDATES:
                self.bodies.add(msg.child_frame_id)

    def report(self, graph):
        with self.lock:
            for topic in sorted(graph):
                kind, type_name, nodes = graph[topic]
                if kind == 'tf':
                    print('%s 发布者=%s' % (topic, nodes or '无'))
                else:
                    print('%s [%s] 发布者=%s' % (topic, type_name, nodes or '无'))
                if kind == 'odom':
                    row = self.rows[topic]
                    stamps = sorted(row['times'] - {0})
                    print('  窗口到达 %d 条；不同非零源戳 %d；首末源戳差 %.3f s；帧=%s%s' %
                          (row['n'], len(stamps), (stamps[-1]-stamps[0])/1e9 if stamps else 0.,
                           sorted(row['frames']), '；实收连接='+str(sorted(row['source']))
                           if row['source'] else ''))
                    if row['n'] and not row['current']:
                        print('  仅收到 ROS1 锁存消息')
                elif kind == 'camera':
                    print('  实收 CameraInfo 帧=%s' % sorted(self.cameras[topic]))
                elif kind == 'state':
                    print('  机身状态话题；此项仅查发布者')
            bodies = self.bodies.copy()
            cameras = set().union(*self.cameras.values())
            lidar_frames = self.livox_frames.copy()
        if lidar_frames:
            print("MID-360 实收 /livox/lidar 坐标系 %s" % sorted(lidar_frames))
        self.tf.report(bodies, lidar_frames, cameras)


def classify(topic, type_name):
    kind = type_name.replace('/msg/', '/')
    if kind == 'nav_msgs/Odometry':
        return 'odom'
    if kind == 'sensor_msgs/CameraInfo':
        return 'camera'
    if topic == '/tf_static' and kind == 'tf2_msgs/TFMessage':
        return 'tf'
    if 'sportmodestate' in kind.lower() or 'sportmodestate' in topic.lower():
        return 'state'


def ros1():
    import rospy
    import rosgraph
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CameraInfo
    from tf2_msgs.msg import TFMessage
    master = rosgraph.Master('/go2_native_audit')
    try:
        master.getPid()
    except Exception as exc:
        sys.exit('ROS1 master 不可达：' + str(exc))
    rospy.init_node('go2_native_audit', anonymous=True)
    state = Observations()
    subs, graph = {}, {}
    def raw_livox(msg):
        # Read just the common Header from the serialized driver message.
        buf = msg._buff
        if len(buf) >= 16:
            n, = struct.unpack_from('<I', buf, 12)
            if 0 < n <= len(buf)-16:
                with state.lock:
                    state.livox_frames.add(buf[16:16+n].decode('utf-8', 'replace'))
    types = {'odom': Odometry, 'camera': CameraInfo, 'tf': TFMessage}
    end = time.monotonic() + WINDOW
    while not rospy.is_shutdown() and time.monotonic() < end:
        publishers = dict(master.getSystemState()[0])
        topic_types = dict(master.getTopicTypes())
        if '/livox/lidar' not in subs and topic_types.get('/livox/lidar') in (
                'livox_ros_driver2/CustomMsg', 'sensor_msgs/PointCloud2'):
            subs['/livox/lidar'] = rospy.Subscriber('/livox/lidar', rospy.AnyMsg, raw_livox,
                                                     queue_size=1, buff_size=1024*1024)
        if '/livox/lidar' in subs and subs['/livox/lidar'] is not None and state.livox_frames:
            subs['/livox/lidar'].unregister()
            subs['/livox/lidar'] = None
        for topic in CANDIDATES:
            topic_types.setdefault(topic, 'nav_msgs/Odometry')
        topic_types.setdefault('/tf_static', 'tf2_msgs/TFMessage')
        for topic, type_name in topic_types.items():
            kind = classify(topic, type_name)
            if not kind:
                continue
            graph[topic] = kind, type_name, publishers.get(topic, [])
            if topic not in subs and kind in types:
                callback = state.tf.add if kind == 'tf' else partial(state.observe, topic, kind)
                subs[topic] = rospy.Subscriber(topic, types[kind], callback, queue_size=30)
        time.sleep(.4)
    for sub in subs.values():
        if sub is not None:
            sub.unregister()
    state.report(graph)


def ros2():
    import rclpy
    from rclpy.qos import (QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy,
                           qos_profile_sensor_data)
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CameraInfo
    from tf2_msgs.msg import TFMessage
    rclpy.init()
    node = rclpy.create_node('go2_native_audit_' + str(os.getpid()) + '_' + str(time.monotonic_ns()))
    state = Observations()
    static_qos = QoSProfile(depth=64, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                            reliability=QoSReliabilityPolicy.BEST_EFFORT)
    types = {'odom': Odometry, 'camera': CameraInfo, 'tf': TFMessage}
    subs, graph = {}, {}
    end = time.monotonic()+WINDOW
    while rclpy.ok() and time.monotonic() < end:
        discovered = dict(node.get_topic_names_and_types())
        for topic in CANDIDATES:
            discovered.setdefault(topic, ['nav_msgs/msg/Odometry'])
        discovered.setdefault('/tf_static', ['tf2_msgs/msg/TFMessage'])
        for topic, options in discovered.items():
            for type_name in options:
                kind = classify(topic, type_name)
                if not kind:
                    continue
                infos = node.get_publishers_info_by_topic(topic)
                names = [i.node_namespace.rstrip('/') + '/' + i.node_name for i in infos
                         if i.topic_type == type_name]
                graph[topic] = kind, type_name, names
                if topic not in subs and kind in types:
                    callback = state.tf.add if kind == 'tf' else partial(state.observe, topic, kind)
                    subs[topic] = node.create_subscription(types[kind], topic, callback,
                                      static_qos if kind == 'tf' else qos_profile_sensor_data)
        until = min(end, time.monotonic() + .5)
        while rclpy.ok() and time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.1)
    state.report(graph)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    {'ros1': ros1, 'ros2': ros2}[sys.argv[1]]()
