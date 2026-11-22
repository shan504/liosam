#!/usr/bin/env python3
"""Compare available native odometry streams while the physical robot is still."""
import math
import time


def main():
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    rclpy.init()
    node = rclpy.create_node('go2_native_readonly_check')
    seen, subscriptions = {}, {}
    print('保持四脚静止。读取 ROS2 原生里程计约12秒；不启动控制节点，不录包。', flush=True)
    print('若没有消息，只代表当前 DDS 环境下未收到，不能据此断定原生里程计不存在。')

    def collect(message, topic):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        position = (p.x, p.y, p.z)
        quat = (q.x, q.y, q.z, q.w)
        if not all(math.isfinite(v) for v in position + quat):
            return
        stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
        row = seen.setdefault(topic, {'first': position, 'last': position, 'stamp': None,
                                     'n': 0, 'travel': 0., 'step': 0., 'order': 0, 'frames': set(),
                                     'ages': []})
        row['n'] += 1
        row['frames'].add((message.header.frame_id, message.child_frame_id))
        row['ages'].append((node.get_clock().now().nanoseconds-stamp)*1e-9)
        if row['stamp'] is not None:
            if stamp <= row['stamp']:
                row['order'] += 1
            row['step'] = max(row['step'], math.sqrt(sum((x-y)**2 for x,y in zip(position,row['last']))))
        row['travel'] = max(row['travel'], math.sqrt(sum((x-y)**2 for x,y in zip(position,row['first']))))
        row['last'], row['stamp'] = position, stamp

    begin, discover = time.monotonic(), 0.
    try:
        while time.monotonic()-begin < 12:
            if time.monotonic() >= discover:
                discover = time.monotonic()+1
                for topic, types in node.get_topic_names_and_types():
                    if 'nav_msgs/msg/Odometry' in types and topic not in subscriptions:
                        subscriptions[topic] = node.create_subscription(
                            Odometry, topic, lambda m,t=topic: collect(m,t), qos_profile_sensor_data)
            rclpy.spin_once(node, timeout_sec=.05)
        for topic in sorted(subscriptions):
            row = seen.get(topic)
            print('\n话题', topic, '发布者', node.count_publishers(topic))
            for endpoint in node.get_publishers_info_by_topic(topic):
                print('  节点', endpoint.node_namespace.rstrip('/')+'/'+endpoint.node_name)
            if row is None:
                print('  未收到消息'); continue
            print('  帧', sorted(row['frames']), '消息', row['n'], '非递增', row['order'])
            print('  静止窗口相对首帧最大位移 %.4fm，相邻最大 %.4fm' % (row['travel'],row['step']))
            print('  ROS时钟减源时间戳范围 %.3f..%.3fs（未做时钟转换）' % (min(row['ages']),max(row['ages'])))
        if not subscriptions:
            print('当前域未发现 nav_msgs/msg/Odometry。需核对 DDS 域、网卡配置及原生发布节点。')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
