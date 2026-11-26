#!/usr/bin/env python3
"""Append map-frame poses to the live mission queue in the given order."""
import argparse
import math
import sys
import time


def waypoint(value):
    try:
        x, y, heading = (float(v) for v in value.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError('使用 x,y,yaw_deg，例如 1.2,0.4,90')
    if not all(math.isfinite(v) for v in (x, y, heading)):
        raise argparse.ArgumentTypeError('坐标和朝向必须为有限数值')
    return x, y, heading


def main():
    parser = argparse.ArgumentParser(description='向正在运行的导航队列依次追加 map 坐标点（米、度）')
    parser.add_argument('goals', metavar='x,y,yaw_deg', type=waypoint, nargs='+')
    args = parser.parse_args()

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.duration import Duration

    rclpy.init(args=[])
    node = rclpy.create_node('go2_waypoint_submitter')
    publisher = node.create_publisher(PoseStamped, '/goal_pose', 100)
    try:
        deadline = time.monotonic() + 5
        while rclpy.ok():
            subscribers = node.get_subscriptions_info_by_topic('/goal_pose')
            if (publisher.get_subscription_count() and
                    any(e.node_name == 'go2_nav_mission' and e.node_namespace == '/'
                        for e in subscribers)):
                break
            if time.monotonic() > deadline:
                sys.exit('未发现导航队列节点 /go2_nav_mission，未发送目标。')
            rclpy.spin_once(node, timeout_sec=.1)
        sent = 0
        for x, y, heading in args.goals:
            if not rclpy.ok():
                sys.exit('ROS 已退出，已发送 %d 个点。' % sent)
            msg = PoseStamped()
            msg.header.frame_id = 'map'
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.pose.position.x, msg.pose.position.y = x, y
            radians = math.radians(heading % 360) / 2
            msg.pose.orientation.z, msg.pose.orientation.w = math.sin(radians), math.cos(radians)
            publisher.publish(msg)
            sent += 1
            if not publisher.wait_for_all_acked(Duration(seconds=3)):
                sys.exit('已发至第 %d 个点，DDS 确认超时。请先查看队列中的编号。' % sent)
            if sent < len(args.goals):
                time.sleep(.05)
        print('已发送 %d 个点，请在 /operator/status 和地图编号上核对入队结果。' % sent)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
