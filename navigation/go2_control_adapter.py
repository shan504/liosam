#!/usr/bin/env python3
"""Connect the original Go2 motion gate to host-timestamped Go2 odometry and MID-360S."""
import math
from pathlib import Path
import sys
import time


class UpstreamTimestamps:
    """The dedicated sensor node already puts both input headers on ROS host time."""

    def __init__(self):
        self.last = {}

    def status(self):
        return {'state': 'host_stamped_inputs', 'odom': '/odom', 'cloud': '/point_cloud2'}

    def check(self, stream, stamp, now):
        if stamp <= 0:
            return 'invalid_timestamp'
        if stamp <= self.last.get(stream, 0):
            return 'duplicate_or_reordered'
        if stamp - now > 100_000_000:
            return 'sensor_time_in_future'
        if now - stamp >= 1_000_000_000:
            return 'sensor_time_stale'
        self.last[stream] = stamp
        return None


class MidSensorInputs:
    """Sensor hook for LocalizedBridge. All command handling belongs to the original bridge."""

    def setup_sensor_relay(self):
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2

        self.clock_map = UpstreamTimestamps()
        self.create_subscription(Odometry, '/odom', self.accept_body_odom, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/point_cloud2', self.accept_mid_cloud, qos_profile_sensor_data)
        self.get_logger().info('运动桥使用 /odom 本体里程计和 /point_cloud2 机身坐标系 MID-360S 回波')

    def check_input(self, stream, msg, reason):
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        now = self.get_clock().now().nanoseconds
        if reason is None:
            reason = self.clock_map.check(stream, stamp, now)
        self.record_sensor_timing(stream, stamp, stamp, now, reason is None, reason)
        if reason is None:
            # A delayed but admissible message only gets its remaining freshness window.
            return time.monotonic() - max(0, now - stamp) / 1e9
        return None

    def accept_body_odom(self, msg):
        reason = None
        position, orientation = msg.pose.pose.position, msg.pose.pose.orientation
        quat = (orientation.x, orientation.y, orientation.z, orientation.w)
        if msg.header.frame_id != 'odom' or msg.child_frame_id != 'base_link':
            reason = 'unexpected_odometry_frame'
        elif not (all(math.isfinite(n) for n in (position.x, position.y, position.z))
                  and .99 <= math.hypot(*quat) <= 1.01):
            reason = 'invalid_odometry_pose'
        at = self.check_input('odom', msg, reason)
        if at is not None:
            self.odom_received = at
            if self.first_odom_stamp_ns is None:
                self.first_odom_stamp_ns = self.clock_map.last['odom']

    def accept_mid_cloud(self, msg):
        reason = None
        if msg.header.frame_id != 'base_link':
            reason = 'unexpected_cloud_frame'
        elif (msg.height < 1 or msg.width < 1 or msg.point_step < 12
              or msg.row_step < msg.width * msg.point_step or len(msg.data) != msg.row_step * msg.height):
            reason = 'empty_or_incomplete_cloud'
        else:
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            if self.first_odom_stamp_ns is None or stamp < self.first_odom_stamp_ns:
                reason = 'before_odom_history'
        at = self.check_input('cloud', msg, reason)
        if at is not None:
            self.cloud_received = at


def main():
    source = Path(__file__).resolve().parent / 'original_go2'
    sys.path[:0] = [str(source / 'go2_3d'), str(source / 'patches/dds')]
    from control_bridge import LocalizedBridge, main as run_bridge

    class MidControlBridge(MidSensorInputs, LocalizedBridge):
        pass

    run_bridge(MidControlBridge)


if __name__ == '__main__':
    main()
