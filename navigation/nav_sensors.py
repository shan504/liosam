#!/usr/bin/env python3
"""Time-align native Go2 odometry and locally transported Noetic MID-360S scans for Nav2."""
import argparse
from array import array
from collections import deque
import json
import math
import os
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'noetic'))
from check_returns import wire_points
from dense_map import transform
from native_cloud import calibration, compose_rotation, interpolate, place
from native_odom import read_sample, validated
from native_sensor_clock import SensorClock
from prepare_map import is_floor

RAW_ODOM = '/utlidar/robot_odom'
SCAN_FRAME = 'go2_nav_scan'
TWO_PI = 2 * math.pi
RANGE_MIN = .12
RANGE_MAX = 10.
BINS = 360


def mount_config(map_dir, override=None):
    options = [override] if override is not None else [map_dir / 'live/body_from_lidar.json',
                                                      map_dir / 'calibration/body_from_lidar.json']
    for path in options:
        if not path.is_file():
            continue
        doc = json.loads(path.read_text())
        if 'transform' in doc:  # Archived, frame-labelled live TransformStamped.
            parent, child = doc['header']['frame_id'], doc['child_frame_id']
            position, quat = doc['transform']['translation'], doc['transform']['rotation']
            xyz, xyzw = [position[k] for k in 'xyz'], [quat[k] for k in 'xyzw']
        else:                   # Original mount_setup.py calibration document.
            parent, child = doc['parent'], doc['child']
            xyz, xyzw = doc['xyz'], doc['xyzw']
        parent, child = parent.lstrip('/'), child.lstrip('/')
        if parent != 'base_link' or child in ('', 'map', 'odom', 'base_link', SCAN_FRAME):
            raise ValueError('需要 base_link <- 实际雷达帧，收到 %s <- %s' % (parent, child))
        return child, calibration(xyz, xyzw), path
    raise ValueError('地图目录中缺少雷达安装外参，请提供实测 --mount 文件')


def yaw_only(q):
    x, y, z, w = q
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return np.array((0., 0., math.sin(.5*yaw), math.cos(.5*yaw)))


def scan_geometry(world, body_pose, lidar_pose, body_height, band=(.085, .85)):
    # Returns selected in a horizontal obstacle slab around the standing robot.
    # The actual emitter origin is retained even when its mount pitches down.
    body = transform(world, body_pose, inverse=True)
    self_hit = ((abs(body[:, 0]) < .37) & (abs(body[:, 1]) < .17) &
                (body[:, 2] < .15) & (body[:, 2] > -.35))
    world = world[~self_hit]
    lidar = transform(world, lidar_pose, inverse=True)
    q_level = yaw_only(body_pose[3:])
    level = transform(world, np.r_[lidar_pose[:3], q_level], inverse=True)
    ground_z = body_pose[2] - body_height
    heights = world[:, 2] - ground_z
    obstacle = (heights >= band[0]) & (heights <= band[1])
    x, y = level[obstacle, 0], level[obstacle, 1]
    d = np.hypot(x, y)
    ok = (d >= RANGE_MIN) & (d <= RANGE_MAX)
    # Nearest beam center, wrapping consistently at +/- pi.
    bins = np.rint((np.arctan2(y[ok], x[ok]) + math.pi) * (BINS / TWO_PI)).astype(int) % BINS
    ranges = np.full(BINS, np.inf, d.dtype)
    np.minimum.at(ranges, bins, d[ok])
    # Retain a real XYZ return for each observed ray, preserving the physical sensor origin.
    # A farther return on that ray must not clear space behind the nearer obstacle.
    nearest = d[ok] == ranges[bins]
    _, first = np.unique(bins[nearest], return_index=True)
    obstacle_rows = np.flatnonzero(obstacle)[ok][nearest][first]
    ranges[~np.isfinite(ranges)] = np.nan  # Missing returns cannot assert clear space.
    inverse_body = body_pose[3:] * np.array([-1, -1, -1, 1])
    return (lidar, ~is_floor(heights), lidar[obstacle_rows], ranges.astype(np.float32, copy=False),
            compose_rotation(inverse_body, q_level))


def stamp_ns(stamp):
    return stamp.sec * 10**9 + stamp.nanosec


def main():
    import rclpy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data as sensor_qos
    from sensor_msgs.msg import LaserScan, PointCloud2, PointField
    from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map-dir', required=True, type=Path)
    parser.add_argument('--scan-dir', required=True, type=Path)
    parser.add_argument('--mount', type=Path, help='current physical mount, if remounted after survey')
    parser.add_argument('--geometry', type=Path, help='prepared map metadata with calibrated height band')
    parser.add_argument('--body-height', type=float,
                        help='standing body origin height in metres, used for the obstacle height band')
    args, ros_args = parser.parse_known_args()
    body_height, band = .327, (.085, .85)
    if args.geometry:
        metadata = json.loads(args.geometry.read_text())
        body_height = float(metadata['initial_base_pose'][2])
        band = tuple(map(float, metadata['obstacle_band_m']))
    if args.body_height is not None:
        body_height = args.body_height
    if not .05 <= body_height <= .65 or len(band) != 2 or not 0. <= band[0] < band[1] <= 2.:
        parser.error('地图中站立机身高度或障碍物高度带无效，请核对保存时的地面与外参')
    sensor_frame, mount, mount_path = mount_config(args.map_dir, args.mount)
    file = args.scan_dir / 'scan.bin'

    def edge(stamp, parent, child, xyz, q):
        msg = TransformStamped()
        msg.header.stamp, msg.header.frame_id, msg.child_frame_id = stamp, parent, child
        p, r = msg.transform.translation, msg.transform.rotation
        p.x, p.y, p.z = map(float, xyz)
        r.x, r.y, r.z, r.w = map(float, q)
        return msg

    class Sensors(Node):
        def __init__(self):
            super().__init__('go2_navigation_sensors')
            self.odom = self.create_publisher(Odometry, '/odom', 10)
            self.cloud = self.create_publisher(PointCloud2, '/navigation/cloud', sensor_qos)
            self.display_cloud = self.create_publisher(PointCloud2, '/navigation/cloud_filtered', sensor_qos)
            self.obstacles = self.create_publisher(PointCloud2, '/navigation/obstacles', 5)
            self.scan = self.create_publisher(LaserScan, '/navigation/scan', sensor_qos)
            self.tf = TransformBroadcaster(self)
            self.static = StaticTransformBroadcaster(self)
            self.static.sendTransform(edge(self.get_clock().now().to_msg(), 'base_link', sensor_frame,
                                           mount[:3], mount[3:]))
            self.fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1)
                           for i,n in enumerate(('x', 'y', 'z', 'intensity'))]
            self.clock = SensorClock()
            self.samples = deque(maxlen=800)
            self.exclusive = False
            self.pending = None
            self.file_tag = None
            self.last_scan_ns = -1
            self.create_subscription(Odometry, RAW_ODOM, self.receive_odom, sensor_qos)
            self.create_timer(.5, self.check_publishers)
            self.create_timer(.025, self.publish_scan)
            self.get_logger().info('Go2 本体里程计与 MID-360S 导航输入已启动，安装外参 ' + str(mount_path))

        def check_publishers(self):
            raw = self.count_publishers(RAW_ODOM)
            output = self.get_publishers_info_by_topic('/odom')
            owner = (len(output) == 1 and output[0].node_name == self.get_name()
                     and output[0].node_namespace == self.get_namespace())
            self.exclusive = raw == 1 and owner
            if not self.exclusive:
                self.samples.clear()
                self.pending = None
                self.get_logger().warning('等待唯一的 %s 发布者与本节点独占 /odom；发布者数 %s/%s。'
                                          % (RAW_ODOM, raw, len(output)), throttle_duration_sec=5.)

        def receive_odom(self, msg):
            if not self.exclusive:
                return
            raw = stamp_ns(msg.header.stamp)
            sample = read_sample(msg, raw, RAW_ODOM)
            try:
                q = validated(sample)
            except (ValueError, TypeError) as exc:
                self.get_logger().warning(str(exc), throttle_duration_sec=2.)
                return
            if not self.clock.observe_odom_sample(raw, sample['now'], sample['before'], sample['after']):
                if self.clock.fault_reason:
                    self.get_logger().error('本体/主机时钟突变，请重新启动导航：' + self.clock.fault_reason,
                                            throttle_duration_sec=3.)
                return
            t = self.clock.convert(raw)
            if self.get_clock().now().nanoseconds - t > 350000000:
                return
            pose = np.asarray(sample['p'] + q)
            self.samples.append((t, pose))
            msg.header.stamp = Time(sec=t//10**9, nanosec=t%10**9)
            msg.header.frame_id, msg.child_frame_id = 'odom', 'base_link'
            self.tf.sendTransform(edge(msg.header.stamp, 'odom', 'base_link', pose[:3], pose[3:]))
            self.odom.publish(msg)

        def load_latest(self):
            try:
                with file.open('rb') as stream:
                    stat = os.fstat(stream.fileno())
                    tag = (stat.st_ino, stat.st_mtime_ns)
                    if self.file_tag == tag:
                        return None
                    self.file_tag = tag
                    if stat.st_size > 4*1024*1024:
                        raise ValueError('异常过大的原始扫描帧')
                    payload = stream.read()
            except FileNotFoundError:
                return None
            start, points = wire_points(payload)
            size = struct.unpack_from('<I', payload, 12)[0]
            frame = payload[16:16+size].decode('utf-8').lstrip('/')
            if frame != sensor_frame:
                raise ValueError('本帧雷达名 %s 与安装标定 %s 不一致' % (frame, sensor_frame))
            if start <= self.last_scan_ns or len(points) == 0:
                return None
            end = start + int(points['offset_time'].max())
            if end - start > 250000000:
                raise ValueError('单帧雷达时间范围异常')
            return start, end, points

        def publish_scan(self):
            if not self.exclusive or len(self.samples) < 2:
                return
            now = self.get_clock().now().nanoseconds
            try:
                if self.pending is None:
                    self.pending = self.load_latest()
                if self.pending is None:
                    return
                start, end, points = self.pending
                if now - start > 450000000 or start - now > 100000000 or self.samples[0][0] > start:
                    raise ValueError('雷达与本体时戳未对齐或当前扫描已过期')
                if self.samples[-1][0] < end:
                    return
                self.pending = None
                self.last_scan_ns = start
                history = list(self.samples)
                # Time-aligned interpolation and lever arm use the same routines as mapping.
                world, lidar_pose = place(start, points, history, mount, RANGE_MIN, 40.)
                pos, q = interpolate(history, np.array([start], dtype=np.int64))
                lidar, visible, obstacles, ranges, level_q = scan_geometry(
                    world, np.r_[pos[0], q[0]], lidar_pose, body_height, band)
                if not len(lidar):
                    raise ValueError('滤除机身回波后无有效点')
            except (OSError, ValueError, struct.error) as exc:
                self.pending = None
                self.get_logger().warning(str(exc), throttle_duration_sec=3.)
                return
            stamp = Time(sec=start//10**9, nanosec=start%10**9)
            self.tf.sendTransform(edge(stamp, 'base_link', SCAN_FRAME, mount[:3], level_q))
            self.publish_cloud(self.cloud, stamp, lidar)
            if self.display_cloud.get_subscription_count():
                self.publish_cloud(self.display_cloud, stamp, lidar[visible])
            if len(obstacles):
                self.publish_cloud(self.obstacles, stamp, obstacles)
            sc = LaserScan()
            sc.header.frame_id, sc.header.stamp = SCAN_FRAME, stamp
            sc.angle_min = -math.pi
            sc.angle_increment = TWO_PI / BINS
            sc.angle_max = sc.angle_min + (BINS-1)*sc.angle_increment
            sc.range_min, sc.range_max = RANGE_MIN, RANGE_MAX
            sc.scan_time = (end-start)/1e9
            sc.time_increment = 0.  # All points have been deskewed to start.
            beam_data = array('f')
            beam_data.frombytes(ranges.tobytes())
            sc.ranges = beam_data
            self.scan.publish(sc)

        def publish_cloud(self, publisher, stamp, points):
            pc = PointCloud2()
            pc.header.frame_id, pc.header.stamp = sensor_frame, stamp
            pc.height, pc.width, pc.fields = 1, len(points), self.fields
            pc.point_step, pc.row_step, pc.is_dense = 16, 16*len(points), True
            payload = array('B')
            payload.frombytes(points.astype('<f4', copy=False).tobytes())
            pc.data = payload
            publisher.publish(pc)

    rclpy.init(args=ros_args)
    node = Sensors()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
