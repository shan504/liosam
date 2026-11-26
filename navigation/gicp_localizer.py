#!/usr/bin/env python3
"""Humble scan-to-map GICP anchored by native odometry at the MID-360S scan stamp."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
from prepare_map import quaternion, rigid

LASER_TOPIC = '/navigation/cloud'
MAX_SCAN_AGE = 1.0
MAX_ODOM_AGE = 0.35
CONFIRMATIONS = 3


def matrix(xyz, xyzw):
    q = np.asarray(xyzw, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(norm) or norm == 0:
        raise ValueError('invalid pose quaternion')
    return rigid(xyz, q / norm)


def vec(p):
    return [p.x, p.y, p.z]


def quat(q):
    return [q.x, q.y, q.z, q.w]


def checked_transform(array):
    value = np.array(array, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5)
            or abs(np.linalg.det(value[:3, :3]) - 1) > 1e-5):
        raise ValueError('invalid rigid transform')
    return value


def yaw(pose):
    return math.atan2(pose[1, 0], pose[0, 0])


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def planar(body):
    footprint = np.eye(4)
    footprint[:3, :3] = rotation_z(yaw(body))
    footprint[:2, 3] = body[:2, 3]
    return footprint


def difference(a, b):
    q = quaternion(a[:3, :3].T @ b[:3, :3])
    angle = 2 * math.atan2(np.linalg.norm(q[:3]), q[3])
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), math.degrees(angle)


def within(a, b, meters, degrees):
    dist, angle = difference(a, b)
    return dist <= meters and angle <= degrees


def blend_at_body(guess, match, weight):
    # Blend around the robot, not around an arbitrary (possibly faraway) native origin.
    result = guess.copy()
    # quaternion() returns w >= 0, giving the shorter arc across the yaw wrap.
    relative = quaternion(guess[:3, :3].T @ match[:3, :3])
    sine = np.linalg.norm(relative[:3])
    if sine > 0.:
        half_angle = weight * math.atan2(sine, relative[3])
        step = np.r_[relative[:3] / sine * math.sin(half_angle), math.cos(half_angle)]
        result[:3, :3] = guess[:3, :3] @ rigid([0., 0., 0.], step)[:3, :3]
    result[:3, 3] += weight * (match[:3, 3] - guess[:3, 3])
    return result


def cloud_points(msg):
    names = {f.name: f for f in msg.fields}
    endian = '>' if msg.is_bigendian else '<'
    coords = []
    if msg.width <= 0 or msg.height <= 0:
        raise ValueError('empty scan')
    for name in ('x', 'y', 'z'):
        f = names.get(name)
        if not f or f.datatype != 7 or f.count != 1 or f.offset + 4 > msg.point_step:
            raise ValueError('MID-360S cloud requires float32 x,y,z')
        coords.append(np.ndarray((msg.height, msg.width), dtype=endian+'f4', buffer=msg.data,
                                 offset=f.offset, strides=(msg.row_step, msg.point_step)).ravel())
    xyz = np.column_stack(coords)
    dist2 = np.einsum('ij,ij->i', xyz, xyz)
    return xyz[np.isfinite(xyz).all(axis=1) & (dist2 > 0.8**2) & (dist2 < 14.**2)]


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class Matcher:
    def __init__(self, directory):
        import open3d as o3d
        self.o3d = o3d
        self.reg = o3d.pipelines.registration
        points = np.load(directory / 'target.npy', mmap_mode='r', allow_pickle=False)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 100
                or not np.isfinite(points).all()):
            raise ValueError('target.npy must contain finite map-frame XYZ')
        self.prior = o3d.geometry.PointCloud()
        self.prior.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype='float64'))
        self.tree = o3d.geometry.KDTreeFlann(self.prior)
        self.center = None
        self.target = None

    def cloud(self, xyz, voxel):
        cloud = self.o3d.geometry.PointCloud()
        cloud.points = self.o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, dtype='float64'))
        return cloud.voxel_down_sample(voxel)

    def submap(self, position):
        if self.center is None or np.linalg.norm(self.center - position) > 2.0:
            count, rows, _ = self.tree.search_radius_vector_3d(position, 18.0)
            if count < 100:
                raise ValueError('初始位姿附近的地图点不足，请在地图上设置初始位姿')
            target = self.cloud(np.asarray(self.prior.points)[np.asarray(rows)], .12)
            target.estimate_normals(self.o3d.geometry.KDTreeSearchParamHybrid(radius=.45, max_nn=24))
            self.target, self.center = target, position.copy()
        return self.target

    def align(self, xyz, guess, tracking):
        target = self.submap(guess[:3, 3])
        source = self.cloud(xyz, .12)
        if len(source.points) < 80:
            raise ValueError('本次扫描的有效点不足')
        solve = self.reg.registration_generalized_icp
        method = self.reg.TransformationEstimationForGeneralizedICP()
        def run(seed, distance, iterations):
            return solve(source, target, distance, seed, method,
                         self.reg.ICPConvergenceCriteria(max_iteration=iterations))
        coarse = run(guess, .45 if tracking else .8, 14)
        if coarse.fitness < .28:
            raise ValueError('GICP 初始重叠不足 %.2f' % coarse.fitness)
        fit = run(coarse.transformation, .30, 18)
        match = checked_transform(fit.transformation)
        if not fit.fitness >= .48 or not .0 <= fit.inlier_rmse <= .17:
            raise ValueError('GICP 质量不足 fitness=%.2f RMSE=%.3fm' % (fit.fitness, fit.inlier_rmse))
        if not within(guess, match, .48 if tracking else 1.15, 15 if tracking else 32):
            raise ValueError('GICP 修正超出范围，请检查初始位姿')
        pairs = np.asarray(fit.correspondence_set)
        if len(pairs) < 65:
            raise ValueError('GICP 对应点不足')
        ids = pairs[:, 1]
        normals = np.asarray(target.normals)[ids, :2]
        arms = np.asarray(target.points)[ids, :2] - match[:2, 3]
        scale = max(1.0, float(np.median(np.linalg.norm(arms, axis=1))))
        jacobian = np.column_stack((normals, (arms[:, 0]*normals[:, 1] - arms[:, 1]*normals[:, 0])/scale))
        if np.linalg.eigvalsh(jacobian.T @ jacobian / len(ids))[0] < .004:
            raise ValueError('当前表面无法约束平移和航向，请补充可辨识结构')
        return match, fit.fitness, fit.inlier_rmse


def main():
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Bool, String
    from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

    class Localizer(Node):
        def __init__(self, directory):
            super().__init__('go2_gicp')
            metadata = json.loads((directory/'metadata.json').read_text())
            if metadata['map_frame'] != 'map':
                raise ValueError('prepared navigation grid and GICP prior must use map')
            self.start_pose = checked_transform(metadata['initial_map_from_body'])
            self.matcher = Matcher(directory)
            self.lock = threading.Lock()
            self.scans = deque(maxlen=12)
            self.odom = None
            self.generation = 0
            self.seed = self.start_pose.copy()
            self.manual_seed = False
            self.correction = None
            self.candidate = None
            self.confirmed = 0
            self.last_scan = 0
            self.last_good = 0
            self.last_accept_mono = -math.inf
            self.last_tf = 0
            self.reason = '等待扫描与本体里程计；默认建图起点，其他位置请发 /initialpose'
            self.buffer = Buffer()
            self.listener = TransformListener(self.buffer, self)
            self.tf_pub = TransformBroadcaster(self)
            self.health_pub = self.create_publisher(Bool, '/localization/valid', 10)
            self.status_pub = self.create_publisher(String, '/localization/status', 10)
            retained = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                  reliability=ReliabilityPolicy.RELIABLE)
            self.map_pub = self.create_publisher(PointCloud2, '/map_cloud', retained)
            self.map_shown = False
            self.map_subscribers = 0
            self.last_map_publish = -math.inf
            self.create_timer(1., self.show_map)
            self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
            self.create_subscription(PointCloud2, LASER_TOPIC, self.on_scan, qos_profile_sensor_data)
            self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.on_hint, 5)
            self.create_timer(.25, self.process, callback_group=MutuallyExclusiveCallbackGroup())
            self.create_timer(.05, self.health)
            self.status_time = -math.inf

        def now_ns(self):
            return self.get_clock().now().nanoseconds

        def show_map(self):
            subscribers = self.map_pub.get_subscription_count()
            joined = subscribers > self.map_subscribers
            self.map_subscribers = subscribers
            if not subscribers or (self.map_shown and not joined and time.monotonic()-self.last_map_publish < 15):
                return
            map_points = np.asarray(self.matcher.prior.points)
            step = max(1, math.ceil(len(map_points)/160000))
            points = np.ascontiguousarray(map_points[::step], dtype='<f4')
            msg = PointCloud2()
            msg.header.frame_id, msg.header.stamp = 'map', self.get_clock().now().to_msg()
            msg.height, msg.width = 1, len(points)
            msg.fields = [PointField(name=k, offset=4*i, datatype=PointField.FLOAT32, count=1)
                          for i,k in enumerate('xyz')]
            msg.point_step, msg.row_step, msg.is_dense = 12, points.nbytes, True
            msg.data = points.tobytes()
            self.map_pub.publish(msg)
            self.map_shown, self.last_map_publish = True, time.monotonic()

        def on_odom(self, msg):
            if msg.header.frame_id != 'odom' or msg.child_frame_id != 'base_link':
                return
            stamp = stamp_ns(msg.header.stamp)
            if not -.10 <= (self.now_ns() - stamp)*1e-9 <= MAX_ODOM_AGE:
                return
            pose = msg.pose.pose
            try:
                body = matrix(vec(pose.position), quat(pose.orientation))
            except ValueError:
                return
            with self.lock:
                if self.odom is None or stamp > self.odom[0]:
                    self.odom = stamp, msg.header.stamp, body, time.monotonic()

        def on_scan(self, msg):
            t = stamp_ns(msg.header.stamp)
            if not msg.header.frame_id or not -.10 <= (self.now_ns()-t)*1e-9 <= MAX_SCAN_AGE:
                return
            with self.lock:
                if t > self.last_scan and (not self.scans or t > stamp_ns(self.scans[-1].header.stamp)):
                    self.scans.append(msg)

        def on_hint(self, msg):
            if msg.header.frame_id.lstrip('/') != 'map':
                self.get_logger().warning('初始位姿需要 map 坐标系')
                return
            try:
                p = msg.pose.pose
                hint = matrix(vec(p.position), quat(p.orientation))
            except ValueError:
                self.get_logger().warning('初始位姿坐标或四元数无效')
                return
            with self.lock:
                self.generation += 1
                self.correction = self.candidate = None
                self.confirmed = self.last_good = 0
                self.last_accept_mono = -math.inf
                self.scans.clear()
                self.seed, self.manual_seed = hint, True
                self.reason = '已接收初始位姿，等待重新配准'
            self.health_pub.publish(Bool(data=False))

        def at(self, target, source, t):
            tf = self.buffer.lookup_transform(target, source, t).transform
            return matrix(vec(tf.translation), quat(tf.rotation))

        def current(self, ns, mono):
            return bool(self.correction is not None and self.last_good and self.odom
                        and -.1 <= (ns-self.last_good)*1e-9 <= MAX_SCAN_AGE
                        and mono-self.last_accept_mono <= MAX_SCAN_AGE
                        and -.1 <= (ns-self.odom[0])*1e-9 <= MAX_ODOM_AGE
                        and mono-self.odom[3] <= MAX_ODOM_AGE)

        def process(self):
            with self.lock:
                packets = list(reversed(self.scans))
                gen = self.generation
            chosen = None
            for msg in packets:
                ns = stamp_ns(msg.header.stamp)
                if not -.1 <= (self.now_ns()-ns)*1e-9 <= MAX_SCAN_AGE:
                    continue
                stamp = Time.from_msg(msg.header.stamp)
                try:
                    odom_body = self.at('odom', 'base_link', stamp)
                    body_laser = self.at('base_link', msg.header.frame_id, stamp)
                except (TransformException, ValueError):
                    continue
                chosen = msg, ns, odom_body, body_laser
                break
            if chosen is None:
                return
            msg, ns, odom_body, body_laser = chosen
            with self.lock:
                if gen != self.generation or ns <= self.last_scan:
                    return
                self.last_scan = ns
                while self.scans and stamp_ns(self.scans[0].header.stamp) <= ns:
                    self.scans.popleft()
                tracking = self.current(self.now_ns(), time.monotonic())
                if self.candidate is not None and not tracking:
                    guess = self.candidate @ odom_body
                elif self.correction is not None:
                    guess = self.correction @ odom_body
                else:
                    guess = self.seed.copy()
                    if self.manual_seed:
                        guess[2, 3] = self.start_pose[2, 3]
                        guess[:3, :3] = rotation_z(yaw(guess)-yaw(odom_body)) @ odom_body[:3, :3]
            try:
                xyz = cloud_points(msg)
                match, fitness, rmse = self.matcher.align(
                    xyz @ body_laser[:3, :3].T + body_laser[:3, 3], guess, tracking)
                if not -.1 <= (self.now_ns()-ns)*1e-9 <= MAX_SCAN_AGE:
                    raise ValueError('配准结果已过时，检查计算负载')
            except (ValueError, RuntimeError) as exc:
                with self.lock:
                    if gen == self.generation:
                        self.reason, self.candidate, self.confirmed = str(exc), None, 0
                        self.last_good = 0
                return
            with self.lock:
                if gen != self.generation:
                    return
                if tracking and self.current(self.now_ns(), time.monotonic()):
                    match = blend_at_body(guess, match, .45)
                else:
                    if self.candidate is None or not within(self.candidate @ odom_body, match, .18, 7):
                        self.confirmed = 0
                    self.candidate = match @ np.linalg.inv(odom_body)
                    self.confirmed += 1
                    if self.confirmed < CONFIRMATIONS:
                        self.reason = 'GICP 正在确认 %d/%d' % (self.confirmed, CONFIRMATIONS)
                        return
                self.correction = match @ np.linalg.inv(odom_body)
                self.last_good = ns
                self.last_accept_mono = time.monotonic()
                self.candidate, self.confirmed = None, 0
                self.reason = 'GICP fitness=%.2f RMSE=%.3fm' % (fitness, rmse)

        def health(self):
            ns, mono = self.now_ns(), time.monotonic()
            with self.lock:
                valid = self.current(ns, mono)
                odom, correction, why = self.odom, self.correction, self.reason
                stamp = odom[0] if odom else 0
                broadcast = valid and stamp > self.last_tf
                if broadcast:
                    self.last_tf = stamp
                if not valid and self.last_good:
                    why = self.reason = '扫描或本体里程计过期，等待连续配准确认'
                    self.last_good = 0
            self.health_pub.publish(Bool(data=valid))
            if mono - self.status_time >= .5:
                self.status_pub.publish(String(data=('定位有效 | ' if valid else '定位未确认 | ') + why))
                self.status_time = mono
            if broadcast:
                body = correction @ odom[2]
                links = []
                for parent, child, value in (
                        ('map', 'odom', correction),
                        ('base_link', 'base_footprint', np.linalg.inv(body) @ planar(body))):
                    tf = TransformStamped()
                    tf.header.stamp, tf.header.frame_id, tf.child_frame_id = odom[1], parent, child
                    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, value[:3, 3])
                    q = quaternion(value[:3, :3])
                    tf.transform.rotation.x,tf.transform.rotation.y,tf.transform.rotation.z,tf.transform.rotation.w = map(float, q)
                    links.append(tf)
                self.tf_pub.sendTransform(links)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = Localizer(args.directory)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
