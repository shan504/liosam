#!/usr/bin/env python3
"""Place individual MID-360 returns using source-timed Go2 body poses."""
from collections import deque
import json
import os
from pathlib import Path
import struct
import sys
from threading import Lock

import numpy as np

from check_returns import wire_points
from dense_map import decode, geometry
from native_mount import TOPICS as MOUNT_TOPICS
from native_inspect_tf import StaticTF, resolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'navigation'))
from obstacle_view import obstacle_mask

WORLD = "go2_native/odom"
BODY = "go2_native/base_link"
GROUND = "go2_ground"
GAP_NS = 250_000_000


def calibration(xyz, xyzw):
    xyz = np.asarray(xyz, dtype=np.float64)
    q = np.asarray(xyzw, dtype=np.float64)
    if (xyz.shape != (3,) or q.shape != (4,) or not np.isfinite(xyz).all()
            or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1.) > .01):
        raise ValueError("安装外参需要米制 XYZ 和有效的单位四元数 XYZW")
    return np.r_[xyz, q / np.linalg.norm(q)]


def same_mount(a, b):
    return (np.linalg.norm(a[:3] - b[:3]) <= .001 and
            abs(np.dot(a[3:], b[3:])) >= np.cos(np.deg2rad(.1) / 2))


def rotate(q, xyz):
    v = q[..., :3]
    v2 = 2 * np.cross(v, xyz)
    return xyz + q[..., 3:4] * v2 + np.cross(v, v2)


def compose_rotation(a, b):
    return np.r_[a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]),
                 a[3] * b[3] - np.dot(a[:3], b[:3])]


def interpolate(samples, targets):
    """Linear position and shortest-arc SLERP. Reject unbracketed or sparse time."""
    times = np.array([s[0] for s in samples], dtype=np.int64)
    if len(times) < 2 or targets.min() < times[0] or targets.max() > times[-1]:
        raise ValueError("原生里程计未覆盖扫描起止时刻")
    rows = np.stack([s[1] for s in samples])
    hi = np.searchsorted(times, targets, side="left")
    lo = np.maximum(0, hi - 1)
    span = times[hi] - times[lo]
    exact = times[hi] == targets
    if np.any((span > GAP_NS) & ~exact):
        raise ValueError("扫描跨越原生里程计数据空洞")
    f = ((targets - times[lo]) / np.maximum(span, 1))[:, None]
    pa, pb = rows[lo, :3], rows[hi, :3]
    qa, qb = rows[lo, 3:], rows[hi, 3:]
    dot = np.sum(qa * qb, axis=1, keepdims=True)
    qb = np.where(dot < 0., -qb, qb)
    angle = np.arccos(np.clip(abs(dot), 0., 1.))
    # sinc(a/pi) = sin(a)/a, including the coincident-rotation case.
    u = angle / np.pi
    q = ((1-f) * np.sinc((1-f)*u) * qa + f * np.sinc(f*u) * qb) / np.sinc(u)
    return pa + f * (pb-pa), q


def place(scan_stamp, points, samples, mount, near, far):
    """Return a deskewed XYZI cloud and the sensor pose at the scan start."""
    xyz = np.stack([points[k] for k in ("x", "y", "z")], axis=1)
    squared = np.einsum("ij,ij->i", xyz, xyz)
    good = np.isfinite(squared) & (squared > 0) & (squared >= near**2) & (squared <= far**2)
    if not np.any(good):
        raise ValueError("本帧没有量程内的有效回波")
    offsets = points["offset_time"][good].astype(np.int64)
    targets = scan_stamp + np.r_[0, offsets]
    p, q = interpolate(samples, targets)
    sensor_xyz = p[0] + rotate(q[0], mount[:3])
    sensor_q = compose_rotation(q[0], mount[3:])
    body_xyz = rotate(mount[3:], xyz[good]) + mount[:3]
    out = np.empty((len(offsets), 4), dtype="<f4")
    out[:, :3] = rotate(q[1:], body_xyz) + p[1:]
    out[:, 3] = points["reflectivity"][good]
    return out, np.r_[sensor_xyz, sensor_q]


def project_ground(points, static_pose, band):
    r, t = geometry(static_pose)
    out = points.copy()
    if np.array_equal(r, np.eye(3)):
        out[:, :3] += t
    else:
        out[:, :3] = points[:, :3] @ r.T + t
    heights = out[:, 2]
    return out[obstacle_mask(heights) & (heights >= band)]


def main():
    import rospy
    import tf2_ros
    from geometry_msgs.msg import PoseStamped
    from livox_ros_driver2.msg import CustomMsg
    from nav_msgs.msg import Odometry, Path as RosPath
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    from tf2_msgs.msg import TFMessage

    rospy.init_node("go2_native_cloud")
    band = float(rospy.get_param("~ground_band", 0.))
    near = float(rospy.get_param("~min_range", .1))
    far = float(rospy.get_param("~max_range", 40.))
    if not (np.isfinite([band, near, far]).all() and 0 <= band and 0 <= near < far):
        raise ValueError("无效的地面过滤带或量程")
    path = Path(os.environ.get("GO2_BODY_FROM_LIDAR") or "~/.config/go2/body_from_lidar.json").expanduser()
    # Physical installation edges and world/floor geometry have separate TF
    # graphs. Never join a prefix from one ROS master to an edge from another.
    static = tf2_ros.Buffer(debug=False)
    mount_graphs = {"/tf_static": StaticTF()}
    graph_lock = Lock()
    revision = 0

    def accept_static(msg, source):
        nonlocal revision
        internal = source != "/tf_static"
        if internal and (msg._connection_header or {}).get("callerid") != "/go2_native_odom_bridge":
            return
        with graph_lock:
            # Each private TFMessage is a complete retained snapshot.
            graph = StaticTF() if internal else mount_graphs[source]
            if not internal:
                for ts in msg.transforms:
                    static.set_transform_static(ts, source)
            graph.add(msg)
            mount_graphs[source] = graph
            revision += 1

    def lookup(parent, child):
        t = static.lookup_transform(parent, child, rospy.Time(0)).transform
        p, q = t.translation, t.rotation
        return calibration([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])

    def mounting(child):
        if path.exists() or os.environ.get("GO2_BODY_FROM_LIDAR"):
            desc = json.loads(path.read_text())
            if (desc["parent"].lstrip("/"), desc["child"].lstrip("/")) != ("base_link", child):
                raise ValueError("安装外参 parent/child 必须与 base_link 和本帧雷达匹配")
            return calibration(desc["xyz"], desc["xyzw"]), str(path)
        choices = []
        for source, graph in mount_graphs.items():
            result = resolve("base_link", child, graph.edges, graph.bad)
            if result is not None:
                _, (xyz, quat) = result
                choices.append((calibration(xyz, quat), source))
        if not choices:
            raise ValueError("没有完整的静态 TF 路径")
        first, source = choices[0]
        for candidate, other in choices[1:]:
            # q and -q represent the same physical alignment.
            if not same_mount(first, candidate):
                raise ValueError("多个静态来源的安装外参不符 (%s, %s)，请核实实测外参文件" % (source, other))
        return first, source

    pubs = [rospy.Publisher(name, PointCloud2, queue_size=1, latch=latch) for name, latch in (
        ("/go2_native/registered_scan", False), ("/go2_visual/ground/scan", False),
        ("/go2_visual/ground/dense_map", True))]
    path_pub = rospy.Publisher("/go2_native/path", RosPath, queue_size=4)
    pose_pub = rospy.Publisher("/go2_native/lidar_pose", Odometry, queue_size=4)
    fields = [PointField(n, i*4, PointField.FLOAT32, 1) for i, n in enumerate(("x","y","z","intensity"))]

    def publish_cloud(pub, stamp, frame, rows):
        pub.publish(PointCloud2(header=Header(stamp=stamp, frame_id=frame), height=1, width=len(rows),
                    fields=fields, is_bigendian=False, point_step=16, row_step=16*len(rows),
                    data=rows.astype("<f4", copy=False).tobytes(), is_dense=True))

    lock = Lock()
    history, pending = deque(maxlen=4096), deque(maxlen=12)
    def odom(m):
        if (m.header.frame_id.lstrip("/"), m.child_frame_id.lstrip("/")) != (WORLD, BODY):
            rospy.logwarn_throttle(5, "原生位姿的坐标系与本次建图不符")
            return
        p, q = m.pose.pose.position, m.pose.pose.orientation
        try:
            pose = calibration([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
        except ValueError as e:
            rospy.logwarn_throttle(5, str(e))
            return
        stamp = m.header.stamp.to_nsec()
        with lock:
            if stamp > 0 and (not history or stamp > history[-1][0]):
                history.append((stamp, pose))
                while len(history) > 2 and stamp - history[1][0] > 3_000_000_000:
                    history.popleft()

    def raw(m):
        if m._connection_header.get("md5sum") != CustomMsg._md5sum:
            rospy.logerr_throttle(5, "需要 livox_ros_driver2/CustomMsg")
            return
        try:
            t, points = wire_points(m._buff)
            frame_len = struct.unpack_from("<I", m._buff, 12)[0]
            frame = m._buff[16:16+frame_len].decode().lstrip("/")
            if not t or not frame or not len(points):
                return
            rospy.loginfo_once("收到 MID-360 原始点云，雷达坐标系 %s", frame)
            end = t + int(points["offset_time"].max())
            if end-t > 500_000_000:
                raise ValueError("扫描回波相对时间跨度超过 0.5 秒")
            with lock:
                if not pending or t > pending[-1][0]:
                    if len(pending) == pending.maxlen:
                        rospy.logwarn_throttle(5, "原生位姿时间未覆盖扫描或处理积压，跳过最旧扫描")
                    pending.append((t, end, frame, points))
        except (ValueError, struct.error) as e:
            rospy.logwarn_throttle(5, "丢弃无效的 MID-360 扫描: %s", e)

    def dense(m):
        if m.header.frame_id.lstrip("/") != WORLD:
            rospy.logwarn_throttle(5, "忽略其他里程计坐标系的稠密图")
            return
        try:
            floor = lookup(GROUND, WORLD)
            publish_cloud(pubs[2], m.header.stamp, GROUND, project_ground(decode(m), floor, band))
        except (tf2_ros.TransformException, ValueError) as e:
            rospy.logwarn_throttle(5, "稠密图等待原生地面参考: %s", e)

    tf_subs = [rospy.Subscriber(src, TFMessage, accept_static, callback_args=src, queue_size=16)
               for src in ["/tf_static"] + list(MOUNT_TOPICS.values())]
    subs = [
            rospy.Subscriber("/go2_native/odometry", Odometry, odom, queue_size=100),
            rospy.Subscriber("/livox/lidar", rospy.AnyMsg, raw, queue_size=3, buff_size=4*1024**2),
            rospy.Subscriber("/go2_visual/dense_map", PointCloud2, dense, queue_size=1, buff_size=64*1024**2)]
    anchor = mount = mount_frame = None
    mount_revision = -1
    last_scan = 0
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        rate.sleep()
        with lock:
            if not pending or len(history) < 2:
                continue
            t, end, frame, pts = pending[0]
            if end > history[-1][0]:
                if end - history[-1][0] > GAP_NS:
                    rospy.logwarn_throttle(5, "原生位姿落后扫描末尾 %.3f 秒",
                                           (end-history[-1][0])/1e9)
                continue
            pending.popleft()
            samples = tuple(history)
        if t <= last_scan:
            continue
        if frame == "base_link" or frame == BODY or frame.startswith("go2_visual/"):
            rospy.logerr_throttle(5, "MID-360 原始帧不能使用机身或显示模型名称作为物理雷达坐标系")
            continue
        if mount_frame is not None and frame != mount_frame:
            rospy.logerr_throttle(5, "本轮雷达原始坐标系已更换")
            continue
        if mount is None or mount_revision != revision:
            try:
                with graph_lock:
                    candidate, source = mounting(frame)
                    if mount is not None and not same_mount(mount, candidate):
                        raise ValueError("运行期间安装外参改变，请核对后开新图")
                    if mount is None:
                        rospy.loginfo("已采用 base_link <- %s 的安装外参 (%s)", frame, source)
                        mount, mount_frame = candidate, frame
                    mount_revision = revision
            except (ValueError, OSError, KeyError, TypeError, AttributeError) as e:
                rospy.logwarn_throttle(5, "等待实测 base_link <- %s 外参 (静态 TF 或 %s): %s", frame, path, e)
                continue
        try:
            rows, pose = place(t, pts, samples, mount, near, far)
        except ValueError as e:
            rospy.logwarn_throttle(5, "跳过未对齐的原生扫描: %s", e)
            continue
        hdr = Header(stamp=rospy.Time(*divmod(t, 10**9)), frame_id=WORLD)
        sensor = Odometry(header=hdr, child_frame_id=frame)
        p, q = sensor.pose.pose.position, sensor.pose.pose.orientation
        p.x, p.y, p.z, q.x, q.y, q.z, q.w = pose
        pose_pub.publish(sensor)
        if anchor is None:
            anchor = PoseStamped(header=hdr, pose=sensor.pose.pose)
        # The existing archive needs a stable anchor and a pairing stamp on
        # each scan. Native odometry provides no retroactive graph corrections.
        path_pub.publish(RosPath(header=hdr, poses=[anchor]))
        publish_cloud(pubs[0], hdr.stamp, WORLD, rows)
        try:
            floor = lookup(GROUND, WORLD)
            publish_cloud(pubs[1], hdr.stamp, GROUND, project_ground(rows, floor, band))
        except tf2_ros.TransformException:
            rospy.logwarn_throttle(5, "本体里程计已配准点云，等待地面参考 go2_ground")
        last_scan = t


if __name__ == "__main__":
    main()
