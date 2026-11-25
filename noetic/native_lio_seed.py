"""Supply LIO-SAM's map optimizer with time-aligned Go2 motion in the LiDAR frame."""
from bisect import bisect_left
from collections import OrderedDict, deque
import json
import math
import os
from pathlib import Path
import struct
import threading
import time

import numpy as np
from native_cloud import calibration, compose_rotation, interpolate, rotate, same_mount
from native_inspect_tf import StaticTF, resolve
from native_mount import TOPICS as MOUNT_TOPICS
from mount_setup import urdf_mount

WORLD = "go2_native/odom"
BODY = "go2_native/base_link"


def raw_header(payload):
    # ROS1 std_msgs/Header is the first field of Livox CustomMsg.
    if len(payload) < 16:
        raise ValueError("雷达消息头太短")
    _, sec, nsec, size = struct.unpack_from("<IIII", payload)
    if size == 0 or size > len(payload) - 16 or nsec >= 10**9:
        raise ValueError("雷达消息头无效")
    return sec*10**9 + nsec, payload[16:16+size].decode("utf8").lstrip("/")


def native_sensor(samples, stamp, mount):
    """Interpolate the native body trajectory before applying the rigid lever arm."""
    i = bisect_left(samples, (stamp,))
    if i < len(samples) and samples[i][0] == stamp:
        body = samples[i][1]
        p, q = body[:3], body[3:]
    else:
        if i == 0 or i == len(samples):
            raise ValueError("扫描时刻缺少前后两条本体位姿")
        pos, quat = interpolate(samples[i-1:i+1], np.array([stamp], dtype=np.int64))
        p, q = pos[0], quat[0]
    return np.r_[p + rotate(q, mount[:3]), compose_rotation(q, mount[3:])]


def rpy(q):
    x, y, z, w = q
    return (math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)),
            math.asin(max(-1., min(1., 2*(w*y - z*x)))),
            math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))


def seed(msg, sensor):
    msg.initialGuessX, msg.initialGuessY, msg.initialGuessZ = sensor[:3]
    angles = rpy(sensor[3:])
    msg.initialGuessRoll, msg.initialGuessPitch, msg.initialGuessYaw = angles
    # The first keyframe reads the attitude-initialization fields before it
    # consults the incremental prediction. Projection/deskew already finished.
    msg.imuRollInit, msg.imuPitchInit, msg.imuYawInit = angles
    msg.odomAvailable = 1


def select_mount(frame, graphs, path, urdf):
    if not frame or frame in ("base_link", BODY) or frame.startswith(("go2_visual/", "go2_native/")):
        raise ValueError("Livox 原始消息头需要物理雷达坐标系，收到 " + repr(frame))
    if path.is_file() or os.environ.get("GO2_BODY_FROM_LIDAR"):
        doc = json.loads(path.read_text())
        if (doc["parent"].strip("/"), doc["child"].strip("/")) != ("base_link", frame):
            raise ValueError("标定文件需要 base_link <- " + frame)
        return calibration(doc["xyz"], doc["xyzw"]), str(path)
    choices = []
    for source, graph in graphs.items():
        result = resolve("base_link", frame, graph.edges, graph.bad)
        if result is not None:
            _, (xyz, quat) = result
            choices.append((calibration(xyz, quat), source))
    if urdf:
        values = urdf_mount(urdf, frame)
        if values is not None:
            choices.append((calibration(values[:3], values[3:]), "物理安装 URDF"))
    if not choices:
        raise ValueError("缺少 base_link <- " + frame + " 的安装外参。另开终端运行 python3 ~/go2-detail/noetic/mount_setup.py --child " + frame + " 核对安装参数；保存后此窗口自动继续。")
    mount, source = choices[0]
    for other, other_source in choices[1:]:
        if not same_mount(mount, other):
            raise ValueError("安装外参不一致：" + source + " 与 " + other_source)
    return mount, source


def main():
    import rospy
    from nav_msgs.msg import Odometry
    from livox_ros_driver2.msg import CustomMsg
    from lio_sam.msg import cloud_info
    from geometry_msgs.msg import TransformStamped
    from tf2_msgs.msg import TFMessage

    rospy.init_node("go2_native_lio_seed")
    output = rospy.Publisher("/lio_sam/feature/cloud_info", cloud_info, queue_size=3)
    matched = rospy.Publisher("/go2_lio/native_lidar_pose", Odometry, queue_size=24)
    fixed = rospy.Publisher("/go2_lio/body_from_lidar", TransformStamped, queue_size=1, latch=True)
    path = Path(os.environ.get("GO2_BODY_FROM_LIDAR") or "~/.config/go2/body_from_lidar.json").expanduser()
    urdf_path = os.environ.get("GO2_ROBOT_URDF")
    graphs = {"/tf_static": StaticTF()}
    revision = 0
    lock = threading.Lock()
    raw_frames = OrderedDict()
    poses = deque(maxlen=1500)
    features = deque(maxlen=6)
    mount = frame = None
    checked_revision = -1

    def add_tf(msg, source):
        nonlocal revision
        private = source != "/tf_static"
        if private and (msg._connection_header or {}).get("callerid") != "/go2_native_odom_bridge":
            return
        with lock:
            tree = StaticTF() if private else graphs[source]
            tree.add(msg)
            graphs[source] = tree
            revision += 1

    def odom(msg):
        if (msg.header.frame_id.lstrip("/"), msg.child_frame_id.lstrip("/")) != (WORLD, BODY):
            return
        stamp = msg.header.stamp.to_nsec()
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        try:
            body = calibration((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
        except ValueError:
            return
        with lock:
            if stamp > 0 and (not poses or stamp > poses[-1][0]):
                poses.append((stamp, body))
                while len(poses) > 2 and stamp - poses[1][0] > 3_000_000_000:
                    poses.popleft()

    def laser(msg):
        if (msg._connection_header or {}).get("md5sum") != CustomMsg._md5sum:
            rospy.logwarn_throttle(5, "原始雷达话题的消息类型与 Livox CustomMsg 不符")
            return
        try:
            stamp, physical = raw_header(msg._buff)
        except (ValueError, struct.error) as error:
            rospy.logwarn_throttle(5, "忽略无效雷达消息头：%s", error)
            return
        with lock:
            raw_frames[stamp] = physical
            if len(raw_frames) > 80:
                raw_frames.popitem(last=False)

    def feature(msg):
        with lock:
            if len(features) == features.maxlen:
                rospy.logwarn_throttle(5, "本体预测积压，跳过最旧特征帧")
            features.append((time.monotonic(), msg))

    subscriptions = [rospy.Subscriber(topic, TFMessage, add_tf, callback_args=topic, queue_size=16)
                     for topic in ["/tf_static"] + list(MOUNT_TOPICS.values())]
    subscriptions.append(rospy.Subscriber("/livox/lidar", rospy.AnyMsg, laser,
                                          queue_size=6, buff_size=4*1024**2))
    subscriptions.extend([rospy.Subscriber("/go2_native/odometry", Odometry, odom, queue_size=80,
                                            tcp_nodelay=True),
                          rospy.Subscriber("/go2_lio/unseeded_features", cloud_info, feature,
                                           queue_size=3, buff_size=8*1024**2)])
    rate = rospy.Rate(100)
    last_stamp = 0
    last_mount_check = 0.
    while not rospy.is_shutdown():
        rate.sleep()
        with lock:
            if not features:
                continue
            receipt, msg = features[0]
            stamp = msg.header.stamp.to_nsec()
            physical = raw_frames.get(stamp)
            if not physical:
                if time.monotonic() - receipt < .4:
                    continue
                features.popleft()
                rospy.logwarn_throttle(5, "特征帧缺少同时间戳 Livox 原始消息头，跳过本帧")
                continue
            if not poses or stamp > poses[-1][0]:
                if time.monotonic() - receipt < .4:
                    continue
                features.popleft()
                rospy.logwarn_throttle(3, "未收到覆盖扫描时刻的 Go2 本体里程计，暂停向 LIO-SAM 送帧")
                continue
            features.popleft()
            samples = tuple(poses)
        if stamp <= last_stamp or stamp < samples[0][0]:
            continue
        if mount is not None and frame != physical:
            rospy.logerr_throttle(3, "雷达原始坐标系改变，暂停本轮建图")
            continue
        if mount is None or revision != checked_revision:
            if time.monotonic()-last_mount_check < .5 and mount is None:
                continue
            last_mount_check = time.monotonic()
            try:
                urdf = (Path(urdf_path).expanduser().read_text() if urdf_path else
                        rospy.get_param("/robot_description", "")) if not path.is_file() else ""
                with lock:
                    candidate, source = select_mount(physical, graphs, path, urdf)
                    if mount is not None and not same_mount(mount, candidate):
                        raise ValueError("运行中安装外参改变，请核对后开新图")
                    checked_revision = revision
                if mount is None:
                    frame, mount = physical, candidate
                    tf = TransformStamped()
                    tf.header.stamp = msg.header.stamp
                    tf.header.frame_id, tf.child_frame_id = "base_link", frame
                    d, q = tf.transform.translation, tf.transform.rotation
                    d.x, d.y, d.z, q.x, q.y, q.z, q.w = mount
                    fixed.publish(tf)
                    rospy.loginfo("建图预测采用 base_link <- %s 安装外参 (%s)", frame, source)
            except (KeyError, ValueError, TypeError, OSError, AttributeError) as error:
                rospy.logwarn_throttle(5, "等待本体到 MID-360 的物理安装变换：%s", error)
                continue
        try:
            sensor = native_sensor(samples, stamp, mount)
        except ValueError as error:
            rospy.logwarn_throttle(3, "跳过本体位姿缺失的扫描：%s", error)
            continue
        pose = Odometry()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id, pose.child_frame_id = WORLD, frame
        p, q = pose.pose.pose.position, pose.pose.pose.orientation
        p.x, p.y, p.z, q.x, q.y, q.z, q.w = sensor
        seed(msg, sensor)
        matched.publish(pose)
        output.publish(msg)
        last_stamp = stamp
        rospy.loginfo_once("已按扫描时间将 Go2 本体位姿用作 LIO-SAM 建图初值")


if __name__ == "__main__":
    main()
