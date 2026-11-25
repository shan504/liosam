#!/usr/bin/env python3
"""Draw the standing Go2 from the native body odometry; own the run's floor frame."""
import io
import math
from pathlib import Path
import sys
import threading
import time
import zipfile

import numpy as np

NATIVE_WORLD = "go2_native/odom"
NATIVE_BODY = "go2_native/base_link"
GROUND = "go2_ground"
BODY = "go2_visual/live_body"
FOCUS = "go2_ground_focus"


def startup_floor(position, quaternion, vertices):
    """Origin of a level floor in native world, at the initial lowest mesh point."""
    x, y, z, w = quaternion
    # World-Z row of R(world <- body). Only applied once to the real standing mesh.
    vertical = (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))
    return np.array([position[0], position[1], position[2] + np.min(vertices @ vertical)])


def main():
    import rospy
    import tf2_ros
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Odometry
    from visualization_msgs.msg import Marker

    archive_path = Path.home() / "go2_model_live.pyz"
    sys.path.insert(0, str(archive_path))
    from original_model import make_mesh
    from surface import FloorSurface

    rospy.init_node("go2_native_model")
    lio_map = rospy.get_param("~lio_map", False)
    if lio_map:
        from lio_alignment import Alignment, WORLD, compose, subscribe
        align = Alignment()
        align_subs = subscribe(align)
    with zipfile.ZipFile(archive_path) as archive:
        with np.load(io.BytesIO(archive.read("standing.npz")), allow_pickle=False) as data:
            vertices = data["triangles"].reshape(-1, 3)
            if not vertices.size or not np.isfinite(vertices).all():
                raise ValueError("Go2 模型顶点无效")
            mesh = make_mesh(data, rospy.Time(0))
    mesh.header.frame_id = BODY
    mesh.frame_locked = True
    remove = Marker()
    remove.header.frame_id = GROUND
    remove.ns, remove.id, remove.action = mesh.ns, mesh.id, Marker.DELETE
    remove.pose.orientation.w = 1.0
    model_pub = rospy.Publisher("/go2_visual/live/model_mesh", Marker, queue_size=1, latch=True)
    pose_pub = rospy.Publisher("/go2_visual/live/model_pose", PoseStamped, queue_size=2)
    tf = tf2_ros.TransformBroadcaster()
    static_tf = tf2_ros.StaticTransformBroadcaster()
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)
    surface = FloorSurface(buffer)
    lock = threading.Lock()
    origin = last_stamp = None
    last_rx = 0.0
    visible = False

    def transform(parent, child, stamp, xyz, quat):
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id, t.header.stamp = parent, child, stamp
        p, q = t.transform.translation, t.transform.rotation
        p.x, p.y, p.z = map(float, xyz)
        q.x, q.y, q.z, q.w = map(float, quat)
        return t

    def native_pose(msg):
        nonlocal origin, last_stamp, last_rx, visible
        if (msg.header.frame_id.lstrip("/"), msg.child_frame_id.lstrip("/")) != (NATIVE_WORLD, NATIVE_BODY):
            rospy.logerr_throttle(5, "Go2 原生位姿帧异常，收到 %s -> %s", msg.header.frame_id, msg.child_frame_id)
            return
        stamp = msg.header.stamp
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        position = np.array([p.x, p.y, p.z])
        quat = np.array([q.x, q.y, q.z, q.w])
        norm = np.linalg.norm(quat)
        if stamp == rospy.Time(0) or not np.isfinite(position).all() or not math.isfinite(norm) or norm < 1e-6:
            return
        quat /= norm
        with lock:
            if last_stamp is not None and stamp <= last_stamp:
                return
            if lio_map:
                map_from_native = align.value()
                if map_from_native is None:
                    rospy.logwarn_throttle(5, "模型等待本体雷达位姿与 LIO 建图同戳对齐")
                    return
                link = transform(WORLD, NATIVE_WORLD, stamp, *map_from_native)
                body = transform(NATIVE_WORLD, BODY, stamp, position, quat)
                tf.sendTransform([link, body])
                xyz, orientation = compose(map_from_native, (tuple(position), tuple(quat)))
                parent = WORLD
            else:
                if origin is None:
                    origin = startup_floor(position, quat, vertices)
                    static_tf.sendTransform(transform(NATIVE_WORLD, GROUND, stamp, origin, (0, 0, 0, 1)))
                    rospy.loginfo("以首帧 Go2 原生位姿和站姿模型脚底建立地面参考系")
                xyz = position - origin
                body = transform(GROUND, BODY, stamp, xyz, quat)
                focus = transform(GROUND, FOCUS, stamp, (xyz[0], xyz[1], 0), (0, 0, 0, 1))
                tf.sendTransform([body, focus])
                orientation = quat
                parent = GROUND
            actual = PoseStamped()
            actual.header.frame_id, actual.header.stamp = parent, stamp
            actual.pose.position.x, actual.pose.position.y, actual.pose.position.z = map(float, xyz)
            q_out = actual.pose.orientation
            q_out.x, q_out.y, q_out.z, q_out.w = map(float, orientation)
            pose_pub.publish(actual)
            last_rx, last_stamp = time.monotonic(), stamp
            if not visible:
                model_pub.publish(mesh)
                visible = True
                rospy.loginfo("Go2 模型位姿已接入 /go2_native/odometry")

    sub = rospy.Subscriber("/go2_native/odometry", Odometry, native_pose, queue_size=8, tcp_nodelay=True)
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        with lock:
            if visible and time.monotonic() - last_rx > 2:
                model_pub.publish(remove)
                visible = False
                rospy.logwarn("Go2 原生位姿超过两秒未更新，已隐藏模型")
            floor_ready = lio_map or origin is not None
        if floor_ready:
            surface.tick()
        rate.sleep()


if __name__ == "__main__":
    main()
