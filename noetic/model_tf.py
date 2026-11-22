"""Standalone Noetic display of the real Go2 standing mesh and calibrated floor."""
import io
import math
import os
from pathlib import Path
import sys
import threading
import time
import zipfile

GROUND = "go2_ground"
OUT = "/go2_visual/live/model_mesh"
ODOM = "/lio_sam/mapping/odometry"


def place_robot(ground_from_odom, odom_from_lidar, lidar_from_body, sole_height):
    import numpy as np
    pose = ground_from_odom @ odom_from_lidar @ lidar_from_body
    x, y = pose[:2, 3]
    fx, fy = pose[:2, 0]
    if not np.isfinite(pose).all() or math.hypot(fx, fy) < 1e-5:
        raise ValueError("无效位姿或机身朝向")
    return float(x), float(y), float(sole_height), math.atan2(fy, fx)


def main():
    if os.environ.get("GO2_LIVE_NOETIC") != "1":
        clean = {k: os.environ[k] for k in ("HOME","USER","TERM","LANG","LC_ALL") if k in os.environ}
        clean.update(GO2_LIVE_NOETIC="1", OPENBLAS_NUM_THREADS="1",
                     PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        setup = ('set -e; source /opt/ros/noetic/setup.bash; '
                 'source "$HOME/mid360s_nk_noetic_ws/devel/setup.bash"; '
                 'export ROS_MASTER_URI=http://127.0.0.1:11321 ROS_IP=127.0.0.1; '
                 'exec /usr/bin/python3 -u "$@"')
        os.execve("/bin/bash", ["bash","--noprofile","--norc","-c",setup,
                                 "go2-live",str(Path(sys.argv[0]).resolve())],clean)

    archive_path = Path.home() / "go2_model_live.pyz"
    sys.path.insert(0, str(archive_path))
    import numpy as np
    import rospy
    import rosgraph
    import tf2_ros
    from tf.transformations import quaternion_matrix, rotation_matrix
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import TransformStamped, PoseStamped
    from visualization_msgs.msg import Marker
    from original_model import make_mesh
    from surface import FloorSurface

    try:
        rosgraph.Master("/go2_live_model_check").getPid()
    except Exception as exc:
        raise SystemExit("无法连接正在建图的 ROS master (11321): " + str(exc))
    rospy.init_node("go2_live_model_view")
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)
    floor = FloorSurface(buffer)
    pub = rospy.Publisher(OUT, Marker, queue_size=1, latch=True)
    with zipfile.ZipFile(archive_path) as archive:
        with np.load(io.BytesIO(archive.read("standing.npz")), allow_pickle=False) as data:
            height = -float(data["triangles"][:,:,2].min())
            marker = make_mesh(data,rospy.Time(0))
    marker.header.frame_id = "go2_visual/live_body"
    marker.frame_locked = True
    marker.header.stamp = rospy.Time(0)
    broadcaster = tf2_ros.TransformBroadcaster()
    pose_pub = rospy.Publisher("/go2_visual/live/model_pose", PoseStamped, queue_size=1)
    rospy.loginfo("已加载内置 Go2 站姿模型：%d 个三角面。", len(marker.points)//3)

    def matrix(v, q):
        quat = np.array([q.x,q.y,q.z,q.w])
        if not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-7:
            raise ValueError("四元数无效")
        m = quaternion_matrix(quat)
        m[:3,3] = (v.x,v.y,v.z)
        return m

    lock = threading.Lock()
    latest = [None, 0.0]
    def received(msg):
        with lock:
            latest[:] = (msg, time.monotonic())
    subscription = rospy.Subscriber(ODOM,Odometry,received,queue_size=1,tcp_nodelay=True)
    exceptions = (tf2_ros.LookupException,tf2_ros.ConnectivityException,tf2_ros.ExtrapolationException)
    last_stamp = None
    sent_mesh = False
    next_floor = 0.
    # Previously used visual mount: T_body_lidar pitch +30 degrees, same origins.
    mount = rotation_matrix(math.radians(-30),(0,1,0))
    got_mount = False
    mount_retry = 0.
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if time.monotonic() >= next_floor:
            floor.tick()
            next_floor = time.monotonic() + .5
        with lock:
            msg, arrival = latest
        if msg is None or time.monotonic()-arrival > 3:
            rospy.logwarn_throttle(8, "正在等待实时建图位姿 %s", ODOM)
        else:
            now = time.monotonic()
            if not got_mount and now >= mount_retry:
                mount_retry = now + 5.
                try:
                    t = buffer.lookup_transform("lidar_link","go2_visual/base_link",rospy.Time(0)).transform
                    mount = matrix(t.translation,t.rotation)
                    got_mount = True
                except exceptions:
                    pass
            try:
                # Both the calibration and optional mounting transform are static.
                t = buffer.lookup_transform(GROUND,msg.header.frame_id.lstrip("/"),rospy.Time(0)).transform
                pose = msg.pose.pose
                x,y,z,yaw = place_robot(matrix(t.translation,t.rotation),
                                         matrix(pose.position,pose.orientation),mount,height)
            except exceptions:
                rospy.logwarn_throttle(8,"等待已标定的地面变换 %s ← %s",GROUND,msg.header.frame_id)
            except ValueError as exc:
                rospy.logwarn_throttle(8,"模型显示：%s",exc)
            else:
                if last_stamp is None or msg.header.stamp > last_stamp:
                    transform = TransformStamped()
                    transform.header.frame_id = GROUND
                    transform.header.stamp = msg.header.stamp
                    transform.child_frame_id = marker.header.frame_id
                    p, q = transform.transform.translation, transform.transform.rotation
                    p.x, p.y, p.z = x, y, z
                    q.x, q.y, q.z, q.w = 0., 0., math.sin(yaw/2), math.cos(yaw/2)
                    broadcaster.sendTransform(transform)
                    actual = PoseStamped()
                    actual.header = transform.header
                    actual.pose.position.x, actual.pose.position.y, actual.pose.position.z = x, y, z
                    actual.pose.orientation = q
                    pose_pub.publish(actual)
                    if not sent_mesh:
                        pub.publish(marker)
                        rospy.loginfo("已发布贴地 Go2 模型：脚底 z=0.000 m，机身 z=%.3f m；网格锁存，TF逐帧更新。", z)
                        sent_mesh = True
                    last_stamp = msg.header.stamp
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            break


if __name__ == "__main__":
    main()
