"""Anchor the native trajectory in the lidar map and follow pose-graph revisions."""
import math
import threading

from native_inspect_tf import compose, invert

WORLD = 'odom'
NATIVE = 'go2_native/odom'


def rigid(pose):
    p, q = pose.position, pose.orientation
    xyz, quat = (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)
    n = math.sqrt(sum(v*v for v in quat))
    if not all(math.isfinite(v) for v in xyz + quat) or abs(n - 1.) > .01:
        raise ValueError('invalid pose')
    return xyz, tuple(v/n for v in quat)


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))


def angle_deg(a, b):
    dot = abs(sum(x*y for x,y in zip(a,b)))
    return math.degrees(2 * math.acos(min(1., dot)))


class Alignment:
    def __init__(self):
        self.lock = threading.Lock()
        self.alignment = None
        self.last_key = None
        self.revision_ns = None
        self.fault = False

    def keys(self, path):
        if not path.poses or path.header.frame_id.lstrip('/') != WORLD:
            return
        with self.lock:
            if self.last_key:
                idx, ns, previous = self.last_key
                if idx >= len(path.poses) or path.poses[idx].header.stamp.to_nsec() != ns:
                    self.fault = True
                    raise ValueError('LIO-SAM 关键帧序列被替换，请重新启动本次建图')
                current = rigid(path.poses[idx].pose)
                if distance(current[0], previous[0]) > .0001 or angle_deg(current[1], previous[1]) > .005:
                    self.revision_ns = path.header.stamp.to_nsec()
            key = path.poses[-1]
            self.last_key = len(path.poses)-1, key.header.stamp.to_nsec(), rigid(key.pose)

    def pair(self, native, mapped, stamp):
        with self.lock:
            if self.fault:
                return None
            changed = self.alignment is None or (self.revision_ns is not None and stamp >= self.revision_ns)
            if changed:
                self.alignment = compose(mapped, invert(native))
                self.revision_ns = None
            estimate = compose(self.alignment, native)
            return changed, tuple(x-y for x,y in zip(estimate[0], mapped[0]))

    def value(self):
        with self.lock:
            return None if self.fault else self.alignment


def subscribe(align):
    import message_filters
    import rospy
    from geometry_msgs.msg import Vector3Stamped
    from nav_msgs.msg import Odometry, Path

    residual = rospy.Publisher('/go2_lio/registration_residual', Vector3Stamped, queue_size=3)
    def keys(path):
        try:
            align.keys(path)
        except ValueError as exc:
            rospy.logerr_throttle(5, str(exc))

    def pair(pred, mapped):
        if pred.header.frame_id.lstrip('/') != NATIVE or mapped.header.frame_id.lstrip('/') != WORLD:
            rospy.logerr_throttle(5, '本体雷达位姿与 LIO 地图坐标系不符')
            return
        try:
            result = align.pair(rigid(pred.pose.pose), rigid(mapped.pose.pose), mapped.header.stamp.to_nsec())
        except ValueError as exc:
            rospy.logerr_throttle(5, str(exc))
            return
        if result is not None:
            changed, diff = result
            if changed:
                rospy.loginfo('已用同一扫描时刻的雷达位姿对齐 LIO 地图和 Go2 原生轨迹')
            m = Vector3Stamped(header=mapped.header)
            m.vector.x, m.vector.y, m.vector.z = diff
            residual.publish(m)

    left = message_filters.Subscriber('/go2_lio/native_lidar_pose', Odometry, queue_size=20)
    right = message_filters.Subscriber('/lio_sam/mapping/odometry', Odometry, queue_size=20)
    sync = message_filters.TimeSynchronizer([left, right], 80)
    sync.registerCallback(pair)
    history = rospy.Subscriber('/lio_sam/mapping/path', Path, keys, queue_size=4)
    return left, right, sync, history
