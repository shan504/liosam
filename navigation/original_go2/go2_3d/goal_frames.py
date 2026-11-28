"""Convert an operator's odom-frame click into an immutable map waypoint.

Only map/odom inputs are supported. This does not grant motion authority or
change occupancy checks. Transform lookup and bounded retries live in Operator.
"""
import copy
import math

from geometry import pose_matrix, quaternion_from_matrix

GOAL_TF_WAIT_SECONDS = 0.5
GOAL_MAX_AGE_NS = 1_000_000_000
LATEST_TF_MAX_AGE_NS = 500_000_000
FUTURE_TOLERANCE_NS = 100_000_000


def goal_frame(frame):
    frame = frame.lstrip('/') if isinstance(frame, str) else ''
    if frame not in ('map', 'odom'):
        raise ValueError('目标坐标系必须是 map 或 odom；请将 Foxglove 显示参考系设为 map')
    return frame


def stamp_ns(stamp):
    sec, ns = stamp.sec, stamp.nanosec
    if (not isinstance(sec, int) or isinstance(sec, bool) or
            not isinstance(ns, int) or isinstance(ns, bool) or sec < 0 or not 0 <= ns < 10**9):
        raise ValueError('目标或 TF 时间戳无效')
    return sec*10**9+ns


def validate_odom_goal_time(message, now_ns):
    stamp = stamp_ns(message.header.stamp)
    if stamp and not -FUTURE_TOLERANCE_NS <= now_ns-stamp <= GOAL_MAX_AGE_NS:
        raise ValueError('odom 目标时间戳过期或超前，请在实时画面重新设置目标')
    return stamp


def _matrix(position, quaternion):
    xyz = [position.x, position.y, position.z]
    q = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    if not all(math.isfinite(v) for v in xyz+q) or abs(sum(v*v for v in q)-1.) > .02:
        raise ValueError('目标或 TF 包含非有限坐标或非单位四元数')
    return pose_matrix(xyz, q)


def odom_goal_to_map(message, transform, now_ns):
    """Apply the actual timestamped map<-odom SE(3), never relabel coordinates."""
    if goal_frame(message.header.frame_id) != 'odom':
        raise ValueError('坐标转换仅用于 odom 目标')
    requested = validate_odom_goal_time(message, now_ns)
    if (transform.header.frame_id.lstrip('/') != 'map' or
            transform.child_frame_id.lstrip('/') != 'odom'):
        raise ValueError('目标转换需要 map <- odom TF')
    actual = stamp_ns(transform.header.stamp)
    if actual <= 0:
        raise ValueError('map <- odom TF 缺少有效时间戳')
    if requested:
        if actual != requested:
            raise ValueError('未获得目标时刻的 map <- odom TF，不能替换为其他时刻')
    elif not -FUTURE_TOLERANCE_NS <= now_ns-actual <= LATEST_TF_MAX_AGE_NS:
        raise ValueError('map <- odom TF 已过期或超前，请等待定位和本体里程计恢复')
    mapped = (_matrix(transform.transform.translation, transform.transform.rotation) @
              _matrix(message.pose.position, message.pose.orientation))
    output = copy.deepcopy(message)
    output.header.frame_id = 'map'
    output.header.stamp = copy.deepcopy(transform.header.stamp)
    p, q = output.pose.position, output.pose.orientation
    p.x, p.y, p.z = map(float, mapped[:3, 3])
    q.x, q.y, q.z, q.w = map(float, quaternion_from_matrix(mapped))
    return output
