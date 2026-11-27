#!/usr/bin/env python3
"""Read-only, simultaneous 12-second observation on Go2. Keep the robot still."""
import argparse
from collections import Counter, defaultdict, namedtuple
import math
import os
from pathlib import Path
import struct
import sys
from threading import Lock
import time

ODOM = '/lio_sam/mapping/odometry'
INC = '/lio_sam/mapping/odometry_incremental'
DESKEW = '/lio_sam/deskew/cloud_deskewed'
RAW = '/livox/lidar'
REG = '/lio_sam/mapping/cloud_registered_raw'
SCAN = '/go2_visual/ground/scan'
FULL_SCAN = '/go2_visual/ground_full/scan'
LOCAL = '/go2_visual/detail/local'
MODEL = '/go2_visual/live/model_mesh'
IMU = '/mid360s/imu_filtered'
GROUND = 'go2_ground'
NS = 1000000000
Event = namedtuple('Event', 'stamp seq frame owner arrival value')
TYPES = {RAW: 'livox_ros_driver2/CustomMsg', DESKEW: 'sensor_msgs/PointCloud2', REG: 'sensor_msgs/PointCloud2',
         SCAN: 'sensor_msgs/PointCloud2',
         MODEL: 'visualization_msgs/Marker'}


def string(buf, pos):
    n, = struct.unpack_from('<I', buf, pos)
    end = pos + 4 + n
    if end > len(buf):
        raise ValueError('truncated ROS string')
    return bytes(buf[pos+4:end]).decode('utf-8'), end


def raw_header(buf):
    seq, sec, ns = struct.unpack_from('<III', buf)
    if ns >= NS:
        raise ValueError('invalid header nsec')
    frame, end = string(buf, 12)
    return seq, sec*NS+ns, frame.lstrip('/'), end


def cloud(buf, pos, sample):
    import numpy as np
    height, width, nfields = struct.unpack_from('<III', buf, pos)
    pos += 12
    if nfields > 256:
        raise ValueError('invalid fields length')
    fields = {}
    for _ in range(nfields):
        name, pos = string(buf, pos)
        offset, dtype, count = struct.unpack_from('<IBI', buf, pos)
        pos += 9
        fields[name] = (offset, dtype, count)
    big, step, row, size = struct.unpack_from('<BIII', buf, pos)
    pos += 13
    if width*step > row or row*height > size or pos+size+1 > len(buf):
        raise ValueError('invalid cloud storage')
    n = width*height
    pts = None
    if sample and n:
        ids = np.linspace(0, n-1, min(n, 17), dtype=np.int64)
        rows, cols = np.divmod(ids, width)
        vals = []
        for axis in 'xyz':
            off, typ, count = fields[axis]
            if typ not in (7, 8) or count != 1:
                raise ValueError('unexpected XYZ encoding')
            dt = np.dtype(('>' if big else '<')+('f4' if typ == 7 else 'f8'))
            if off+dt.itemsize > step:
                raise ValueError('invalid XYZ offset')
            vals.append(np.ndarray((height, width), dt, buf, pos+off, (row, step))[rows, cols])
        pts = np.column_stack(vals)
    return n, pts


def rotation(q):
    import numpy as np
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError('invalid quaternion')
    x, y, z, w = q/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def marker(buf, pos):
    """Sample triangle vertices as well as pose: some publishers bake vertices into ground."""
    import numpy as np
    namespace, pos = string(buf, pos)
    ident, typ, action = struct.unpack_from('<iii', buf, pos)
    pos += 12
    pose = np.array(struct.unpack_from('<7d', buf, pos))
    scale = np.array(struct.unpack_from('<3d', buf, pos+56))
    pos += 56+24+16+8+1  # pose, scale, RGBA, duration, frame_locked
    count, = struct.unpack_from('<I', buf, pos)
    pos += 4
    if pos+count*24 > len(buf):
        raise ValueError('invalid Marker point count')
    center = pose[:3].copy()
    if count and not action:
        points = np.ndarray((count, 3), '<f8', buf, pos)
        idx = np.linspace(0, count-1, min(97, count), dtype=np.int64)
        valid = points[idx]
        if not np.isfinite(valid).all():
            raise ValueError('invalid marker vertices')
        center += rotation(pose[3:]) @ (valid.mean(axis=0)*scale)
    return (namespace, ident, typ, action, count), pose, center


def quat_angle(a, b):
    import numpy as np
    return math.degrees(2*math.acos(float(np.clip(abs(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)), 0, 1))))


def show_stream(topic, rows, source_topic=None):
    import numpy as np
    label = source_topic or topic
    if not rows:
        print(label, '本窗口无消息')
        return
    owners = sorted(set(r.owner for r in rows))
    frames = sorted(set(r.frame for r in rows))
    stamps = [r.stamp for r in rows]
    dt = np.diff(stamps)/1e6
    wall = np.diff([r.arrival for r in rows])*1000
    print('%s n=%d pub=%s frame=%s stamp_ns=%d..%d' %
          (label, len(rows), ','.join(owners), ','.join(frames), stamps[0], stamps[-1]))
    if len(dt):
        print('  源戳间隔 中位/最大 %.1f/%.1f ms，到达最大间隔 %.1f ms；重复/倒序 %d/%d' %
              (np.median(dt), max(dt), max(wall), len(stamps)-len(set(stamps)), sum(dt < 0)))
    if topic in (DESKEW, REG, SCAN, RAW):
        sizes = [r.value[0] for r in rows]
        print('  每帧点数 min/median/max %d/%d/%d' % (min(sizes), np.median(sizes), max(sizes)))


def pose_report(rows):
    import numpy as np
    group = defaultdict(list)
    for r in rows:
        group[r.owner, r.frame, r.value[1]].append(r)
    for key, subset in group.items():
        print('建图父/子帧 %s→%s，发布者 %s' % (key[1], key[2], key[0]))
        if len(subset) < 2:
            continue
        poses = np.array([r.value[0] for r in subset])
        dp = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
        dq = [quat_angle(a, b) for a, b in zip(poses[:-1, 3:], poses[1:, 3:])]
        ix = int(np.argmax(dp))
        print('  相邻位置最大 %.3f m (stamp_ns %d→%d，本步 dt %.3f s)，相邻旋转最大 %.2f°；' %
              (dp[ix], subset[ix].stamp, subset[ix+1].stamp,
               (subset[ix+1].stamp-subset[ix].stamp)/NS, max(dq)))
        print('  从首帧最大位移 %.3f m、最大转角 %.2f°，末帧位移 %.3f m' %
              (np.linalg.norm(poses[:, :3]-poses[0, :3], axis=1).max(),
               max(quat_angle(poses[0, 3:], q) for q in poses[:, 3:]),
               np.linalg.norm(poses[-1, :3]-poses[0, :3])))


def unique_stamps(rows):
    count = Counter(r.stamp for r in rows)
    return {r.stamp: r for r in rows if r.stamp and count[r.stamp] == 1}


def livox_window(buf):
    # In this fork the conversion loops over [0, point_num - 1), then uses
    # the last copied point (the input's penultimate return) as scan end.
    from check_returns import wire_points
    _, points = wire_points(buf)
    return len(points), (int(points['offset_time'][-2]) if len(points) >= 2 else None)


def imu_coverage(raw, imu):
    import numpy as np
    if not raw or not imu:
        print('雷达/IMU 时间覆盖：缺少扫描或滤波 IMU，无法比较。')
        return
    if any(len({r.owner for r in stream}) > 1 or len({r.frame for r in stream}) > 1
           for stream in (raw, imu)):
        print('雷达/IMU 时间覆盖：存在多个发布者或坐标系，无法合并不同来源的时间戳。')
        return
    known = [r for r in raw if r.value[1] is not None and r.stamp > 0]
    times = np.unique([r.stamp for r in imu if r.stamp > 0])
    # Classify interior by receipt times, independently of sensor stamp. A clock
    # offset must remain visible, not be excluded by selecting on IMU headers.
    lo = max(min(r.arrival for r in raw), min(r.arrival for r in imu))+.25
    hi = min(max(r.arrival for r in raw), max(r.arrival for r in imu))-.25
    take = [r for r in known if lo <= r.arrival <= hi]
    if not take or len(times) < 2:
        print('雷达/IMU 时间覆盖：去除接收窗口首尾各 0.25 s 后，没有足够的独立样本。')
        return
    gaps, endpoint_errors, no_inside = [], [], 0
    outside_start = outside_end = 0
    for r in take:
        start, end = r.stamp, r.stamp+r.value[1]
        i = int(np.searchsorted(times, start, side='right'))-1
        j = int(np.searchsorted(times, end, side='left'))
        outside_start += int(i < 0)
        outside_end += int(j >= len(times))
        if i < 0 or j >= len(times):
            continue
        interior = times[i:j+1]
        gaps.append(int(np.diff(interior).max()) if len(interior) > 1 else 0)
        endpoint_errors.append(max(min(abs(int(v)-start) for v in interior[:2]),
                                   min(abs(int(v)-end) for v in interior[-2:])))
        no_inside += int(not np.any((interior >= start) & (interior <= end)))
    print('扫描使用倒数第 2 个回波偏移。接收窗口首尾各去 0.25 s；内部扫描 %d，扫描时长中位 %.2f ms。' %
          (len(take), np.median([r.value[1] for r in take])/1e6))
    print('  滤波 IMU 时间戳包住扫描起止 %d/%d；起点前缺样本 %d，终点后缺样本 %d。' %
          (len(gaps), len(take), outside_start, outside_end))
    if gaps:
        print('  已包住的扫描：跨越扫描区间的相邻 IMU 最大间隔 %.2f ms，端点距最近 IMU 最大 %.2f ms，扫描内无 IMU %d 帧。' %
              (max(gaps)/1e6, max(endpoint_errors)/1e6, no_inside))
    print('  以上是诊断订阅端收到的源戳，建图回调当时的 IMU 队列状态无法从这里直接读取。')
    return len(take), len(gaps), max(gaps) if gaps else None


def paired_report(data, lookup):
    import numpy as np
    o, reg, scan, markers = [unique_stamps(data[t]) for t in (ODOM, REG, SCAN, MODEL)]
    inc = unique_stamps(data[INC])
    both = sorted(o.keys() & inc.keys())
    if len(both) >= 2 and all(len({r.owner for r in data[t]}) == 1 for t in (ODOM, INC)):
        pairs = [(a, b) for a, b in zip(both, both[1:]) if
                 o[a].frame == o[b].frame and inc[a].frame == inc[b].frame and
                 o[a].value[1] == o[b].value[1] and inc[a].value[1] == inc[b].value[1]]
        if pairs:
            a, b = max(pairs, key=lambda ab: np.linalg.norm(o[ab[1]].value[0][:3]-o[ab[0]].value[0][:3]))
            print('在两路共有的时间戳中，全局位移最大的同一段 dt=%.3f s，stamp_ns=%d→%d：' % ((b-a)/NS, a, b))
            for name, stream in [('全局', o), ('增量', inc)]:
                p, q = stream[a].value[0], stream[b].value[0]
                print('  %s %.3f m / %.2f°' % (name, np.linalg.norm(p[:3]-q[:3]), quat_angle(p[3:], q[3:])))
    for label, a, b in [('配准/建图', reg, o), ('贴地/配准', scan, reg), ('模型/建图', markers, o)]:
        count = len(a.keys() & b.keys())
        print('%s 严格同源戳 %d 对；各自唯一非零戳 %d/%d' % (label, count, len(a), len(b)))
    triples = sorted(reg.keys() & scan.keys() & o.keys())
    dx, dz, zclamped, failed, changed_size = [], [], 0, 0, 0
    origins, rotations = [], []
    for stamp in triples:
        a, b = reg[stamp], scan[stamp]
        if b.frame != GROUND or a.frame != o[stamp].frame:
            continue
        tf = lookup(GROUND, a.frame, stamp)
        if tf is None:
            failed += 1
            continue
        R, t = tf
        origins.append(t)
        rotations.append(R)
        n, pa = a.value
        k, pb = b.value
        if n != k:
            changed_size += 1
            continue
        if not n:
            continue
        pred = pa @ R.T + t
        good = np.isfinite(pred).all(axis=1) & np.isfinite(pb).all(axis=1)
        dx.extend(np.linalg.norm(pred[good, :2]-pb[good, :2], axis=1))
        use = good & (pb[:, 2] != 0)  # ground node intentionally snaps near/below-floor Z to zero
        zclamped += int((good & ~use).sum())
        dz.extend(np.abs(pred[use, 2]-pb[use, 2]))
    print('配准/贴地/建图三者同戳 %d；同戳贴地 TF 命中 %d，查询失败 %d，扫描点数不同 %d' %
          (len(triples), len(origins), failed, changed_size))
    if origins:
        tf_dist = np.linalg.norm(np.asarray(origins)-origins[0], axis=1)
        tf_deg = [math.degrees(math.acos(float(np.clip((np.trace(rotations[0].T@r)-1)/2, -1, 1))))
                  for r in rotations]
        print('  地面←建图 TF 首次平移 %s；本窗口标定变化最大 %.4f m / %.3f°' %
              (np.round(origins[0], 3), max(tf_dist), max(tf_deg)))
    if dx:
        print('  同序点按同戳 TF 换算：%d 点 XY 误差 P95/最大 %.4f/%.4f m；Z=0 点 %d、其余点 Z 误差最大 %s m' %
              (len(dx), np.percentile(dx, 95), max(dx), zclamped, '%.4f' % max(dz) if dz else '无'))
    if data[MODEL]:
        last = data[MODEL][-1]
        meta, pose, center = last.value
        print('最后模型 ns/id/type/action/顶点数 %s，header=%s，Marker 原点 %s，绘制顶点采样中心 %s' %
              (meta, last.frame, np.round(pose[:3], 3), np.round(center, 3)))
    matched = defaultdict(list)
    for stamp in sorted(markers.keys() & o.keys()):
        m, od = markers[stamp], o[stamp]
        meta, pose, center = m.value
        if meta[3]:
            continue
        mt, ot = lookup(GROUND, m.frame, stamp), lookup(GROUND, od.frame, stamp)
        if mt is None or ot is None:
            continue
        xy = (mt[0]@center + mt[1])[:2]
        od_xy = (ot[0]@od.value[0][:3]+ot[1])[:2]
        matched[(m.owner, od.owner, meta[:3], meta[4])].append((xy, od_xy))
    for key, obs in matched.items():
        if len(obs) > 1:
            xy, od_xy = np.array(obs).transpose(1, 0, 2)
            print('  %d 个同戳模型/里程计：相对各自首帧的 XY 最大位移 %.3f/%.3f m，位移向量差最大 %.3f m（转身时两者参考点不同）。' %
                  (len(obs), np.linalg.norm(xy-xy[0], axis=1).max(),
                   np.linalg.norm(od_xy-od_xy[0], axis=1).max(),
                   np.linalg.norm((xy-xy[0])-(od_xy-od_xy[0]), axis=1).max()))
    if not any(len(obs) > 1 for obs in matched.values()):
        print('  缺少两次可比较的同戳模型更新；静止时模型可只收到历史锁存消息。')


def vector(s):
    return (s.x, s.y, s.z)


def run(seconds):
    import numpy as np
    import rospy
    import tf2_ros
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    from rosgraph_msgs.msg import Log
    from tf2_msgs.msg import TFMessage
    from stationary_surfaces import SurfaceAudit
    # Fail promptly if master is unavailable instead of blocking inside init_node.
    from xmlrpc.client import ServerProxy
    from xmlrpc.client import Transport

    class TimedTransport(Transport):
        def make_connection(self, host):
            conn = super().make_connection(host)
            conn.timeout = 2
            return conn
    master = ServerProxy(os.environ['ROS_MASTER_URI'], transport=TimedTransport())
    code, message, _ = master.getPid('/go2_alignment_probe')
    if code != 1:
        raise RuntimeError(message)
    rospy.init_node('go2_alignment_probe', anonymous=True, disable_rosout=True)
    tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(40))
    listener = tf2_ros.TransformListener(tfbuf)
    audit = SurfaceAudit()
    data, issues, warnings, broadcasters = defaultdict(list), Counter(), Counter(), Counter()
    lock = Lock()
    closed = False
    subscriptions = []

    def callback(msg, topic):
        try:
            ch = msg._connection_header or {}
            source = ch.get('callerid', '?')
            if topic in ('/tf', '/tf_static'):
                with lock:
                    for t in msg.transforms:
                        p, c = t.header.frame_id.lstrip('/'), t.child_frame_id.lstrip('/')
                        if c in (GROUND, 'go2_ground_focus', 'lidar_link') or p == GROUND:
                            broadcasters[(topic, p, c, source)] += 1
                return
            if topic == '/rosout_agg':
                text = (msg.name+' '+msg.msg).lower()
                known = ('large velocity', 'large bias', 'waiting for imu',
                         'not enough features', 'send buffer limit')
                label = next((k for k in known if k in text), None)
                if label or (msg.level >= Log.WARN and any(k in text for k in ('lio_sam', 'go2_', 'livox', 'foxglove'))):
                    # Upstream's waiting-for-IMU and some feature messages can be INFO.
                    with lock:
                        warnings[(msg.name, label or msg.msg[:140])] += 1
                return
            now = time.monotonic()
            if topic in TYPES:
                if ch.get('type') != TYPES[topic]:
                    raise ValueError('收到意外类型 '+str(ch.get('type')))
                buf = msg._buff
                seq, stamp, frame, pos = raw_header(buf)
                if topic == MODEL:
                    value = marker(buf, pos)
                elif topic == RAW:
                    value = livox_window(buf)
                else:
                    value = cloud(buf, pos, topic in (REG, SCAN))
            else:
                h = msg.header
                seq, stamp, frame = h.seq, h.stamp.to_nsec(), h.frame_id.lstrip('/')
                if topic in (INC, ODOM):
                    p, q = msg.pose.pose.position, msg.pose.pose.orientation
                    pose = np.array((*vector(p), *vector(q), q.w))
                    if not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
                        raise ValueError('里程计位姿无效')
                    value = pose, msg.child_frame_id, msg.pose.covariance[0]
                else:
                    value = np.linalg.norm(vector(msg.linear_acceleration)
                                           ), np.linalg.norm(vector(msg.angular_velocity))
            if topic in (DESKEW, REG):
                try:
                    audit.observe('deskewed' if topic == DESKEW else 'registered', msg)
                except (ValueError, KeyError, struct.error) as exc:
                    with lock:
                        issues[('墙面抽样', str(exc))] += 1
            with lock:
                if not closed:
                    data[topic].append(Event(stamp, seq, frame, source, now, value))
        except (KeyError, ValueError, struct.error, OverflowError) as exc:
            with lock:
                issues[(topic, str(exc))] += 1

    _, _, graph = master.getSystemState('/go2_alignment_probe')
    publishers = dict(graph[0])
    # Compare every point before display filtering. Older running stacks publish
    # the full levelled scan on the original topic.
    use_full = bool(publishers.get(FULL_SCAN) or '/go2_mapping_obstacles' in publishers.get(SCAN, ()))
    scan_input = FULL_SCAN if use_full else SCAN
    for topic, typ in [(n, rospy.AnyMsg) for n in TYPES]+[(ODOM, Odometry), (INC, Odometry), (IMU, Imu),
                                                          ('/rosout_agg', Log), ('/tf', TFMessage), ('/tf_static', TFMessage)]:
        cap = 3 if topic in TYPES else 100
        subscriptions.append(rospy.Subscriber(scan_input if topic == SCAN else topic,
                                              typ, callback, callback_args=topic, queue_size=cap,
                                              buff_size=8*1024**2 if topic in TYPES else 65536))
    print('显示细节层 %s 发布者=%s，订阅者=%s（只查注册表）。' %
          (LOCAL, dict(graph[0]).get(LOCAL, []), dict(graph[1]).get(LOCAL, [])))
    print('保持静止。现在同时记录 %.1f 秒；只读取现有话题与 TF。' % seconds, flush=True)
    try:
        until = time.monotonic()+seconds
        while not rospy.is_shutdown() and time.monotonic() < until:
            time.sleep(.05)
    finally:
        with lock:
            closed = True
        for sub in subscriptions:
            sub.unregister()
    with lock:
        data = defaultdict(list, {k: list(v) for k, v in data.items()})
        issues, warnings, broadcasters = issues.copy(), warnings.copy(), broadcasters.copy()
    lookup_errors = Counter()

    def lookup(to, frm, stamp):
        if to == frm:
            return np.eye(3), np.zeros(3)
        try:
            t = tfbuf.lookup_transform(to, frm, rospy.Time(stamp//NS, stamp % NS)).transform
            return rotation((*vector(t.rotation), t.rotation.w)), np.array(vector(t.translation))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, ValueError) as exc:
            lookup_errors[str(exc)[:110]] += 1
            return None
    print('=== ALIGNMENT BEGIN ===')
    for topic in (RAW, IMU, ODOM, INC, DESKEW, REG, SCAN, MODEL):
        show_stream(topic, data[topic], scan_input if topic == SCAN else topic)
    if data[INC]:
        flags = Counter(float(r.value[2]) for r in data[INC])
        print('增量校正 pose.covariance[0] 退化标记 1：%d/%d；实际取值 %s' %
              (flags[1.], len(data[INC]), dict(flags)))
    if data[MODEL] and data[ODOM]:
        last_m = data[MODEL][-1].stamp
        print('最后里程计源戳减最后模型源戳 %.3f s（模型可因未达更新阈值而保留旧戳）' %
              ((data[ODOM][-1].stamp-last_m)/NS))
    if data[IMU]:
        vals = np.array([r.value for r in data[IMU]])
        good = np.isfinite(vals).all(axis=1)
        if good.any():
            print('滤波 IMU |a| 中位 %.3f m/s²，|ω| 中位/P95 %.4f/%.4f rad/s，无效 %d' %
                  (np.median(vals[good, 0]), np.median(vals[good, 1]), np.percentile(vals[good, 1], 95), sum(~good)))
    imu_coverage(data[RAW], data[IMU])
    for topic in (ODOM, INC):
        if data[topic]:
            print(topic, '位姿变化')
            pose_report(data[topic])
    paired_report(data, lookup)
    audit.report()
    for (topic, parent, child, source), count in sorted(broadcasters.items()):
        print('TF %s %s→%s 来源=%s n=%d' % (topic, parent, child, source, count))
    for (topic, s), cnt in issues.items():
        print('读取异常 %s ×%d %s' % (topic, cnt, s))
    for s, cnt in lookup_errors.most_common(2):
        print('TF 查询异常 ×%d %s' % (cnt, s))
    for (node, s), cnt in warnings.most_common(5):
        print('窗口内告警 %s ×%d %s' % (node, cnt, s))
    print('此次收到的相关告警/关键日志共 %d 条。' % sum(warnings.values()))
    print('=== ALIGNMENT END ===')
    return 0 if len(data[ODOM]) >= 2 and data[REG] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=12)
    args = parser.parse_args()
    if not 3 <= args.seconds <= 30:
        parser.error('采样时长须为 3–30 秒')
    if os.environ.get('GO2_ALIGNMENT_CLEAN') != '1' and os.environ.get('GO2_DETAIL_CLEAN_ENV') != '1':
        clean = {k: os.environ[k] for k in ('HOME', 'USER', 'TERM', 'LANG', 'LC_ALL') if k in os.environ}
        clean.update(GO2_ALIGNMENT_CLEAN='1', OPENBLAS_NUM_THREADS='1', PATH='/usr/local/bin:/usr/bin:/bin')
        launch = 'source /opt/ros/noetic/setup.bash && export ROS_MASTER_URI=http://127.0.0.1:11321 ROS_IP=127.0.0.1 && exec /usr/bin/python3 -u "$@"'
        os.execve('/bin/bash', ['bash', '--noprofile', '--norc', '-c', launch, 'go2-check',
                                str(Path(__file__).resolve()), *sys.argv[1:]], clean)
    try:
        return run(args.seconds)
    except (OSError, RuntimeError, ImportError) as exc:
        print('检查未完成：%s' % exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
