"""Stationary patch-width evidence from same-stamp, same-order deskewed/registered scans."""
from collections import Counter, OrderedDict
import struct
import threading
import numpy as np

MAX_FRAMES = 6
MAX_POINTS = 14000
MIN_POINTS = 20
RADIUS = .32
DEPTH = .22
STEP_NS = 750000000


def read_xyz(msg, indices):
    height, width, step = msg.height, msg.width, msg.point_step
    if step <= 0 or msg.row_step < width*step or len(msg.data) < height*msg.row_step:
        raise ValueError('点云行或字节数异常')
    fields = {f.name: f for f in msg.fields}
    xyz = np.empty((len(indices), 3), dtype=np.float32)
    for i, k in enumerate(('x', 'y', 'z')):
        f = fields.get(k)
        if f is None or f.datatype not in (7, 8) or f.count != 1:
            raise ValueError('缺少浮点 XYZ')
        typ = np.dtype(('>' if msg.is_bigendian else '<')+('f4' if f.datatype == 7 else 'f8'))
        if f.offset < 0 or f.offset+typ.itemsize > step:
            raise ValueError('点字段偏移异常')
        chan = np.ndarray((height, width), typ, buffer=msg.data, offset=f.offset, strides=(msg.row_step, step))
        xyz[:, i] = chan[indices//width, indices % width]
    return xyz


def pair_arrays(t, a, b):
    if a.header.stamp.to_nsec() != t or b.header.stamp.to_nsec() != t:
        raise ValueError('时间戳不同')
    n = a.width*a.height
    if n != b.width*b.height or n < 150:
        raise ValueError('同帧点数不同或不足')
    if n > MAX_POINTS:
        # One matching original index per stratum in both messages; randomize
        # within strata to avoid systematic subsampling of the Livox lines.
        ends = np.arange(MAX_POINTS+1, dtype=np.int64)*n//MAX_POINTS
        frac = np.random.RandomState(int(t) & 0xffffffff).random_sample(MAX_POINTS)
        idx = ends[:-1]+(np.diff(ends)*frac).astype(np.int64)
    else:
        idx = np.arange(n)
    sensor, world = read_xyz(a, idx), read_xyz(b, idx)
    norm2 = np.einsum('ij,ij->i', sensor, sensor)
    valid = (norm2 >= .45**2) & (norm2 <= 10.**2) & np.isfinite(world).all(axis=1)
    if np.count_nonzero(valid) < 150:
        raise ValueError('相同有效回波不足')
    return sensor[valid], world[valid]


def verified_rigid(src, dst):
    # The upstream point-by-point transform preserves index order. This is a
    # correspondence check, not ICP. Align every later map to the first basis.
    a, b = src[::max(1, len(src)//320)].astype(float), dst[::max(1, len(src)//320)].astype(float)
    s, t = a.mean(axis=0), b.mean(axis=0)
    u, val, vt = np.linalg.svd((a-s).T@(b-t))
    if val[1] < 1.e-4:
        raise ValueError('回波几何不足以验证同帧刚体变换')
    if np.linalg.det(u@vt) < 0:
        u[:, -1] *= -1.
    r = u@vt
    offset = t-s@r
    d = a@r+offset-b
    if np.percentile(np.sqrt(np.einsum('ij,ij->i', d, d)), 95) > .010:
        raise ValueError('同索引回波未通过 1 cm 刚体一致性检查')
    return r, offset


def residuals(points, center, n):
    delta = points-center
    dd = np.einsum('ij,ij->i', delta, delta)
    # Cheap sphere first, followed by an identical cylindrical aperture in
    # both domains. Do not refit, re-center or truncate to a noise band.
    delta = delta[dd <= RADIUS*RADIUS+DEPTH*DEPTH]
    h = np.einsum('ij,j->i', delta, n)
    tangential = np.einsum('ij,ij->i', delta, delta)-h*h
    return h[(tangential <= RADIUS*RADIUS) & (np.abs(h) <= DEPTH)]


def pick_patches(points):
    cells = np.floor(points/.58).astype(np.int64)
    offset = 32
    k = (cells[:, 0]+offset)*4096+(cells[:, 1]+offset)*64+(cells[:, 2]+offset)
    ids, inv, ns = np.unique(k, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    means = np.column_stack([np.bincount(inv, weights=v, minlength=len(ids))/ns for v in points.T])
    groups = [[] for _ in range(3)]
    for j in np.argsort(-ns)[:96]:
        if ns[j] < 8:
            break
        origin = means[j]
        d = points-origin
        sq = np.einsum('ij,ij->i', d, d)
        x = points[sq < RADIUS**2]
        if len(x) < MIN_POINTS:
            continue
        center = x.mean(axis=0)
        delta = x-center
        lam, v = np.linalg.eigh(delta.T@delta/len(x))
        if lam[1] < .003 or lam[0] > .022**2 or lam[0] > .10*lam[1]:
            continue
        normal = v[:, 0]
        h = residuals(points, center, normal)
        if len(h) < MIN_POINTS or np.diff(np.percentile(h, [5, 95]))[0] > .070:
            continue
        group = groups[int(np.argmax(np.abs(normal)))]
        if all(np.linalg.norm(center-p[0]) > .75 for p in group):
            group.append((center, normal))
    # Prefer walls as well as horizontal surfaces where there is enough support.
    out = []
    for level in range(8):
        for group in groups:
            if level < len(group) and len(out) < 8 and all(np.linalg.norm(group[level][0]-p[0]) > .60 for p in out):
                out.append(group[level])
    return out


def measure(frames):
    if len(frames) < 4:
        return [], 0
    sensor = [a for _, a, _ in frames]
    r, t = verified_rigid(*frames[0][1:])
    world = [((b-t)@r.T).astype(np.float32) for _, _, b in frames]
    patches = pick_patches(sensor[0])
    out = []
    for center, n in patches:
        measures = [[], []]
        for a, b in zip(sensor, world):
            h, k = residuals(a, center, n), residuals(b, center, n)
            if min(len(h), len(k)) < MIN_POINTS:
                continue
            measures[0].append(h)
            measures[1].append(k)
        if len(measures[0]) < 4:
            continue
        metrics = []
        for rows in measures:
            single = np.median([np.diff(np.percentile(h, [5, 95]))[0] for h in rows])
            stack = np.diff(np.percentile(np.concatenate(rows), [5, 95]))[0]
            shift = np.ptp([np.median(h) for h in rows])
            metrics.append(np.array([single, stack, shift])*1000.)
        out.append((metrics, len(measures[0]), float(np.linalg.norm(center))))
    return out, len(patches)


class SurfaceAudit:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = []
        self.failed = Counter()
        self.last = None
        self.names = None
        self.wire = {'deskewed': OrderedDict(), 'registered': OrderedDict()}

    def observe(self, side, msg):
        """Hooks for a rospy.AnyMsg subscriber. Deserialize only selected exact pairs."""
        try:
            sec, ns = struct.unpack_from('<II', msg._buff, 4)
        except (struct.error, TypeError):
            return
        t = sec*1000000000+ns
        other = 'registered' if side == 'deskewed' else 'deskewed'
        with self.lock:
            if len(self.frames) >= MAX_FRAMES or (self.last is not None and t-self.last < STEP_NS):
                return
            if t not in self.wire[other]:
                hold = self.wire[side]
                hold[t] = msg
                if len(hold) > 12:
                    hold.popitem(last=False)
                return
            peer = self.wire[other].pop(t)
            a, b = (msg, peer) if side == 'deskewed' else (peer, msg)
            self.wire[side].clear()
            self.wire[other].clear()
        from sensor_msgs.msg import PointCloud2
        from genpy import DeserializationError
        try:
            self.add_pair(t, PointCloud2().deserialize(a._buff), PointCloud2().deserialize(b._buff))
        except (DeserializationError, ValueError, struct.error):
            with self.lock:
                self.failed['三维点云字节解码失败'] += 1

    def add_pair(self, stamp_ns, deskewed, registered):
        stamp_ns = int(stamp_ns)
        with self.lock:
            if len(self.frames) >= MAX_FRAMES or (self.last is not None and stamp_ns-self.last < STEP_NS):
                return
            self.last = stamp_ns
            try:
                names = (deskewed.header.frame_id, registered.header.frame_id)
                if self.names is not None and names != self.names:
                    raise ValueError('采样中坐标系名称改变')
                a, b = pair_arrays(stamp_ns, deskewed, registered)
                verified_rigid(a, b)
                self.frames.append((stamp_ns, a, b))
                self.names = names
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                self.failed[str(exc)] += 1

    def report(self):
        with self.lock:
            frames = list(self.frames)
            failures = dict(self.failed)
        metrics, candidates = measure(frames)
        print('静止实测表面：同时间戳、点序刚体核对通过 %d 帧，固定参考平面有效 %d/%d 处。' %
              (len(frames), len(metrics), candidates))
        for reason, n in failures.items():
            print('  排除 %d 帧：%s。' % (n, reason))
        if len(metrics) < 2:
            print('  可重复观测的局部平面不足，无法比较叠帧增厚。')
            return None
        stats = np.median(np.array([v for v, _, _ in metrics]), axis=0)
        duration = (frames[-1][0]-frames[0][0])*1e-9
        print('  观测 %.1f 秒；各区域用至少 4 帧，每帧至少 %d 回波。以下是各区域结果的中位数。' %
              (duration, MIN_POINTS))
        for label, row in zip(('去畸变传感器系', '配准世界系'), stats):
            print('  %s：单帧厚度 %.1f mm，叠帧厚度 %.1f mm，各帧法向中位偏移的峰峰值 %.1f mm。' %
                  (label, row[0], row[1], row[2]))
        # Also print the worst differential and the two views of that exact
        # patch so a median cannot hide a single visibly bad local wall.
        where = max(metrics, key=lambda item: (item[0][1][1]-item[0][1][0])-(item[0][0][1]-item[0][0][0]))
        print('  世界系相对增厚最多的区域（距雷达 %.1f m）：去畸变 %.1f→%.1f mm，配准 %.1f→%.1f mm。' %
              (where[2], *where[0][0][:2], *where[0][1][:2]))
        print('  厚度为同一物理区域点到首帧平面的 P95−P05；两路使用相同固定窗口。')
        return stats
