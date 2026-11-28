#!/usr/bin/env python3
"""Short-window measured XYZI geometry from the running dense mapper's SQLite archive."""
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
import threading
import time
import xmlrpc.client
from collections import deque
from urllib.parse import urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'navigation'))
from obstacle_view import keep_connected, obstacle_mask
from dense_map import cell_ids, geometry, hashes
from corridor_export import load_corridor_document
from corridor_recorder import pose_matrix
from surveyed_corridor import CorridorIndex

GLOBAL = "/go2_visual/dense_map"
REGISTERED = "/lio_sam/mapping/cloud_registered_raw"
FALLBACK = "/go2_visual/ground/scan"
OUTPUT = "/go2_visual/detail/local"
FRAME = "go2_ground"
AGE_NS = 6000000000
MAX_SCANS = 30
RADIUS = 7.0
VOXEL = .005
CAP = 120000


def to_ground(local, pose, tf):
    r, t = geometry(pose)
    g, u = geometry(tf)
    world = local.copy()
    # Compose in double before applying to each new frame. Preserve intensity.
    world[:, :3] = local[:, :3] @ (g@r).T + (g@t+u)
    return world


def choose(candidates, capacity, dist2, k):
    if len(candidates) <= capacity:
        return candidates
    if capacity <= 0:
        return np.empty(0, np.intp)
    # Fixed per-voxel noise with a mild preference for nearby detail. Avoid
    # retaining only the nearest wall or only early frames when over budget.
    prior = 1.+.75*dist2[candidates]/(RADIUS*RADIUS)
    priority = (hashes(k[candidates]) >> np.uint64(11)).astype(float)*prior
    return candidates[np.argpartition(priority, capacity-1)[:capacity]]


def detail_mask(heights, ground_band):
    # Keep any stricter private detail setting on top of the common floor filter.
    return obstacle_mask(heights) & (heights >= ground_band)


def local_cloud(clouds, center, ground_band, limit=CAP):
    chunks = []
    for p in clouds:
        delta = p[:, :3]-center
        d2 = np.einsum('ij,ij->i', delta, delta)
        inside = np.isfinite(d2) & (d2 <= RADIUS**2)
        inside &= detail_mask(p[:, 2], ground_band)
        if inside.any():
            chunks.append(p[inside])
    if not chunks:
        return np.empty((0, 4), '<f4'), 0, 0
    p = np.concatenate(chunks)
    # Classify the whole measured window before display sampling. Splitting
    # by scan or capping first would break real connected objects into speckle.
    p = p[keep_connected(p[:, :2])]
    total = len(p)
    if not total:
        return np.empty((0, 4), '<f4'), 0, 0
    k = cell_ids(np.floor(p[:, :3].astype(np.float64)/VOXEL))
    unique, idx = np.unique(k, return_index=True)
    p = p[idx]  # newest scans came first; every output is a real observation
    if len(p) <= limit:
        return p, total, len(p)
    delta = p[:, :3]-center
    dist = np.einsum('ij,ij->i', delta, delta)
    # Reserve spatial coverage prior to filling remaining slots with 5-mm
    # detail. No coarsening of surviving voxels or interpolation of corners.
    coarse = cell_ids(np.floor(p[:, :3].astype(np.float64)/.025))
    _, reserve = np.unique(coarse, return_index=True)
    reserve = choose(reserve, max(1, int(.65 * limit)), dist, unique)
    available = np.ones(len(p), dtype=bool)
    available[reserve] = False
    detail = choose(np.flatnonzero(available), limit - len(reserve), dist, unique)
    selected = np.concatenate((reserve, detail))
    return p[selected], total, len(p)


class Window:
    def __init__(self, path):
        self.conn = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True, timeout=.3)
        self.conn.execute('PRAGMA query_only=ON')
        self.frames = {}

    def close(self):
        self.conn.close()

    def read(self, stamp, live, frame, tf, band, live_pose=None):
        db = self.conn
        db.execute('BEGIN')
        try:
            meta = db.execute("SELECT value FROM meta WHERE name='frame'").fetchone()
            if meta != (frame,):
                raise ValueError('归档坐标系与在线扫描不一致')
            rows = db.execute("SELECT s.stamp,s.anchor,p.pose FROM scans AS s JOIN poses AS p ON p.stamp=s.anchor "
                              "WHERE s.stamp BETWEEN ? AND ? ORDER BY s.stamp DESC LIMIT ?",
                              (stamp-AGE_NS, stamp, MAX_SCANS)).fetchall()
            # The queried file must contain a scan we actually saw from the
            # current registration publisher (and no older than two seconds).
            if not rows or not any(t in live and stamp-t <= 2000000000 for t, _, _ in rows):
                raise ValueError('等候与当前配准扫描同时间戳的归档')
            cache = {}
            for t, anchor, encoded in rows:
                old = self.frames.get(t)
                if old is not None and old[:3] == (anchor, encoded, tf):
                    cache[t] = old
                else:
                    if old is not None:
                        raw = old[3]
                    else:
                        blob = db.execute('SELECT points FROM scans WHERE stamp=?', (t,)).fetchone()[0]
                        raw = np.frombuffer(blob, dtype='<f4').reshape(-1, 4)
                    corrected = to_ground(raw, json.loads(encoded), tf)
                    cache[t] = (anchor, encoded, tf, raw, corrected)
            # This view reads raw historical scans, not the dense publisher's
            # already filtered points. Read corridor events and optimized poses
            # in the same transaction, so a loop closure cannot mix versions.
            document = load_corridor_document(db, frame)
            corridor = None
            if document is not None and document['poses']:
                ground_from_source = pose_matrix(tf)
                corridor = CorridorIndex([ground_from_source @ np.asarray(event['body_pose'])
                                          for event in document['poses']])
        finally:
            db.rollback()  # release the short read transaction (including on errors)
        self.frames = cache
        rg, ug = geometry(tf)
        center_xyz = json.loads(rows[0][2])[:3]
        if live_pose and live_pose[0] == frame and abs(live_pose[1]-rows[0][0]) <= 500000000:
            if np.isfinite(live_pose[2]).all():
                center_xyz = live_pose[2]
        center = rg@np.asarray(center_xyz)+ug
        clouds = [entry[4] for entry in cache.values()]
        if corridor is not None:
            clouds = [points[~corridor.contains(points)] for points in clouds]
        pts, n, vox = local_cloud(clouds, center, band)
        return rows[0][0], pts, (len(rows), n, vox)


def single_xyzi(msg, band):
    if msg.header.frame_id.lstrip('/') != FRAME:
        raise ValueError('单帧输入未贴地')
    fields = {f.name: f for f in msg.fields}
    types = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    if msg.row_step < msg.width*msg.point_step or len(msg.data) < msg.row_step*msg.height:
        raise ValueError('单帧记录长度异常')
    count = msg.width*msg.height
    out = np.empty((count, 4), dtype='<f4')
    for i, name in enumerate(('x', 'y', 'z', 'intensity')):
        f = fields.get(name)
        if f is None or f.count != 1 or f.datatype not in types:
            raise ValueError('单帧点云缺少有效 XYZI 字段')
        dtype = np.dtype(('>' if msg.is_bigendian else '<')+types[f.datatype])
        if not 0 <= f.offset <= msg.point_step-dtype.itemsize:
            raise ValueError('单帧字段偏移越界')
        if count:
            out[:, i] = np.ndarray((msg.height, msg.width), dtype, buffer=msg.data, offset=f.offset,
                                   strides=(msg.row_step, msg.point_step)).reshape(-1)
    valid = np.isfinite(out[:, :3]).all(axis=1)
    valid &= detail_mask(out[:, 2], band)
    valid[valid] &= keep_connected(out[valid, :2])
    return out[valid]


def owner_db(master, rospy):
    owners = dict(master.getSystemState()[0]).get(GLOBAL, [])
    if len(owners) != 1:
        raise ValueError('未找到唯一的在线稠密建图发布者')
    owner = owners[0]
    root = rospy.get_param(owner.rstrip('/')+'/archive_dir', '')
    if root:
        p = Path(root).expanduser()/'scans.sqlite'
        if p.is_file():
            return p
        raise ValueError('当前稠密节点的 archive_dir 尚无 scans.sqlite')
    uri = master.lookupNode(owner)
    host = urlparse(uri).hostname
    if host not in ('localhost', '127.0.0.1', '::1', socket.gethostname(), os.getenv('ROS_IP'), os.getenv('ROS_HOSTNAME')):
        raise ValueError('archive_dir 缺失，无法核对本机稠密节点进程')
    ret, _, pid = xmlrpc.client.ServerProxy(uri).getPid(rospy.get_name())
    proc = Path('/proc')/str(int(pid))
    if ret != 1 or proc.stat().st_uid != os.geteuid():
        raise ValueError('无法核对稠密节点进程')
    paths = set()
    for fd in (proc/'fd').iterdir():
        try:
            p = Path(os.readlink(str(fd)))
            if p.name == 'scans.sqlite' and p.is_absolute() and p.is_file() and os.path.samestat(fd.stat(), p.stat()):
                paths.add(p)
        except OSError:
            continue
    if len(paths) != 1:
        raise ValueError('运行中的稠密节点未唯一打开 scans.sqlite')
    return paths.pop()


class Worker:
    def __init__(self):
        import rospy
        import rosgraph
        from rosgraph.masterapi import MasterException
        import tf2_ros
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import PointCloud2, PointField
        self.rospy, self.tf_errors = rospy, (tf2_ros.LookupException,
                                             tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException)
        self.master_error = MasterException
        self.master = rosgraph.Master(rospy.get_name())
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.lock = threading.Lock()
        self.live = deque(maxlen=128)
        self.fallback = None
        self.pose = None
        self.frame = ''
        self.last = 0.
        self.pub = rospy.Publisher(OUTPUT, PointCloud2, queue_size=1, latch=False)
        self.fields = [PointField(k, 4*i, PointField.FLOAT32, 1)
                       for i, k in enumerate(('x', 'y', 'z', 'intensity'))]
        self.subs = [rospy.Subscriber(REGISTERED, PointCloud2, self.remember, queue_size=1, buff_size=4*1024**2),
                     rospy.Subscriber(FALLBACK, PointCloud2, self.single,
                                      queue_size=1, buff_size=4*1024**2),
                     rospy.Subscriber('/lio_sam/mapping/odometry', Odometry, self.odometry, queue_size=2)]
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def remember(self, msg):
        t = msg.header.stamp.to_nsec()
        with self.lock:
            frame = msg.header.frame_id
            if self.live and (self.frame != frame or t < self.live[-1]-1000000000):
                self.live.clear()
                self.fallback = None
            self.live.append(t)
            self.last = time.monotonic()
            self.frame = frame

    def single(self, msg):
        with self.lock:
            if msg.header.frame_id.lstrip('/') == FRAME:
                self.fallback = (time.monotonic(), msg)

    def odometry(self, msg):
        p = msg.pose.pose.position
        with self.lock:
            self.pose = (msg.header.frame_id, msg.header.stamp.to_nsec(), (p.x, p.y, p.z))

    def publish(self, p, header):
        from sensor_msgs.msg import PointCloud2
        self.pub.publish(PointCloud2(header=header, height=1, width=len(p), fields=self.fields,
                                     is_bigendian=False, point_step=16, row_step=len(p)*16, data=p.astype('<f4', copy=False).tobytes(),
                                     is_dense=True))

    def work(self):
        from std_msgs.msg import Header
        rospy = self.rospy
        path = None
        window = None
        next_check = 0.
        band = 0.
        try:
            while not rospy.is_shutdown():
                if self.pub.get_num_connections() == 0:
                    time.sleep(.3)
                    continue
                begin = time.monotonic()
                with self.lock:
                    live = list(self.live)
                    frame, last, fallback, pose = self.frame, self.last, self.fallback, self.pose
                try:
                    if not live or begin-last > 2.5:
                        raise ValueError('在线配准扫描暂未更新')
                    if begin >= next_check:
                        next_check = begin+3.
                        try:
                            actual = owner_db(self.master, rospy)
                        except (ValueError, OSError, self.master_error, xmlrpc.client.Error):
                            if window is not None:
                                window.close()
                            path, window = None, None
                            raise
                        band = float(rospy.get_param('~local_ground_band', 0.))
                        if not np.isfinite(band) or band < 0:
                            raise ValueError('无效的贴地筛选参数')
                        if path != actual or window is None:
                            if window is not None:
                                window.close()
                            path, window = None, None
                            window = Window(actual)
                            path = actual
                    if window is None:
                        raise ValueError('稠密归档尚未就绪')
                    tr = self.buffer.lookup_transform(
                        FRAME, frame, rospy.Time(0), rospy.Duration(.1)).transform
                    a, b = tr.translation, tr.rotation
                    tf = (a.x, a.y, a.z, b.x, b.y, b.z, b.w)
                    t, p, (n, input_n, voxel_n) = window.read(live[-1], set(live), frame, tf, band, pose)
                    hdr = Header(stamp=rospy.Time(t//10**9, t % 10**9), frame_id=FRAME)
                    self.publish(p, hdr)
                    rospy.loginfo_throttle(12, "局部实测：%d 帧，%.1f m 内 %d 回波，%.0f mm 格 %d，显示 %d 点",
                                           n, RADIUS, input_n, VOXEL*1000, voxel_n, len(p))
                except (ValueError, sqlite3.Error, OSError, RuntimeError, self.master_error, xmlrpc.client.Error)+self.tf_errors as exc:
                    rospy.logwarn_throttle(6, "局部多帧暂不可用：%s；有实时贴地扫描时显示单帧", exc)
                    if fallback and begin-fallback[0] < 1.5:
                        try:
                            self.publish(single_xyzi(fallback[1], band), fallback[1].header)
                        except ValueError as err:
                            rospy.logwarn_throttle(6, '当前贴地单帧不可显示：%s', err)
                wait = max(.10, 1.3-(time.monotonic()-begin))
                time.sleep(wait)
        finally:
            if window is not None:
                window.close()


def main():
    import rospy
    rospy.init_node('go2_detail_local')
    worker = Worker()
    rospy.spin()


if __name__ == '__main__':
    main()

