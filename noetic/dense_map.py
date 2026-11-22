#!/usr/bin/env python3
"""Noetic all-route dense map from LIO-SAM's already registered high-res scans."""
import json
import math
import sqlite3
import time
from collections import OrderedDict, deque
from pathlib import Path
from threading import Lock

import numpy as np

CLOUD = "/lio_sam/mapping/cloud_registered_raw"
PATH = "/lio_sam/mapping/path"
OUTPUT = "/go2_visual/dense_map"
OFFSET = 1 << 20


def geometry(pose):
    tx, ty, tz, x, y, z, w = pose
    norm = x*x + y*y + z*z + w*w
    if norm < 1e-10 or not np.isfinite(pose).all():
        raise ValueError("Invalid keyframe pose")
    s = 2.0 / norm
    R = np.array([[1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
                  [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
                  [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)]], dtype=np.float64)
    return R, np.array([tx, ty, tz], dtype=np.float64)


def transform(points, pose, inverse=False):
    R, t = geometry(pose)
    result = points.copy()
    if inverse:
        result[:, :3] = (points[:, :3] - t) @ R
    else:
        result[:, :3] = points[:, :3] @ R.T + t
    return result


def decode(msg):
    fields = {f.name: f for f in msg.fields}
    xyz = ("x", "y", "z")
    if any(k not in fields or fields[k].datatype != 7 for k in xyz):
        raise ValueError("Expected PointCloud2 FLOAT32 XYZ")
    endian = ">" if msg.is_bigendian else "<"
    dims = (msg.height, msg.width)
    pts = np.zeros((msg.height*msg.width, 4), dtype="<f4")
    if len(pts):
        for j, name in enumerate(xyz + ("intensity",)):
            if name in fields and fields[name].datatype == 7:
                pts[:, j] = np.ndarray(dims, dtype=endian+"f4", buffer=msg.data,
                                       offset=fields[name].offset,
                                       strides=(msg.row_step, msg.point_step)).reshape(-1)
    return pts[np.isfinite(pts[:, :3]).all(axis=1)]


def cell_ids(cells):
    if len(cells) and (cells.min() < -OFFSET or cells.max() >= OFFSET):
        raise ValueError("Position exceeds visualization coordinate range; inspect mapping odometry")
    c = cells.astype(np.int64, copy=False) + OFFSET
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def hashes(keys):
    # SplitMix64 gives a stable, spatially uniform retention decision per voxel.
    z = keys.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


class Display:
    def __init__(self, voxel, limit):
        self.voxel, self.limit = voxel, limit
        self.keys = np.empty(0, np.int64)
        self.points = np.empty((0, 4), '<f4')
        self.scores = np.empty(0, np.float32)
        self.hash_cutoff = None

    def add(self, pts, *, stop_at=None):
        scaled = pts[:, :3].astype(np.float64) / self.voxel
        cells = np.floor(scaled)
        try:
            keys = cell_ids(cells)
        except ValueError:
            if stop_at is not None:
                return False
            raise
        offset = scaled - cells - 0.5
        scores = np.einsum("ij,ij->i", offset, offset).astype(np.float32)
        if self.hash_cutoff is not None:
            keep = hashes(keys) <= self.hash_cutoff
            keys, pts, scores = keys[keep], pts[keep], scores[keep]
        if not len(keys):
            return True
        # Use a measured return closest to each voxel center, including its
        # intensity. Averaging observations across a thin railing can put a
        # synthetic point in the gap, and a first-hit policy keeps noisy edges.
        order = np.lexsort((scores, keys))
        sorted_keys = keys[order]
        idx = order[np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]]
        fresh = keys[idx]
        pos = self.keys.searchsorted(fresh)
        exists = pos < len(self.keys)
        exists[exists] = self.keys[pos[exists]] == fresh[exists]
        dst, src = pos[exists], idx[exists]
        better = scores[src] < self.scores[dst]
        self.points[dst[better]] = pts[src[better]]
        self.scores[dst[better]] = scores[src[better]]
        if exists.all():
            return True
        new_k, new_idx = fresh[~exists], idx[~exists]
        if stop_at is not None and len(self.keys) + len(new_k) > stop_at:
            return False
        # Merge the two sorted arrays without a full re-sort of the map.
        dest = self.keys.searchsorted(new_k) + np.arange(len(new_k))
        merged_k = np.empty(len(self.keys)+len(new_k), dtype=np.int64)
        merged_p = np.empty((len(merged_k), 4), dtype='<f4')
        merged_s = np.empty(len(merged_k), dtype=np.float32)
        old = np.ones(len(merged_k), dtype=bool)
        old[dest] = False
        merged_k[old], merged_p[old], merged_s[old] = self.keys, self.points, self.scores
        merged_k[dest], merged_p[dest], merged_s[dest] = new_k, pts[new_idx], scores[new_idx]
        if len(merged_k) > self.limit:
            # Keep ~90% of the budget, with headroom for new voxels. Stable
            # hash sampling covers the whole route without halving the map.
            weights = hashes(merged_k)
            first_excess = int(np.partition(weights, self.limit)[self.limit])
            cutoff = (1 << 64) - 1 if self.hash_cutoff is None else int(self.hash_cutoff)
            while cutoff >= first_excess:
                cutoff = (9 * cutoff) // 10
            self.hash_cutoff = np.uint64(cutoff)
            retain = weights <= self.hash_cutoff
            merged_k, merged_p, merged_s = merged_k[retain], merged_p[retain], merged_s[retain]
        self.keys, self.points, self.scores = merged_k, merged_p, merged_s
        return True


def pose_changed(first, latest, distance2, min_cos2):
    """Compare rigid transforms, treating q and -q as the same rotation."""
    if first == latest:
        return False
    dx, dy, dz = (latest[i] - first[i] for i in range(3))
    if dx*dx + dy*dy + dz*dz >= distance2:
        return True
    a, b = first[3:], latest[3:]
    dot = sum(x*y for x,y in zip(a,b))
    norm_product = sum(x*x for x in a) * sum(x*x for x in b)
    return dot * dot < min_cos2 * norm_product


class AnchoredDisplay(Display):
    """The pose actually used to render each contributing keyframe."""
    def __init__(self, voxel, limit, fine_voxel=0.):
        super().__init__(voxel, limit)
        self.anchors = {}
        self.fine = Display(fine_voxel, limit) if fine_voxel else None
        self.fine_limit = max(1, 9 * limit // 10)

    def add(self, pts):
        super().add(pts)
        if self.fine is not None and not self.fine.add(pts, stop_at=self.fine_limit):
            self.fine = None

    def display_points(self):
        if self.fine is None:
            return self.points
        # Preserve the parent cell's measured winner even when the nearest
        # fine-cell winner would have been a different return. Do this only
        # when publishing, with one output row per occupied fine cell.
        points = self.fine.points.copy()
        scaled = self.points[:, :3].astype(np.float64) / self.fine.voxel
        idx = self.fine.keys.searchsorted(cell_ids(np.floor(scaled)))
        points[idx] = self.points
        return points

    def add_scan(self, anchor, local, pose, world=None, world_pose=None):
        baseline = self.anchors.setdefault(anchor, pose)
        if world is not None and world_pose == baseline:
            self.add(world)
        else:
            self.add(transform(local, baseline))


class MapPreview:
    """Continue growing the active map while a finite archive snapshot is replayed."""
    def __init__(self, voxel, limit, distance, angle_deg, interval, fine_voxel=0.):
        if fine_voxel not in (0., voxel, voxel / 2):
            raise ValueError("fine_voxel must be 0 (disabled), voxel, or half of voxel")
        self.fine_voxel = fine_voxel if fine_voxel == voxel / 2 else 0.
        self.active = AnchoredDisplay(voxel, limit, self.fine_voxel)
        self.building = None
        self.snapshot = {}
        self.distance2 = distance ** 2
        self.min_cos2 = math.cos(math.radians(angle_deg) * .5) ** 2
        self.interval = interval
        self.next_check = 0.
        self.through = -1
        self.cursor = -1
        self.last_stamp = -1

    def sync_resolution(self):
        # Once either copy exhausts its detail allowance, keep the continuously
        # maintained baseline for the rest of this session (including replays).
        if self.fine_voxel and (self.active.fine is None or
                               (self.building is not None and self.building.fine is None)):
            self.fine_voxel = 0.
            self.active.fine = None
            if self.building is not None:
                self.building.fine = None

    def add(self, stamp, anchor, local, latest, world, at_capture):
        self.last_stamp = stamp
        self.active.add_scan(anchor, local, latest, world, at_capture)
        if self.building is not None:
            # This scan was appended after the snapshot. Include it in both
            # views exactly once so a growing input never moves the replay fence.
            pose = self.snapshot.setdefault(anchor, latest)
            self.building.add_scan(anchor, local, pose, world, at_capture)
        self.sync_resolution()

    def maybe_begin(self, store, now):
        if self.building is not None or self.last_stamp < 0 or now < self.next_check:
            return False
        self.next_check = now + 1.
        if not any(pose_changed(base, store.poses[key], self.distance2, self.min_cos2)
                   for key,base in self.active.anchors.items()):
            return False
        self.snapshot = store.poses.copy()
        self.through, self.cursor = self.last_stamp, -1
        self.building = AnchoredDisplay(self.active.voxel, self.active.limit, self.fine_voxel)
        return True

    def advance(self, store, now, max_rows=2):
        if self.building is None:
            return False
        rows = store.db.execute(
            "SELECT stamp,anchor,points FROM scans WHERE stamp>? AND stamp<=? "
            "ORDER BY stamp LIMIT ?", (self.cursor, self.through, max_rows))
        for stamp, anchor, blob in rows:
            local = np.frombuffer(blob, dtype='<f4').reshape(-1, 4)
            self.building.add_scan(anchor, local, self.snapshot[anchor])
            self.cursor = stamp
        self.sync_resolution()
        if self.cursor < self.through:
            return False
        self.active, self.building = self.building, None
        self.snapshot = {}
        self.next_check = now + self.interval
        return True


class Archive:
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.file = directory / "scans.sqlite"
        # Atomically reserve the path so a new session never mixes with old frames.
        with self.file.open("xb"):
            pass
        self.db = sqlite3.connect(str(self.file))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE scans (stamp INTEGER PRIMARY KEY, anchor INTEGER NOT NULL, points BLOB NOT NULL);
            CREATE TABLE poses (stamp INTEGER PRIMARY KEY, pose TEXT NOT NULL);
            CREATE TABLE meta (name TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        self.poses = {}
        self.origin = None
        self.frame = None

    def path(self, frame, rows):
        if not rows:
            return False
        if self.origin is not None and (self.origin != next(iter(rows)) or frame != self.frame):
            raise ValueError("LIO-SAM restarted or changed world frame. Start a fresh dense-map session.")
        if self.origin is None:
            self.origin, self.frame = next(iter(rows)), frame
            self.db.execute("INSERT INTO meta VALUES('frame', ?)", (frame,))
        if self.poses.keys() - rows.keys():
            raise ValueError("LIO-SAM path lost old anchors. Start a fresh dense-map session.")
        changed = [(t, p) for t,p in rows.items() if self.poses.get(t) != p]
        corrected = any(t in self.poses for t,p in changed)
        self.db.executemany("INSERT OR REPLACE INTO poses VALUES(?,?)",
                            ((t, json.dumps(p, allow_nan=False)) for t,p in changed))
        self.poses.update(changed)
        return corrected

    def add(self, stamp, anchor, points):
        self.db.execute("INSERT INTO scans VALUES(?,?,?)",
                        (stamp, anchor, points.astype('<f4', copy=False).tobytes()))


def pose_values(row):
    p, q = row.pose.position, row.pose.orientation
    return (p.x, p.y, p.z, q.x, q.y, q.z, q.w)


class Inputs:
    """Join by header.stamp (same mapping callback); each scan retains its contemporaneous anchor."""
    def __init__(self):
        self.guard = Lock()
        self.clouds, self.paths = OrderedDict(), OrderedDict()
        self.ready = deque(maxlen=48)
        self.latest_path = None
        self.latest_cloud = None
        self.dropped = 0

    def pair(self, stamp):
        if stamp in self.clouds and stamp in self.paths:
            if len(self.ready) == self.ready.maxlen:
                self.dropped += 1
            self.ready.append((self.clouds.pop(stamp), self.paths.pop(stamp)))
        for table in (self.clouds, self.paths):
            if len(table) > 32:
                table.popitem(last=False)
                self.dropped += 1

    def cloud(self, msg):
        with self.guard:
            t = msg.header.stamp.to_nsec()
            self.clouds[t] = msg
            self.latest_cloud = msg
            self.pair(t)

    def path(self, msg):
        if not msg.poses:
            return
        row = msg.poses[-1]
        with self.guard:
            t = msg.header.stamp.to_nsec()
            if self.latest_path is None or t >= self.latest_path.header.stamp.to_nsec():
                self.latest_path = msg
            self.paths[t] = (msg.header.frame_id, row.header.stamp.to_nsec(), pose_values(row))
            self.pair(t)

    def take(self):
        with self.guard:
            pairs, self.ready = self.ready, deque(maxlen=48)
            return self.latest_path, pairs, self.latest_cloud, self.dropped


def run():
    import rospy
    from nav_msgs.msg import Path as RosPath
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    import tempfile

    rospy.init_node("go2_l1_style_dense")
    voxel = float(rospy.get_param("~voxel", .015))
    fine_voxel = float(rospy.get_param("~fine_voxel", 0.))
    limit = int(rospy.get_param("~max_points", 450000))
    period = float(rospy.get_param("~period", 3.5))
    sample = float(rospy.get_param("~scan_period", .09))
    distance = float(rospy.get_param("~reproject_translation", .02))
    angle = float(rospy.get_param("~reproject_rotation_deg", .15))
    interval = float(rospy.get_param("~reproject_interval", 5.))
    if (not all(math.isfinite(v) and v > 0 for v in
                (voxel, limit, period, sample, distance, angle, interval)) or angle > 180):
        raise ValueError("Positive finite map settings and rotation <= 180 degrees required")
    folder = str(rospy.get_param("~archive_dir", ""))
    if not folder:
        root = Path.home() / "go2_dense_runs"
        root.mkdir(exist_ok=True)
        folder = tempfile.mkdtemp(prefix="mid360s_", dir=str(root))
    store = Archive(Path(folder).expanduser())
    rospy.loginfo("Archive=%s; registered scan -> world map, voxel %.3f m, display cap %d", store.file, voxel, limit)
    preview = MapPreview(voxel, limit, distance, angle, interval, fine_voxel)
    inputs = Inputs()
    target = rospy.Publisher(OUTPUT, PointCloud2, queue_size=1, latch=True)
    fields = [PointField(n, 4*i, PointField.FLOAT32, 1) for i,n in enumerate(("x","y","z","intensity"))]
    subs = [rospy.Subscriber(CLOUD, PointCloud2, inputs.cloud, queue_size=6, buff_size=8388608),
            rospy.Subscriber(PATH, RosPath, inputs.path, queue_size=6, buff_size=8388608)]
    last_path = None
    last_capture = -10**18
    last_pub = 0.
    count = 0
    last_scan_points = 0
    dirty = False
    fallback_stamp = None
    rate = rospy.Rate(20)
    try:
        while not rospy.is_shutdown():
            path, pairs, recent, dropped = inputs.take()
            if path is not None and path is not last_path:
                rows = {row.header.stamp.to_nsec(): pose_values(row) for row in path.poses}
                store.path(path.header.frame_id, rows)
                last_path = path
            for msg,(frame,anchor,at_capture) in sorted(pairs, key=lambda b:b[0].header.stamp.to_nsec()):
                stamp = msg.header.stamp.to_nsec()
                if stamp - last_capture < sample*1e9 or anchor not in store.poses:
                    continue
                if frame != msg.header.frame_id or frame != store.frame:
                    raise ValueError("Path/cloud frame mismatch")
                points = decode(msg)
                if not len(points):
                    continue
                local = transform(points, at_capture, inverse=True)
                store.add(stamp, anchor, local)
                last_scan_points = len(points)
                preview.add(stamp, anchor, local, store.poses[anchor], points, at_capture)
                last_capture, count, dirty = stamp, count + 1, True
            store.db.commit()
            now = time.monotonic()
            if preview.maybe_begin(store, now):
                rospy.loginfo("Updating dense preview through scan %d; current map remains live", count)
            if preview.advance(store, now):
                dirty = True
            if dirty and now-last_pub >= period and len(preview.active.points):
                p = preview.active.display_points()
                hdr = Header(stamp=rospy.Time(last_capture//10**9, last_capture%10**9), frame_id=store.frame)
                target.publish(PointCloud2(header=hdr, height=1, width=len(p), fields=fields,
                                   is_bigendian=False, point_step=16, row_step=16*len(p),
                                   data=p.astype('<f4',copy=False).tobytes(), is_dense=True))
                dirty,last_pub = False,now
                rospy.loginfo_throttle(
                    12, "Dense: %d scans saved, %d displayed, %d unmatched/queue losses; latest registered scan: %d finite points, display voxel %.1f mm",
                    count, len(p), dropped, last_scan_points, 1000. * (preview.fine_voxel or voxel),
                )
            elif not count and recent is not None and recent.header.stamp != fallback_stamp and now-last_pub >= period:
                target.publish(recent)
                fallback_stamp,last_pub = recent.header.stamp,now
                rospy.logwarn_throttle(5, "Showing one registered scan; waiting for matching mapping/path")
            rate.sleep()
    finally:
        for sub in subs: sub.unregister()
        store.db.close()


if __name__ == "__main__":
    import rospy
    try:
        run()
    except rospy.ROSInterruptException:
        pass
