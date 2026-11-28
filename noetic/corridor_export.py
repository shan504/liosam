"""Freeze continuous surveyed corridors without scanning/fitting terrain."""
import json
from pathlib import Path
import sys

import numpy as np

from dense_map import geometry, transform


class SourceFrameCorridor:
    """Test source-frame returns against gravity-aligned, physical body strips."""
    def __init__(self, ground, document):
        from surveyed_corridor import index_from_document
        self.ground = rigid_matrix(ground)
        self.index = index_from_document(document, self.ground)

    def contains(self, points):
        xyz = np.asarray(points[:, :3], dtype=np.float64)
        level = xyz @ self.ground[:3, :3].T + self.ground[:3, 3]
        return self.index.contains(level)


def pose_matrix(values):
    rotation, translation = geometry(values)
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, translation
    return result


def rigid_matrix(value):
    result = np.asarray(value, dtype=np.float64)
    if (result.shape != (4, 4) or not np.isfinite(result).all() or
            not np.allclose(result[3], [0., 0., 0., 1.], atol=1e-8) or
            not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(result[:3, :3]), 1., atol=1e-6)):
        raise ValueError('人工通行区的锚点相对机身位姿无效')
    return result


def world_scans(db, poses):
    """Replay a caller-owned frozen SQLite snapshot, one measured scan at a time."""
    for stamp, anchor, blob in db.execute('SELECT stamp,anchor,points FROM scans ORDER BY stamp'):
        if not blob or len(blob) % 16 or anchor not in poses:
            raise ValueError(f'扫描 {stamp} 的点数据损坏或缺少对应关键帧 {anchor}')
        local = np.frombuffer(blob, dtype='<f4').reshape(-1, 4)
        world = transform(local, poses[anchor])
        if not np.isfinite(world[:, :3]).all():
            raise ValueError(f'扫描 {stamp} 的地图坐标无效')
        yield stamp, anchor, world


def load_corridor_document(db, frame, poses=None):
    """Read corrected events in a caller-owned SQLite snapshot; no files written."""
    marker = db.execute("SELECT value FROM meta WHERE name='surveyed_corridor'").fetchone()
    if marker is None:
        return None
    from surveyed_corridor import POLICY, make_document

    marker = json.loads(marker[0])
    if (not isinstance(marker, dict) or marker.get('enabled') is not True or
            marker.get('policy') != POLICY):
        raise ValueError('新建图人工通行区策略无效；不能按旧地图猜测清理范围')
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'corridor_events' not in tables:
        raise ValueError('新建图已声明人工通行区，但缺少逐帧确认记录')
    if poses is None:
        poses = {stamp: json.loads(pose) for stamp, pose in db.execute('SELECT stamp,pose FROM poses ORDER BY stamp')}
    events = []
    for stamp, anchor, relative, scan_anchor in db.execute(
            'SELECT e.stamp,e.anchor,e.body_relative,s.anchor FROM corridor_events e '
            'LEFT JOIN scans s ON s.stamp=e.stamp ORDER BY e.stamp'):
        if stamp <= 0 or anchor not in poses or scan_anchor != anchor:
            raise ValueError('人工通行区缺少同时间戳扫描或对应的优化锚点')
        body = pose_matrix(poses[anchor]) @ rigid_matrix(json.loads(relative))
        events.append(dict(stamp_ns=int(stamp), anchor_stamp_ns=int(anchor), body_pose=body.tolist()))
    return make_document(frame, events)


def continuous_paths(db, poses, events, ground, mount):
    """Follow recorded optimized keyframes, never shortcut a curved trajectory.

    Confirmed forward events bound this operation. Scan outages or implausible
    tracking jumps break a path; a missing forward event alone does not. These
    transport checks do not impose any map-size or total route-length limit.
    """
    if not events:
        return []
    first, last = events[0]['stamp_ns'], events[-1]['stamp_ns']
    inverse_mount = np.linalg.inv(mount)
    route = {stamp: dict(stamp_ns=int(stamp), anchor_stamp_ns=int(stamp),
                         body_pose=(pose_matrix(pose) @ inverse_mount).tolist())
             for stamp, pose in poses.items() if first <= stamp <= last}
    route.update((event['stamp_ns'], event) for event in events)
    stamps = np.array([row[0] for row in db.execute(
        'SELECT stamp FROM scans WHERE stamp BETWEEN ? AND ? ORDER BY stamp',
        (first, last))], dtype=np.int64)
    # Counting large source-stamp gaps by prefix sum keeps per-edge checks O(log N).
    outages = np.r_[0, np.cumsum(np.diff(stamps) > 2_000_000_000)]
    paths, current = [], []
    previous = None
    for event in (route[stamp] for stamp in sorted(route)):
        stamp = event['stamp_ns']
        body = ground @ np.asarray(event['body_pose'], dtype=float)
        broken = False
        if previous is not None:
            old_stamp, old_body = previous
            lo = max(0, int(np.searchsorted(stamps, old_stamp, side='right'))-1)
            hi = min(len(stamps)-1, int(np.searchsorted(stamps, stamp, side='left')))
            dt = (stamp-old_stamp)*1e-9
            delta = body[:3, 3]-old_body[:3, 3]
            forward = old_body[:2, 0]+body[:2, 0]
            heading_norm = float(np.linalg.norm(forward))
            reversed_motion = (heading_norm > 1e-6 and
                               float(delta[:2] @ (forward/heading_norm)) < -.05)
            broken = (dt <= 0 or outages[hi] > outages[lo] or
                      float(np.linalg.norm(delta)) > .5 + 3.*dt or reversed_motion)
        if broken and current:
            paths.append(current)
            current = []
        current.append(event)
        previous = stamp, body
    if current:
        paths.append(current)
    # After a tracking break, only connect portions bounded by actual forward
    # confirmations. Unconfirmed leading/trailing keyframes do not grant free space.
    confirmed = {event['stamp_ns'] for event in events}
    bounded = []
    for path in paths:
        confirmations = [i for i, event in enumerate(path) if event['stamp_ns'] in confirmed]
        if confirmations:
            bounded.append(path[confirmations[0]:confirmations[-1]+1])
    return bounded


def prepare_corridor(db, poses, frame, directory, created, build_index=True):
    """Return (document, index), or (None, None) for unmarked historical runs.

    Save compact source-frame path/calibration metadata independently of the
    complete localization PCD; no point-cloud terrain fit or NPZ.
    """
    document = load_corridor_document(db, frame, poses)
    if document is None:
        return None, None
    from surveyed_corridor import make_document
    events = document['poses']
    marker = json.loads(db.execute("SELECT value FROM meta WHERE name='surveyed_corridor'").fetchone()[0])

    # Calibration reads only small metadata files. It does not replay raw points.
    navigation = Path(__file__).resolve().parents[1] / 'navigation'
    sys.path.insert(0, str(navigation))
    from prepare_map import transforms

    ground, mount, sensor = transforms(Path(directory), frame)
    physical = marker.get('mount')
    if (not isinstance(physical, dict) or physical.get('parent') != 'base_link' or
            str(physical.get('child', '')).lstrip('/') != sensor):
        raise ValueError('人工通行区缺少建图时的物理安装外参')
    recorded_mount = pose_matrix([*physical['xyz'], *physical['xyzw']])
    if not np.allclose(recorded_mount, mount, atol=1e-6):
        raise ValueError('保存时安装外参与人工通行区使用的外参不同')
    inverse_mount = np.linalg.inv(mount)
    bodies = [ground @ pose_matrix(pose) @ inverse_mount for _, pose in sorted(poses.items())]
    body_height = float(bodies[0][2, 3])
    if not np.isfinite(body_height) or not .1 < body_height < 1.:
        raise ValueError('人工通行区地图的建图起点站立高度无效')
    calibration = dict(map_from_source_map=ground.tolist(), source_frame=frame.strip('/'),
                       body_height_above_ground=body_height, body_from_lidar=mount.tolist())
    paths = continuous_paths(db, poses, events, ground, mount)
    document = make_document(frame, events, paths=paths, calibration=calibration)
    index = SourceFrameCorridor(ground, document) if build_index else None
    print(f'正在沿优化轨迹连接 {len(events):,} 段通行区（{len(paths):,} 条连续记录）；'
          '仅保存导航通行带，保留完整定位点云，不执行地形拟合。', flush=True)
    return document, index
