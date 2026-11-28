"""Freeze surveyed mapping corridors and preserve pre-clearance terrain evidence."""
import json
from pathlib import Path
import sys
import time

import numpy as np

from dense_map import geometry, transform


class SourceFrameCorridor:
    """Test source-frame returns against gravity-aligned, physical body strips."""
    def __init__(self, ground, events):
        from surveyed_corridor import CorridorIndex
        self.ground = rigid_matrix(ground)
        self.index = CorridorIndex([self.ground @ np.asarray(event['body_pose']) for event in events])

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


def prepare_corridor(db, poses, frame, directory, created):
    """Return (document, index), or (None, None) for unmarked historical runs.

    Raw scans remain in the run database. A marked new map stores only one PCD;
    its terrain model is fitted before the user's surveyed strip is erased.
    """
    document = load_corridor_document(db, frame, poses)
    if document is None:
        return None, None
    from surveyed_corridor import CorridorIndex, make_document
    events = document['poses']
    if not events:
        return document, CorridorIndex([])
    marker = json.loads(db.execute("SELECT value FROM meta WHERE name='surveyed_corridor'").fetchone()[0])

    # These modules are pure NumPy/SQLite helpers; saving needs no ROS2 runtime,
    # Open3D, SciPy or C++ compilation on the Noetic host.
    navigation = Path(__file__).resolve().parents[1] / 'navigation'
    sys.path.insert(0, str(navigation))
    sys.path.insert(0, str(navigation / 'original_go2/go2_3d'))
    from prepare_map import transforms, apply
    from terrain import build_terrain

    ground, mount, sensor = transforms(Path(directory), frame)
    index = SourceFrameCorridor(ground, events)
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
    anchors = np.array([body[:3, 3] for body in bodies])
    anchors[:, 2] -= body_height
    print(f'正在保存 {len(events):,} 段人工确认通行区；先从原始扫描保留局部地形依据……', flush=True)
    last_report = [time.monotonic()]
    def progress(stage, count):
        now = time.monotonic()
        if stage == 'supported_cells' or now-last_report[0] >= 5.:
            print(f'保存地形：{stage} {count:,}', flush=True)
            last_report[0] = now
    model = build_terrain((apply(world, ground) for _, _, world in world_scans(db, poses)),
                          anchors, work_directory=directory, progress=progress)
    target = Path(directory) / 'survey_terrain.npz'
    if target.exists():
        raise FileExistsError(target)
    terrain = model.save(target)
    created.append(target)
    terrain.update(map_from_source_map=ground.tolist(), source_frame=frame.strip('/'),
                   body_height_above_ground=body_height, body_from_lidar=mount.tolist(),
                   evidence='raw_snapshot_before_surveyed_clearance')
    document = make_document(frame, events, terrain=terrain)
    return document, index
