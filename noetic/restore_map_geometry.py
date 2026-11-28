#!/usr/bin/env python3
"""Re-export a saved map's original registered geometry without ROS or a farm visit."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np

from dense_map import geometry, transform
from save_navigation_dense import export_dense
from save_navigation_map import write_checksums
from surveyed_corridor import read_document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'navigation'))
from select_map import select


class SavedSnapshot:
    """Reconstruct exactly the saved scan/pose snapshot, not the run's later poses."""
    def __init__(self, bundle, manifest):
        self.bundle = Path(bundle)
        self.manifest = manifest
        dense = manifest['dense']
        self.frame = dense['frame_id']
        self.poses = {}
        with (self.bundle / 'keyframes.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                stamp = int(row['stamp_ns'])
                pose = [float(row[k]) for k in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')]
                geometry(pose)
                if stamp in self.poses or stamp <= 0:
                    raise ValueError('存档关键帧时间戳重复或无效')
                self.poses[stamp] = pose
        self.scans = []
        offset, previous = 0, 0
        with (self.bundle / 'scans.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                stamp, anchor = int(row['stamp_ns']), int(row['anchor_stamp_ns'])
                count = int(row['point_count'])
                if (stamp <= previous or anchor not in self.poses or count < 0 or
                        int(row['first_point_index']) != offset):
                    raise ValueError('存档扫描索引不连续或缺少关键帧')
                self.scans.append((stamp, anchor, count))
                offset += count
                previous = stamp
        if not self.scans or not self.poses:
            raise ValueError('存档没有扫描/关键帧')
        self.scan_count = len(self.scans)
        self.first, self.last = self.scans[0][0], self.scans[-1][0]
        if (dense['scans'] != self.scan_count or dense['points'] != offset or
                dense['first_stamp_ns'] != self.first or dense['last_stamp_ns'] != self.last):
            raise ValueError('扫描索引与地图清单不一致')
        corridor_name = dense.get('surveyed_corridor')
        if corridor_name and Path(corridor_name).name != corridor_name:
            raise ValueError('通行带文件路径必须在地图版本内')
        self.corridor = read_document(self.bundle / corridor_name, expected_frame=self.frame) if corridor_name else None
        self.source_points = 0

    def validate(self, db, frame):
        if frame != self.frame:
            raise ValueError('运行归档坐标系与已保存地图不一致')
        source_points = 0
        # Only cheap lengths/anchors here. Raw blobs are read once by world_scans.
        for stamp, anchor, saved_count in self.scans:
            row = db.execute('SELECT anchor,length(points) FROM scans WHERE stamp=?', (stamp,)).fetchone()
            if row is None or row[0] != anchor or row[1] <= 0 or row[1] % 16 or row[1] // 16 < saved_count:
                raise ValueError(f'原始扫描 {stamp} 不存在、锚点不匹配或数据长度异常')
            source_points += row[1] // 16
        dense = self.manifest['dense']
        expected = dense.get('source_points', dense['points'])
        if expected != source_points:
            raise ValueError('原始扫描点数与本版本保存时记录不符，不能混用不同运行归档')
        self.source_points = source_points
        if self.corridor:
            marker = db.execute("SELECT value FROM meta WHERE name='surveyed_corridor'").fetchone()
            if marker is None:
                raise ValueError('原始运行缺少本版本通行带的来源记录')
            marker = json.loads(marker[0])
            if marker.get('policy') != self.corridor['policy']:
                raise ValueError('原始运行与存档的通行带规则不同')
            # For v2, compare the real mount, not only the label/child frame.
            if self.corridor.get('calibration'):
                from corridor_export import pose_matrix
                mount = marker.get('mount', {})
                physical = pose_matrix([*mount['xyz'], *mount['xyzw']])
                if mount.get('parent') != 'base_link' or not np.allclose(
                        physical, self.corridor['calibration']['body_from_lidar'], atol=1e-6):
                    raise ValueError('原始运行的物理安装外参与本版本不同')

    def world_scans(self, db):
        for stamp, anchor, _ in self.scans:
            row = db.execute('SELECT points FROM scans WHERE stamp=?', (stamp,)).fetchone()
            world = transform(np.frombuffer(row[0], dtype='<f4').reshape(-1, 4), self.poses[anchor])
            if not np.isfinite(world[:, :3]).all():
                raise ValueError(f'原始扫描 {stamp} 有无效地图坐标')
            yield stamp, anchor, world


def _metadata_checksums(bundle):
    """Check recorded snapshot inputs; no redundant full point-cloud read."""
    path = bundle / 'SHA256SUMS'
    if not path.is_file():
        raise ValueError('原存档缺少 SHA256SUMS，无法校验用于恢复的轨迹和标定')
    required = {'manifest.json', 'scans.csv', 'keyframes.csv'}
    required.update(str(p.relative_to(bundle)) for p in (bundle / 'calibration').rglob('*') if p.is_file())
    if (bundle / 'surveyed_corridor.json').exists():
        required.add('surveyed_corridor.json')
    entries = {}
    for line in path.read_text().splitlines():
        digest, name = line.split('  ', 1)
        entries[name] = digest
    for name in required:
        if name not in entries or hashlib.sha256((bundle / name).read_bytes()).hexdigest() != entries[name]:
            raise ValueError('原存档元数据校验失败：' + name)


def restore(argument, root):
    root = Path(root).expanduser().resolve()
    bundle = select(argument, root)
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    name = manifest.get('name', '')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('存档地图名无效')
    if manifest.get('dense', {}).get('pc_geometry') == 'full_registered_returns':
        print('当前地图已保留完整配准几何，无需恢复：' + str(bundle))
        return bundle
    _metadata_checksums(bundle)
    source_text = manifest.get('source_run', {}).get('source_archive_dir')
    session_text = manifest.get('session', {}).get('archive_dir')
    if not source_text or not session_text:
        raise ValueError('存档未记录原始运行目录，不能猜测使用其他扫描')
    archive = Path(source_text).expanduser().resolve()
    if archive != Path(session_text).expanduser().resolve():
        raise ValueError('清单来源目录与建图会话不符')
    database = archive / 'scans.sqlite'
    if not database.is_file():
        raise ValueError('本次建图的原始扫描已不在本机：' + str(database) + '；已删除几何无法从旧 PCD 还原，需要重新建图')
    snapshot = SavedSnapshot(bundle, manifest)
    parent = root / name
    parent.mkdir(parents=True, exist_ok=True)
    # Full raw point count was saved even when the former PCD was carved.
    payload = int(manifest['dense'].get('source_points', manifest['dense']['points'])) * 16
    if payload >= shutil.disk_usage(parent).free:
        raise OSError('磁盘空间不足以保存恢复后的完整地图')
    stamp = datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')
    stage = Path(tempfile.mkdtemp(prefix='.restoring-' + stamp + '_', dir=parent))
    final = parent / stage.name[len('.restoring-'):]
    try:
        # Preserve the original calibration/live provenance, but rebuild indices
        # and main PCD together. Never copy/delete/overwrite the existing PCD.
        for source in bundle.iterdir():
            if source.name in {manifest['map_file'], 'manifest.json', 'scans.csv', 'keyframes.csv',
                               'SHA256SUMS', 'surveyed_corridor.json'}:
                continue
            if source.is_symlink():
                raise ValueError('地图存档包含符号链接，不能复制：' + source.name)
            if source.is_dir():
                shutil.copytree(source, stage / source.name)
            elif source.is_file():
                shutil.copy2(source, stage / source.name)
        print('正在恢复原版本的完整配准几何；使用原扫描范围和原优化轨迹，不读取实时 ROS。', flush=True)
        dense = export_dense(database, stage / manifest['map_file'], snapshot=snapshot)
        manifest['dense'] = dense
        manifest['completed_at'] = datetime.now(timezone.utc).isoformat()
        manifest['geometry_restoration'] = dict(source_bundle=str(bundle), database=str(database),
                                               saved_scan_and_anchor_snapshot=True)
        manifest['limitations'] = [
            '主 PCD 保留完整配准回波用于定位和三维环境显示；通行带仅用于派生导航栅格。',
            '本版本由原扫描 SQLite 按旧版本的扫描索引及优化关键帧重新导出；不包含新采集扫描或后续优化位姿。',
            '无法补回原建图输入已过滤掉的后方回波，无法补录原始 LiDAR 包、IMU 或 RGB。']
        (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        write_checksums(stage)
        stage.rename(final)
    except BaseException:
        print('恢复未完成，旧版本未修改；临时目录：' + str(stage), file=sys.stderr)
        raise
    print('完整定位地图恢复完成：' + str(final / manifest['map_file']))
    print('导航将使用同名地图的这个新版本；旧版本保持原样。')
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map', help='地图名或已保存版本目录')
    parser.add_argument('--root', type=Path, default=Path.home() / 'go2_maps')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / '.save.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('另一个地图保存或恢复任务正在运行')
        restore(args.map, args.root)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, KeyboardInterrupt) as exc:
        sys.exit('恢复失败：' + str(exc or '中断'))
