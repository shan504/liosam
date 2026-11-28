#!/usr/bin/env python3
"""Adapt a completed LIO-SAM archive to the original Go2 navigation map contract."""
import argparse
import csv
import fcntl
import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import yaml

from prepare_map import apply, point_array, read_json, rigid, transforms

HERE = Path(__file__).resolve().parent
GO2 = HERE / 'original_go2' / 'go2_3d'
sys.path.insert(0, str(GO2))
from denoise import clean_directory
from geometry import ObservedGrid
from ground import floor_and_obstacles
from map_library import map_name, validate_bundle
from map_start import validate_start_pose
from mapping_store import _keys
from navigation_grid import current_settings


class Progress:
    """Keep preparation visible on stderr; stdout is the cache-path protocol."""
    def __init__(self, stage, total):
        self.stage, self.total = stage, total
        self.started = self.last = time.monotonic()
        self.report(0, force=True)

    def report(self, done, force=False):
        now = time.monotonic()
        if force or done == self.total or now - self.last >= 5.:
            print(f'{self.stage}: {done:,}/{self.total:,} '
                  f'({100.*done/max(1,self.total):.1f}%)，耗时 {now-self.started:.1f}s',
                  file=sys.stderr, flush=True)
            self.last = now


def cache_key(version, inputs):
    digest = hashlib.sha256()
    # The container always sees /map_source. Include actual per-version input
    # identities and the original grid policy used to generate the cached map.
    for path in [Path(__file__), HERE / 'prepare_map.py',
                 *[GO2 / p for p in ('geometry.py', 'grid_io.py', 'ground.py', 'denoise.py',
                                     'free_space.py', 'navigation_grid.py',
                                     'navigation_grid.yaml', 'navigation_filter.yaml', 'map_start.py',
                                     'mapping_store.py', 'gicp_dense.yaml')]]:
        digest.update(path.read_bytes())
    for name in inputs:
        path = version / name
        if path.exists():
            stat = path.stat()
            digest.update(repr((name, stat.st_dev, stat.st_ino, stat.st_size,
                                stat.st_mtime_ns, stat.st_ctime_ns)).encode())
    if (version / 'SHA256SUMS').is_file():
        digest.update((version / 'SHA256SUMS').read_bytes())
    return 'original-go2-' + digest.hexdigest()[:24]


def export_grid(chunks, grid, directory, policy, model):
    """Compress duplicate XY evidence before invoking Go2's grid exporter.

    XYZI archives keep repeated scans of every surface. Go2's export only
    needs one obstacle and two floor returns per cell. Work on bounded chunks
    and retain one real obstacle sample per cell, including single hits.
    """
    floor_once = set()
    occupied = {}
    low, high = policy['obstacle_min_height'], policy['obstacle_max_height']
    for cloud in chunks:
        xyz = cloud[:, :3]
        floor, obstacles = floor_and_obstacles(xyz, model, low, high)
        floor_cells, counts = np.unique(np.floor(floor[:, :2]/grid.resolution).astype(np.int64),
                                        axis=0, return_counts=True)
        for cell, count in zip(map(tuple, floor_cells), counts):
            if cell in grid.free:
                continue
            if count > 1 or cell in floor_once:
                grid.free.add(cell)
                floor_once.discard(cell)
            else:
                floor_once.add(cell)
        cells, indices = np.unique(np.floor(obstacles[:, :2]/grid.resolution).astype(np.int64),
                                   axis=0, return_index=True)
        occupied.update(zip(map(tuple, cells), map(tuple, obstacles[indices])))
    print(f'正在导出完整导航栅格：实测通行 {len(grid.free):,} 格，'
          f'障碍 {len(occupied):,} 格，分辨率 {grid.resolution:g} m',
          file=sys.stderr, flush=True)
    samples = np.asarray(list(occupied.values()), dtype=np.float32).reshape(-1, 3)
    grid.export(samples, directory, floor_z=0., min_height=low, max_height=high, ground_model=model)


def accumulate_voxels(db, points, resolution):
    # Canonical floor(XYZ/resolution), including signed coordinates. Combine
    # a whole batch in NumPy before upserting its distinct keys on disk.
    values = points.astype(np.float64)
    keys = _keys(values[:, :3], resolution)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys, values = keys[order], values[order]
    starts = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
    finite_i = np.isfinite(values[:, 3])
    values[~finite_i, 3] = 0.
    totals = np.add.reduceat(values, starts, axis=0).tolist()
    hits = np.diff(np.r_[starts, len(points)]).tolist()
    valid_i = np.add.reduceat(finite_i.astype(np.int64), starts).tolist()
    rows = ((*key, *total, count, valid) for key, total, count, valid in
            zip(keys[starts].tolist(), totals, hits, valid_i))
    db.executemany("""INSERT INTO voxels VALUES (?,?,?,?,?,?,?,?,?)
                      ON CONFLICT(x,y,z) DO UPDATE SET
                      sx=sx+excluded.sx, sy=sy+excluded.sy, sz=sz+excluded.sz,
                      si=si+excluded.si, hits=hits+excluded.hits, ni=ni+excluded.ni""", rows)


def write_voxel_map(db, dest, voxel):
    retained = db.execute('SELECT count(*) FROM voxels').fetchone()[0]
    if retained < 100:
        raise ValueError('转换后不足 100 个独立体素，原 Go2 GICP 无法加载此地图')
    header = (f'VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n'
              f'COUNT 1 1 1 1\nWIDTH {retained}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
              f'POINTS {retained}\nDATA binary\n')
    cursor = db.execute('SELECT x,y,z,sx/hits,sy/hits,sz/hits,si/NULLIF(ni,0) '
                        'FROM voxels ORDER BY x,y,z')
    progress = Progress('写入定位地图体素', retained)
    written = 0
    with dest.open('xb') as out:
        out.write(header.encode('ascii'))
        while True:
            block = cursor.fetchmany(32768)
            if not block:
                break
            data = np.asarray(block, dtype=np.float64)
            centroids = np.ascontiguousarray(data[:, 3:], dtype='<f4')
            # The archive uses float32. Rounding a centroid at a voxel boundary
            # can push it to the neighbour. There is a representable interior
            # sample because each contributing source point is float32.
            xyz = centroids[:, :3]
            rounded = _keys(xyz.astype(np.float64), voxel)
            keys = data[:, :3].astype(np.int64)
            xyz[rounded < keys] = np.nextafter(xyz[rounded < keys], np.float32(np.inf))
            xyz[rounded > keys] = np.nextafter(xyz[rounded > keys], np.float32(-np.inf))
            out.write(memoryview(centroids).cast('B'))
            written += len(block)
            progress.report(written)
    return retained


def aligned_chunks(raw, transform, db, voxel):
    progress = Progress('转换全部原始回波并聚合定位体素', len(raw))
    for offset in range(0, len(raw), 131072):
        points = np.array(raw[offset:offset+131072], copy=True)
        points[:, :3] = apply(points, transform)
        accumulate_voxels(db, points, voxel)
        yield points
        progress.report(min(offset+len(points), len(raw)))


def trace_archive(version, dest, manifest, files):
    source = manifest['dense']['frame_id'].strip('/')
    ground, mount, sensor = transforms(version, source)
    raw = point_array(version / files[0])
    if len(raw) != manifest['dense']['points'] or len(raw) < 100:
        raise ValueError('地图点数不足或与 manifest 不符')
    print(f'存档：{len(raw):,} 个原始回波，{manifest["dense"]["scans"]:,} 帧，'
          f'XYZI 数据 {raw.nbytes/(1024**3):.2f} GiB；完整读取，不裁剪地图。',
          file=sys.stderr, flush=True)
    with (version / files[2]).open(newline='') as stream:
        keyframes = {}
        for row in csv.DictReader(stream):
            stamp = int(row['stamp_ns'])
            if stamp in keyframes:
                raise ValueError('关键帧时间戳重复')
            keyframes[stamp] = ground @ rigid([float(row[k]) for k in 'xyz'],
                                               [float(row['q' + k]) for k in 'xyzw'])
    if not keyframes:
        raise ValueError('缺少已优化的关键帧位姿')
    start_pose = validate_start_pose(keyframes[min(keyframes)] @ np.linalg.inv(mount))
    body_height = float(start_pose[2, 3])
    if not 0.1 < body_height < 1.0:
        raise ValueError('建图起点机身离地高度与已保存的地面/安装外参不符')

    policy = current_settings()
    grid = ObservedGrid(policy['grid_resolution'])
    n = len(raw)
    n_scans = first = exact = 0
    progress = Progress('恢复实测通行区域（扫描帧）', manifest['dense']['scans'])
    with (version / files[1]).open(newline='') as stream:
        for row in csv.DictReader(stream):
            stamp, anchor = int(row['stamp_ns']), int(row['anchor_stamp_ns'])
            offset, count = int(row['first_point_index']), int(row['point_count'])
            if offset != first or count <= 0 or offset + count > n or anchor not in keyframes:
                raise ValueError('逐帧索引与地图或关键帧不一致')
            # Other scan poses are unavailable in this archive. Their endpoints
            # are valid; their keyframe anchors are not their beam origins.
            if stamp == anchor:
                xyz = apply(raw[offset:offset+count], ground).astype('<f4')
                if not np.isfinite(xyz).all():
                    raise ValueError('点云包含无效 XYZ')
                grid.observe(xyz, keyframes[stamp][:3, 3])
                exact += 1
            first += count
            n_scans += 1
            progress.report(n_scans)
    if first != n or n_scans != manifest['dense']['scans']:
        raise ValueError('逐帧索引数量与地图不符')

    dense = yaml.safe_load((GO2 / 'gicp_dense.yaml').read_text())['go2_gicp']['ros__parameters']
    voxel = float(dense['map_voxel_size'])
    if not np.isfinite(voxel) or voxel <= 0.:
        raise ValueError('原 Go2 dense 的体素尺寸无效')
    model = dict(plane=[0., 0., 1., 0.], height_at_origin=0.)
    # This temporary on-disk index bounds RAM independently of occupied 3D
    # voxel count. The old grid receives *all* original endpoints, never the
    # voxel centers. The disposable DB lives outside the published bundle.
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix='voxels-') as tmp:
        with closing(sqlite3.connect(Path(tmp)/'map.sqlite')) as db:
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('PRAGMA synchronous=OFF')
            db.execute('PRAGMA cache_size=-32768')
            db.execute('CREATE TABLE voxels (x INTEGER,y INTEGER,z INTEGER,sx REAL,sy REAL,'
                       'sz REAL,si REAL,hits INTEGER,ni INTEGER, PRIMARY KEY(x,y,z)) WITHOUT ROWID')
            export_grid(aligned_chunks(raw, ground, db, voxel), grid, dest, policy, model)
            retained = write_voxel_map(db, dest/'map.pcd', voxel)
    print('正在按原规则去除导航栅格孤立噪点……', file=sys.stderr, flush=True)
    report = clean_directory(dest)
    name = map_name(manifest.get('name', Path(files[0]).stem))
    metadata = dict(frame_id='map', map_type='lio_sam', point_count=retained, source_point_count=n,
                    ground_aligned=True,
                    body_height_above_ground=body_height, ground_model=model,
                    map_start_pose=start_pose.tolist(), map_from_source_map=ground.tolist(),
                    body_from_lidar=mount.tolist(), sensor_frame=sensor, source_frame=source,
                    parameters=dict(policy, map_name=name, floor_z=0., map_voxel_size=voxel), free_area_version=2,
                    exact_origin_scans=exact, scans=n_scans, navigation_denoise=report)
    (dest / 'metadata.yaml').write_text(yaml.safe_dump(metadata))
    print(f'原始回波 {n}，定位地图 {retained} 个体素（{voxel:g} m）', file=sys.stderr, flush=True)


def prepare(version, cache):
    version = Path(version).resolve(strict=True)
    cache = Path(cache).resolve()
    if version == cache or version in cache.parents:
        raise ValueError('缓存必须在原始地图目录之外')
    manifest = read_json(version / 'manifest.json')
    files = [manifest[field] for field in ('map_file', 'scan_index', 'optimized_keyframes')]
    if any(not isinstance(v, str) or not v or v != Path(v).name or v in ('.', '..') for v in files):
        raise ValueError('地图清单包含无效的输入文件名')
    key = cache_key(version, files + ['manifest.json', 'live/reference_transforms.json',
                                      'live/body_from_lidar.json', 'calibration/floor.json',
                                      'calibration/body_from_lidar.json'])
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / key
    with (cache / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            validate_bundle(output)
            if not (output / 'observed_free.npz').is_file():
                raise ValueError(f'导航缓存缺少通行区域文件: {output}')
            print(f'复用已完成的导航地图缓存：{output.name}', file=sys.stderr, flush=True)
            return output
        print('正在适配点云坐标，并使用原 Go2 的实测通行栅格和去噪策略生成导航地图……',
              file=sys.stderr, flush=True)
        with tempfile.TemporaryDirectory(dir=cache, prefix='.preparing-go2-') as tmp:
            stage = Path(tmp) / 'ready'
            stage.mkdir()
            trace_archive(version, stage, manifest, files)
            validate_bundle(stage)
            stage.rename(output)
    return output


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('version', type=Path)
    ap.add_argument('cache', type=Path)
    args = ap.parse_args()
    try:
        print(prepare(args.version, args.cache))
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        ap.exit(1, f'导航地图适配失败：{exc}\n')

