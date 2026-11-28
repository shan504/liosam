#!/usr/bin/env python3
"""Derive a level GICP target and observed occupancy from a saved Go2 run."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import tempfile
import sys

import numpy as np
from obstacle_view import obstacle_mask

RESOLUTION = 0.05
VOXEL = 0.10
DISPLAY_VOXEL = 0.04
MAX_DISPLAY_POINTS = 160_000
FLOOR_TOLERANCE = 0.045
OBSTACLE_MIN_HEIGHT = 0.085
OBSTACLE_MAX_HEIGHT = 0.85
MAX_VOXELS = 1_500_000


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def is_floor(height):
    """Test height in metres relative to the measured floor plane."""
    return np.abs(height) <= FLOOR_TOLERANCE


def rigid(xyz, xyzw):
    xyz, q = np.asarray(xyz, dtype=float), np.asarray(xyzw, dtype=float)
    if xyz.shape != (3,) or q.shape != (4,) or not np.all(np.isfinite(xyz)) or not .99 <= np.linalg.norm(q) <= 1.01:
        raise ValueError('位姿平移或四元数无效')
    matrix = np.eye(4)
    x, y, z, w = q / np.linalg.norm(q)
    matrix[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    matrix[:3, 3] = xyz
    return matrix


def quaternion(rotation):
    # Eigenvector form avoids dividing by a near-zero trace at 180 degrees.
    xx, xy, xz = rotation[0]
    yx, yy, yz = rotation[1]
    zx, zy, zz = rotation[2]
    k = np.array([[xx-yy-zz, xy+yx, xz+zx, zy-yz],
                  [xy+yx, yy-xx-zz, yz+zy, xz-zx],
                  [xz+zx, yz+zy, zz-xx-yy, yx-xy],
                  [zy-yz, xz-zx, yx-xy, xx+yy+zz]])
    q = np.linalg.eigh(k)[1][:, -1]
    return q if q[3] >= 0 else -q


def tf_matrix(tf):
    t, q = tf['transform']['translation'], tf['transform']['rotation']
    return rigid([t[k] for k in 'xyz'], [q[k] for k in 'xyzw'])


def transforms(version, source_frame):
    # Saved ROS TF is target <- source. floor.json describes the inverse.
    tf_path = version / 'live/reference_transforms.json'
    ground = None
    if tf_path.is_file():
        for tf in read_json(tf_path)['transforms']:
            if (tf['header']['frame_id'].strip('/'), tf['child_frame_id'].strip('/')) == ('go2_ground', source_frame):
                ground = tf_matrix(tf)
                break
    floor_path = version / 'calibration/floor.json'
    if floor_path.is_file():
        floor = read_json(floor_path)
        if (floor['frame'], floor['ground_frame']) != (source_frame, 'go2_ground'):
            raise ValueError('地面标定与点云坐标系不同')
        rotation = np.asarray(floor['rotation'], dtype=float)
        if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4):
            raise ValueError('地面标定旋转无效')
        matrix = rigid(floor['origin'], quaternion(rotation))
        from_floor = np.linalg.inv(matrix)
        if ground is not None and not np.allclose(ground, from_floor, atol=1e-3):
            raise ValueError('本轮地面 TF 与 floor.json 不一致')
        ground = from_floor
    if ground is None:
        raise ValueError('这版地图缺少实测地面标定 (go2_ground <- %s)' % source_frame)
    mount_path = version / 'live/body_from_lidar.json'
    if mount_path.is_file():
        tf = read_json(mount_path)
        parent, child = tf['header']['frame_id'], tf['child_frame_id']
        mount = tf_matrix(tf)
    else:
        cfg_path = version / 'calibration/body_from_lidar.json'
        if not cfg_path.is_file():
            raise ValueError('这版地图缺少建图时的 base_link 到 MID-360 安装外参')
        cfg = read_json(cfg_path)
        parent, child = cfg['parent'], cfg['child']
        mount = rigid(cfg['xyz'], cfg['xyzw'])
    if parent.strip('/') != 'base_link' or not child.strip('/') or child.strip('/') == 'base_link':
        raise ValueError('安装外参父子坐标系错误')
    return ground, mount, child.strip('/')


def point_array(path):
    with path.open('rb') as stream:
        header = {}
        for _ in range(32):
            line = stream.readline(1024).decode('ascii').split()
            if line:
                header[line[0]] = line[1:]
                if line[0] == 'DATA':
                    break
        offset = stream.tell()
    expected = {'FIELDS': 'x y z intensity', 'SIZE': '4 4 4 4', 'TYPE': 'F F F F',
                'COUNT': '1 1 1 1', 'DATA': 'binary'}
    if any(header.get(k) != v.split() for k, v in expected.items()):
        raise ValueError('需要本项目 save_map.sh 生成的二进制 XYZI 地图')
    n = int(header['POINTS'][0])
    if n <= 0 or path.stat().st_size != offset + 16*n:
        raise ValueError('点云文件不完整')
    return np.memmap(path, dtype='<f4', mode='r', offset=offset, shape=(n, 4))


def apply(points, matrix):
    return np.asarray(points[:, :3], dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def add_display_voxels(samples, xyz, rows, voxel):
    # Byte keys keep all three signed cell indices, including distant map origins.
    cells = np.ascontiguousarray(np.floor(xyz / voxel), dtype=np.int32)
    keys = cells.view(np.dtype((np.void, 3 * cells.itemsize))).reshape(-1)
    unique, first = np.unique(keys, return_index=True)
    samples.update(zip(unique.tolist(), rows[first].tolist()))


def survey_clouds(cloud, ground):
    sample_indices = {}
    display_indices = {}
    display_voxel = DISPLAY_VOXEL
    lower, upper = np.full(2, np.inf), np.full(2, -np.inf)
    for start in range(0, len(cloud), 131072):
        xyz = apply(cloud[start:start+131072], ground)
        if not np.isfinite(xyz).all() or np.abs(xyz).max() >= VOXEL * (1 << 20):
            raise ValueError('点云含非有限或超出体素整数索引范围的坐标')
        lower = np.minimum(lower, xyz[:, :2].min(axis=0))
        upper = np.maximum(upper, xyz[:, :2].max(axis=0))
        ijk = np.floor(xyz / VOXEL).astype(np.int64) + (1 << 20)
        keys = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
        unique, ids = np.unique(keys, return_index=True)
        sample_indices.update(zip(unique.tolist(), (ids + start).tolist()))
        if len(sample_indices) > MAX_VOXELS:
            raise ValueError('10 cm 配准目标过大，超出本机地图准备预算')
        rows = np.flatnonzero(obstacle_mask(xyz[:, 2]))
        add_display_voxels(display_indices, xyz[rows], rows + start, display_voxel)
        while len(display_indices) > MAX_DISPLAY_POINTS:
            # Doubling gives nested cells. One real return per child voxel
            # suffices to cover every occupied parent, including old map regions.
            display_voxel *= 2
            keep = np.sort(np.fromiter(display_indices.values(), dtype=np.int64))
            display_indices.clear()
            add_display_voxels(display_indices, apply(cloud[keep], ground), keep, display_voxel)
    samples = np.fromiter(sample_indices.values(), dtype=np.int64, count=len(sample_indices))
    if len(samples) < 30:
        raise ValueError('GICP 目标有效体素不足 30 个')
    display_rows = np.sort(np.fromiter(display_indices.values(), dtype=np.int64))
    target = apply(cloud[np.sort(samples)], ground).astype('<f4')
    display = apply(cloud[display_rows], ground).astype('<f4')
    return target, lower, upper, display, display_voxel


def observe(grid, xy, lower_cell, height_only):
    # Accumulate independent observations across scans. One obstacle observation
    # leaves an unknown cell (never free); two support marking it occupied.
    if not len(xy):
        return
    cells = np.floor(xy / RESOLUTION).astype(np.int64) - lower_cell
    valid = (cells >= 0).all(axis=1) & (cells[:, 0] < grid.shape[1]) & (cells[:, 1] < grid.shape[0])
    cells = cells[valid]
    ids = np.unique(cells[:, 1] * grid.shape[1] + cells[:, 0])
    flat = grid.reshape(-1)
    if height_only:
        flat[ids] = np.minimum(flat[ids] + 1, 2)
    else:
        flat[ids] = 1


def observe_rays(free, origin, xyz, lower):
    # Only scan times exactly equal to an optimized LiDAR keyframe have a
    # measured origin in the archive. Other scans still contribute endpoints.
    hits = xyz[(xyz[:, 2] >= -FLOOR_TOLERANCE) & (xyz[:, 2] <= OBSTACLE_MAX_HEIGHT)]
    delta = hits[:, :2] - origin[:2]
    distance = np.linalg.norm(delta, axis=1)
    near = (distance >= .1) & (distance < 16.)
    delta, distance = delta[near], distance[near]
    if not len(distance):
        return
    bins = np.floor(np.arctan2(delta[:, 1], delta[:, 0]) / np.deg2rad(1.)).astype(int)
    # A nearby floor return does not block a measured ray in the same sector.
    # Stop at its nearest potential obstacle, or the farthest observed floor
    # if the sector has only floor returns. Unobserved sectors remain unknown.
    floor = is_floor(hits[near, 2])
    order = np.lexsort((np.where(floor, -distance, distance), floor, bins))
    ids = order[np.r_[True, bins[order][1:] != bins[order][:-1]]]
    directions = delta[ids] / distance[ids, None]
    # Sample below half a grid cell and stop short of the observed surface.
    distances = np.arange(0, distance[ids].max(), RESOLUTION * .45)
    for group in np.array_split(np.arange(len(ids)), max(1, math.ceil(len(ids) / 64))):
        mask = distances[None, :] < distance[ids[group], None] - RESOLUTION
        rays = origin[:2] + directions[group, None, :] * distances[None, :, None]
        observe(free, rays[mask], lower, False)


def build(version, dest, layout):
    manifest = read_json(version / 'manifest.json')
    source_frame = manifest['dense']['frame_id'].strip('/')
    ground, mount, sensor = transforms(version, source_frame)
    cloud = point_array(version / layout[0])
    if len(cloud) != manifest['dense']['points']:
        raise ValueError('manifest 点数与 PCD 不一致')
    with (version / layout[2]).open(newline='') as f:
        poses = {}
        for row in csv.DictReader(f):
            stamp = int(row['stamp_ns'])
            if stamp in poses:
                raise ValueError('关键帧时间戳重复')
            poses[stamp] = ground @ rigid([float(row[k]) for k in ('x', 'y', 'z')],
                                           [float(row['q'+k]) for k in 'xyzw'])
    if not poses:
        raise ValueError('缺少已优化的关键帧位姿')
    initial = poses[min(poses)] @ np.linalg.inv(mount)
    print('正在从存档点云生成障碍物显示层、10 cm GICP 目标和 5 cm 通行栅格……', file=sys.stderr, flush=True)
    target, lower, upper, display, display_voxel = survey_clouds(cloud, ground)
    lidar_xy = np.asarray([pose[:2, 3] for pose in poses.values()])
    lower = np.floor(np.minimum(lower, lidar_xy.min(axis=0)) / RESOLUTION).astype(np.int64) - 2
    upper = np.floor(np.maximum(upper, lidar_xy.max(axis=0)) / RESOLUTION).astype(np.int64) + 2
    w, h = map(int, upper-lower+1)
    free, obstacles = np.zeros((h, w), 'u1'), np.zeros((h, w), 'u1')
    processed = n_scans = exact = 0
    with (version / layout[1]).open(newline='') as f:
        for row in csv.DictReader(f):
            start, n = int(row['first_point_index']), int(row['point_count'])
            stamp, anchor = int(row['stamp_ns']), int(row['anchor_stamp_ns'])
            if start != processed or n <= 0 or start+n > len(cloud) or anchor not in poses:
                raise ValueError('逐帧索引与点云或关键帧不一致')
            xyz = apply(cloud[start:start+n], ground)
            # Real endpoints on this measured floor; no distant ceiling clears floor.
            observe(free, xyz[is_floor(xyz[:, 2]), :2], lower, False)
            obs = (xyz[:, 2] >= OBSTACLE_MIN_HEIGHT) & (xyz[:, 2] <= OBSTACLE_MAX_HEIGHT)
            observe(obstacles, xyz[obs, :2], lower, True)
            if stamp == anchor:
                observe_rays(free, poses[anchor][:3, 3], xyz, lower)
                exact += 1
            processed += n
            n_scans += 1
    if processed != len(cloud) or n_scans != manifest['dense']['scans']:
        raise ValueError('扫描索引长度或总帧数与清单不一致')
    if not np.any(free) or obstacles.max() == 0:
        raise ValueError('缺少地面或障碍物实测点，该地图暂时无法用于导航')
    image = np.full((h, w), 205, dtype='u1')
    image[(free != 0) & (obstacles == 0)] = 254
    image[obstacles >= 2] = 0
    if not np.any(image == 0) or not np.any(image == 254):
        raise ValueError('导航栅格缺少可通行区或重复观测到的障碍物')
    with (dest / 'nav.pgm').open('wb') as out:
        out.write(('P5\n%d %d\n255\n' % (w, h)).encode('ascii'))
        out.write(image[::-1].tobytes())
    (dest / 'nav.yaml').write_text(
        'image: nav.pgm\nmode: trinary\nresolution: %.10g\norigin: [%.10g, %.10g, 0.0]\n'
        'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n'
        % (RESOLUTION, lower[0]*RESOLUTION, lower[1]*RESOLUTION))
    np.save(dest / 'target.npy', target, allow_pickle=False)
    np.save(dest / 'obstacles.npy', display, allow_pickle=False)
    metadata = dict(map_frame='map', map_axes='saved go2_ground', sensor_frame=sensor,
                    source_frame=source_frame, map_from_source=ground.tolist(),
                    body_from_lidar=mount.tolist(), initial_map_from_body=initial.tolist(),
                    initial_base_pose=initial[:3, 3].tolist() + quaternion(initial[:3, :3]).tolist(),
                    grid_resolution=RESOLUTION, voxel_size=VOXEL, scans=n_scans, exact_origin_scans=exact,
                    floor_band_m=[-FLOOR_TOLERANCE, FLOOR_TOLERANCE],
                    obstacle_band_m=[OBSTACLE_MIN_HEIGHT, OBSTACLE_MAX_HEIGHT],
                    display_voxel_m=display_voxel, display_points=len(display),
                    target_points=len(target), source_version=str(version))
    (dest / 'metadata.json').write_text(json.dumps(metadata, allow_nan=False, indent=2)+'\n')


def prepare(version, cache):
    version = Path(version).resolve(strict=True)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = read_json(version / 'manifest.json')
    layout = [manifest['map_file'], manifest['scan_index'], manifest['optimized_keyframes']]
    if any(not isinstance(name, str) or name != Path(name).name or name in ('', '.', '..') for name in layout):
        raise ValueError('地图清单包含无效的输入文件名')
    names = layout + ['manifest.json', 'calibration/floor.json', 'live/reference_transforms.json',
                      'live/body_from_lidar.json', 'calibration/body_from_lidar.json']
    key = hashlib.sha256(Path(__file__).read_bytes() + str(version).encode())
    key.update(Path(__file__).with_name('obstacle_view.py').read_bytes())
    checksums = version / 'SHA256SUMS'
    if checksums.is_file():
        key.update(checksums.read_bytes())
    for name in names:
        path = version / name
        if path.exists():
            st = path.stat()
            key.update(repr((name, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)).encode())
    output = cache / key.hexdigest()[:24]
    with (cache / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            if not all((output / name).is_file() for name in ('nav.yaml', 'nav.pgm', 'target.npy', 'obstacles.npy', 'metadata.json')):
                raise ValueError('导航缓存不完整: ' + str(output))
            return output
        with tempfile.TemporaryDirectory(dir=str(cache), prefix='.preparing-') as tmp:
            stage = Path(tmp)/'result'
            stage.mkdir()
            build(version, stage, layout)
            stage.rename(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version', type=Path)
    parser.add_argument('cache', type=Path)
    args = parser.parse_args()
    try:
        print(prepare(args.version, args.cache).resolve())
    except (OSError, KeyError, ValueError, TypeError) as exc:
        sys.exit('导航地图准备失败：' + str(exc))

