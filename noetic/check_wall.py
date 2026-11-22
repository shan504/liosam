#!/usr/bin/env python3
"""Measure one planar wall in a saved Go2 dense archive without starting ROS.

Example (replace the run path and coordinates with a wall in your scan):
  python3 check_wall.py ~/go2_mapping_runs/run_001/dense --wall 1 2 3 2 --z 0 1

Supply segment endpoints and absolute Z limits in the archive frame (usually
odom, not go2_ground). Pick the face of a straight, unobstructed wall. Avoid
corners, doors, pipes, opposite faces, floors and ceilings. With a single face
in the strip, changes in signed distance can reveal misalignment. A clipped or
mixed strip, changing incidence or surface relief can also change the metrics.
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np

from dense_map import transform


def sample_patch(world, start, tangent, length, normal, height, band, rng):
    """Give all four quadrants equal weight and reject partial observations."""
    low, high = height
    xyz = world[(world[:, 2] >= low) & (world[:, 2] <= high), :3]
    delta = xyz[:, :2] - start
    along = delta @ tangent
    distance = delta @ normal
    keep = (along >= 0) & (along <= length) & (np.abs(distance) <= band)
    patch = np.column_stack((along[keep] - length/2, xyz[keep, 2] - (low+high)/2,
                             distance[keep]))
    quadrant = (patch[:, 0] >= 0).astype(int) + 2*(patch[:, 1] >= 0)
    groups = [np.flatnonzero(quadrant == k) for k in range(4)]
    count = min(64, *(len(g) for g in groups))
    if count < 8:
        return None
    return patch[np.concatenate([rng.choice(g, count, replace=False) for g in groups])]


def read_patches(path, wall, height, band):
    delta = wall[2:] - wall[:2]
    length = np.linalg.norm(delta)
    tangent = delta / length
    normal = np.array([-tangent[1], tangent[0]])
    rng = np.random.RandomState(1729)
    patches, stamps = [], []
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")  # All scan data and poses must come from one snapshot.
        frame = db.execute("SELECT value FROM meta WHERE name='frame'").fetchone()
        poses = {key: json.loads(pose) for key, pose in db.execute("SELECT stamp,pose FROM poses")}
        total = 0
        for stamp, key, blob in db.execute("SELECT stamp,anchor,points FROM scans ORDER BY stamp"):
            total += 1
            if key not in poses or len(blob) % 16:
                raise ValueError("Incomplete archive row at stamp %d" % stamp)
            local = np.frombuffer(blob, dtype='<f4').reshape(-1, 4)
            world = transform(local, poses[key])
            patch = sample_patch(world, wall[:2], tangent, length, normal, height, band, rng)
            if patch is not None:
                stamps.append(stamp)
                patches.append(patch)
    print("坐标系=%s  归档扫描=%d  同一墙面四区均有足够回波的扫描=%d" %
          (frame[0] if frame else "未知", total, len(patches)))
    if len(patches) < 2:
        raise ValueError("至少需要两帧各覆盖四区（每区 8 点）。请核对坐标系、墙段及高度范围。")
    return np.array(stamps, dtype=np.int64), patches


def report(stamps, patches, band):
    # Remove one common lean of the physical wall. Fit only within individual
    # scans, otherwise a shift between visits could itself determine the slope.
    slopes = []
    for patch in patches:
        centered = patch - patch.mean(axis=0)
        slope, _, rank, _ = np.linalg.lstsq(centered[:, :2], centered[:, 2], rcond=None)
        if rank == 2:
            slopes.append(slope)
    if not slopes:
        raise ValueError("点在墙面内的分布退化，无法估计共同法向。")
    slope = np.median(slopes, axis=0)
    normal_scale = np.sqrt(1 + slope @ slope)
    residuals = [(p[:, 2] - p[:, :2] @ slope) / normal_scale for p in patches]
    # Approximate the equal-scan mixture with the same number of quantiles per
    # scan. A close-up with more returns cannot dominate the merged width.
    q = (np.arange(101) + 0.5) * 100 / 101
    pooled = np.concatenate([np.percentile(r, q) for r in residuals])
    def width(r): return np.diff(np.percentile(r, [10, 90]))[0]
    medians = np.array([np.median(r) for r in residuals])
    print("单帧墙厚中位数 (P90−P10): %.2f cm" % (100*np.median([width(r) for r in residuals])))
    print("合并墙厚 (P90−P10，按帧等权近似): %.2f cm" % (100*width(pooled)))
    print("各帧墙面中位位置的 P90−P10: %.2f cm" % (100*width(medians)))
    times = (stamps - stamps[0]) * 1e-9
    for group in np.array_split(np.arange(len(stamps)), min(4, len(stamps))):
        print("%7.1f–%7.1f s | %d 帧 | 法向位置中位数 %+.2f cm" %
              (times[group[0]], times[group[-1]], len(group), 100*np.median(medians[group])))
    edge = np.mean([np.mean(np.abs(p[:, 2]) > 0.9 * band) for p in patches])
    if edge > 0.05 or np.linalg.norm(slope) > 0.27:
        print("注意：墙面靠近截取边界，或与选定的竖直面夹角较大。请复核选区后再比较。")
    print("仅测该墙片上保留帧的法向一致性；时段按样本数分组，不能代替已知往返时刻或真值。")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path, help="归档目录或 scans.sqlite")
    parser.add_argument("--wall", required=True, type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--z", required=True, type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="归档坐标系中的 z 下界与上界，单位 m")
    parser.add_argument("--band", type=float, default=.25, help="墙面两侧各保留的距离，默认 0.25 m")
    args = parser.parse_args()
    wall = np.array(args.wall)
    if (not np.isfinite([*wall, *args.z, args.band]).all() or args.band <= 0
            or args.z[0] >= args.z[1] or np.linalg.norm(wall[2:] - wall[:2]) < 1e-4):
        parser.error("请输入有限的不同端点、递增 z 区间及正半宽。")
    path = args.archive.expanduser().resolve()
    if path.is_dir():
        path /= "scans.sqlite"
    try:
        stamps, patches = read_patches(path, wall, args.z, args.band)
        report(stamps, patches, args.band)
    except (sqlite3.Error, ValueError, OSError) as exc:
        parser.exit(2, "无法完成墙面检查：%s\n" % exc)


if __name__ == "__main__":
    main()
