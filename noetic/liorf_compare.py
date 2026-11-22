#!/usr/bin/env python3
"""Compare two ROS1 stationary runs from the same LiDAR packet sequence."""
import argparse
from collections import Counter
import struct
import sys

import numpy as np

from stationary_surfaces import MIN_POINTS, pair_arrays, pick_patches, residuals, verified_rigid

TOPICS = {
    'pose': 'mapping/odometry',
    'sensor': 'deskew/cloud_deskewed',
    'world': 'mapping/cloud_registered_raw',
}


def topic_map(namespace):
    return {f'/{namespace}/{suffix}': name for name, suffix in TOPICS.items()}


def bag_index(bag, namespace):
    topics = topic_map(namespace)
    counts = {key: Counter() for key in TOPICS}
    for topic, msg, _ in bag.read_messages(topics=list(topics), raw=True):
        key = topics[topic]
        expected = 'nav_msgs/Odometry' if key == 'pose' else 'sensor_msgs/PointCloud2'
        if msg[0] != expected:
            raise ValueError(f'{topic} 类型为 {msg[0]}，需要 {expected}')
        sec, ns = struct.unpack_from('<II', msg[1], 4)
        if ns >= 10**9:
            raise ValueError(f'{topic} 时间戳非法')
        counts[key][sec * 10**9 + ns] += 1
        if len(counts[key]) > 10000:
            raise ValueError('请使用短静止片段，每个话题最多 10000 个时间戳')
    # Duplicate headers could indicate overlapping publishers or repeated data.
    return {key: {stamp for stamp, n in c.items() if n == 1 and stamp > 0}
            for key, c in counts.items()}


def read_selected(bag, namespace, pose_stamps, cloud_stamps):
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    topics = topic_map(namespace)
    poses, pending, pairs, frames = {}, {}, {}, {}
    for topic, record, _ in bag.read_messages(topics=list(topics), raw=True):
        sec, ns = struct.unpack_from('<II', record[1], 4)
        stamp = sec * 10**9 + ns
        key = topics[topic]
        if stamp not in (pose_stamps if key == 'pose' else cloud_stamps):
            continue
        msg = (Odometry() if key == 'pose' else PointCloud2()).deserialize(record[1])
        frame = msg.header.frame_id.lstrip('/')
        if not frame or frames.setdefault(key, frame) != frame:
            raise ValueError(f'{topic} 的参考系为空或变化')
        if key == 'pose':
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            poses[stamp] = ([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
        else:
            pair = pending.setdefault(stamp, {})
            pair[key] = msg
            if 'sensor' in pair and 'world' in pair:
                sensor, world = pair_arrays(stamp, pair['sensor'], pair['world'])
                # Validate indexed correspondence every frame. Only the first
                # transform will be applied; later pose errors remain measurable.
                rigid = verified_rigid(sensor, world)
                pairs[stamp] = (sensor, world, rigid)
                del pending[stamp]
    return poses, pairs, frames


def pose_metric(poses, stamps):
    if len(stamps) < 2:
        return None
    xyz = np.asarray([poses[t][0] for t in stamps], dtype=float)
    quat = np.asarray([poses[t][1] for t in stamps], dtype=float)
    lengths = np.linalg.norm(quat, axis=1)
    if not (np.isfinite(xyz).all() and np.isfinite(quat).all()) or np.min(lengths) < 1e-8:
        raise ValueError('里程计的位置或四元数无效')
    quat /= lengths[:, None]
    dots = np.abs(np.sum(quat[1:] * quat[:-1], axis=1))
    return (np.linalg.norm(np.diff(xyz, axis=0), axis=1).max() * 1000,
            np.rad2deg(2 * np.arccos(np.clip(dots, 0, 1))).max(),
            np.linalg.norm(xyz - xyz[0], axis=1).max() * 1000)


def thickness(rows):
    widths = [np.ptp(np.percentile(r, [5, 95])) for r in rows]
    return 1000 * np.array([np.median(widths), np.ptp(np.percentile(np.concatenate(rows), [5, 95])),
                            np.ptp([np.median(r) for r in rows])])


def shared_surfaces(a, b, stamps):
    if len(stamps) < 4:
        return 0, []
    first = stamps[0]
    # Both physical sensor frames have the same origin/axes. Derive a
    # balanced, symmetric common aperture from real first-scan samples.
    count = min(len(a[first][0]), len(b[first][0]))
    samples = [v[first][0][np.linspace(0, len(v[first][0]) - 1, count, dtype=int)] for v in (a, b)]
    planes = pick_patches(np.vstack(samples))
    worlds = []
    for data in (a, b):
        rot, offset = data[first][2]
        worlds.append([(data[t][1] - offset) @ rot.T for t in stamps])
    results = []
    for center, normal in planes:
        rows = [[], []]
        for x, y in zip(*worlds):
            ra, rb = (residuals(cloud, center, normal) for cloud in (x, y))
            if min(len(ra), len(rb)) >= MIN_POINTS:
                rows[0].append(ra)
                rows[1].append(rb)
        if len(rows[0]) >= max(4, int(np.ceil(.8 * len(stamps)))):
            results.append((thickness(rows[0]), thickness(rows[1]), len(rows[0])))
    return len(planes), results


def compare(first, second):
    indices = [bag_index(bag, ns) for bag, ns in ((first, 'lio_sam'), (second, 'liorf'))]
    pose_stamps = sorted(indices[0]['pose'] & indices[1]['pose'])
    common = sorted(set.intersection(*(v[k] for v in indices for k in ('sensor', 'world'))))
    take = np.linspace(0, len(common) - 1, min(12, len(common)), dtype=int)
    selected = sorted({common[i] for i in take})
    print(f'前提为实机静止。两路同源戳里程计 {len(pose_stamps)}，两路均有去畸变和配准云 {len(common)} 帧。')
    a, b = [read_selected(bag, ns, set(pose_stamps), set(selected))
            for bag, ns in ((first, 'lio_sam'), (second, 'liorf'))]
    if len(pose_stamps) >= 2:
        print('共同位姿窗口 %.2f s，最大相邻源戳间隔 %.1f ms。' %
              ((pose_stamps[-1] - pose_stamps[0]) * 1e-9, np.max(np.diff(pose_stamps)) * 1e-6))
        for label, data in zip(('LIO-SAM', 'LIORF'), (a, b)):
            print(('%s 相邻共同采样最大平移 %.1f mm / 最大转角 %.3f°，'
                   '距各自起点最大位移 %.1f mm。') % ((label,) + pose_metric(data[0], pose_stamps)))
    else:
        print('共同里程计样本不足。')
    if len(selected) < 4:
        print('共同扫描不足 4 帧，无法比较轮廓。')
        return False
    if a[2]['sensor'] != b[2]['sensor']:
        raise ValueError('两路去畸变云传感器参考系不同，无法共用首帧参考平面')
    total, metrics = shared_surfaces(a[1], b[1], selected)
    print('共同采样 %d 帧 / %.2f s，首帧固定平面 %d 处，可重复观测 %d 处。' %
          (len(selected), (selected[-1]-selected[0])*1e-9, total, len(metrics)))
    if len(metrics) < 2:
        print('可比平面不足两处，无法判断轮廓厚度。')
        return False
    for i, label in enumerate(('LIO-SAM', 'LIORF')):
        metrics_i = np.median([row[i] for row in metrics], axis=0)
        print('%s 单帧厚度 %.1f mm，叠帧厚度 %.1f mm，法向中位位置峰峰值 %.1f mm。' %
              ((label,) + tuple(metrics_i)))
    print('厚度为同一物理区域法向距离的 P95−P05，各区域取中位数。')
    return len(pose_stamps) >= 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline')
    parser.add_argument('liorf')
    args = parser.parse_args()
    import rosbag
    try:
        with rosbag.Bag(args.baseline) as first, rosbag.Bag(args.liorf) as second:
            return 0 if compare(first, second) else 2
    except (OSError, ValueError, rosbag.ROSBagException, struct.error, np.linalg.LinAlgError) as e:
        print(f'对比未完成：{e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
