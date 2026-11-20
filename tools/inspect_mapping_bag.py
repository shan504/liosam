"""Summarize a standalone mapping cold-start recording (ROS 2 sqlite3 bag)."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import sqlite3
from statistics import median

from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, PointCloud2


TOPICS = {
    '/mid360s/imu': ('imu', Imu),
    '/mid360s/points': ('cloud', PointCloud2),
    '/lio_sam/feature/cloud_corner': ('corner', PointCloud2),
    '/lio_sam/feature/cloud_surface': ('surface', PointCloud2),
    '/lio_sam/mapping/odometry': ('map', Odometry),
    '/lio_sam/mapping/odometry_incremental': ('inc', Odometry),
}


def run(directory):
    files = sorted(directory.glob('*.db3'))
    if not files:
        raise SystemExit('No *.db3 files found in ' + str(directory))
    ages = defaultdict(list)
    features = defaultdict(dict)
    poses = defaultdict(dict)
    last_stamp = {}
    nonpositive = defaultdict(int)
    counts = defaultdict(int)
    start = None
    for path in files:
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            query = ('SELECT t.name, m.timestamp, m.data FROM messages m '
                     'JOIN topics t ON t.id=m.topic_id WHERE t.name IN (' +
                     ','.join('?' for _ in TOPICS) + ') ORDER BY m.timestamp, m.id')
            for topic, recorded, data in db.execute(query, tuple(TOPICS)):
                label, kind = TOPICS[topic]
                msg = deserialize_message(data, kind)
                stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                if start is None:
                    start = recorded
                elapsed = (recorded-start) * 1e-9
                counts[label] += 1
                if label in last_stamp and stamp <= last_stamp[label]:
                    nonpositive[label] += 1
                last_stamp[label] = stamp
                if label in ('imu', 'cloud'):
                    ages[int(elapsed)//2, label].append((recorded-stamp) * 1e-6)
                elif label in ('corner', 'surface'):
                    features[stamp][label] = msg.width * msg.height
                else:
                    p = msg.pose.pose.position
                    poses[stamp][label] = (elapsed, (p.x, p.y, p.z), msg.pose.covariance[0])
    print('received:', dict(counts), 'non-increasing stamps:', dict(nonpositive))
    print('record time minus message stamp (ms); compare cloud minus IMU per 2s bin')
    print('sec    imu_p50   cloud_p50   difference')
    for index in sorted({i for i, _ in ages}):
        imu, cloud = ages[index, 'imu'], ages[index, 'cloud']
        if imu and cloud:
            a, b = median(imu), median(cloud)
            print(f'{2*index:3d} {a:10.1f} {b:11.1f} {b-a:12.1f}')
    print('first 5 paired poses; then first 3 jumps >=1 m or nonfinite, in stamp order')
    print('sec    map_step_m    inc_step_m   degen   corners   surfaces')
    previous = None
    paired = jumps = shown_jumps = 0
    for stamp, row in sorted(poses.items()):
        if 'map' not in row or 'inc' not in row:
            continue
        steps = [math.dist(row[k][1], previous[k][1]) for k in ('map','inc')] if previous else [0., 0.]
        event = any(not math.isfinite(d) or d >= 1.0 for d in steps)
        if paired < 5 or (event and shown_jumps < 3):
            f = features.get(stamp, {})
            print(f'{row["inc"][0]:5.1f} {steps[0]:12.3f} {steps[1]:12.3f}'
                  f' {int(row["inc"][2] == 1):7d} {f.get("corner", "-"):>9} {f.get("surface", "-"):>10}')
            shown_jumps += int(event)
        paired += 1
        jumps += int(event)
        previous = row
    print('paired pose stamps:', paired, 'steps >=1 m or nonfinite:', jumps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag_directory', type=Path)
    run(parser.parse_args().bag_directory)
