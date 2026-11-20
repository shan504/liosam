"""Monitor a stationary LIO-SAM cold start with live ROS 2 topics."""
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Log
from rclpy.qos import qos_profile_sensor_data
from visualization_msgs.msg import MarkerArray


def main():
    rclpy.init()
    node = rclpy.create_node('live_mapping_check')
    start = time.monotonic()
    pending = {}
    previous = {}
    origin = {}
    first_events = set()
    pairs = 0
    degenerate = 0
    last_stamp = -1
    edges = 0
    resets = 0
    stats = {'map': 0.0, 'inc': 0.0}

    def emit(line):
        print(f'+{time.monotonic()-start:5.1f}s {line}', flush=True)

    def pose(msg, label):
        nonlocal pairs, last_stamp, degenerate
        ts = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        if ts <= last_stamp:
            return
        p = msg.pose.pose.position
        row = pending.setdefault(ts, {})
        row[label] = (p.x, p.y, p.z)
        if label == 'inc':
            row['degen'] = int(msg.pose.covariance[0] == 1)
        if len(pending) > 128:
            pending.pop(min(pending))
        if 'map' not in row or 'inc' not in row:
            return
        pairs += 1
        degenerate += row['degen']
        gap = (ts-last_stamp)/1e9 if previous else 0.0
        last_stamp = ts
        fields = []
        events = []
        for name in ('map', 'inc'):
            pos = row[name]
            step = math.dist(previous.get(name, pos), pos)
            drift = math.dist(origin.setdefault(name, pos), pos)
            fields.append(f'{name}_step={step:.3f} {name}_from_start={drift:.3f}')
            if math.isfinite(step):
                stats[name] = max(stats[name], step)
            if not math.isfinite(step) or not math.isfinite(drift):
                events.append(name + '_invalid')
            if step >= 0.25:
                events.append(name + '_step>=0.25m')
            if drift >= 0.25:
                events.append(name + '_drift>=0.25m')
            previous[name] = pos
        if pairs == 1 or pairs % 20 == 0 or set(events)-first_events:
            emit(f'pair={pairs} dt={gap:.3f}s ' + ' '.join(fields) +
                 f' degen={row["degen"]} edges_seen={edges} ' +
                 ' '.join(sorted(set(events)-first_events)))
        first_events.update(events)
        for old in [key for key in pending if key <= ts]:
            del pending[old]

    def loop(msg):
        nonlocal edges
        n = sum(len(m.points)//2 for m in msg.markers if m.ns == 'loop_edges')
        if n > edges:
            edges = n
            if 'loop' not in first_events:
                first_events.add('loop')
                emit(f'first published loop marker: edges={n} pair={pairs}')

    def log(msg):
        nonlocal resets
        if 'Large velocity, reset IMU-preintegration' in msg.msg:
            resets += 1
            if resets == 1:
                emit(f'first velocity reset: latest_pair={pairs} edges_seen={edges}')

    for label, topic in [('map', 'odometry'), ('inc', 'odometry_incremental')]:
        node.create_subscription(Odometry, '/lio_sam/mapping/'+topic,
                                 lambda m, name=label: pose(m, name), qos_profile_sensor_data)
    node.create_subscription(MarkerArray, '/lio_sam/mapping/loop_closure_constraints',
                             loop, qos_profile_sensor_data)
    node.create_subscription(Log, '/rosout', log, qos_profile_sensor_data)
    emit('live monitor ready')
    try:
        while rclpy.ok() and time.monotonic() - start < 70:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        emit(f'SUMMARY pairs={pairs} degenerate={degenerate} max_map_step={stats["map"]:.3f} '
             f'max_inc_step={stats["inc"]:.3f} loop_edges_seen={edges} velocity_resets_seen={resets}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
