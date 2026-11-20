"""Sample the running bench mapping pipeline while the sensor is stationary."""
import math
import struct
import time
from bisect import bisect_left
from statistics import median

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2, PointField


def stamp(msg):
    t = msg.header.stamp
    return t.sec + t.nanosec * 1e-9


def xyz(v):
    return (v.x, v.y, v.z)


def acceleration_in_world(msg):
    q = msg.orientation
    norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
    if not math.isfinite(norm) or norm < 1e-8 or msg.orientation_covariance[0] < 0:
        return None
    x, y, z, w = (v / norm for v in (q.x, q.y, q.z, q.w))
    a, b, c = xyz(msg.linear_acceleration)
    u, v, t = 2*(y*c-z*b), 2*(z*a-x*c), 2*(x*b-y*a)
    return (a+w*u+y*t-z*v, b+w*v+z*u-x*t, c+w*t+x*v-y*u)


def scan_duration(msg):
    if not msg.width or not msg.height:
        return None
    field = next((f for f in msg.fields if f.name == 'time'), None)
    if field is None or field.datatype != PointField.FLOAT32:
        return None
    last = (msg.height-1)*msg.row_step + (msg.width-1)*msg.point_step + field.offset
    if last + 4 > len(msg.data):
        return None
    return struct.unpack_from('>f' if msg.is_bigendian else '<f', msg.data, last)[0]


def main():
    rclpy.init()
    node = rclpy.create_node('mapping_health_probe')
    imu, clouds, corrections = [], [], []
    frames = set()

    def on_imu(msg):
        imu.append((stamp(msg), acceleration_in_world(msg), math.hypot(*xyz(msg.angular_velocity))))
        frames.add(('imu', msg.header.frame_id))

    def on_cloud(msg):
        clouds.append((time.monotonic(), stamp(msg), scan_duration(msg)))
        frames.add(('cloud', msg.header.frame_id))

    def on_correction(msg):
        corrections.append((stamp(msg), xyz(msg.pose.pose.position), msg.pose.covariance[0]))

    topics = [(Imu, '/mid360s/imu', on_imu),
              (PointCloud2, '/mid360s/points', on_cloud),
              (Odometry, '/lio_sam/mapping/odometry_incremental', on_correction)]
    for kind, topic, callback in topics:
        node.create_subscription(kind, topic, callback, qos_profile_sensor_data)
    print('Sampling for 12 seconds. Keep the sensor still.', flush=True)
    start = time.monotonic()
    while time.monotonic() < start + 12 and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

    for (_, topic, _), rows in zip(topics, (imu, clouds, corrections)):
        print(topic, 'publishers:', node.count_publishers(topic), 'samples:', len(rows))
    print('frames:', sorted(frames))
    times = [row[0] for row in imu]
    if len(times) >= 2:
        gaps = [(b-a)*1000 for a, b in zip(times, times[1:])]
        print('IMU header dt ms median/max/nonpositive:',
              round(median(gaps), 2), round(max(gaps), 2), sum(d <= 0 for d in gaps))
        print('gyro magnitude rad/s median:', round(median(row[2] for row in imu), 4))
    world = [row[1] for row in imu if row[1] is not None]
    if world:
        print('world accel xyz m/s^2 median:', tuple(round(median(v), 3) for v in zip(*world)))

    ordered = sorted(times)
    scans = [(s, s+d) for arrival, s, d in clouds if start+2 < arrival < start+10
             and d is not None and math.isfinite(d) and d >= 0]
    if ordered and scans:
        covered = sum(ordered[0] <= s and e <= ordered[-1] for s, e in scans)
        distances = []
        for first, last in scans:
            for endpoint in (first, last):
                i = bisect_left(ordered, endpoint)
                neighbors = ordered[max(i-1, 0):i+1]
                distances.append(min(abs(t-endpoint) for t in neighbors))
        print('scan duration ms median:', round(median(e-s for s, e in scans)*1000, 2))
        print('scans inside captured IMU stamp range:', covered, '/', len(scans))
        print('scan endpoint to nearest IMU ms median/max:',
              round(median(distances)*1000, 2), round(max(distances)*1000, 2))
    if len(corrections) >= 2:
        jumps = [math.dist(a[1], b[1]) for a, b in zip(corrections, corrections[1:])]
        print('mapping correction max position step m:', round(max(jumps), 3))
        print('mapping corrections marked degenerate:', sum(c[2] == 1 for c in corrections),
              '/', len(corrections))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
