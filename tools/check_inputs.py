#!/usr/bin/env python3
"""Inspect Livox scan fields and stationary IMU values before configuring LIO-SAM."""
import argparse
from collections import Counter
from math import sqrt
from statistics import median
import time

import rclpy
from livox_ros_driver2.msg import CustomMsg
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2, PointField


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapted", action="store_true", help="Inspect both LIO-SAM inputs")
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node("check_livox_inputs")
    got_cloud = False
    accel_norms = []
    imu_target = 200

    def cloud(msg):
        nonlocal got_cloud
        if got_cloud or not msg.points:
            return
        lines = Counter(p.line for p in msg.points)
        offsets = [p.offset_time for p in msg.points]
        print("Cloud frame:", msg.header.frame_id, "points:", len(msg.points))
        print("Points per line:", dict(sorted(lines.items())))
        print("offset_time range (ns):", min(offsets), max(offsets))
        got_cloud = True

    def adapted_cloud(msg):
        nonlocal got_cloud
        if got_cloud or msg.width * msg.height == 0:
            return
        fields = {f.name: f.datatype for f in msg.fields}
        expected = {name: PointField.FLOAT32 for name in ("x", "y", "z", "intensity", "time")}
        expected["ring"] = PointField.UINT16
        got_cloud = all(fields.get(name) == dtype for name, dtype in expected.items())
        print("Adapted cloud frame:", msg.header.frame_id, "points:", msg.width * msg.height)
        print("Point fields:", fields, "LIO-SAM schema OK:", got_cloud)

    def imu(msg):
        if len(accel_norms) >= imu_target:
            return
        if not accel_norms:
            q = msg.orientation
            print("IMU frame:", msg.header.frame_id)
            print("Orientation xyzw:", (q.x, q.y, q.z, q.w))
            print("orientation_covariance[0]:", msg.orientation_covariance[0])
        a = msg.linear_acceleration
        accel_norms.append(sqrt(a.x*a.x + a.y*a.y + a.z*a.z))

    if args.adapted:
        cloud_sub = node.create_subscription(PointCloud2, "/mid360s/points", adapted_cloud,
                                             qos_profile_sensor_data)
        imu_topic = "/mid360s/imu"
    else:
        cloud_sub = node.create_subscription(CustomMsg, "/livox/lidar", cloud, qos_profile_sensor_data)
        imu_topic = "/livox/imu"
    node.create_subscription(Imu, imu_topic, imu, qos_profile_sensor_data)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and (not got_cloud or len(accel_norms) < imu_target):
            rclpy.spin_once(node, timeout_sec=0.25)
            if got_cloud and cloud_sub is not None:
                node.destroy_subscription(cloud_sub)
                cloud_sub = None
        if accel_norms:
            print("Stationary acceleration magnitude (median):", round(median(accel_norms), 4))
        print("Received cloud:", got_cloud, "IMU samples:", len(accel_norms))
        return 0 if got_cloud and len(accel_norms) == imu_target else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
