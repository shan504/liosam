#!/usr/bin/env python3
"""Use selected-map local terrain for obstacles, preserving sensor stamps."""
import json
import time
from collections import deque
from pathlib import Path
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, String
from tf2_ros import Buffer,TransformListener,TransformException
from geometry import cloud_xyz,pose_matrix,transform_points
from ground import heights
from ros_cloud import xyz_message
from navigation_grid import current_settings
from terrain import load_terrain
from live_ground import measured_ground, relative_heights


def obstacle_selection(points_map, plane, min_height, max_height, terrain=None, measured=None):
    """Classify in map geometry without moving or relabeling the output points.

    Missing local terrain support is not evidence of free space. Keep those
    returns, and returns substantially below the saved surface, as obstacles.
    """
    if terrain is None and measured is None:
        h = heights(points_map, plane)
        return (h >= min_height) & (h <= max_height)
    h, supported, _ = relative_heights(points_map, plane, terrain, measured)
    return (~supported | ((h >= min_height) & (h <= max_height)) |
            (supported & (h <= -min_height)))


class ObstacleCloud(Node):
    def __init__(self):
        super().__init__('go2_ground_obstacles')
        self.declare_parameter('map_directory','/maps/latest')
        directory = Path(self.get_parameter('map_directory').value)
        metadata = yaml.safe_load((directory/'metadata.yaml').read_text())
        self.model = metadata['ground_model']
        self.terrain = load_terrain(directory, metadata)
        self.standing_height = metadata.get('body_height_above_ground')
        p = current_settings()
        self.min_height,self.max_height = p['obstacle_min_height'],p['obstacle_max_height']
        self.valid,self.valid_received = False,None
        self.pending = deque(maxlen=20)
        self.last_received_stamp = None
        self.last_processed_stamp = None
        self.state, self.counts = 'waiting_for_localization', {}
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer,self)
        self.pub = self.create_publisher(PointCloud2,'/navigation/obstacle_cloud',qos_profile_sensor_data)
        # Visualization keeps the original vertical extent; Nav2's separate
        # collision-height policy is not a truncation of the lidar or scene.
        self.full_pub = self.create_publisher(PointCloud2,'/navigation/measured_scan_obstacles',qos_profile_sensor_data)
        self.status_pub = self.create_publisher(String,'/navigation/obstacle_status',1)
        self.create_subscription(Bool,'/localization/valid',self.on_valid,1)
        self.create_subscription(PointCloud2,'/point_cloud2',self.receive,qos_profile_sensor_data)
        self.create_timer(0.05,self.process)
        self.create_timer(1.0,self.report)
        if self.terrain is None:
            self.get_logger().warning('旧地图缓存没有局部地形模型，仍使用保存的地面平面；请重新准备导航地图以支持起伏路面。')
        self.get_logger().info(f'Live obstacle heights relative to '
            f'{"local terrain" if self.terrain is not None else "legacy ground plane"}: '
            f'{self.min_height}..{self.max_height}m; measured scan ground enabled=' 
            f'{self.standing_height is not None}; unsupported terrain remains an obstacle')

    def report(self):
        age = None if self.last_processed_stamp is None else (
            self.get_clock().now().nanoseconds-self.last_processed_stamp)/1e9
        msg = String()
        msg.data = json.dumps(dict(state=self.state, source_age_s=age,
            localization_valid=bool(self.valid), **self.counts), ensure_ascii=False)
        self.status_pub.publish(msg)

    def on_valid(self,msg):
        self.valid,self.valid_received = msg.data,time.monotonic()
        if not self.valid:
            self.pending.clear()
            self.state = 'localization_invalid; obstacle output paused'

    def receive(self,msg):
        if (not self.valid or self.valid_received is None or time.monotonic()-self.valid_received>0.5):
            return
        if msg.header.frame_id!='base_link':
            self.state = 'wrong_input_frame:'+msg.header.frame_id
            return
        stamp = Time.from_msg(msg.header.stamp)
        if not -0.1 <= (self.get_clock().now()-stamp).nanoseconds/1e9 <= 1.0:
            self.state = 'scan_timestamp_out_of_window'
            return
        if self.last_received_stamp is not None and stamp.nanoseconds<=self.last_received_stamp:
            return
        self.last_received_stamp = stamp.nanoseconds
        self.pending.append(msg)

    def process(self):
        if (not self.valid or self.valid_received is None or time.monotonic()-self.valid_received>0.5):
            self.pending.clear()
            self.state = 'localization_invalid_or_stale; obstacle output paused'
            return
        while self.pending:
            msg = self.pending[0]
            stamp = Time.from_msg(msg.header.stamp)
            if (self.get_clock().now()-stamp).nanoseconds/1e9>1.0:
                self.pending.popleft()
                self.state = 'scan_expired_waiting_for_tf'
                continue
            try:
                tf = self.buffer.lookup_transform('map','base_link',stamp)
                break
            except TransformException:
                self.state = 'waiting_for_timestamped_map_tf'
                try:
                    latest = self.buffer.lookup_transform('map','base_link',Time())
                    if Time.from_msg(latest.header.stamp)>stamp:
                        # Older than the available history, not waiting on a
                        # future transform; drop this packet and try the next.
                        self.pending.popleft()
                        continue
                except TransformException:
                    pass
                return  # Wait for timestamped TF without relabeling the cloud.
        else:
            return
        self.pending.popleft()
        try:
            t,q = tf.transform.translation,tf.transform.rotation
            matrix = pose_matrix([t.x,t.y,t.z],[q.x,q.y,q.z,q.w])
            points = cloud_xyz(msg)
            points = points[np.isfinite(points).all(axis=1)]
            points_map = transform_points(points,matrix)
            # A cleared, trajectory-only map cannot identify live ground
            # outside its narrow road strip. Use actual returns anchored to
            # the calibrated body height; never clear a blind front box.
            measured = (measured_ground(points_map,matrix[:3,3],self.standing_height)
                        if self.standing_height is not None else None)
            keep = obstacle_selection(points_map,self.model,self.min_height,
                                      self.max_height,self.terrain,measured)
            selected = points[keep]
            self.pub.publish(xyz_message(msg.header,selected))
            h,supported,live = relative_heights(points_map,self.model,self.terrain,measured)
            visual = ~supported | (h>=self.min_height) | (supported & (h<=-self.min_height))
            self.full_pub.publish(xyz_message(msg.header,points[visual]))
            front = (points[:,0]>=0.) & (points[:,0]<=1.) & (np.abs(points[:,1])<=.5)
            self.counts = dict(input_points=len(points), obstacle_points=int(keep.sum()),
                display_points_all_heights=int(visual.sum()),
                measured_ground_points=int(live.sum()), unsupported_points=int((~supported).sum()),
                below_reference_points=int((supported & (h<=-self.min_height)).sum()),
                front_obstacle_points=int((front & keep).sum()),
                front_unknown_points=int((front & ~supported).sum()),
                source_frame=msg.header.frame_id, source_stamp_ns=stamp.nanoseconds)
            self.last_processed_stamp = stamp.nanoseconds
            self.state = 'publishing_current_obstacles'
        except (TransformException,ValueError) as error:
            self.get_logger().warning(f'Ground obstacle filter waiting for timestamped map TF: {error}',throttle_duration_sec=5.0)


if __name__ == '__main__':
    rclpy.init()
    node = ObstacleCloud()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
