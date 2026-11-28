#!/usr/bin/env python3
"""Ground-free display clouds; never used as GICP or collision inputs."""
import copy
from pathlib import Path
import sys

import numpy as np
import yaml

# The launch mounts this source tree at /opt/go2_project and the complete repo
# at /repo. Local tests use the same sibling repository helper.
NAVIGATION = Path(__file__).resolve().parents[2]
if not (NAVIGATION / 'obstacle_view.py').is_file():
    NAVIGATION = Path('/repo/navigation')
sys.path.insert(0, str(NAVIGATION))
from obstacle_view import obstacle_mask
from ground import heights

MAP_TOPIC = '/navigation/display/map_obstacles'
SCAN_TOPIC = '/navigation/display/scan_obstacles'


def ground_model(metadata):
    """Require the selected map's physical height model, never guess Z=0."""
    if metadata.get('frame_id', '').lstrip('/') != 'map':
        raise ValueError('Display metadata must describe the selected map frame')
    model = metadata['ground_model']
    plane = np.asarray(model['plane'], dtype=float)
    if (plane.shape != (4,) or not np.isfinite(plane).all() or plane[2] <= 0 or
            not np.isclose(np.linalg.norm(plane[:3]), 1., atol=1e-5)):
        raise ValueError('Display needs a finite, normalized upward saved ground plane')
    return dict(model, plane=plane.tolist())


def xyz_blocks(message, block_size=65536):
    """Parse PointCloud2 in bounded blocks, including row padding/endian fields.

    block_size only bounds temporary arrays. It does not truncate map extents,
    point counts or height. The original message buffer is never modified.
    """
    if type(block_size) is not int or block_size <= 0:
        raise ValueError('Display block size must be a positive integer')
    width, height = int(message.width), int(message.height)
    point_step, row_step = int(message.point_step), int(message.row_step)
    if (min(width, height, point_step, row_step) < 0 or point_step == 0 or
            row_step < width * point_step or len(message.data) != row_step * height):
        raise ValueError('Invalid display PointCloud2 dimensions/data')
    fields = {field.name: field for field in message.fields}
    sizes = {7: ('f4', 4), 8: ('f8', 8)}
    endian = '>' if message.is_bigendian else '<'
    formats, offsets = [], []
    for name in ('x', 'y', 'z'):
        field = fields.get(name)
        if field is None or field.datatype not in sizes or field.count != 1:
            raise ValueError('Display needs scalar FLOAT32/FLOAT64 XYZ fields')
        kind, size = sizes[field.datatype]
        if field.offset < 0 or field.offset + size > point_step:
            raise ValueError('Display XYZ field outside point stride')
        formats.append(endian + kind)
        offsets.append(field.offset)
    dtype = np.dtype(dict(names=['x', 'y', 'z'], formats=formats,
                         offsets=offsets, itemsize=point_step))
    data = np.ndarray((height, width), dtype=dtype, buffer=message.data,
                      strides=(row_step, point_step))
    for row in data:
        for start in range(0, width, block_size):
            chunk = row[start:start + block_size]
            yield np.column_stack([chunk[name] for name in ('x', 'y', 'z')])


def display_points(message, model, block_size=65536):
    """Select only display obstacles in map coordinates at all retained heights."""
    if message.header.frame_id.lstrip('/') != 'map':
        raise ValueError('Display input must already be registered in map; refusing to relabel coordinates')

    def selected(block):
        finite = np.isfinite(block).all(axis=1)
        result = np.zeros(len(block), dtype=bool)
        result[finite] = obstacle_mask(heights(block[finite], model))
        return result

    # Two bounded passes allocate exactly one final XYZ array, rather than a
    # full source XYZ array, full-map mask and multiple retained-cloud copies.
    count = sum(int(np.count_nonzero(selected(block)))
                for block in xyz_blocks(message, block_size))
    output = np.empty((count, 3), dtype='<f4')
    offset = 0
    for block in xyz_blocks(message, block_size):
        kept = block[selected(block)]
        output[offset:offset + len(kept)] = kept
        offset += len(kept)
    return output


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from ros_cloud import xyz_message

    class DisplayCloud(Node):
        def __init__(self):
            super().__init__('go2_navigation_display')
            self.declare_parameter('map_directory', '/runtime/navigation_map')
            directory = Path(self.get_parameter('map_directory').value)
            self.model = ground_model(yaml.safe_load((directory / 'metadata.yaml').read_text()))
            retained = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.map_output = self.create_publisher(PointCloud2, MAP_TOPIC, retained)
            self.scan_output = self.create_publisher(PointCloud2, SCAN_TOPIC, qos_profile_sensor_data)
            self.pending_map = self.pending_scan = None
            self.map_published = False
            self.last_scan_stamp = -1
            self.create_subscription(PointCloud2, '/map_cloud', self.receive_map, retained)
            self.create_subscription(PointCloud2, '/registered_cloud', self.receive_scan, qos_profile_sensor_data)
            self.create_timer(.2, self.process)
            self.get_logger().info('导航显示：使用本地图地面标定去除低于 10cm 的点；保留全部高度障碍。定位和代价地图输入不变。')

        def receive_map(self, message):
            if not self.map_published:
                self.pending_map = message

        def receive_scan(self, message):
            stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
            if stamp > self.last_scan_stamp:
                self.last_scan_stamp = stamp
                self.pending_scan = message  # Only newest accepted registered scan; no growing queue.

        def publish(self, message, publisher):
            points = display_points(message, self.model)
            header = copy.deepcopy(message.header)
            header.frame_id = 'map'
            publisher.publish(xyz_message(header, points))
            return len(points)

        def process(self):
            if self.pending_map is not None and not self.map_published:
                message, self.pending_map = self.pending_map, None
                try:
                    count = self.publish(message, self.map_output)
                    self.map_published = True
                    self.get_logger().info(f'已发布导航障碍显示图：{count} 点；原始完整地图保留供定位。')
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    self.get_logger().warning(f'导航地图显示失败：{error}', throttle_duration_sec=5.)
            if self.pending_scan is not None:
                message, self.pending_scan = self.pending_scan, None
                try:
                    # An empty result is published too, clearing old obstacles
                    # from the display when the current scan is all ground.
                    self.publish(message, self.scan_output)
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    self.get_logger().warning(f'导航实时显示失败：{error}', throttle_duration_sec=5.)

    rclpy.init()
    node = DisplayCloud()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
