#!/usr/bin/env python3
"""Ground-free display clouds; never used as GICP or collision inputs."""
import copy
from pathlib import Path
import sys
import time

import numpy as np
import yaml

# The launch mounts this source tree at /opt/go2_project and the complete repo
# at /repo. Local tests use the same sibling repository helper.
NAVIGATION = Path(__file__).resolve().parents[2]
if not (NAVIGATION / 'obstacle_view.py').is_file():
    NAVIGATION = Path('/repo/navigation')
sys.path.insert(0, str(NAVIGATION))
from ground import obstacle_mask
from point_noise import cell_keys, retained_cells
from navigation_grid import current_settings
from geometry import pose_matrix, transform_points

NOETIC = NAVIGATION.parent / 'noetic'
sys.path.insert(0, str(NOETIC))
from surveyed_corridor import index_from_document, read_document, rigid_matrix

MAP_TOPIC = '/navigation/display/map_obstacles'
SCAN_TOPIC = '/navigation/display/scan_obstacles'
OBSTACLE_TOPIC = '/navigation/measured_scan_obstacles'


def scan_is_current(stamp_ns, now_ns, valid, valid_received, monotonic_now):
    """Do not present a frozen registration as a current obstacle observation."""
    return (valid and valid_received is not None and
            0. <= monotonic_now - valid_received <= .5 and
            stamp_ns > 0 and -.1e9 <= now_ns - stamp_ns <= 1.e9)


def load_display_corridor(directory, metadata):
    """Static road presentation uses the same selected-map geometry as its grid."""
    descriptor = metadata.get('surveyed_corridor')
    if descriptor is None:
        return None
    name = descriptor.get('file')
    if (not isinstance(name, str) or not name or Path(name).name != name or
            name in ('.', '..')):
        raise ValueError('Display corridor must be local to the selected map bundle')
    directory = Path(directory).resolve()
    path = (directory / name).resolve()
    if path.parent != directory:
        raise ValueError('Display corridor must be local to the selected map bundle')
    document = read_document(path, expected_frame=metadata['source_frame'])
    return index_from_document(document, metadata['map_from_source_map'])


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


def display_points(message, model, block_size=65536, corridor=None, *, noise_min_cells=None):
    """Select only display obstacles in map coordinates at all retained heights."""
    if message.header.frame_id.lstrip('/') != 'map':
        raise ValueError('Display input must already be registered in map; refusing to relabel coordinates')

    policy = current_settings()

    def selected(block):
        finite = np.isfinite(block).all(axis=1)
        result = np.zeros(len(block), dtype=bool)
        result[finite] = obstacle_mask(block[finite], model, policy['obstacle_min_height'])
        if corridor is not None:
            # Only the static navigation presentation omits surveyed road
            # returns. Full map geometry and live sensing keep their points.
            result[finite] &= ~corridor.contains(block[finite])
        return result

    # Components span the complete map, not each parsing block. Sparse cell
    # sets avoid allocating the farm's entire bounding rectangle or imposing
    # an extent cap. XYZ buffers stay bounded until the final output array.
    cells = set()
    for block in xyz_blocks(message, block_size):
        cells.update(map(tuple, cell_keys(block[selected(block), :2], policy['grid_resolution'])))
    retained = retained_cells(cells, min_cells=noise_min_cells)

    def clean(block):
        chosen = block[selected(block)]
        mask = np.fromiter((tuple(cell) in retained for cell in
            cell_keys(chosen[:, :2], policy['grid_resolution'])), dtype=bool, count=len(chosen))
        return chosen[mask]

    count = sum(len(clean(block)) for block in xyz_blocks(message, block_size))
    output = np.empty((count, 3), dtype='<f4')
    offset = 0
    for block in xyz_blocks(message, block_size):
        kept = clean(block)
        output[offset:offset + len(kept)] = kept
        offset += len(kept)
    return output


def live_display_points(message, map_from_body):
    """Display the Nav2 ground classifier's all-height companion without refiltering."""
    if message.header.frame_id != 'base_link':
        raise ValueError('Live obstacle display requires base_link input')
    matrix = rigid_matrix(map_from_body)
    blocks = [block[np.isfinite(block).all(axis=1)] for block in xyz_blocks(message)]
    points = np.concatenate(blocks) if blocks else np.empty((0, 3), dtype='<f4')
    return transform_points(points, matrix).astype('<f4')


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Bool
    from tf2_ros import Buffer, TransformListener, TransformException
    from ros_cloud import xyz_message

    class DisplayCloud(Node):
        def __init__(self):
            super().__init__('go2_navigation_display')
            self.declare_parameter('map_directory', '/runtime/navigation_map')
            directory = Path(self.get_parameter('map_directory').value)
            metadata = yaml.safe_load((directory / 'metadata.yaml').read_text())
            self.model = ground_model(metadata)
            self.corridor = load_display_corridor(directory, metadata)
            retained = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.map_output = self.create_publisher(PointCloud2, MAP_TOPIC, retained)
            self.scan_output = self.create_publisher(PointCloud2, SCAN_TOPIC, qos_profile_sensor_data)
            self.pending_map = self.pending_scan = None
            self.map_published = False
            self.last_scan_stamp = -1
            self.valid, self.valid_received = False, None
            self.scan_header = None
            self.last_clock_ns = None
            self.buffer = Buffer()
            self.listener = TransformListener(self.buffer, self)
            self.create_subscription(PointCloud2, '/map_cloud', self.receive_map, retained)
            self.create_subscription(PointCloud2, OBSTACLE_TOPIC, self.receive_scan, qos_profile_sensor_data)
            self.create_subscription(Bool, '/localization/valid', self.receive_valid, 1)
            self.create_timer(.2, self.process)
            self.get_logger().info('导航显示：静态地图按初始地面 15cm 过滤、去除孤立簇并保留通行区；实时扫描使用 Nav2 同源过滤的全高度点云，按源时间戳变换。')

        def receive_map(self, message):
            if not self.map_published:
                self.pending_map = message

        def receive_scan(self, message):
            stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
            if stamp > self.last_scan_stamp:
                self.last_scan_stamp = stamp
                self.pending_scan = message  # Newest classified scan only; no growing queue.

        def receive_valid(self, message):
            self.valid, self.valid_received = bool(message.data), time.monotonic()
            if not self.valid:
                self.pending_scan = None
                self.clear_scan()

        def clear_scan(self):
            # This is visualization only: never clear a costmap or claim that
            # the physical space is empty when localization is unavailable.
            if self.scan_header is not None:
                header = copy.deepcopy(self.scan_header)
                header.stamp = self.get_clock().now().to_msg()
                self.scan_output.publish(xyz_message(header, np.empty((0, 3), dtype='<f4')))
                self.scan_header = None

        def publish(self, message, publisher, *, static=False):
            points = display_points(message, self.model, corridor=self.corridor if static else None)
            header = copy.deepcopy(message.header)
            header.frame_id = 'map'
            publisher.publish(xyz_message(header, points))
            return len(points)

        def publish_scan(self, message):
            if message.header.frame_id != 'base_link':
                raise ValueError('Live obstacle display requires base_link input')
            transform = self.buffer.lookup_transform('map', 'base_link', Time.from_msg(message.header.stamp))
            t, q = transform.transform.translation, transform.transform.rotation
            matrix = pose_matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
            points = live_display_points(message, matrix)
            header = copy.deepcopy(message.header)
            header.frame_id = 'map'
            self.scan_output.publish(xyz_message(header, points))
            self.scan_header = header

        def process(self):
            now_ns = self.get_clock().now().nanoseconds
            if self.last_clock_ns is not None and now_ns < self.last_clock_ns:
                # A replay restart must not retain points from the previous lap.
                self.last_scan_stamp = -1
                self.pending_scan = None
                self.valid, self.valid_received = False, None
                self.clear_scan()
            self.last_clock_ns = now_ns
            if self.pending_map is not None and not self.map_published:
                message, self.pending_map = self.pending_map, None
                try:
                    count = self.publish(message, self.map_output, static=True)
                    self.map_published = True
                    self.get_logger().info(f'已发布导航障碍显示图：{count} 点；原始完整地图保留供定位。')
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    self.get_logger().warning(f'导航地图显示失败：{error}', throttle_duration_sec=5.)
            if self.pending_scan is not None:
                message, self.pending_scan = self.pending_scan, None
                stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
                if not scan_is_current(stamp, self.get_clock().now().nanoseconds,
                                       self.valid, self.valid_received, time.monotonic()):
                    self.clear_scan()
                    return
                try:
                    # An empty result is published too, clearing old obstacles
                    # from the display when the current scan is all ground.
                    self.publish_scan(message)
                except TransformException as error:
                    # Retry the newest still-fresh packet when its exact TF
                    # reaches this subscriber. Never substitute latest TF.
                    self.pending_scan = message
                    self.clear_scan()
                    self.get_logger().warning(f'实时障碍显示等待源时间戳 TF：{error}', throttle_duration_sec=5.)
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    self.get_logger().warning(f'导航实时显示失败：{error}', throttle_duration_sec=5.)
                    self.clear_scan()
            elif self.scan_header is not None:
                stamp = self.scan_header.stamp.sec * 10**9 + self.scan_header.stamp.nanosec
                if not scan_is_current(stamp, self.get_clock().now().nanoseconds,
                                       self.valid, self.valid_received, time.monotonic()):
                    self.clear_scan()

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
