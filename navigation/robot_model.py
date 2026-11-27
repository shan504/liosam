"""Publish the Go2 standing mesh used by mapping in the navigation body frame."""
from pathlib import Path
import time
import zipfile
import zlib

import numpy as np

MESH = Path(__file__).resolve().parent / 'assets' / 'go2' / 'standing.npz'
TOPIC = '/go2_visual/live/model_mesh'


def read_mesh(path):
    with np.load(path, allow_pickle=False) as mesh:
        triangles, ids, rgba = mesh['triangles'], mesh['colors'], mesh['palette']
    if (triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0
            or not np.isfinite(triangles).all() or ids.shape != (len(triangles),)
            or not np.issubdtype(ids.dtype, np.integer)
            or rgba.ndim != 2 or rgba.shape[1] != 4 or not np.isfinite(rgba).all()
            or not ((rgba >= 0) & (rgba <= 1)).all() or ids.min() < 0 or ids.max() >= len(rgba)):
        raise ValueError('standing.npz 的三角面或颜色索引无效')
    return triangles, ids, rgba


def publish_model(node):
    from geometry_msgs.msg import Point
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.serialization import serialize_message
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker

    try:
        triangles, ids, rgba = read_mesh(MESH)
    except (OSError, KeyError, TypeError, ValueError, EOFError, RuntimeError, zipfile.BadZipFile, zlib.error) as exc:
        node.get_logger().warning(f'未加载 Go2 模型，请检查 {MESH}：{exc}')
        return

    marker = Marker()
    marker.header.frame_id = 'base_link'
    marker.frame_locked = True
    marker.ns, marker.id = 'go2_edu_urdf', 0
    marker.type, marker.action = Marker.TRIANGLE_LIST, Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = marker.scale.y = marker.scale.z = 1.0
    marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
    marker.points = [Point(x=x, y=y, z=z) for x, y, z in triangles.reshape(-1, 3).tolist()]
    palette = [ColorRGBA(r=r, g=g, b=b, a=a) for r, g, b, a in rgba.tolist()]
    marker.colors = [palette[i] for i in np.repeat(ids, 3)]
    data = serialize_message(marker)

    pub = node.create_publisher(Marker, TOPIC, QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    pub.publish(data)
    last_send = time.monotonic()
    previous = 0

    def refresh():
        nonlocal previous, last_send
        count = pub.get_subscription_count()
        now = time.monotonic()
        # Older bridges can subscribe with volatile durability or share one ROS subscription.
        if count and (count > previous or now - last_send >= 15.0):
            pub.publish(data)
            last_send = now
        previous = count

    node.create_timer(1.0, refresh)
    node.get_logger().info(f'已发布 Go2 站姿模型 {TOPIC}，{len(triangles)} 个三角面，跟随 base_link')


if __name__ == '__main__':
    import rclpy
    from rclpy.node import Node
    rclpy.init()
    node = Node('go2_nav_model')
    try:
        publish_model(node)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
