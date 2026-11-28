#!/usr/bin/env python3
"""Publish floor-filtered copies of levelled mapping clouds for Foxglove."""
from copy import copy
import json
from pathlib import Path
import sys
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'navigation'))
from obstacle_view import obstacle_mask
from corridor_recorder import pose_matrix
from surveyed_corridor import CorridorIndex, rigid_matrix, validate_document


def filtered_cloud(msg, corridor=None):
    """Compact complete point records, preserving fields, timestamp and byte order."""
    if msg.header.frame_id.lstrip('/') != 'go2_ground':
        raise ValueError('等待地面标定，输入点云需要在 go2_ground 坐标系中')
    width, height, step = msg.width, msg.height, msg.point_step
    if msg.row_step < width * step or len(msg.data) < height * msg.row_step:
        raise ValueError('PointCloud2 行长或数据长度异常')
    out = copy(msg)
    out.height, out.width, out.row_step, out.data = 1, 0, 0, b''
    if not width or not height:
        return out

    fields = {f.name: f for f in msg.fields}
    endian = '>' if msg.is_bigendian else '<'
    keep = np.ones((height, width), dtype=bool)
    coordinates = []
    for name in ('x', 'y', 'z'):
        f = fields.get(name)
        if f is None or f.count != 1 or f.datatype not in (7, 8):
            raise ValueError('PointCloud2 缺少浮点坐标 ' + name)
        dtype = np.dtype(endian + ('f4' if f.datatype == 7 else 'f8'))
        if not 0 <= f.offset <= step - dtype.itemsize:
            raise ValueError('PointCloud2 坐标偏移越界')
        values = np.ndarray((height, width), dtype=dtype, buffer=msg.data, offset=f.offset,
                            strides=(msg.row_step, step))
        keep &= obstacle_mask(values) if name == 'z' else np.isfinite(values)
        if corridor is not None:
            coordinates.append(values)
    if corridor is not None and keep.any():
        # Only the membership test uses XYZ arrays. Publish original records,
        # including all extra fields and the original timestamp/byte order.
        xyz = np.column_stack([value[keep] for value in coordinates])
        keep[keep] &= ~corridor.contains(xyz)
    records = np.ndarray((height, width), dtype='V%d' % step, buffer=msg.data,
                         strides=(msg.row_step, step))
    selected = records[keep]
    out.width, out.row_step, out.data = len(selected), selected.nbytes, selected.tobytes()
    return out


class CorridorView:
    """A mapping-session descriptor, evaluated in the levelled display frame."""
    def __init__(self):
        self.lock = threading.RLock()
        self.document = None
        self.cached = None

    def update(self, payload):
        # Validate fully before replacing the working snapshot. An incomplete
        # message must never turn into a blanket 'clear all' rectangle.
        document = validate_document(json.loads(payload))
        with self.lock:
            self.document, self.cached = document, None

    def index(self, ground_from_source):
        with self.lock:
            document = self.document
            if document is None or not document['poses']:
                return None
            matrix = rigid_matrix(ground_from_source)
            if self.cached is not None and np.array_equal(self.cached[0], matrix):
                return self.cached[1]
            index = CorridorIndex([matrix @ np.asarray(event['body_pose'])
                                   for event in document['poses']])
            self.cached = (matrix, index)
            return index

    def for_message(self, msg, buffer, rospy):
        # Keep descriptor, transform and index from one version, even if the
        # recorder publishes an optimized path while this callback runs.
        with self.lock:
            document = self.document
        if document is None or not document['poses']:
            return None
        if not msg.header.stamp.to_nsec():
            raise ValueError('通行区显示需要点云的有效源时间戳')
        source = document['frame_id']
        if source == 'go2_ground':
            matrix = np.eye(4)
        else:
            tf = buffer.lookup_transform('go2_ground', source, msg.header.stamp,
                                         rospy.Duration(.05)).transform
            t, q = tf.translation, tf.rotation
            matrix = pose_matrix([t.x, t.y, t.z, q.x, q.y, q.z, q.w])
        with self.lock:
            if self.document is not document:
                raise ValueError('通行区优化版本正在更新，保留本帧原始障碍显示')
            return self.index(matrix)


def main():
    import rospy
    import tf2_ros
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String

    rospy.init_node('go2_mapping_obstacles')
    view = CorridorView()
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)
    tf_errors = (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                 tf2_ros.ExtrapolationException)

    def surveyed(msg):
        try:
            view.update(msg.data)
        except (ValueError, TypeError, KeyError) as exc:
            rospy.logwarn_throttle(5., '建图通行区记录不可用，保留现有筛选：%s', exc)

    def publish(msg, publisher):
        index = None
        try:
            index = view.for_message(msg, buffer, rospy)
        except (ValueError,) + tf_errors as exc:
            rospy.logwarn_throttle(5., '通行区显示变换暂不可用，保留障碍显示：%s', exc)
        try:
            out = filtered_cloud(msg, index)
        except ValueError as exc:
            rospy.logwarn_throttle(5., '贴地点云筛选失败：%s', exc)
            return
        publisher.publish(out)

    subscriptions = [rospy.Subscriber('/go2_mapping/surveyed_corridor', String,
                                     surveyed, queue_size=1)]
    for name in ('scan', 'dense_map'):
        publisher = rospy.Publisher('/go2_visual/ground/' + name, PointCloud2,
                                    queue_size=1, latch=name == 'dense_map')
        subscriptions.append(rospy.Subscriber('/go2_visual/ground_full/' + name, PointCloud2, publish,
                                               callback_args=publisher, queue_size=1, buff_size=32*1024*1024))
    rospy.spin()


if __name__ == '__main__':
    main()

