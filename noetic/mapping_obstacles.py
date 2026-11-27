#!/usr/bin/env python3
"""Publish floor-filtered copies of levelled mapping clouds for Foxglove."""
from copy import copy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'navigation'))
from obstacle_view import obstacle_mask


def filtered_cloud(msg):
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
    records = np.ndarray((height, width), dtype='V%d' % step, buffer=msg.data,
                         strides=(msg.row_step, step))
    selected = records[keep]
    out.width, out.row_step, out.data = len(selected), selected.nbytes, selected.tobytes()
    return out


def main():
    import rospy
    from sensor_msgs.msg import PointCloud2

    rospy.init_node('go2_mapping_obstacles')

    def publish(msg, publisher):
        try:
            out = filtered_cloud(msg)
        except ValueError as exc:
            rospy.logwarn_throttle(5., '贴地点云筛选失败：%s', exc)
            return
        publisher.publish(out)

    subscriptions = []
    for name in ('scan', 'dense_map'):
        publisher = rospy.Publisher('/go2_visual/ground/' + name, PointCloud2,
                                    queue_size=1, latch=name == 'dense_map')
        subscriptions.append(rospy.Subscriber('/go2_visual/ground_full/' + name, PointCloud2, publish,
                                               callback_args=publisher, queue_size=1, buff_size=32*1024*1024))
    rospy.spin()


if __name__ == '__main__':
    main()
