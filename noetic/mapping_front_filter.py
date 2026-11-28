#!/usr/bin/env python3
"""Mapping-only Livox gate: retain the front half-space in physical base_link."""
import os
from pathlib import Path
import struct
from threading import Lock

import numpy as np

from check_returns import wire_points
from dense_map import geometry
from native_cloud import calibration, same_mount
from native_inspect_tf import StaticTF
from native_lio_seed import raw_header, select_mount
from native_mount import TOPICS as MOUNT_TOPICS

INPUT = '/livox/lidar'
OUTPUT = '/go2_mapping/lidar_front'


def filter_payload(payload, mount):
    """Zero rejected XYZ, preserving every scan/return timestamp and other byte.

    The installed MID360 LIO-SAM converter uses point_num-1 returns and reads
    scan end from the penultimate original return. Removing/reordering points
    would change deskew timing. Finite zero-range placeholders are discarded
    by its positive lidarMinRange and by native_cloud.place's range>0 check.
    They are never infinity/free-space rays. Coordinates of kept points stay
    in the original LiDAR frame; only the selection uses body coordinates.
    """
    values = np.asarray(mount, dtype=np.float64)
    if values.shape != (7,):
        raise ValueError('安装外参需要 7 个 XYZ/XYZW 数值')
    values = calibration(values[:3], values[3:])
    stamp, points = wire_points(payload)
    if stamp <= 0 or len(points) < 2:
        raise ValueError('扫描需要有效源戳和至少两个带时序的回波')
    rotation, translation = geometry(values)
    xyz = np.column_stack([points[name] for name in ('x', 'y', 'z')]).astype(np.float64)
    with np.errstate(invalid='ignore'):
        body_x = xyz @ rotation[0] + translation[0]
    keep = np.isfinite(xyz).all(axis=1) & (body_x >= 0.) & np.any(xyz != 0., axis=1)
    retained = int(np.count_nonzero(keep))
    if retained == len(points):
        return bytes(payload), retained, 0
    result = bytearray(payload)
    _, writable = wire_points(result)
    for name in ('x', 'y', 'z'):
        writable[name][~keep] = 0.
    return bytes(result), retained, len(points)-retained


def main():
    import rospy
    from livox_ros_driver2.msg import CustomMsg
    from tf2_msgs.msg import TFMessage

    rospy.init_node('go2_mapping_front_filter')
    uses_lio = bool(rospy.get_param('~uses_lio', True))
    path = Path(os.environ.get('GO2_BODY_FROM_LIDAR') or
                '~/.config/go2/body_from_lidar.json').expanduser()
    urdf_path = os.environ.get('GO2_ROBOT_URDF')
    publisher = rospy.Publisher(OUTPUT, CustomMsg, queue_size=2)
    graphs = {'/tf_static': StaticTF()}
    guard = Lock()
    revision = 0
    checked = None
    mount = physical_frame = None
    rospy.set_param('~policy', dict(version=1, frame='base_link',
                    keep='x >= 0', input=INPUT, output=OUTPUT,
                    rejected_return='zero_range_preserve_timing', mapping_only=True))

    def accept_tf(msg, source):
        nonlocal revision
        private = source != '/tf_static'
        if private and (msg._connection_header or {}).get('callerid') != '/go2_native_odom_bridge':
            return
        with guard:
            graph = StaticTF() if private else graphs[source]
            graph.add(msg)
            graphs[source] = graph
            revision += 1

    def receive(msg):
        nonlocal mount, physical_frame, checked
        if (msg._connection_header or {}).get('md5sum') != CustomMsg._md5sum:
            rospy.logerr_throttle(5., '前方建图过滤需要 livox_ros_driver2/CustomMsg')
            return
        try:
            # This check is required for zero-range timing placeholders. Do not
            # silently pass through raw returns if a deployment changes it.
            if uses_lio:
                minimum = float(rospy.get_param('/lio_sam/lidarMinRange', 0.))
                if not np.isfinite(minimum) or minimum <= 0.:
                    raise ValueError('等待 LIO-SAM 正数 lidarMinRange，确保无效回波不进入地图')
            _, frame = raw_header(msg._buff)
            with guard:
                if physical_frame is not None and frame != physical_frame:
                    raise ValueError('雷达坐标系改变，请核对后重新建图')
                stat = path.stat() if path.is_file() else None
                # Revalidate URDF-only installations too, even when editing a
                # file/parameter does not emit a new /tf_static snapshot.
                urdf = '' if stat is not None else (
                    Path(urdf_path).expanduser().read_text() if urdf_path else
                    rospy.get_param('/robot_description', ''))
                key = (revision, None if stat is None else (stat.st_ino, stat.st_mtime_ns, stat.st_size), urdf)
                if mount is None or key != checked:
                    candidate, source = select_mount(frame, graphs, path, urdf)
                    if mount is not None and not same_mount(mount, candidate):
                        raise ValueError('运行中机身到雷达安装外参改变，请重新建图')
                    if mount is None:
                        mount, physical_frame = candidate, frame
                        rospy.set_param('~mount', dict(parent='base_link', child=frame,
                                        xyz=mount[:3].tolist(), xyzw=mount[3:].tolist(), source=source))
                        rospy.loginfo('前方建图过滤已启用：机身 base_link X>=0；安装外参 %s', source)
                    checked = key
                current_mount = mount.copy()
            payload, retained, rejected = filter_payload(msg._buff, current_mount)
            if not retained:
                rospy.logwarn_throttle(5., '本扫描没有有效前方回波，跳过；未把后方当作空闲空间')
                return
            # rospy's typed Publisher explicitly accepts AnyMsg. Its wire
            # serializer preserves the original header/offsets without creating
            # tens of thousands of Python CustomPoint objects for every scan.
            output = rospy.AnyMsg()
            output._buff = payload
            publisher.publish(output)
            rospy.loginfo_throttle(10., '前方建图过滤：保留 %d 点，排除后方/无效 %d 点', retained, rejected)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, struct.error) as exc:
            rospy.logwarn_throttle(5., '暂停送入建图，前方过滤未就绪：%s', exc)

    subscriptions = [rospy.Subscriber(topic, TFMessage, accept_tf, callback_args=topic, queue_size=16)
                     for topic in ['/tf_static'] + list(MOUNT_TOPICS.values())]
    subscriptions.append(rospy.Subscriber(INPUT, rospy.AnyMsg, receive, queue_size=2,
                                          buff_size=4*1024**2, tcp_nodelay=True))
    rospy.spin()


if __name__ == '__main__':
    main()
