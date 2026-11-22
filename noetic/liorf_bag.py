#!/usr/bin/env python3
"""Convert a recorded Go2 Noetic session for an offline LIORF comparison."""
import argparse
import os
import struct
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from check_returns import wire_points

# The ROS1 MD5 of the driver2 CustomMsg and its embedded CustomPoint definition.
CUSTOM_TYPE = "livox_ros_driver2/CustomMsg"
CUSTOM_MD5 = "e4d6829bdfe657cb6c21a746c86b21a6"
NS_PER_SEC = 1_000_000_000
LIDAR_OUT = "/liorf_ab/lidar"
IMU_OUT = "/liorf_ab/imu"

# LIORF consumes XYZ, intensity, ring and time. Retain the original tag and
# integer nanosecond offset as additional fields for independent timing audits.
OUTPUT = np.dtype({
    "names": ["x", "y", "z", "intensity", "ring", "tag", "time", "offset_time"],
    "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "u1", "<f4", "<u4"],
    "offsets": [0, 4, 8, 12, 16, 18, 20, 24],
    "itemsize": 28,
})


def convert_points(data, span_limit_ns):
    stamp_ns, points = wire_points(data)
    seq, _, _, frame_len = struct.unpack_from("<4I", data)
    frame = bytes(data[16:16 + frame_len]).decode("utf-8")
    if not frame or stamp_ns == 0:
        raise ValueError("雷达消息缺少坐标系或有效时间戳")
    base_ns = struct.unpack_from("<Q", data, 16 + frame_len)[0]
    finite = np.isfinite(points["x"]) & np.isfinite(points["y"]) & np.isfinite(points["z"])
    nonzero = (points["x"] != 0) | (points["y"] != 0) | (points["z"] != 0)
    valid = points[finite & nonzero]
    if len(valid) and int(valid["offset_time"].max()) > span_limit_ns:
        raise ValueError("雷达回波相对时间超出 --max-span-ms；请核对采集频率和单位")
    offsets = valid["offset_time"]
    reordered = bool(np.any(offsets[1:] < offsets[:-1]))
    if reordered:
        valid = valid[np.argsort(offsets, kind="stable")]
    out = np.zeros(len(valid), dtype=OUTPUT)
    for axis in ("x", "y", "z", "tag", "offset_time"):
        out[axis] = valid[axis]
    out["intensity"] = valid["reflectivity"]
    out["ring"] = valid["line"]
    out["time"] = valid["offset_time"].astype(np.float64) / NS_PER_SEC
    return stamp_ns, seq, frame, base_ns, out, len(points) - len(out), reordered


def make_fields(PointField):
    kinds = {"f": PointField.FLOAT32, "u2": PointField.UINT16,
             "u1": PointField.UINT8, "u4": PointField.UINT32}
    fields = []
    for name in OUTPUT.names:
        dtype, offset = OUTPUT.fields[name]
        kind = "f" if dtype.kind == "f" else "u%d" % dtype.itemsize
        fields.append(PointField(name=name, offset=offset, datatype=kinds[kind], count=1))
    return fields


def check_imu(msg):
    stamp = msg.header.stamp
    if not msg.header.frame_id or not 0 <= stamp.nsecs < NS_PER_SEC or stamp.to_nsec() <= 0:
        raise ValueError("IMU 消息缺少坐标系或有效时间戳")
    vectors = (msg.linear_acceleration, msg.angular_velocity, msg.orientation)
    if not all(np.isfinite([v.x, v.y, v.z]).all() for v in vectors):
        raise ValueError("IMU 含非有限数值")
    q = msg.orientation
    if not np.isfinite(q.w) or np.linalg.norm([q.x, q.y, q.z, q.w]) < 0.1 or msg.orientation_covariance[0] == -1:
        raise ValueError("所选 IMU 没有有效姿态；请选择已滤波、使用 SI 单位的 IMU")


def convert_bag(args):
    import genpy
    import rosbag
    from sensor_msgs.msg import Imu, PointCloud2, PointField
    from std_msgs.msg import Header

    source = Path(args.input).expanduser()
    dest = Path(args.output).expanduser()
    if os.path.lexists(dest):
        raise FileExistsError("输出文件已存在，另取文件名：" + str(dest))
    counts = Counter()
    scan_times = []
    imu_first = imu_last = None
    fields = make_fields(PointField)
    with rosbag.Bag(str(source), "r") as src:
        topics = src.get_type_and_topic_info().topics
        for topic, expected in ((args.lidar, CUSTOM_TYPE), (args.imu, "sensor_msgs/Imu")):
            if topic not in topics or topics[topic].msg_type != expected:
                raise ValueError("输入包缺少 %s 类型话题 %s" % (expected, topic))
        fd, stage = tempfile.mkstemp(dir=str(dest.parent), prefix="." + dest.name + "-", suffix=".bag")
        os.close(fd)
        try:
            with rosbag.Bag(stage, "w") as dst:
                for topic, raw, recorded_at in src.read_messages(topics=[args.lidar, args.imu], raw=True):
                    datatype, data, md5sum = raw[:3]
                    if topic == args.imu:
                        if datatype != Imu._type or md5sum != Imu._md5sum:
                            raise ValueError("IMU 消息类型或 MD5 不符")
                        msg = Imu().deserialize(data)
                        check_imu(msg)
                        stamp = msg.header.stamp.to_nsec()
                        imu_first = stamp if imu_first is None else min(imu_first, stamp)
                        imu_last = stamp if imu_last is None else max(imu_last, stamp)
                        dst.write(IMU_OUT, msg, t=recorded_at)
                        counts["imu"] += 1
                        continue
                    if datatype != CUSTOM_TYPE or md5sum != CUSTOM_MD5:
                        raise ValueError("Livox 消息类型或 MD5 不符，不能使用该字节布局")
                    stamp, seq, frame, base, pts, removed, reordered = convert_points(data, args.span_ns)
                    counts["removed"] += removed
                    if not len(pts):
                        counts["empty"] += 1
                        continue
                    counts["reordered"] += reordered
                    counts["no_spread"] += int(len(pts) > 1 and pts["offset_time"][0] == pts["offset_time"][-1])
                    counts["laser"] += 1
                    counts["points"] += len(pts)
                    counts["max_ring"] = max(counts["max_ring"], int(pts["ring"].max()))
                    counts["base_delta"] = max(counts["base_delta"], abs(base - stamp))
                    end = stamp + int(pts["offset_time"][-1])
                    scan_times.append((stamp, end))
                    header = Header(seq=seq, stamp=genpy.Time(*divmod(stamp, NS_PER_SEC)), frame_id=frame)
                    cloud = PointCloud2(header=header, height=1, width=len(pts), fields=fields,
                                        is_bigendian=False, point_step=OUTPUT.itemsize,
                                        row_step=pts.nbytes, data=pts.tobytes(), is_dense=True)
                    dst.write(LIDAR_OUT, cloud, t=recorded_at)
            if not counts["laser"] or not counts["imu"]:
                raise ValueError("包中没有可转换的有效扫描或 IMU 消息")
            covered = sum(imu_first <= start and end <= imu_last for start, end in scan_times)
            if not covered:
                raise ValueError("没有一帧雷达时段落在所选 IMU 的首末消息时间之间，请检查时间基准")
            # Same-directory hard link commits a complete bag without replacing an existing file.
            os.link(stage, dest)
        finally:
            os.unlink(stage)
    print("输出 %s" % dest)
    print("雷达 %d 帧、%d 个有效回波；最大线号 %d；IMU %d 条" %
          (counts["laser"], counts["points"], counts["max_ring"], counts["imu"]))
    print("剔除非有限或原点回波 %d；跳过空帧 %d；帧内按时间重排 %d 帧" %
          (counts["removed"], counts["empty"], counts["reordered"]))
    if counts["no_spread"]:
        print("注意：%d 帧中多个回波的时间完全相同，请核对逐点时间。" % counts["no_spread"])
    print("雷达时段落入 IMU 首末时间范围 %d/%d；雷达 header 和 timebase 最大差值 %.3f ms" %
          (covered, counts["laser"], counts["base_delta"] / 1e6))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="原始 ROS1 .bag，采集雷达以及已滤波 SI IMU")
    parser.add_argument("output", help="新建对比包的路径")
    parser.add_argument("--lidar", default="/livox/lidar")
    parser.add_argument("--imu", default="/mid360s/imu_filtered", help="含有效姿态且使用 SI 单位的 IMU")
    parser.add_argument("--max-span-ms", type=float, default=1000, help="单帧最大允许回波偏移，毫秒")
    args = parser.parse_args()
    if not 0 < args.max_span_ms <= (2**32 - 1) / 1e6:
        parser.error("--max-span-ms 超出 ROS1 Livox uint32 纳秒字段范围")
    args.span_ns = int(round(args.max_span_ms * 1e6))
    try:
        convert_bag(args)
    except (ImportError, OSError, ValueError, struct.error) as exc:
        sys.exit("转换失败：" + str(exc))


if __name__ == "__main__":
    main()
