#!/usr/bin/env python3
"""Read-only Noetic audit for nkymzsy/LIO-SAM-MID360 (Livox-ros-driver2)."""
import argparse
import struct
import time
from collections import Counter, OrderedDict
from threading import Lock
import numpy as np

RAW = "/livox/lidar"
REGISTERED = "/lio_sam/mapping/cloud_registered_raw"
PARAMS = ('N_SCAN', 'Horizon_SCAN', 'downsampleRate', 'lidarMinRange', 'lidarMaxRange')
POINT = np.dtype([('offset_time', '<u4'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                  ('reflectivity', 'u1'), ('tag', 'u1'), ('line', 'u1')])
RAW_NAMES = ('header', 'timebase', 'point_num', 'lidar_id', 'rsvd', 'points')
RAW_TYPES = ('std_msgs/Header', 'uint64', 'uint32', 'uint8', 'uint8[3]', 'livox_ros_driver2/CustomPoint[]')
POINT_TYPES = ('uint32', 'float32', 'float32', 'float32', 'uint8', 'uint8', 'uint8')


def wire_points(payload):
    # ROS 1 header: seq, seconds, nanoseconds, frame_id string length + bytes.
    if len(payload) < 16:
        raise ValueError("原始消息头不足 16 字节")
    _, sec, ns, frame_len = struct.unpack_from('<4I', payload)
    base = 16+frame_len
    if len(payload) < base+20 or ns >= 1000000000:
        raise ValueError("原始消息头或时间戳异常")
    declared = struct.unpack_from('<I', payload, base+8)[0]
    count = struct.unpack_from('<I', payload, base+16)[0]
    if len(payload) != base+20+count*POINT.itemsize or count != declared:
        raise ValueError("point_num、序列化数组长度或字节数不一致，已排除该帧")
    return sec*1000000000+ns, np.frombuffer(payload, dtype=POINT, count=count, offset=base+20)


def classify(points, config):
    nscan, horizon, stride, lo, hi = config
    c = Counter(raw=len(points))
    # This fork iterates i < point_num - 1; don't silently count that return as a range rejection.
    usable = points[:-1]
    c['last'] = int(bool(len(points)))
    with np.errstate(over="ignore", invalid="ignore"):
        r = np.sqrt(usable['x']**2+usable['y']**2+usable['z']**2)
    good = np.isfinite(r)
    c['invalid'] = int((~good).sum())
    c['near'] = int(np.count_nonzero(good & (r < lo)))
    c['far'] = int(np.count_nonzero(good & (r > hi)))
    line = usable['line'][good & (r >= lo) & (r <= hi)]
    eligible = line < nscan
    c['ring'] = int(np.count_nonzero(~eligible))
    line = line[eligible].astype(np.intp, copy=False)
    chosen = line % stride == 0
    c['stride'] = int(np.count_nonzero(~chosen))
    counts = np.bincount(line[chosen], minlength=256)
    c['cap'] = int(np.maximum(counts-horizon, 0).sum())
    c['pred'] = int(np.minimum(counts, horizon).sum())
    return c


class Matches:
    def __init__(self, capacity=64):
        self.wait = {'raw': OrderedDict(), 'reg': OrderedDict()}
        self.seen = {'raw': set(), 'reg': set()}
        self.events = Counter()
        self.totals = Counter()
        self.rows = []
        self.capacity = capacity
        self.lock = Lock()
        self.closed = False

    def add(self, side, t, value):
        other = 'reg' if side == 'raw' else 'raw'
        with self.lock:
            if self.closed:
                return
            self.events[side] += 1
            if t in self.seen[side]:
                self.events[side+'_duplicate'] += 1
                return
            self.seen[side].add(t)
            if t not in self.wait[other]:
                hold = self.wait[side]
                hold[t] = value
                if len(hold) > self.capacity:
                    hold.popitem(last=False)
                    self.events[side+'_evicted'] += 1
                return
            peer = self.wait[other].pop(t)
            counts, reg = (value, peer) if side == 'raw' else (peer, value)
            self.totals.update(counts)
            self.totals['reg'] += reg
            self.rows.append((counts['raw'], counts['pred'], reg))

    def report(self):
        with self.lock:
            self.closed = True
        n = len(self.rows)
        print("收到原始 %d 帧、配准 %d 帧，严格同时间戳配对 %d 帧。" %
              (self.events['raw'], self.events['reg'], n))
        if self.events['raw_duplicate'] or self.events['reg_duplicate'] or self.events['raw_evicted'] or self.events['reg_evicted']:
            print("观测缓存：原始/配准重复帧 %d/%d，超窗淘汰 %d/%d。" %
                  tuple(self.events[k] for k in ('raw_duplicate', 'reg_duplicate', 'raw_evicted', 'reg_evicted')))
        if not n:
            if self.seen['raw'] and self.seen['reg']:
                near = min(abs(a-b) for a in self.seen['raw'] for b in self.seen['reg'])
                print("最接近的时间戳仍相差 %.3f ms；无法计算同帧保留率。" % (near/1e6))
            return False
        c = self.totals
        denom = max(1, c['raw'])
        for label, k in [('末点（按此分支源码）', 'last'), ('非有限坐标或距离', 'invalid'),
                         ('低于最小量程', 'near'), ('超出最大量程', 'far'), ('线号越界', 'ring'),
                         ('按线抽样剔除', 'stride'), ('每线列数超限', 'cap')]:
            print("  %-14s %8d 点  %5.1f%%" % (label, c[k], 100*c[k]/denom))
        print("同帧点数合计：原始 %d，按当前参数预计保留的有限点 %d，配准话题实收 %d（%.1f%%）。" %
              (c['raw'], c['pred'], c['reg'], 100*c['reg']/denom))
        diff = [row[2]-row[1] for row in self.rows]
        print("逐帧 实收−预计：最小 %+d，最大 %+d 点；有差异 %d/%d 帧。" %
              (min(diff), max(diff), sum(d != 0 for d in diff), n))
        if c['invalid']:
            print("含非有限点：分支中的距离比较不能可靠剔除 NaN，预计值只统计有限回波。")
        print("该比例仅适用于配对帧。一次建图更新可跨过多个雷达帧，未配对帧不计为丢点。")
        return True


def main():
    import rospy
    from livox_ros_driver2.msg import CustomMsg, CustomPoint
    from sensor_msgs.msg import PointCloud2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=8)
    args = parser.parse_args(rospy.myargv()[1:])
    if not 1 <= args.seconds <= 30:
        parser.error('--seconds 支持 1 至 30 秒')
    rospy.init_node('go2_check_returns', anonymous=True)
    if ((tuple(CustomMsg.__slots__), tuple(CustomMsg._slot_types)) != (RAW_NAMES, RAW_TYPES) or
            (tuple(CustomPoint.__slots__), tuple(CustomPoint._slot_types)) != (POINT.names, POINT_TYPES)):
        raise SystemExit("本机 Livox 消息结构不同，已终止原始字节解析。")
    try:
        values = [rospy.get_param('/lio_sam/'+k) for k in PARAMS]
        cfg = tuple([int(v) for v in values[:3]]+[float(v) for v in values[3:]])
    except (KeyError, ValueError, TypeError) as exc:
        raise SystemExit("未读到此分支有效参数：%s" % exc)
    if min(cfg[:3]) < 1 or not (np.isfinite(cfg[-1]) and 0 <= cfg[-2] <= cfg[-1]):
        raise SystemExit("参数范围异常：%s" % (cfg,))
    print("按实机参数核对："+'，'.join('%s=%s' % p for p in zip(PARAMS, cfg)), flush=True)
    print("正在同时采样 %.1f 秒……" % args.seconds, flush=True)
    pairs = Matches()
    failures = Counter()

    def raw(msg):
        try:
            if msg._connection_header.get('md5sum') != CustomMsg._md5sum:
                raise ValueError('原始话题类型或 MD5 与本机 Livox 消息不一致')
            t, points = wire_points(msg._buff)
            pairs.add('raw', t, classify(points, cfg))
        except (ValueError, struct.error) as exc:
            with pairs.lock:
                failures[str(exc)] += 1

    def reg(msg):
        if msg.row_step < msg.width*msg.point_step or len(msg.data) < msg.row_step*msg.height:
            with pairs.lock:
                failures['配准点云行长度异常'] += 1
            return
        pairs.add('reg', msg.header.stamp.to_nsec(), msg.width*msg.height)
    sub1 = rospy.Subscriber(RAW, rospy.AnyMsg, raw, queue_size=3, buff_size=4*1024**2, tcp_nodelay=True)
    sub2 = rospy.Subscriber(REGISTERED, PointCloud2, reg, queue_size=3,
                            buff_size=4*1024**2, tcp_nodelay=True)
    try:
        until = time.monotonic()+args.seconds
        while not rospy.is_shutdown() and time.monotonic() < until:
            time.sleep(.05)
    finally:
        sub1.unregister()
        sub2.unregister()
    found = pairs.report()
    with pairs.lock:
        for issue, count in failures.items():
            print("异常 %d 帧：%s" % (count, issue))
    return 0 if found else 2


if __name__ == '__main__':
    raise SystemExit(main())
