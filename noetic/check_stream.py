#!/usr/bin/env python3
"""Read-only sample of ROS payloads requested by Foxglove; never records a bag."""
import argparse
from collections import Counter
import math
import threading
import time


def bridge_node(name):
    return 'foxglove' in name.rsplit('/', 1)[-1].lower()


class Traffic:
    def __init__(self):
        self.lock = threading.Lock()
        self.rows = {}
        self.open = False

    def receive(self, message, topic):
        # Retain only counters, never payloads. First latched message is counted.
        size = len(message._buff)
        with self.lock:
            if not self.open:
                return
            row = self.rows.setdefault(topic, [0, 0, 0])
            row[0] += 1
            row[1] += size
            row[2] = max(row[2], size)

    def snapshot(self):
        with self.lock:
            self.open = False
            return {key: tuple(value) for key, value in self.rows.items()}


def main():
    import rosgraph
    import rospy
    from rosgraph_msgs.msg import Log

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=8.)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 4 <= args.seconds <= 30:
        parser.error('--seconds must be between 4 and 30')
    rospy.init_node('go2_stream_check', anonymous=True, disable_signals=True)
    master = rosgraph.Master(rospy.get_name())
    _, subscriptions, _ = master.getSystemState()
    requested = {topic: sorted(n for n in nodes if bridge_node(n))
                 for topic, nodes in subscriptions if any(bridge_node(n) for n in nodes)}
    if not requested:
        raise SystemExit('未发现 Foxglove 的 ROS 订阅；请连接现有布局后重试。')
    types = dict(master.getTopicTypes())
    print('只采样当前 Foxglove 订阅的 ROS 消息大小，不录包、不修改节点。', flush=True)
    print('下表是 bridge 输入负载，不是 WebSocket 成功传输速度；包括首次锁存消息。')
    for name in sorted(rospy.get_param_names()):
        if 'foxglove' in name and name.rsplit('/', 1)[-1] in (
                'send_buffer_limit', 'topic_whitelist', 'max_update_ms'):
            print(name, '=', rospy.get_param(name))
    traffic = Traffic()
    warnings = Counter()
    warning_lock = threading.Lock()

    def log(message):
        if bridge_node(message.name) and 'send buffer limit' in message.msg.lower():
            with warning_lock:
                warnings[message.msg] += 1

    handles = [rospy.Subscriber('/rosout', Log, log, queue_size=100)]
    begin = time.monotonic()
    traffic.open = True
    try:
        for topic in requested:
            handles.append(rospy.Subscriber(topic, rospy.AnyMsg, traffic.receive,
                                             callback_args=topic, queue_size=1,
                                             buff_size=32 * 1024 * 1024))
        deadline = begin + args.seconds
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            time.sleep(.05)
    finally:
        rows = traffic.snapshot()
        elapsed = time.monotonic() - begin
        for handle in handles:
            handle.unregister()
    print('\n采样 %.2f 秒；MiB=1048576 字节' % elapsed)
    print(' MiB/s   最大消息MiB   条数   话题')
    for topic in sorted(requested, key=lambda t: rows.get(t, (0, 0, 0))[1], reverse=True):
        count, size, peak = rows.get(topic, (0, 0, 0))
        print('%7.2f %12.2f %6d   %s [%s]' % (
            size / elapsed / 1048576, peak / 1048576, count, topic, types.get(topic, '?')))
    total = sum(row[1] for row in rows.values()) / elapsed / 1048576
    print('合计采样输入 %.2f MiB/s；每个话题只计一次，不按客户端数量翻倍。' % total)
    with warning_lock:
        for message, count in warnings.items():
            print('缓冲区告警 ×%d：%s' % (count, message))
        if not warnings:
            print('本窗口未收到发送缓冲区告警。')
    _, after, _ = master.getSystemState()
    current = {t for t, nodes in after if any(bridge_node(n) for n in nodes)}
    if current != set(requested):
        print('采样期间 Foxglove 订阅发生变化；请固定布局后再比较。')
    print('采样可能漏掉本机订阅队列溢出的消息；本报告不用于判断雷达丢包。')


if __name__ == '__main__':
    main()
