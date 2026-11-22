#!/usr/bin/env python3
"""Online motion telemetry only: no bag recording, pose clamping or command output."""
import math
import threading
import time


def step(a, b):
    dt = (b[0] - a[0]) * 1e-9
    distance = math.sqrt(sum((x-y)**2 for x, y in zip(a[1], b[1])))
    qa, qb = a[2], b[2]
    norm = math.sqrt(sum(x*x for x in qa) * sum(x*x for x in qb))
    if norm < 1e-12:
        raise ValueError('invalid quaternion')
    cosine = min(1., abs(sum(x*y for x, y in zip(qa, qb))) / norm)
    return distance, dt, math.degrees(2*math.acos(cosine))


class Watch:
    def __init__(self):
        self.lock = threading.Lock()
        self.previous = {}
        self.stats = {}

    def receive(self, message, label):
        pose = message.pose if label == '模型' else message.pose.pose
        p, q = pose.position, pose.orientation
        value = (message.header.stamp.to_nsec(), (p.x, p.y, p.z), (q.x, q.y, q.z, q.w),
                 message.header.frame_id,
                 int(message.pose.covariance[0]) if label == '预测' else None)
        if not all(math.isfinite(v) for v in value[1] + value[2]):
            return
        with self.lock:
            row = self.stats.setdefault(label, {'n': 0, 'max': (0., 0., 0.),
                                                'rot': 0., 'reset': 0, 'order': 0})
            row['n'] += 1
            previous = self.previous.get(label)
            if previous:
                if value[0] <= previous[0]:
                    row['order'] += 1
                    return
                if value[3:] != previous[3:]:
                    row['reset'] += 1
                else:
                    try:
                        metrics = step(previous, value)
                    except ValueError:
                        return
                    if metrics[0] > row['max'][0]:
                        row['max'] = metrics
                    row['rot'] = max(row['rot'], metrics[2])
            self.previous[label] = value

    def report(self):
        with self.lock:
            stats, self.stats = self.stats, {}
            pose = self.previous.get('建图')
        text = []
        for label in ('建图', '增量', '预测', '模型'):
            row = stats.get(label)
            if not row:
                text.append(label + '无新消息')
                continue
            d, dt, _ = row['max']
            text.append('%s n=%d 最大步长=%.3fm/%.3fs 最大转角=%.2f° 重置=%d 非递增=%d' %
                        (label, row['n'], d, dt, row['rot'], row['reset'], row['order']))
        if pose:
            text.append('建图XYZ=(%.2f,%.2f,%.2f)' % pose[1])
        return '运动检查 | ' + ' | '.join(text)


def main():
    import rospy
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
    from rosgraph_msgs.msg import Log
    rospy.init_node('go2_motion_watch')
    watch = Watch()
    predictor = str(rospy.get_param('/lio_sam/odomTopic', 'odometry/imu')).rstrip('/') + '_incremental'
    topics = [('建图', '/lio_sam/mapping/odometry', Odometry),
              ('增量', '/lio_sam/mapping/odometry_incremental', Odometry),
              ('预测', predictor, Odometry),
              ('模型', '/go2_visual/live/model_pose', PoseStamped)]
    subs = [rospy.Subscriber(t, cls, watch.receive, callback_args=label, queue_size=400,
                            buff_size=4*1024*1024, tcp_nodelay=True) for label, t, cls in topics]

    def log(message):
        if 'lio_sam' in message.name and message.level >= Log.WARN:
            print('运动告警 | %s | %s' % (message.name, message.msg), flush=True)
    subs.append(rospy.Subscriber('/rosout', Log, log, queue_size=50))
    print('运动检查已启动：每4秒输出源时间戳步长。预测流含优化回写，跨回写步长不能直接换算为实机速度。', flush=True)
    while not rospy.is_shutdown():
        for _ in range(40):
            if rospy.is_shutdown():
                return
            time.sleep(.1)
        print(watch.report(), flush=True)


if __name__ == '__main__':
    main()
