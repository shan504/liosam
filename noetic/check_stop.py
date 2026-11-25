#!/usr/bin/env python3
"""Measure a manually stopped Go2 in an active LIO/native mapping run."""
import math
import time
from threading import Lock

from lio_alignment import angle_deg, distance, rigid


def main():
    import rospy
    from geometry_msgs.msg import PoseStamped, Vector3Stamped
    from nav_msgs.msg import Odometry
    rospy.init_node('go2_stopped_check', anonymous=True)
    sources = [
        ('本体', '/go2_native/odometry', Odometry),
        ('地图中的模型', '/go2_visual/live/model_pose', PoseStamped),
        ('LIO 建图雷达', '/lio_sam/mapping/odometry', Odometry)]
    samples = {name: [] for name, _, _ in sources}
    residual = []
    guard = Lock()
    def receive(msg, name):
        pose = msg.pose if isinstance(msg, PoseStamped) else msg.pose.pose
        try:
            value = rigid(pose)
        except ValueError:
            return
        with guard:
            samples[name].append((msg.header.stamp.to_sec(), value))
    def error(msg):
        v = msg.vector
        with guard:
            residual.append(math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z))
    print('保持实机静止 10 秒；开始观察实际位姿。', flush=True)
    subs = [rospy.Subscriber(topic, cls, receive, callback_args=name, queue_size=50)
            for name, topic, cls in sources]
    subs.append(rospy.Subscriber('/go2_lio/registration_residual', Vector3Stamped, error, queue_size=10))
    time.sleep(10.)
    for sub in subs:
        sub.unregister()
    complete = True
    with guard:
        for name, _, _ in sources:
            rows = samples[name]
            span = rows[-1][0]-rows[0][0] if rows else 0.
            gaps = [b[0]-a[0] for a,b in zip(rows,rows[1:])]
            print('%s %d 条，源时间覆盖 %.2f s' % (name, len(rows), span), end='；')
            if len(rows) < 2 or span < 8. or min(gaps) <= 0. or max(gaps) > 1.:
                print('采样覆盖不足或时间不连续')
                complete = False
                continue
            p0, q0 = rows[0][1]
            print('距停稳首帧最大 %.3f m / %.2f°，单步最大 %.3f m' % (
                max(distance(p, p0) for _,(p,q) in rows),
                max(angle_deg(q, q0) for _,(p,q) in rows),
                max(distance(a[1][0], b[1][0]) for a,b in zip(rows,rows[1:]))))
        if residual:
            print('同戳本体预测雷达位置与 LIO 配准结果最大间距 %.3f m（%d 帧）' %
                  (max(residual), len(residual)))
        else:
            print('缺少同戳配准结果')
            complete = False
    if not complete:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
