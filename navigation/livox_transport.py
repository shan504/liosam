#!/usr/bin/env python3
"""Expose the latest full driver scan on a per-run, host-local shared filesystem."""
import os
from pathlib import Path
import sys


def main():
    import rospy
    target = Path(sys.argv[1]) / 'scan.bin'
    pending = target.with_suffix('.pending')
    rospy.init_node('go2_nav_livox_sender')

    def scan(msg):
        if (msg._connection_header or {}).get('type') != 'livox_ros_driver2/CustomMsg':
            rospy.logerr_throttle(5., '导航需要 livox_ros_driver2/CustomMsg')
            return
        # One reader obtains either the previous complete message or the next.
        try:
            with pending.open('wb') as stream:
                stream.write(msg._buff)
            os.replace(pending, target)
        except OSError as exc:
            rospy.logerr_throttle(2., '写入雷达共享内存失败: %s', exc)

    subscriber = rospy.Subscriber('/livox/lidar', rospy.AnyMsg, scan,
                                  queue_size=1, buff_size=2**22, tcp_nodelay=True)
    rospy.spin()


if __name__ == '__main__':
    main()
