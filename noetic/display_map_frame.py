#!/usr/bin/env python3
"""Expose the mapping floor to the shared Foxglove layout on this ROS1 master."""


def connect_display_map(world):
    """The navigation map uses the saved go2_ground axes and origin."""
    import rospy
    import tf2_ros
    import yaml
    from geometry_msgs.msg import TransformStamped

    buffer = tf2_ros.Buffer(debug=False)
    listener = tf2_ros.TransformListener(buffer)
    try:
        while not rospy.is_shutdown():
            if buffer.can_transform('go2_ground', world, rospy.Time(0), rospy.Duration(.5)):
                break
        else:
            return
        if buffer.can_transform('map', 'go2_ground', rospy.Time(0)):
            rospy.loginfo('Foxglove 使用已有 map 与 go2_ground 变换')
            return
        frames = yaml.safe_load(buffer.all_frames_as_yaml()) or {}
        if 'map' in frames or any(row['parent'] == 'map' for row in frames.values()):
            rospy.logerr('已有 map 坐标树与 go2_ground 断开，请检查现有 map 变换的发布节点')
            return

        alias = TransformStamped()
        alias.header.stamp = rospy.Time.now()
        alias.header.frame_id, alias.child_frame_id = 'go2_ground', 'map'
        alias.transform.rotation.w = 1.
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        broadcaster.sendTransform(alias)
        return broadcaster
    finally:
        listener.unregister()


def main():
    import rospy

    rospy.init_node('go2_display_map_frame')
    world = 'go2_native/odom' if rospy.get_param('~mapping_mode') == 'native' else 'odom'
    # Keep the latched broadcaster alive for Foxglove clients that reconnect.
    broadcaster = connect_display_map(world)
    rospy.spin()


if __name__ == '__main__':
    main()
