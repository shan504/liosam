"""Explicit acquisition and replay contracts; actuator commands are never replayed."""
REPLAY_TOPICS = ('/point_cloud2', '/odom', '/camera/image/compressed', '/camera/display/compressed')
REQUIRED_TOPICS = ('/point_cloud2', '/odom')
RECORD_TOPICS = REPLAY_TOPICS + (
    '/utlidar/robot_odom', '/tf', '/tf_static', '/initialpose', '/goal_pose',
    '/localization/valid', '/localization/status', '/localization/state',
    '/navigation/startup_status', '/navigation/obstacle_cloud',
    '/navigation/obstacle_status', '/navigation/measured_scan_obstacles', '/rosout',
    '/localization/recovery', '/operator/status', '/control/status',
    '/plan', '/local_plan', '/scan', '/cmd_vel', '/cmd_vel_nav', '/cmd_vel_smoothed')


def replay_command(directory, rate=1.0):
    if not 0.5 <= float(rate) <= 1.0:
        raise ValueError('回放速度必须在 0.5 到 1.0 之间')
    return ['ros2', 'bag', 'play', str(directory), '--clock', '100', '--rate', str(rate),
            '--delay', '5', '--qos-profile-overrides-path', '/repo/navigation/bag_qos.yaml',
            '--topics', *REPLAY_TOPICS]
