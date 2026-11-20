from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="go2_mid360s_bridge", executable="cloud_adapter", output="screen"),
        Node(package="go2_mid360s_bridge", executable="imu_si", output="screen"),
        Node(
            package="imu_filter_madgwick",
            executable="imu_filter_madgwick_node",
            output="screen",
            parameters=[{
                "use_mag": False,
                "publish_tf": False,
                "world_frame": "enu",
                "remove_gravity_vector": False,
            }],
            remappings=[
                ("imu/data_raw", "/mid360s/imu_raw_si"),
                ("imu/data", "/mid360s/imu"),
            ],
        ),
    ])
