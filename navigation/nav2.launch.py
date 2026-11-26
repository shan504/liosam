"""Humble navigation on a saved occupancy grid and an external GICP map transform."""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    here = Path(__file__).resolve().parent
    params = str(here / 'nav2.yaml')
    raw_command = '/navigation/controller_cmd_vel'
    managed = []

    def server(package, name, parameters=(), remappings=()):
        if name != 'map_server':
            managed.append(name)
        return Node(package=package, executable=name, name=name, output='screen',
                    parameters=[params, *parameters], remappings=list(remappings),
                    on_exit=EmitEvent(event=Shutdown(reason=name + ' exited')))

    return LaunchDescription([
        DeclareLaunchArgument('map', description='Absolute path to the derived Nav2 map YAML'),
        server('nav2_map_server', 'map_server', [
            {'yaml_filename': ParameterValue(LaunchConfiguration('map'), value_type=str)}]),
        server('nav2_planner', 'planner_server'),
        server('nav2_controller', 'controller_server', remappings=[('cmd_vel', raw_command)]),
        server('nav2_behaviors', 'behavior_server', remappings=[('cmd_vel', raw_command)]),
        server('nav2_velocity_smoother', 'velocity_smoother',
               remappings=[('cmd_vel', raw_command), ('cmd_vel_smoothed', '/cmd_vel_nav')]),
        server('nav2_bt_navigator', 'bt_navigator', [{
            'default_nav_to_pose_bt_xml': str(here / 'navigate.xml'),
            'default_nav_through_poses_bt_xml': str(here / 'navigate_through.xml'),
        }], [('goal_pose', '/navigation/nav2_direct_goal_unused')]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_map', output='screen',
             parameters=[{'autostart': True, 'node_names': ['map_server']}],
             on_exit=EmitEvent(event=Shutdown(reason='map lifecycle manager exited'))),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[{'autostart': False, 'node_names': managed}],
             on_exit=EmitEvent(event=Shutdown(reason='Nav2 lifecycle manager exited'))),
        Node(executable='/usr/bin/python3', name='go2_nav2_ready', output='screen',
             arguments=['-u', str(here / 'nav2_ready.py')],
             on_exit=EmitEvent(event=Shutdown(reason='Nav2 readiness monitor exited'))),
    ])
