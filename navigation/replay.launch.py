"""Offline-only launch: same perception/planner, no SDK bridge or physical sensors."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction,IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource,FrontendLaunchDescriptionSource
from launch_ros.actions import Node,SetRemap


def generate_launch_description():
    root=Path('/opt/go2_project/go2_3d');repo=Path('/repo/navigation')
    params='/runtime/config/nav2_3d.yaml';sim={'use_sim_time':True}
    def python(name,path,extra=None):
        return Node(executable='/usr/bin/python3',name=name,output='screen',
                    arguments=[str(path)],parameters=[sim,*(extra or [])])
    return LaunchDescription([
        python('go2_replay_inputs',repo/'replay_inputs.py'),
        python('go2_nav_model',repo/'robot_model.py'),
        Node(package='robot_state_publisher',executable='robot_state_publisher',output='screen',
             parameters=[sim,{'robot_description':Path('/runtime/config/go2_edu.urdf').read_text()}]),
        python('go2_gicp',root/'mapper.py',[str(root/'gicp.yaml'),str(root/'gicp_dense.yaml'),
            {'mode':'localization','map_directory':'/runtime/navigation_map','auto_initialize':False,
             'initial_from_map_start':True,'scan_window':.5}]),
        python('go2_navigation_display',root/'display_cloud.py',[{'map_directory':'/runtime/navigation_map'}]),
        python('go2_ground_obstacles',root/'obstacle_cloud.py',[{'map_directory':'/runtime/navigation_map'}]),
        python('go2_nav2_ready',root/'nav2_ready.py'),
        Node(package='pointcloud_to_laserscan',executable='pointcloud_to_laserscan_node',output='screen',
             remappings=[('cloud_in','/navigation/obstacle_cloud'),('scan','/scan')],
             parameters=[sim,{'target_frame':'base_link','min_height':-10.,'max_height':10.,
                 'range_min':.35,'range_max':20.,'scan_time':.1,'angle_min':-3.14159,'angle_max':3.14159,
                 'angle_increment':.01,'transform_tolerance':.2,'use_inf':False}]),
        Node(package='nav2_map_server',executable='map_server',name='map_server',output='screen',
             parameters=[params,sim,{'yaml_filename':'/runtime/navigation_map/nav.yaml'}]),
        Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',name='lifecycle_manager_map',
             parameters=[sim,{'autostart':True,'node_names':['map_server']}]),
        GroupAction(actions=[
            SetRemap(src='goal_pose',dst='/replay/nav2_direct_goal_unused'),
            SetRemap(src='cmd_vel',dst='/replay/cmd_vel'),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(str(Path(get_package_share_directory('nav2_bringup'))/'launch/navigation_launch.py')),
                launch_arguments={'params_file':params,'use_sim_time':'true','autostart':'false','use_composition':'False'}.items()),
        ]),
        IncludeLaunchDescription(FrontendLaunchDescriptionSource(str(Path(get_package_share_directory('foxglove_bridge'))/'launch/foxglove_bridge_launch.xml')),
            launch_arguments={'port':'8766','send_buffer_limit':'268435456',
                'capabilities':'[clientPublish,connectionGraph,assets]',
                'client_topic_whitelist':'["^/(initialpose|goal_pose)$"]'}.items()),
    ])
