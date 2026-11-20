"""Run standalone sensor-frame mapping from already adapted data (live or bag)."""
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    upstream = Path(get_package_share_directory("lio_sam")) / "config"
    bridge = Path(get_package_share_directory("go2_mid360s_bridge")) / "config"
    params = {}
    for path in (upstream / "params.yaml", bridge / "mid360s_inputs.yaml",
                 bridge / "lidar_only_mapping.yaml"):
        with path.open() as stream:
            params.update(yaml.safe_load(stream)["/**"]["ros__parameters"])

    executables = ["lio_sam_imuPreintegration", "lio_sam_imageProjection",
                   "lio_sam_featureExtraction", "lio_sam_mapOptimization"]
    # imuPreintegration hosts two nodes; retain their distinct upstream names.
    return LaunchDescription([
        Node(package="lio_sam", executable=exe, output="screen", parameters=[params],
             remappings=[("/tf", "/lio_sam/tf"), ("/tf_static", "/lio_sam/tf_static")])
        for exe in executables
    ])
