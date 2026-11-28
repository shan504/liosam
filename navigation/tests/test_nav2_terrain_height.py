"""Generated Nav2 YAML keeps terrain-filtered returns at any global elevation."""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import yaml


GO2 = Path(__file__).resolve().parents[1] / 'original_go2' / 'go2_3d'


def sdk_config():
    """Only SDK-provided structure is stubbed; actual prepare.main generates YAML."""
    return {
        'amcl': {'ros__parameters': {}},
        'local_costmap': {'local_costmap': {'ros__parameters': {
            'robot_radius': 0.25, 'obstacle_layer': {'max_obstacle_height': 2.0}}}},
        'global_costmap': {'global_costmap': {'ros__parameters': {
            'robot_radius': 0.25, 'obstacle_layer': {'min_obstacle_height': 0.0}}}},
        'planner_server': {'ros__parameters': {}},
        'controller_server': {'ros__parameters': {
            'FollowPath': {'critics': ['BaseObstacle', 'PathDist', 'RotateToGoal']},
            'progress_checker': {}, 'general_goal_checker': {}}},
        'velocity_smoother': {'ros__parameters': {}},
        'bt_navigator': {'ros__parameters': {}},
        'behavior_server': {'ros__parameters': {}},
    }


def generated_config(directory):
    directory = Path(directory)
    packages = types.ModuleType('ament_index_python.packages')
    packages.get_package_share_directory = lambda name: '/sdk/go2_robot_sdk'
    legacy = types.ModuleType('prepare')

    def prepare_sdk(_source, output):
        output.mkdir(parents=True, exist_ok=True)
        (output / 'go2_edu.urdf').write_text(
            '<robot name="go2"><link name="base_link"/>'
            '<link name="base_footprint"/><joint name="footprint" type="fixed">'
            '<parent link="base_link"/><child link="base_footprint"/></joint></robot>')
        (output / 'nav2_edu.yaml').write_text(yaml.safe_dump(sdk_config()))

    legacy.prepare = prepare_sdk
    modules = {'ament_index_python': types.ModuleType('ament_index_python'),
               'ament_index_python.packages': packages, 'prepare': legacy}
    spec = importlib.util.spec_from_file_location('nav2_terrain_prepare_under_test', GO2 / 'prepare.py')
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path[:]
    missing = object()
    previous_modules = {name: sys.modules.get(name, missing) for name in modules}
    try:
        sys.modules.update(modules)
        try:
            spec.loader.exec_module(module)
        finally:
            # Restore only our ROS/SDK stand-ins. Preserve genuine imports such
            # as NumPy; unloading and reimporting its extension is unsafe.
            for name, previous in previous_modules.items():
                if previous is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
        # The production entry point owns a fixed container output directory.
        # Redirect only that filesystem destination for this offline test.
        with mock.patch.object(module, 'Path', side_effect=lambda path: (
                directory if str(path) == '/runtime/config' else Path(path))), \
                contextlib.redirect_stdout(io.StringIO()):
            module.main('mapping')
    finally:
        sys.path[:] = old_path
    text = (directory / 'nav2_3d.yaml').read_text()
    return yaml.safe_load(text), text


class Nav2TerrainHeight(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config, self.serialized = generated_config(Path(self.temp.name) / 'config')

    def test_both_costmaps_and_both_global_z_gates_admit_uphill_and_downhill(self):
        # ObservationBuffer first filters source bounds after its map/odom TF;
        # ObstacleLayer then checks its own bounds. Fixing only one loses data.
        for name in ('local_costmap', 'global_costmap'):
            layer = self.config[name][name]['ros__parameters']['obstacle_layer']
            for elevation in (-10.0, 0.0, 10.0):
                for relative_height in (0.10, 0.30, 1.5):
                    point_z = elevation + relative_height
                    with self.subTest(costmap=name, z=point_z):
                        self.assertTrue(all(gate['min_obstacle_height'] <= point_z <=
                                            gate['max_obstacle_height']
                                            for gate in (layer, layer['scan'])))

    def test_bounds_are_finite_double_extrema_after_yaml_roundtrip(self):
        self.assertNotIn('.inf', self.serialized.lower())
        self.assertNotIn('.nan', self.serialized.lower())
        for name in ('local_costmap', 'global_costmap'):
            layer = self.config[name][name]['ros__parameters']['obstacle_layer']
            for gate in (layer, layer['scan']):
                self.assertEqual(gate['min_obstacle_height'], -sys.float_info.max)
                self.assertEqual(gate['max_obstacle_height'], sys.float_info.max)
                self.assertTrue(math.isfinite(gate['min_obstacle_height']))
                self.assertTrue(math.isfinite(gate['max_obstacle_height']))

    def test_physical_height_policy_and_collision_configuration_stay_separate(self):
        policy = yaml.safe_load((GO2 / 'navigation_grid.yaml').read_text())
        self.assertEqual(policy['obstacle_min_height'], 0.10)
        self.assertEqual(policy['obstacle_max_height'], 1.5)
        for name in ('local_costmap', 'global_costmap'):
            params = self.config[name][name]['ros__parameters']
            layer = params['obstacle_layer']
            self.assertEqual(layer['plugin'], 'nav2_costmap_2d::ObstacleLayer')
            self.assertEqual(layer['observation_sources'], 'scan')
            scan = {key: value for key, value in layer['scan'].items()
                    if key not in ('min_obstacle_height', 'max_obstacle_height')}
            self.assertEqual(scan, dict(topic='/scan', data_type='LaserScan', clearing=True,
                                        marking=True, inf_is_valid=False,
                                        raytrace_min_range=0.35, raytrace_max_range=15.0,
                                        obstacle_min_range=0.35, obstacle_max_range=12.0))
            self.assertEqual(params['footprint'],
                             '[[-0.40,-0.22],[-0.40,0.22],[0.40,0.22],[0.40,-0.22]]')
            self.assertEqual(params['footprint_padding'], 0.01)
            self.assertEqual(params['inflation_layer'], dict(
                plugin='nav2_costmap_2d::InflationLayer', inflation_radius=0.05,
                cost_scaling_factor=12.0))
            self.assertEqual(params['denoise_layer'], dict(
                plugin='nav2_costmap_2d::DenoiseLayer', enabled=True,
                minimal_group_size=8, group_connectivity_type=8))
        planner = self.config['planner_server']['ros__parameters']['GridBased']
        self.assertFalse(planner['allow_unknown'])


if __name__ == '__main__':
    unittest.main()
