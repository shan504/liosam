"""Exercise the original Go2 configuration generator against an SDK-style install."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import yaml


NAV = Path(__file__).resolve().parents[1]
ORIGINAL = NAV / 'original_go2'


def load_file(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_sdk_share(directory):
    config = directory / 'config'
    urdf = directory / 'urdf'
    config.mkdir(parents=True)
    urdf.mkdir()
    # ROS packages may ship fixed links for world/odom/footprint. These need to
    # be released to the live Go2 odometer and the old Mapper respectively.
    robot = ET.Element('robot', name='go2')
    names = ('map', 'odom', 'base_link', 'base_footprint', 'arm_mount')
    for name in names:
        ET.SubElement(robot, 'link', name=name)
    for parent, child in [('map', 'odom'), ('odom', 'base_link'),
                          ('base_link', 'base_footprint'), ('base_link', 'arm_mount')]:
        joint = ET.SubElement(robot, 'joint', name=child+'_joint', type='fixed')
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
    ET.ElementTree(robot).write(urdf/'go2.urdf')

    def params(**values):
        return {'ros__parameters': values}

    base = {
        'planner_server': params(GridBased={}),
        'controller_server': params(FollowPath={'critics': ['Oscillation', 'BaseObstacle', 'RotateToGoal']},
                                    progress_checker={}, general_goal_checker={}),
        'behavior_server': params(),
        'bt_navigator': params(),
    }
    for key in ('local_costmap', 'global_costmap'):
        base[key] = {key: params(voxel_layer={'scan': {}}, robot_radius=0.3)}
    (config/'nav2_params.yaml').write_text(yaml.safe_dump(base))
    (config/'mapper_params_online_async.yaml').write_text(
        yaml.safe_dump({'slam_toolbox': params(use_sim_time=True)}))


class OriginalNavigationIntegration(unittest.TestCase):
    def test_actual_sdk_config_generation_and_tf_ownership(self):
        with tempfile.TemporaryDirectory() as work:
            sdk = Path(work)/'sdk'
            output = Path(work)/'runtime'/'config'
            create_sdk_share(sdk)
            ament = types.ModuleType('ament_index_python.packages')
            ament.get_package_share_directory = lambda name: str(sdk)
            sdk_adapter = load_file('sdk_prepare', ORIGINAL/'patches/dds/prepare.py')
            # Only redirect the container's absolute output and dependency
            # lookup. Run both layers of the real configuration generator.
            modules = {'ament_index_python.packages': ament, 'prepare': sdk_adapter}
            with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, 'path', sys.path[:]):
                generator = load_file('go2_prepare', ORIGINAL/'go2_3d/prepare.py')
                real_path = generator.Path
                generator.Path = lambda path: output if str(path) == '/runtime/config' else real_path(path)
                generator.main('mapping')

            root = ET.parse(output/'go2_edu.urdf')
            self.assertEqual({link.get('name') for link in root.findall('link')},
                             {'base_link', 'arm_mount'})
            self.assertEqual([joint.find('child').get('link') for joint in root.findall('joint')],
                             ['arm_mount'])
            config = yaml.safe_load((output/'nav2_3d.yaml').read_text())
            self.assertFalse(config['planner_server']['ros__parameters']['GridBased']['allow_unknown'])
            for key in ('local_costmap', 'global_costmap'):
                params = config[key][key]['ros__parameters']
                self.assertEqual(params['robot_base_frame'], 'base_footprint')
                self.assertEqual(params['obstacle_layer']['scan']['topic'], '/scan')
                self.assertEqual(params['inflation_layer']['inflation_radius'], .05)
                self.assertIn('denoise_layer', params['plugins'])
                self.assertEqual('static_layer' in params['plugins'], key == 'global_costmap')
            follow = config['controller_server']['ros__parameters']['FollowPath']
            self.assertEqual(follow['plugin'], 'nav2_rotation_shim_controller::RotationShimController')
            self.assertEqual(follow['primary_controller'], 'dwb_core::DWBLocalPlanner')
            self.assertIn('ObstacleFootprint', follow['critics'])
            # Test the generated runtime file, not the unused legacy YAML.
            # Restore the original stop/turn-or-0.30m/s forward candidates.
            # Verify the generated file overrides the SDK's legacy 0.15 cap
            # without removing zero-X turning or the acceleration ramp.
            self.assertEqual(follow['vx_samples'], 2)
            self.assertEqual(follow['min_vel_x'], 0.)
            self.assertEqual(follow['min_speed_xy'], .01)
            self.assertEqual(follow['min_speed_theta'], .10)
            self.assertEqual(follow['max_speed_xy'], .30)
            self.assertEqual(follow['trajectory_generator_name'], 'dwb_plugins::StandardTrajectoryGenerator')
            self.assertFalse(follow['limit_vel_cmd_in_traj'])
            self.assertTrue(follow['debug_trajectory_details'])
            self.assertEqual(follow['acc_lim_x'], .40)
            self.assertEqual(follow['decel_lim_x'], -.40)
            # From rest the simulation window must reach the 0.30 candidate;
            # a one-tick dynamic window would silently restore weak commands.
            self.assertGreaterEqual(follow['acc_lim_x'] * follow['sim_time'], .30)
            self.assertEqual(follow['vtheta_samples'], 7)
            self.assertEqual(follow['max_vel_x'], .30)
            self.assertEqual(follow['max_vel_theta'], 1.20)
            self.assertEqual(follow['rotate_to_heading_angular_vel'], 1.20)
            self.assertEqual(follow['acc_lim_theta'], 1.60)
            self.assertEqual(follow['decel_lim_theta'], -1.60)
            self.assertEqual(follow['acc_lim_theta'], follow['max_angular_accel'])
            self.assertIn('Oscillation', follow['critics'])
            self.assertEqual(follow['Oscillation.oscillation_reset_time'], 2.)
            self.assertEqual(follow['Oscillation.oscillation_reset_dist'], .05)
            self.assertEqual(follow['Oscillation.oscillation_reset_angle'], .20)
            self.assertFalse(follow['closed_loop'])
            smoother = config['velocity_smoother']['ros__parameters']
            self.assertEqual(smoother['feedback'], 'OPEN_LOOP')
            self.assertEqual(smoother['smoothing_frequency'], 20.)
            self.assertEqual(smoother['max_velocity'][0], .30)
            self.assertEqual(smoother['min_velocity'][0], 0.)
            self.assertEqual(smoother['max_accel'][0], follow['acc_lim_x'])
            self.assertEqual(smoother['max_decel'][0], follow['decel_lim_x'])
            self.assertGreaterEqual(smoother['max_velocity'][2], follow['rotate_to_heading_angular_vel'])
            self.assertLessEqual(smoother['min_velocity'][2], -follow['rotate_to_heading_angular_vel'])
            self.assertEqual(smoother['max_accel'][2], follow['max_angular_accel'])
            # Acceptance and stopping rules cannot be loosened by turn tuning.
            self.assertEqual(config['controller_server']['ros__parameters']['progress_checker']['movement_time_allowance'], 10.)
            self.assertNotIn('RotateToGoal', follow['critics'])
            bt_path = Path(config['bt_navigator']['ros__parameters']['default_nav_to_pose_bt_xml'])
            installed = ORIGINAL / bt_path.relative_to('/opt/go2_project')
            recovery = ET.parse(installed).find('./BehaviorTree/RecoveryNode')
            self.assertEqual(recovery.get('number_of_retries'), '3')
            self.assertEqual(recovery.find('Wait').get('wait_duration'), '2')

    def test_turn_envelope_reaches_bridge_without_amplifying_small_commands(self):
        motion = load_file('motion_limits', ORIGINAL/'go2_3d/motion_profile.py')
        logic = load_file('command_logic', ORIGINAL/'patches/dds/bridge_logic.py')
        gate = logic.CommandGate()
        gate.max_linear, gate.max_yaw = motion.LINEAR_SPEED, motion.YAW_SPEED
        gate.set_enabled(True)
        for x in (0., motion.LINEAR_SPEED):
            for yaw in [-1.2, -.8, -.4, -.173684, 0., .173684, .4, .8, 1.2]:
                with self.subTest(x=x, yaw=yaw):
                    gate.receive(x, 0., yaw, 10.)
                    action, command = gate.poll(10.01, True)
                    self.assertEqual(action, 'move')
                    self.assertEqual(command[0], x)
                    self.assertAlmostEqual(command[2], yaw)
        # Limits still cap an excessive request rather than disabling the gate.
        for sign in (-1., 1.):
            gate.receive(0., 0., sign * 3., 10.)
            self.assertEqual(gate.poll(10.01, True), ('move', (0., 0., sign * 1.2)))
        gate.set_enabled(False)
        self.assertNotEqual(gate.poll(10.02, True)[0], 'move')

    def test_forward_speed_ramp_and_stop_survive_bridge(self):
        motion = load_file('forward_motion_limits', ORIGINAL/'go2_3d/motion_profile.py')
        logic = load_file('forward_command_logic', ORIGINAL/'patches/dds/bridge_logic.py')
        gate = logic.CommandGate(max_linear=motion.LINEAR_SPEED, max_yaw=motion.YAW_SPEED)
        gate.set_enabled(True)
        # Representative smoother ramp, cruise, slowdown, then stop. The
        # actuator bridge must neither retain an old cruise nor boost a ramp.
        for i, vx in enumerate([.02, .18, .30, .30, .18, .02, 0.]):
            now = 10. + i * .05
            gate.receive(vx, 0., 0., now)
            action, command = gate.poll(now + .01, True)
            self.assertEqual(action, 'move')
            self.assertEqual(command, (vx, 0., 0.))
        gate.receive(.30, 0., 0., 11.)
        self.assertEqual(gate.poll(11.01, True), ('move', (.30, 0., 0.)))
        self.assertEqual(gate.poll(11.02, False), ('stop', None))
        self.assertEqual(gate.poll(11.03, True), (None, None))
        gate.receive(.30, 0., 0., 12.)
        self.assertEqual(gate.poll(12.01, True)[0], 'move')
        self.assertEqual(gate.poll(12.6, True), ('stop', None))

    def test_original_operator_layout_endpoints(self):
        layout = json.loads((NAV/'foxglove.json').read_text())
        panels = layout['configById'].values()
        buttons = {(p['topicName'], p.get('value')) for p in panels if 'topicName' in p}
        self.assertTrue({('/control/enable','{"data":true}'), ('/control/enable','{"data":false}'),
                         ('/navigation/cancel','{}'), ('/navigation/resume','{}')} <= buttons)
        picks = {p['publish']['type']: p['publish'] for p in panels if p.get('publish')}
        self.assertEqual(picks['pose']['poseTopic'], '/goal_pose')
        self.assertEqual(picks['pose_estimate']['poseEstimateTopic'], '/initialpose')
        for name, panel in layout['configById'].items():
            if name.startswith('3D!'):
                self.assertTrue(panel['topics']['/go2_visual/live/model_mesh']['visible'])
                self.assertTrue(panel['topics']['/navigation/waypoints']['visible'])

    def test_session_starts_one_sensor_and_the_original_navigation(self):
        supervisor = load_file('go2_supervisor', NAV/'run_stack.py')
        children, stops = [], []
        def spawn(argv, start_new_session):
            self.assertTrue(start_new_session)
            child = mock.Mock(argv=[str(x) for x in argv])
            child.poll.return_value = None
            children.append(child)
            return child
        # Withdraw the lease as soon as all children reach the run loop. This
        # exercises normal startup and shutdown without opening a ROS graph.
        with mock.patch.object(Path, 'is_file', side_effect=lambda: len(children) < 3), \
             mock.patch.object(supervisor.signal, 'signal'), \
             mock.patch.object(supervisor.subprocess, 'Popen', side_effect=spawn), \
             mock.patch.object(supervisor, 'stop', side_effect=lambda group, grace: stops.extend(group)):
            self.assertEqual(supervisor.run(), 0)
        sensor, model, nav = children
        self.assertIn(str(NAV/'nav_sensors.py'), sensor.argv)
        self.assertIn('--go2-cloud', sensor.argv)
        self.assertIn('/map_source', sensor.argv)
        self.assertIn(str(NAV/'robot_model.py'), model.argv)
        self.assertIn('/opt/go2_project/go2_3d/stack.launch.py', nav.argv)
        self.assertIn('mode:=navigation', nav.argv)
        self.assertIn('initial_from_map_start:=true', nav.argv)
        self.assertIn('port:=8766', nav.argv)
        self.assertEqual(stops[0], nav)



if __name__ == '__main__':
    unittest.main()
