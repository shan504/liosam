"""Exercise production Mapper.process acceptance without ROS/Open3D hardware.

Only the expensive solver, cloud containers and ROS messages are fakes. The
production process, SE(3) geometry, match acceptance, stationary guard and odom
budget run unchanged. These tests check commit/freshness behavior, not GICP
convergence or execution speed; those need the real backend/robot.
"""
import ast
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as Obj
import unittest
from unittest.mock import Mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1] / 'original_go2/go2_3d'
sys.path.insert(0, str(ROOT))
from geometry import (filter_points, first_map_pose, pose_matrix, quaternion_from_matrix,
                      rotation_angle, transform_points)
from localization_hold import OdometryBudget


class ClockTime:
    def __init__(self, ns=0):
        self.nanoseconds = ns

    @classmethod
    def from_msg(cls, stamp):
        return cls(stamp.sec * 10**9 + stamp.nanosec)

    def __sub__(self, other):
        return ClockTime(self.nanoseconds - other.nanoseconds)

    def to_msg(self):
        sec, nanosec = divmod(self.nanoseconds, 10**9)
        return Obj(sec=sec, nanosec=nanosec)


def stamped_cloud(ns, points):
    sec, nanosec = divmod(ns, 10**9)
    return Obj(header=Obj(stamp=Obj(sec=sec, nanosec=nanosec), frame_id='base_link'),
               points=np.array(points, dtype=float).reshape(-1, 3))


def production_process(namespace):
    tree = ast.parse((ROOT / 'mapper.py').read_text())
    mapper = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Mapper')
    mapper.bases = []
    names = {'process', 'publish_map', 'localization_readiness'}
    mapper.body = [n for n in mapper.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in mapper.body} == names
    errors = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ScanMatchUnavailable']
    require = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'require_match']
    transforms = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'transform_matrix']
    registration = ast.parse((ROOT / 'registration.py').read_text())
    helpers = [n for n in registration.body if isinstance(n, ast.FunctionDef) and n.name in
               {'static_correction_is_consistent', 'blend_correction', 'odometry_is_stationary'}]
    assert len(errors) == len(require) == 1 and len(helpers) == 3
    exec(compile(ast.Module(body=[*helpers, *errors, *require, *transforms, mapper], type_ignores=[]),
                 str(ROOT / 'mapper.py'), 'exec'), namespace)
    return namespace['Mapper']


class LocalizationDeadline(unittest.TestCase):
    def setUp(self):
        self.reset()

    def reset(self, origin=(0., 0., 0.), mode='localization'):
        self.monotonic = 500.
        self.ros_ns = 100_000_000_000
        self.align_hook = self.serialize_hook = None
        self.accepted = True
        self.result_reason = None
        self.align_calls = []
        # Asymmetric real coordinates; this harness never substitutes an
        # all-zero scan that could accidentally pass the finite/range filter.
        rng = np.random.default_rng(930)
        self.scan = rng.uniform([1., -.8, -.2], [4., 1.3, 1.4], (240, 3))
        self.odom = pose_matrix([2., -.4, .32], [0., 0., 0., 1.])
        heading = .3 if origin[0] == 0 else -.7
        self.initial_correction = pose_matrix(origin, [0., 0., math.sin(heading/2), math.cos(heading/2)])
        self.solution = self.initial_correction @ self.odom
        self.solution[:3, 3] += [.1, -.08, .02]
        self.map_points = transform_points(self.scan, self.solution)
        coarse = Obj(points=self.map_points.copy())
        fine = Obj(points=self.map_points.copy())

        def cloud(points, voxel=None):
            return Obj(points=np.array(points, dtype=float, copy=True))

        def bounded(points, voxel, limit, prepare_covariances=True):
            return cloud(points, voxel)

        def align(source, target, guess, **kwargs):
            self.align_calls.append(kwargs)
            if self.align_hook is not None:
                self.align_hook()
            return Obj(accepted=self.accepted, transform=self.solution.copy(), fitness=.9,
                       rmse=.03 if self.accepted else 1.,
                       reason=self.result_reason or ('accepted' if self.accepted else 'overlap/residual rejected'))

        def serialized(header, points):
            if self.serialize_hook is not None:
                self.serialize_hook()
            return Obj(header=header, points=np.array(points, copy=True))

        self.serialize = Mock(side_effect=serialized)
        ns = dict(np=np, time=Obj(monotonic=lambda: self.monotonic), Time=ClockTime,
                  Header=lambda **kw: Obj(**kw), String=lambda **kw: Obj(**kw), TransformException=LookupError,
                  cloud=cloud, bounded_cloud=bounded,
                  cloud_xyz=lambda message: message.points.copy(),
                  filter_points=filter_points,
                  align=align, xyz_message=self.serialize, transform_points=transform_points,
                  first_map_pose=first_map_pose, pose_matrix=pose_matrix,
                  OdometryBudget=OdometryBudget,
                  quaternion_from_matrix=quaternion_from_matrix, rotation_angle=rotation_angle)
        Mapper = production_process(ns)
        self.node = n = Mapper()
        n.p = yaml.safe_load((ROOT / 'gicp.yaml').read_text())['go2_gicp']['ros__parameters']
        n.p.update(mode=mode, scan_window=.5, map_frame='map')
        n.p.setdefault('localization_coarse_voxel', .3)
        n.p.setdefault('localization_max_source_points', 6000)
        n.p.setdefault('localization_max_target_points', 20000)
        n.p.setdefault('localization_initial_correspondence', 1.2)
        n.p.setdefault('localization_initial_max_translation', 3.)
        n.p.setdefault('localization_initial_max_rotation_degrees', 90.)
        n.p.setdefault('localization_coarse_iterations', 20)
        n.p.setdefault('localization_fine_iterations', 20)
        n.lock = threading.RLock()
        n.external_backend = False
        n.initialized = True
        n.seeded = False
        n.recovery_episode = False
        n.recovery_worker = None
        n.generation = 0
        n.processed_stamp = None
        n.last_accepted = n.accepted_odom = n.accepted_match_correction = None
        n.correction = self.initial_correction.copy()
        n.last_failure = n.hold_exhausted = ''
        n.map_point_count = len(self.map_points)
        n._registration_points = self.map_points.copy()
        n.points = self.map_points.copy()
        n.localization_points = self.map_points.copy()
        n.static_map_published = False
        n.localization_target = Obj(get=Mock(return_value=Obj(coarse=coarse, fine=fine)))
        n.mapping_store = None
        n.odom_budget = OdometryBudget()
        n.registered_pub = Obj(publish=Mock())
        n.map_pub = Obj(publish=Mock(), get_subscription_count=Mock(return_value=0))
        n.status_pub = Obj(publish=Mock())
        self.log = Obj(info=Mock(), warning=Mock())
        n.get_logger = lambda: self.log
        n.get_clock = lambda: Obj(now=lambda: ClockTime(self.ros_ns))
        n.normal_recovery_confirmation = Mock()
        n.localization_readiness = lambda: (False, 'registration_invalid', '')
        n.begin_recovery = Mock()
        n.queue_recovery = Mock()
        n.cloud_diagnostics = lambda: 'test cloud diagnostics'
        n.last_keyframe = n.last_keyframe_time = n.mapping_start_pose = None
        n.loop_worker = None
        n.capacity_warned = False
        self.set_input(self.ros_ns - 200_000_000, 499.8)
        return n

    def set_input(self, stamp_ns, receipt, latest_empty=False):
        n = self.node
        actual = (stamped_cloud(stamp_ns, self.scan), self.odom.copy(), receipt)
        n.history = [actual]
        n.latest = actual
        if latest_empty:
            empty = (stamped_cloud(self.ros_ns - 50_000_000, []), self.odom.copy(), 499.95)
            n.history.append(empty)
            n.latest = empty

    def assert_not_committed(self):
        n = self.node
        np.testing.assert_array_equal(n.correction, self.initial_correction)
        self.assertIsNone(n.last_accepted)
        self.assertIsNone(n.accepted_odom)
        self.assertIsNone(n.accepted_match_correction)
        self.assertFalse(n.seeded)
        self.assertIsNone(n.odom_budget.stamp)
        n.registered_pub.publish.assert_not_called()
        n.normal_recovery_confirmation.assert_not_called()

    def test_fresh_bootstrap_commits_actual_scan_anchor_on_two_map_origins(self):
        for origin in [(0., 0., 0.), (200., -70., 2.3)]:
            with self.subTest(map_origin=origin):
                n = self.reset(origin)
                n.process()
                self.assertEqual(n.last_failure, '')
                self.assertTrue(n.seeded)
                self.assertEqual(n.last_accepted, 499.8)
                self.assertEqual(n.odom_budget.stamp, 99_800_000_000)
                np.testing.assert_allclose(n.correction @ self.odom, self.solution, atol=1e-12)
                np.testing.assert_array_equal(n.accepted_odom, self.odom)
                n.registered_pub.publish.assert_called_once()
                self.assertEqual(n.registered_pub.publish.call_args.args[0].header.stamp,
                                 n.latest[0].header.stamp)

    def test_source_stamp_and_receipt_deadlines_are_independent(self):
        for cause in ['stamp', 'receipt']:
            with self.subTest(cause=cause):
                n = self.reset()
                # For old receipt, a newer empty packet keeps callbacks live
                # but must not make the old contributing geometry fresh.
                self.set_input(self.ros_ns - (2_200_000_000 if cause == 'stamp' else 200_000_000),
                               499.8 if cause == 'stamp' else 497.8,
                               latest_empty=cause == 'receipt')
                n.process()
                self.assert_not_committed()
                self.assertIn('too old', n.last_failure)

    def test_solver_and_output_preparation_timeouts_both_prevent_commit(self):
        for phase in ['solver', 'serialization']:
            with self.subTest(phase=phase):
                n = self.reset()
                def delay():
                    self.monotonic += 2.1
                    self.ros_ns += 2_100_000_000
                if phase == 'solver':
                    self.align_hook = delay
                else:
                    self.serialize_hook = delay
                n.process()
                self.assert_not_committed()
                self.assertIn('too old', n.last_failure)

    def test_manual_initialpose_generation_supersedes_inflight_match(self):
        n = self.node
        self.align_hook = lambda: setattr(n, 'generation', n.generation + 1)
        n.process()
        self.assertGreater(n.generation, 0)
        self.assert_not_committed()
        self.assertEqual(n.last_failure, '')

    def test_new_empty_scan_cannot_advance_accepted_time_or_motion_anchor(self):
        n = self.node
        self.set_input(self.ros_ns - 400_000_000, 499.6, latest_empty=True)
        n.latest[1][0, 3] += .2
        n.process()
        self.assertEqual(n.last_failure, '')
        self.assertEqual(n.last_accepted, 499.6)
        self.assertAlmostEqual(n.odom_budget.distance, .2)
        self.assertEqual(n.normal_recovery_confirmation.call_args.args[3], 99_600_000_000)

    def test_poor_match_stays_invalid(self):
        self.accepted = False
        self.node.process()
        self.assert_not_committed()
        self.assertIn('overlap/residual rejected', self.node.last_failure)

    def test_tracking_timeout_does_not_renew_or_erase_previous_measurement(self):
        n = self.node
        n.process()
        accepted_time = n.last_accepted
        correction = n.correction.copy()
        accepted_match = n.accepted_match_correction.copy()
        accepted_odom = n.accepted_odom.copy()
        budget_stamp = n.odom_budget.stamp
        n.registered_pub.publish.reset_mock()
        n.normal_recovery_confirmation.reset_mock()
        self.align_calls.clear()
        self.monotonic += .2
        self.ros_ns += 200_000_000
        self.set_input(self.ros_ns - 100_000_000, 500.1)

        def delay():
            self.monotonic += 2.1
            self.ros_ns += 2_100_000_000

        self.align_hook = delay
        n.process()
        self.assertEqual(len(self.align_calls), 1)  # Tracking skips bootstrap.
        self.assertIn('too old', n.last_failure)
        self.assertEqual(n.last_accepted, accepted_time)
        self.assertEqual(n.odom_budget.stamp, budget_stamp)
        np.testing.assert_array_equal(n.correction, correction)
        np.testing.assert_array_equal(n.accepted_match_correction, accepted_match)
        np.testing.assert_array_equal(n.accepted_odom, accepted_odom)
        n.registered_pub.publish.assert_not_called()
        n.normal_recovery_confirmation.assert_not_called()

    def test_mapping_uses_original_target_and_commits_without_localization_cache(self):
        n = self.reset(mode='mapping')
        update = object()
        n.mapping_store = Obj(nearby=Mock(return_value=self.map_points.copy()),
                              prepare=Mock(return_value=update), commit=Mock())
        n.grid = Obj(prepare_observation=Mock(return_value={(1, 2)}), free=set())
        n.process()
        self.assertEqual(n.last_failure, '')
        n.mapping_store.nearby.assert_called_once()
        n.localization_target.get.assert_not_called()
        n.mapping_store.commit.assert_called_once_with(update)
        self.assertEqual(n.grid.free, {(1, 2)})
        self.assertEqual(len(self.align_calls), 1)
        self.assertEqual(self.align_calls[0]['correspondence'], .7)
        self.assertEqual(self.align_calls[0]['min_fitness'], .55)
        self.assertEqual(self.align_calls[0]['max_rmse'], .2)
        self.assertTrue(n.seeded)

    def test_localization_map_serializes_once_even_before_first_subscriber(self):
        n = self.node
        n.publish_map()
        self.assertTrue(n.static_map_published)
        self.serialize.assert_called_once()
        n.map_pub.publish.assert_called_once()
        np.testing.assert_array_equal(n.map_pub.publish.call_args.args[0].points, self.map_points)
        # A new subscriber uses the retained publication; neither repeated
        # timers nor a new initialpose changes the immutable selected map.
        n.map_pub.get_subscription_count.return_value = 1
        for _ in range(3):
            n.publish_map()
        n.generation += 1
        n.publish_map()
        self.serialize.assert_called_once()
        n.map_pub.publish.assert_called_once()

    def test_mapping_keeps_republishing_updated_full_cloud(self):
        n = self.reset(mode='mapping')
        n.publish_map()
        n.points = np.vstack((n.points, [[9., -3., .2]]))
        n.generation += 1
        n.publish_map()
        self.assertFalse(n.static_map_published)
        self.assertEqual(self.serialize.call_count, 2)
        self.assertEqual(n.map_pub.publish.call_count, 2)
        np.testing.assert_array_equal(n.map_pub.publish.call_args.args[0].points, n.points)

    def test_initial_publication_invalidated_during_serialization_retries(self):
        n = self.node
        self.serialize_hook = lambda: setattr(n, 'generation', n.generation + 1)
        n.publish_map()
        n.map_pub.publish.assert_not_called()
        self.assertFalse(n.static_map_published)
        self.serialize_hook = None
        n.publish_map()
        n.map_pub.publish.assert_called_once()
        self.assertTrue(n.static_map_published)


if __name__ == '__main__':
    unittest.main()
