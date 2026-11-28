"""Production tracking continuity, bounded planar hold and hard failure checks.

Solvers/ROS transport are mocked; these are temporal/geometry regressions, not
claims about field GICP accuracy or robot executor timing.
"""
import math
from types import SimpleNamespace as NS
import unittest

import numpy as np

import test_localization_deadline as process_harness
from geometry import quaternion_from_matrix
from localization_hold import OdometryBudget


def pose(x=0., y=0., z=0., roll=0., pitch=0., yaw=0.):
    r, p, a = np.radians([roll, pitch, yaw])
    cr, sr, cp, sp, ca, sa = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(a), np.sin(a)
    matrix = np.eye(4)
    matrix[:3, :3] = [[ca*cp, ca*sp*sr-sa*cr, ca*sp*cr+sa*sr],
                      [sa*cp, sa*sp*sr+ca*cr, sa*sp*cr-ca*sr],
                      [-sp, cp*sr, cp*cr]]
    matrix[:3, 3] = [x, y, z]
    return matrix


class PlanarBudget(unittest.TestCase):
    def test_gait_bounce_roll_pitch_do_not_consume_navigation_progress(self):
        for origin in [(0., 0.), (100., -70.)]:
            with self.subTest(origin=origin):
                budget = OdometryBudget()
                budget.reset(pose(*origin, z=.3), 1)
                for i in range(1, 51):
                    budget.observe(pose(*origin, z=.3+.04*math.sin(i),
                                        roll=8*math.sin(i), pitch=10*math.cos(i)), i+1)
                self.assertAlmostEqual(budget.distance, 0.)
                self.assertAlmostEqual(budget.angle, 0.)
                self.assertEqual(budget.exceeded(1.5, math.pi), '')

    def test_reversals_wrapped_yaw_and_duplicate_samples_never_refill_budget(self):
        budget = OdometryBudget()
        budget.reset(pose(yaw=179), 1)
        budget.observe(pose(x=.8, yaw=-179), 2)
        budget.observe(pose(x=0., yaw=179), 3)
        self.assertAlmostEqual(budget.distance, 1.6)
        self.assertAlmostEqual(math.degrees(budget.angle), 4.)
        self.assertEqual(budget.exceeded(1.5, math.pi), 'odometry_distance_limit')
        self.assertFalse(budget.observe(pose(), 2))
        budget.observe(pose(x=1., yaw=0), 3)
        self.assertAlmostEqual(budget.distance, 1.6)
        budget.reset(pose(), 1)
        for i in range(1, 5):
            budget.observe(pose(yaw=50*i), i+1)
        self.assertAlmostEqual(math.degrees(budget.angle), 200.)
        self.assertEqual(budget.exceeded(1.5, math.pi), 'odometry_angle_limit')

    def test_invalid_pose_and_undefined_heading_never_corrupt_anchor(self):
        bads = [np.full((4, 4), np.nan), np.eye(3), pose(pitch=90)]
        bad_rotation = np.eye(4); bad_rotation[0, 0] = 2.; bads.append(bad_rotation)
        for bad in bads:
            with self.subTest(pose=bad):
                budget = OdometryBudget()
                budget.reset(pose(), 1)
                with self.assertRaises(ValueError):
                    budget.observe(bad, 2)
                self.assertEqual(budget.stamp, 1)
                self.assertEqual(budget.distance, 0.)
                with self.assertRaises(ValueError):
                    budget.reset(bad, 2)
                self.assertEqual(budget.stamp, 1)


class TrackingContinuity(unittest.TestCase):
    def make_tracker(self, origin=(0., 0., 0.)):
        harness = process_harness.LocalizationDeadline(methodName='runTest')
        node = harness.reset(origin)
        node.process()
        self.assertTrue(node.seeded)
        node.pending_initial_pose = None
        del node.localization_readiness  # Exercise the actual readiness method.
        harness.odom_stamp_override = None

        def lookup(*_):
            stamp = (harness.ros_ns-10_000_000 if harness.odom_stamp_override is None
                     else harness.odom_stamp_override)
            q = quaternion_from_matrix(harness.odom)
            return NS(header=NS(stamp=process_harness.ClockTime(stamp).to_msg()),
                      transform=NS(translation=NS(**dict(zip('xyz', harness.odom[:3, 3]))),
                                   rotation=NS(**dict(zip('xyzw', q)))))

        node.buffer = NS(lookup_transform=lookup)
        return harness, node

    def advance(self, harness, seconds):
        harness.monotonic += seconds
        harness.ros_ns += round(seconds*1e9)
        harness.set_input(harness.ros_ns-20_000_000, harness.monotonic-.02)

    def test_finite_candidate_outliers_keep_old_anchor_then_exhaust_five_second_hold(self):
        for origin in [(0., 0., 0.), (200., -70., 2.3)]:
            with self.subTest(map_origin=origin):
                harness, node = self.make_tracker(origin)
                accepted, correction = node.last_accepted, node.correction.copy()
                reference = node.accepted_match_correction.copy()
                anchor_stamp = node.odom_budget.stamp
                node.normal_recovery_confirmation.reset_mock()
                harness.accepted, harness.result_reason = False, 'innovation rejected'
                harness.solution[:3, 3] += [4., 0., 0.]
                elapsed = 0.
                for when in [.6, 1.4, 2.5, 3.6, 4.7, 5.1]:
                    self.advance(harness, when-elapsed); elapsed = when
                    node.process()
                    valid, state, _ = node.localization_readiness()
                    self.assertEqual(valid, when < 5.)
                    if valid:
                        self.assertEqual(state, 'odometry_hold')
                    else:
                        self.assertEqual(state, 'registration_timeout')
                    self.assertEqual(node.last_accepted, accepted)
                    np.testing.assert_array_equal(node.correction, correction)
                    np.testing.assert_array_equal(node.accepted_match_correction, reference)
                self.assertGreater(node.odom_budget.stamp, anchor_stamp)  # Real odom tracked, not reset.
                node.normal_recovery_confirmation.assert_not_called()
                node.begin_recovery.assert_called()

    def test_stationary_outlier_is_discarded_but_nonfinite_solution_invalidates_immediately(self):
        harness, node = self.make_tracker()
        accepted, correction = node.last_accepted, node.correction.copy()
        self.advance(harness, .2)
        harness.solution[:3, 3] += [.5, 0., 0.]
        node.process()
        self.assertIn('jump while odometry stationary', node.last_failure)
        self.assertEqual(node.last_accepted, accepted)
        self.assertTrue(node.localization_readiness()[0])
        np.testing.assert_array_equal(node.correction, correction)
        self.advance(harness, .2)
        harness.accepted, harness.result_reason = False, 'nonfinite transform'
        harness.solution[0, 0] = np.nan
        node.process()
        self.assertIsNone(node.last_accepted)
        self.assertFalse(node.localization_readiness()[0])
        np.testing.assert_array_equal(node.correction, correction)

    def test_slow_result_keeps_original_source_anchor_requires_live_stream_and_uses_own_rmse(self):
        harness, node = self.make_tracker()
        self.advance(harness, .3)
        source_received = node.latest[2]
        source_stamp = process_harness.ClockTime.from_msg(node.latest[0].header.stamp).nanoseconds
        harness.align_hook = lambda: self.advance(harness, 1.2)
        node.process()
        self.assertEqual(node.last_failure, '')
        self.assertEqual(node.last_accepted, source_received)
        self.assertEqual(node.odom_budget.stamp, source_stamp)
        self.assertEqual(harness.align_calls[-1]['max_rmse'], .25)
        self.assertEqual(node.p['max_rmse'], .20)  # Mapping threshold unaffected.
        self.assertTrue(node.localization_readiness()[0])
        accepted = node.last_accepted
        self.advance(harness, .1)

        def no_more_clouds():
            harness.monotonic += 1.2
            harness.ros_ns += 1_200_000_000

        harness.align_hook = no_more_clouds
        node.process()
        self.assertIn('Live cloud stream too old', node.last_failure)
        self.assertEqual(node.last_accepted, accepted)
        self.assertFalse(node.localization_readiness()[0])

    def test_stale_sensor_and_regressed_odometry_still_stop_during_wider_hold(self):
        for fault in ['cloud_receipt', 'cloud_stamp', 'odom_stale', 'clock_regressed', 'invalid_heading']:
            with self.subTest(fault=fault):
                harness, node = self.make_tracker()
                self.advance(harness, 2.)
                if fault == 'cloud_receipt':
                    node.latest = (*node.latest[:2], harness.monotonic-1.1)
                elif fault == 'cloud_stamp':
                    node.latest[0].header.stamp = process_harness.ClockTime(harness.ros_ns-1_100_000_000).to_msg()
                elif fault == 'odom_stale':
                    harness.odom_stamp_override = harness.ros_ns-400_000_000
                elif fault == 'clock_regressed':
                    node.odom_budget.stamp = harness.ros_ns
                else:
                    harness.odom = pose(pitch=90)
                accepted = node.last_accepted
                self.assertFalse(node.localization_readiness()[0])
                self.assertEqual(node.last_accepted, accepted)

    def test_mapping_result_deadline_remains_one_second(self):
        harness = process_harness.LocalizationDeadline(methodName='runTest')
        node = harness.reset(mode='mapping')
        node.mapping_store = NS(nearby=lambda *_: harness.map_points.copy())
        harness.align_hook = lambda: self.advance(harness, 1.2)
        node.process()
        self.assertIn('limit=1.000s', node.last_failure)
        self.assertIsNone(node.last_accepted)

    def test_invalid_planar_anchor_prevents_partial_pose_commit(self):
        harness = process_harness.LocalizationDeadline(methodName='runTest')
        node = harness.reset()
        original = node.correction.copy()
        harness.odom = pose(pitch=90)
        harness.set_input(harness.ros_ns-20_000_000, harness.monotonic-.02)
        node.process()
        self.assertIn('undefined planar heading', node.last_failure)
        self.assertIsNone(node.last_accepted)
        self.assertFalse(node.seeded)
        self.assertIsNone(node.accepted_odom)
        self.assertIsNone(node.accepted_match_correction)
        self.assertIsNone(node.odom_budget.stamp)
        np.testing.assert_array_equal(node.correction, original)
        node.registered_pub.publish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
