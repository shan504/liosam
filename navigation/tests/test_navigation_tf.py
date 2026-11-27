"""Run the production Mapper TF methods with a deterministic ROS message/cache shim.

These tests cover geometry, freshness, callback registration and gating. They do
not measure executor scheduling or physical robot response (ROS is not installed
in the test workspace). GICP itself is outside the TF hot path under test.
"""
import ast
import copy
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as Obj
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]/'original_go2/go2_3d'
sys.path.insert(0, str(ROOT))
from geometry import pose_matrix, planar_pose, quaternion_from_matrix
from localization_hold import OdometryBudget


class ClockTime:
    def __init__(self, ns=0):
        self.nanoseconds = ns

    @classmethod
    def from_msg(cls, stamp):
        return cls(stamp.sec*10**9+stamp.nanosec)

    def __sub__(self, other):
        return ClockTime(self.nanoseconds-other.nanoseconds)


def header(ns, frame='odom'):
    sec, nanosec = divmod(ns, 10**9)
    return Obj(stamp=Obj(sec=sec, nanosec=nanosec), frame_id=frame)


def transform_message():
    return Obj(header=header(0), child_frame_id='', transform=Obj(
        translation=Obj(x=0., y=0., z=0.), rotation=Obj(x=0., y=0., z=0., w=1.)))


def odometry(ns, matrix):
    xyz, q = matrix[:3, 3], quaternion_from_matrix(matrix)
    return Obj(header=header(ns), child_frame_id='base_link', pose=Obj(pose=Obj(
        position=Obj(**dict(zip('xyz', xyz))), orientation=Obj(**dict(zip('xyzw', q))))))


class Cache:
    def __init__(self):
        self.latest = None

    def set_transform(self, value, authority):
        if (self.latest is None or ClockTime.from_msg(value.header.stamp).nanoseconds >
                ClockTime.from_msg(self.latest.header.stamp).nanoseconds):
            self.latest = value

    def lookup_transform(self, parent, child, stamp):
        assert (parent, child) == ('odom', 'base_link')
        if self.latest is None:
            raise LookupError('No odometry')
        return self.latest


def mapper_methods():
    # Compile the actual methods unchanged, without importing Open3D/ROS or
    # constructing a map. Fail if a method disappears; never copy its logic.
    tree = ast.parse((ROOT/'mapper.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Mapper')
    names = {'configure_navigation_tf', 'receive_navigation_odometry',
             'publish_correction', 'localization_readiness'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in cls.body} == names
    cls.bases = []
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'transform_matrix']
    constant = next(n for n in tree.body if isinstance(n, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == 'NAVIGATION_TF_PERIOD' for t in n.targets))
    import time
    ns = dict(np=np, time=time, Time=ClockTime, TransformStamped=transform_message,
              TransformException=LookupError, pose_matrix=pose_matrix, planar_pose=planar_pose,
              quaternion_from_matrix=quaternion_from_matrix, MutuallyExclusiveCallbackGroup=object,
              Odometry=Obj, QoSProfile=lambda **kw: Obj(**kw),
              ReliabilityPolicy=Obj(BEST_EFFORT='best_effort'))
    exec(compile(ast.Module(body=[constant, *functions, cls], type_ignores=[]),
                 str(ROOT/'mapper.py'), 'exec'), ns)
    return ns['Mapper'], ns['transform_matrix']


Mapper, matrix = mapper_methods()


class NavigationTF(unittest.TestCase):
    def setUp(self):
        self.reset_node()
        self.time_patch = patch('time.monotonic', return_value=500.)
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)

    def reset_node(self):
        self.now = 100_000_000_000
        self.node = Mapper()
        n = self.node
        n.lock = threading.RLock()
        n.buffer, n.broadcaster = Cache(), Obj(sendTransform=Mock())
        n.external_backend = False
        n.p = dict(mode='localization', validity_timeout=1., localization_odom_timeout=.3,
                   localization_hold_timeout=3., localization_hold_distance=1.,
                   localization_hold_rotation_degrees=60.)
        n.seeded = n.initialized = True
        n.pending_initial_pose = None
        n.last_failure = n.hold_exhausted = ''
        n.last_accepted = 499.5
        n.latest = (Obj(header=header(self.now-200_000_000)), np.eye(4), 499.8)
        n.last_tf_stamp = None
        n.correction = np.eye(4)
        n.odom_budget = OdometryBudget()
        n.get_clock = lambda: Obj(now=lambda: ClockTime(self.now))

    def prime(self, body):
        self.node.odom_budget.reset(body, self.now-100_000_000)

    def test_callback_group_not_shared_with_cloud_or_map_work(self):
        n = self.node
        n.create_subscription, n.create_timer = Mock(), Mock()
        n.configure_navigation_tf()
        self.assertIs(n.create_subscription.call_args.kwargs['callback_group'], n.navigation_tf_group)
        self.assertIs(n.create_timer.call_args.kwargs['callback_group'], n.navigation_tf_group)
        self.assertEqual(n.create_subscription.call_args.args[3].depth, 1)
        # Inject through the registered callback while the TF listener has no
        # native edge yet. Delivery of the second /tf message is unnecessary.
        self.prime(np.eye(4))
        n.create_subscription.call_args.args[2](odometry(self.now-10_000_000, np.eye(4)))
        n.broadcaster.sendTransform.assert_called_once()

    def test_same_stamped_edges_project_tilted_body_on_two_map_origins(self):
        for origin, heading in [((0., 0., 0.), math.pi/2), ((7., -2., .4), -math.pi)]:
            with self.subTest(map_origin=origin, heading=heading):
                n = self.node
                n.correction = pose_matrix(origin, [0., 0., math.sin(heading/2), math.cos(heading/2)])
                body = pose_matrix([1., .3, .32], [.1, -.15, .2, .95])
                self.prime(body)
                n.last_tf_stamp = None
                msg = odometry(self.now-10_000_000, body)
                n.receive_navigation_odometry(msg)
                correction, footprint = n.broadcaster.sendTransform.call_args.args[0]
                self.assertEqual((correction.header.frame_id, correction.child_frame_id), ('map', 'odom'))
                self.assertEqual((footprint.header.frame_id, footprint.child_frame_id), ('base_link', 'base_footprint'))
                self.assertEqual(correction.header.stamp, msg.header.stamp)
                self.assertEqual(footprint.header.stamp, msg.header.stamp)
                np.testing.assert_allclose(matrix(correction.transform)@body@matrix(footprint.transform),
                                           planar_pose(n.correction@body), atol=1e-12)

    def test_drops_bad_frames_poses_old_and_future_odometry(self):
        self.prime(np.eye(4))
        good = odometry(self.now-10_000_000, np.eye(4))
        cases = [odometry(self.now-400_000_000, np.eye(4)), odometry(self.now+1, np.eye(4))]
        for field in ['frame', 'child', 'position', 'quaternion']:
            bad = copy.deepcopy(good)
            if field == 'frame': bad.header.frame_id = 'map'
            if field == 'child': bad.child_frame_id = 'lidar'
            if field == 'position': bad.pose.pose.position.x = float('nan')
            if field == 'quaternion': bad.pose.pose.orientation.w = float('nan')
            cases.append(bad)
        for msg in cases:
            self.node.receive_navigation_odometry(msg)
        self.assertIsNone(self.node.buffer.latest)
        self.node.broadcaster.sendTransform.assert_not_called()

    def test_invalid_localization_and_exhausted_budget_cannot_renew_tf(self):
        for fault in ['no_match', 'match_timeout', 'cloud_timeout', 'budget']:
            with self.subTest(fault=fault):
                self.reset_node()
                n = self.node
                self.prime(np.eye(4))
                body = np.eye(4)
                if fault == 'no_match': n.last_accepted = None
                if fault == 'match_timeout': n.last_accepted = 496.
                if fault == 'cloud_timeout': n.latest = (*n.latest[:2], 498.)
                if fault == 'budget': body[0, 3] = 2.
                n.receive_navigation_odometry(odometry(self.now-10_000_000, body))
                n.broadcaster.sendTransform.assert_not_called()

    def test_decimates_repeats_and_never_restamps_old_measurements(self):
        n = self.node
        self.prime(np.eye(4))
        n.receive_navigation_odometry(odometry(self.now-50_000_000, np.eye(4)))
        n.receive_navigation_odometry(odometry(self.now-40_000_000, np.eye(4)))
        n.receive_navigation_odometry(odometry(self.now-60_000_000, np.eye(4)))
        n.publish_correction()
        self.assertEqual(n.broadcaster.sendTransform.call_count, 1)
        n.receive_navigation_odometry(odometry(self.now-20_000_000, np.eye(4)))
        self.assertEqual(n.broadcaster.sendTransform.call_count, 2)
        # A live timer alone cannot make stopped odometry fresh.
        self.now += 400_000_000
        n.publish_correction()
        self.assertEqual(n.broadcaster.sendTransform.call_count, 2)


if __name__ == '__main__':
    unittest.main()
