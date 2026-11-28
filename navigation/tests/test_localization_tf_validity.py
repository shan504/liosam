"""Exercise production TF publication without ROS, including loss of GICP."""
import ast
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
import unittest

import numpy as np

GO2 = Path(__file__).resolve().parents[1] / 'original_go2/go2_3d'
sys.path.insert(0, str(GO2))
from geometry import pose_matrix, quaternion_from_matrix, planar_pose


class Time:
    def __init__(self, nanoseconds=0):
        self.nanoseconds = nanoseconds

    @classmethod
    def from_msg(cls, msg):
        return cls(msg.sec*10**9 + msg.nanosec)

    def __sub__(self, other):
        return Time(self.nanoseconds-other.nanoseconds)


def transform_message(matrix=None, stamp=100.0):
    matrix = np.eye(4) if matrix is None else matrix
    q = quaternion_from_matrix(matrix)
    return NS(header=NS(stamp=NS(sec=int(stamp), nanosec=round((stamp-int(stamp))*1e9)),
                        frame_id='odom'), child_frame_id='base_link',
              transform=NS(translation=NS(**dict(zip('xyz', matrix[:3, 3]))),
                           rotation=NS(**dict(zip('xyzw', q)))))


def matrix_from_message(transform):
    t, q = transform.translation, transform.rotation
    return pose_matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])


class LocalizationTfValidity(unittest.TestCase):
    def setUp(self):
        module = ast.parse((GO2/'mapper.py').read_text())
        mapper = next(x for x in module.body if isinstance(x, ast.ClassDef) and x.name == 'Mapper')
        namespace = dict(Node=object, np=np, Time=Time, time=NS(monotonic=lambda: 100.),
                         TransformStamped=transform_message, TransformException=ValueError,
                         transform_matrix=matrix_from_message, planar_pose=planar_pose,
                         quaternion_from_matrix=quaternion_from_matrix, NAVIGATION_TF_PERIOD=.02)
        exec(compile(ast.Module(body=[mapper], type_ignores=[]), str(GO2/'mapper.py'), 'exec'), namespace)
        self.node = namespace['Mapper'].__new__(namespace['Mapper'])
        node = self.node
        node.lock = threading.RLock()
        node.external_backend = False
        node.p = dict(mode='localization', validity_timeout=1., localization_odom_timeout=.3)
        node.seeded, node.last_accepted, node.last_tf_stamp = True, 50., None
        node.correction = pose_matrix([19., -8., 1.3], [0., 0., .2, .98])
        node.get_clock = lambda: NS(now=lambda: Time(100_000_000_000))
        self.odom = transform_message(pose_matrix([2., 4., .4], [.06, -.03, .1, .99]), 99.95)
        node.buffer = NS(lookup_transform=lambda *_: self.odom)
        self.published = []
        node.broadcaster = NS(sendTransform=self.published.append)
        node.localization_readiness = lambda: (False, 'registration_timeout', 'residual rejected')

    def test_invalid_match_keeps_native_projection_current_without_renewing_map(self):
        node = self.node
        before = node.correction.copy()
        node.publish_correction()
        self.assertEqual(len(self.published), 1)
        edges = self.published[0]
        self.assertEqual([(e.header.frame_id, e.child_frame_id) for e in edges],
                         [('base_link', 'base_footprint')])
        self.assertEqual(edges[0].header.stamp, self.odom.header.stamp)
        self.assertEqual(node.last_accepted, 50.)
        np.testing.assert_array_equal(node.correction, before)
        body = before @ matrix_from_message(self.odom.transform)
        np.testing.assert_allclose(body @ matrix_from_message(edges[0].transform), planar_pose(body), atol=1e-12)
        self.assertFalse(node.localization_readiness()[0])

    def test_fresh_match_publishes_both_edges(self):
        self.node.localization_readiness = lambda: (True, 'tracking', '')
        self.node.publish_correction()
        self.assertEqual([(e.header.frame_id, e.child_frame_id) for e in self.published[0]],
                         [('map', 'odom'), ('base_link', 'base_footprint')])

    def test_no_projection_with_stale_odometry_or_missing_accepted_pose(self):
        for age, seeded, accepted in [(0.31, True, 50.), (.05, False, 50.), (.05, True, None)]:
            with self.subTest(age=age, seeded=seeded, accepted=accepted):
                self.odom = transform_message(stamp=100.-age)
                self.node.seeded, self.node.last_accepted = seeded, accepted
                self.node.last_tf_stamp = None
                self.node.publish_correction()
                self.assertEqual(self.published, [])

    def test_mapping_timeout_still_stops_both_edges(self):
        self.node.p['mode'] = 'mapping'
        self.node.publish_correction()
        self.assertEqual(self.published, [])

    def test_duplicate_native_sample_does_not_restamp_projection(self):
        self.node.publish_correction()
        self.node.publish_correction()
        self.assertEqual(len(self.published), 1)


class MatchDiagnostics(unittest.TestCase):
    def test_residual_failure_retains_recoverable_type_and_reports_limits(self):
        module = ast.parse((GO2/'mapper.py').read_text())
        definitions = [x for x in module.body if getattr(x, 'name', '') in ('ScanMatchUnavailable', 'require_match')]
        namespace = {}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), 'mapper_diagnostics', 'exec'), namespace)
        result = NS(accepted=False, reason='overlap/residual rejected', fitness=.99, rmse=.204)
        with self.assertRaisesRegex(namespace['ScanMatchUnavailable'], 'rmse=0.204; limits rmse<=0.200m'):
            namespace['require_match'](result, 'Fine GICP ', diagnostics='limits rmse<=0.200m')
        result.accepted = True
        namespace['require_match'](result)


if __name__ == '__main__':
    unittest.main()
