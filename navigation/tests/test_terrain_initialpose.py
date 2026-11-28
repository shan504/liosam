"""Exercise real initial-pose handling without ROS or robot movement."""
import ast
from collections import deque
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
import unittest

import numpy as np

GO2 = Path(__file__).resolve().parents[1] / 'original_go2/go2_3d'
sys.path.insert(0, str(GO2))
from geometry import pose_matrix, quaternion_from_matrix
from ground import body_seed_position
from map_start import start_pose_from_metadata
from terrain import TerrainModel


GROUND = {'plane': [0., 0., 1., 0.]}
BODY_HEIGHT = .37


def terrain_at(xy, ground_z, slope=(.12, -.04)):
    """A fitted observed local cell; fitting itself is covered in test_terrain."""
    xy = np.asarray(xy, dtype=float)
    cell = np.floor(xy / .4).astype(np.int64)
    center = (cell + .5) * .4
    z_center = ground_z - np.dot(slope, xy-center)
    return TerrainModel([cell], [[*slope, z_center]])


def hint(position=(0., 0., 0.), quaternion=(0., 0., 0., 1.)):
    return NS(header=NS(frame_id='map'), pose=NS(pose=NS(
        position=NS(**dict(zip(('x', 'y', 'z'), position))),
        orientation=NS(**dict(zip(('x', 'y', 'z', 'w'), quaternion))))))


class TerrainInitialHeight(unittest.TestCase):
    def test_uphill_and_downhill_use_local_ground_in_two_map_origins(self):
        for xy, ground_z in [((7.1, 30.76), 1.3), ((-140.23, 72.15), -.85)]:
            with self.subTest(xy=xy, ground_z=ground_z):
                model = terrain_at(xy, ground_z)
                position = np.array([*xy, 0.])
                result = body_seed_position(position, GROUND, BODY_HEIGHT, terrain=model)
                np.testing.assert_allclose(result, [*xy, ground_z+BODY_HEIGHT])
                np.testing.assert_array_equal(position, [*xy, 0.])

    def test_unknown_ground_refuses_zero_height_hint(self):
        model = terrain_at((3., 4.), .7)
        with self.assertRaisesRegex(ValueError, 'no observed local terrain support'):
            body_seed_position([30., 40., 0.], GROUND, BODY_HEIGHT, terrain=model)

    def test_explicit_nonzero_body_z_is_never_lifted(self):
        model = terrain_at((3., 4.), .7)
        for position in ([3., 4., .45], [300., -400., -.25]):
            with self.subTest(position=position):
                result = body_seed_position(position, GROUND, BODY_HEIGHT, terrain=model)
                np.testing.assert_array_equal(result, position)

    def test_legacy_plane_and_missing_height_contract_are_unchanged(self):
        normal = np.array([.08, -.04, 1.])
        normal /= np.linalg.norm(normal)
        plane = {'plane': [*normal, -.6]}
        position = [3., -2., 0.]
        expected_z = (BODY_HEIGHT+.6-normal[:2] @ position[:2]) / normal[2]
        result = body_seed_position(position, plane, BODY_HEIGHT)
        np.testing.assert_allclose(result, [3., -2., expected_z])
        model = terrain_at((3., -2.), 1.2)
        np.testing.assert_array_equal(body_seed_position(position, terrain=model), position)
        np.testing.assert_array_equal(body_seed_position(position, plane, terrain=model), position)


class MapperInitialPose(unittest.TestCase):
    def setUp(self):
        # Compile production method bodies with only ROS IO substituted. This
        # exercises the auto-start branch as well as the actual hint handler.
        path = GO2/'mapper.py'
        mapper = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == 'Mapper')
        namespace = dict(Node=object, np=np, body_seed_position=body_seed_position,
                         pose_matrix=pose_matrix, quaternion_from_matrix=quaternion_from_matrix,
                         time=NS(monotonic=lambda: 12.), PoseWithCovarianceStamped=hint,
                         INITIAL_POSE_WAIT_SECONDS=10.)
        exec(compile(ast.Module(body=[mapper], type_ignores=[]), str(path), 'exec'), namespace)
        self.node = namespace['Mapper'].__new__(namespace['Mapper'])
        node = self.node
        node.p = dict(mode='localization', auto_initialize=True, initial_from_map_start=True)
        node.lock = threading.RLock()
        node.map_metadata = dict(ground_model=GROUND, body_height_above_ground=BODY_HEIGHT)
        node.terrain = terrain_at((7.1, 30.76), 1.3)
        node.pending_initial_pose = None
        node.initial_hint_used = False
        node.initialized = node.seeded = True
        node.generation = node.recovery_serial = 0
        node.odom_budget = NS(reset=lambda *_: None)
        node.apply_pending_initial_pose = lambda: False
        node.recovery_status = lambda *_: None
        self.errors, self.reports = [], []
        node.get_logger = lambda: NS(error=self.errors.append, warning=self.errors.append)
        node.report = self.reports.append

    def test_manual_2d_hint_lifts_locally_and_invalidates_old_match(self):
        node = self.node
        node.initial_pose(hint((7.1, 30.76, 0.)))
        self.assertEqual(self.errors, [])
        np.testing.assert_allclose(node.pending_initial_pose[0][:3, 3], [7.1, 30.76, 1.67])
        self.assertFalse(node.initialized)
        self.assertFalse(node.seeded)
        self.assertTrue(node.initial_hint_used)
        self.assertEqual(node.generation, 1)
        self.assertIsNone(node.last_accepted)

    def test_unknown_hint_does_not_replace_previous_pose_or_trust(self):
        node = self.node
        previous = (np.eye(4), 9.)
        node.pending_initial_pose = previous
        node.initial_pose(hint((100., 200., 0.)))
        self.assertIn('no observed local terrain support', self.errors[0])
        self.assertIs(node.pending_initial_pose, previous)
        self.assertTrue(node.initialized)
        self.assertEqual(node.generation, 0)

    def test_saved_full_start_pose_including_zero_z_bypasses_2d_lift(self):
        node = self.node
        # A retained exact body pose with z=0 is valid in its own map frame.
        # It must not be mistaken for a browser's ground-height sentinel.
        quaternion = np.array([.04, -.03, .21, .97])
        quaternion /= np.linalg.norm(quaternion)
        exact = pose_matrix([7.1, 30.76, 0.], quaternion)
        node.map_metadata['map_start_pose'] = exact.tolist()
        node.saved_start_pose = start_pose_from_metadata(node.map_metadata)
        node.pending_clouds = deque()
        node.latest = object()  # A fresh aligned packet exists; no new TF read.
        node.process_pending_clouds()
        self.assertEqual(self.errors, [])
        np.testing.assert_allclose(node.pending_initial_pose[0], exact, atol=1e-12)
        np.testing.assert_array_equal(node.saved_start_pose, exact)
        self.assertFalse(node.initialized)


if __name__ == '__main__':
    unittest.main()
