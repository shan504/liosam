"""Historical trajectory-reference utility remains separate from runtime policy."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'original_go2/go2_3d'))
sys.path.insert(0, str(ROOT.parent / 'noetic'))
from corridor_reference import trajectory_reference
from ground import floor_and_obstacles
from terrain import load_terrain


def body(x, y, z):
    result = np.eye(4)
    result[:3, 3] = [x, y, z]
    return result


class CorridorReference(unittest.TestCase):
    def test_hill_reference_uses_trajectory_only_and_stays_in_requested_cells(self):
        for offset in [(0., 0.), (80., -120.)]:
            with self.subTest(origin=offset):
                x, y = offset
                poses = [body(x, y, .37), body(x+2., y, .57)]
                cells = {(int(np.floor((x+dx)/.05)), int(np.floor((y+dy)/.05)))
                         for dx in np.arange(.025, 2., .05) for dy in (-.025, .025)}
                model = trajectory_reference(SimpleNamespace(paths=[poses]), .37, .05, cells)
                xy = np.array([[x+.525, y+.025], [x+1.525, y+.025], [x+1., y+.6]])
                z, valid = model.height_at(xy)
                np.testing.assert_array_equal(valid, [True, True, False])
                np.testing.assert_allclose(z[:2], [.0525, .1525], atol=1e-9)
                self.assertEqual(model.statistics['evidence'], 'operator_trajectory_reference')
                # The utility can still read old reference data, but the
                # runtime must classify against its saved initial plane.
                points = np.array([[*xy[0], z[0]], [*xy[0], z[0]+.2],
                                   [*xy[2], 0.], [*xy[1], z[1]]])
                with patch.object(model, 'evaluate', side_effect=AssertionError('terrain used at runtime')):
                    floor, obstacles = floor_and_obstacles(points,
                        {'plane': [0., 0., 1., 0.], '_terrain': model})
                np.testing.assert_allclose(floor, points[[0, 2]])
                np.testing.assert_allclose(obstacles, points[[1, 3]])

    def test_curved_route_uses_nearby_recorded_segments_and_does_not_bridge_gap(self):
        # U-shaped body trace; opposite sides have different heights. Road
        # reference may not shortcut diagonally from first to final body pose.
        path = [body(0., 0., .37), body(3., 0., .67),
                body(3., 3., .97), body(0., 3., 1.27)]
        query = np.array([[1.025, .025], [1.025, 3.025]])
        cells = {tuple(v) for v in np.floor(query/.05).astype(int)}
        model = trajectory_reference(SimpleNamespace(paths=[path]), .37, .05, cells)
        z, valid = model.height_at(query)
        np.testing.assert_array_equal(valid, True)
        np.testing.assert_allclose(z, [.1025, .7975], atol=1e-9)
        _, mid = model.height_at([[1.025, 1.525]])
        self.assertFalse(mid[0])

    def test_crossing_levels_remain_unsupported_and_reference_roundtrips(self):
        paths = [[body(-1., 0., .37), body(1., 0., .37)],
                 [body(0., -1., 1.37), body(0., 1., 1.37)]]
        cells = {(0, 0), (10, 0)}
        model = trajectory_reference(SimpleNamespace(paths=paths), .37, .05, cells)
        _, valid = model.height_at([[.025, .025], [.525, .025]])
        np.testing.assert_array_equal(valid, [False, True])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'road_reference.npz'
            descriptor = model.save(path)
            restored = load_terrain(Path(tmp), {'terrain_model': descriptor})
            np.testing.assert_array_equal(restored.cells, model.cells)
            np.testing.assert_allclose(restored.planes, model.planes)
            self.assertEqual(descriptor['evidence'], 'operator_trajectory_reference')

    def test_start_and_rear_foot_support_is_reference_only(self):
        start = body(0., 0., .37)
        path = [body(.2, 0., .39), body(2., 0., .57)]
        original_free = {(x, y) for x in range(4, 60) for y in range(-10, 10)}
        before = original_free.copy()
        model = trajectory_reference(SimpleNamespace(paths=[path]), .37, .05,
                                     original_free, start_pose=start)
        self.assertEqual(original_free, before)
        query = np.array([[-.375, -.175], [1.025, .175], [0., .575]])
        z, supported = model.height_at(query)
        np.testing.assert_array_equal(supported, [True, True, False])
        self.assertAlmostEqual(z[0], 0.)
        points = np.array([[*query[0], 0.], [*query[0], .20], [*query[2], 0.]])
        with patch.object(model, 'evaluate', side_effect=AssertionError('terrain used at runtime')):
            floor, obstacles = floor_and_obstacles(points,
                {'plane': [0., 0., 1., 0.], '_terrain': model})
        np.testing.assert_allclose(floor, points[[0, 2]])
        np.testing.assert_allclose(obstacles, points[1:2])


if __name__ == '__main__':
    unittest.main()
