"""Shared fixed-plane policy and sparse component filtering in actual geometry."""
from pathlib import Path
import sys
import unittest

import numpy as np

GO2 = Path(__file__).resolve().parents[1] / 'original_go2/go2_3d'
sys.path.insert(0, str(GO2))
from denoise import small_components
from ground import body_seed_position, floor_and_obstacles, height_obstacle_mask, obstacle_mask
from navigation_grid import current_settings
from point_noise import cell_keys, keep_connected, retained_cells


class ForbiddenTerrain:
    def evaluate(self, *args):
        raise AssertionError('current policy must not inspect terrain')

    def height_at(self, *args):
        raise AssertionError('current policy must not inspect terrain')


class InitialPlanePolicy(unittest.TestCase):
    def test_filter_includes_exact_fifteen_centimeters_in_two_world_frames(self):
        heights = np.array([-.04, -.02, 0., .05, .10, .15, .151, .4, 1.5, 1.51, 9.], dtype='f4')
        xyz = np.column_stack((np.linspace(-.3, .3, len(heights)), np.zeros(len(heights)), heights))
        for name, offset, angle in [('orchard', [0., 0., 0.], 0.), ('building', [100., -80., 2.], .23)]:
            with self.subTest(map=name):
                c, s = np.cos(angle), np.sin(angle)
                rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
                points = xyz @ rotation.T + offset
                normal = rotation[:, 2]
                model = dict(plane=[*normal, -normal @ np.asarray(offset)], _terrain=ForbiddenTerrain())
                before = points.copy()
                floor, obstacle = floor_and_obstacles(points, model)
                np.testing.assert_allclose(floor, points[1:6])
                np.testing.assert_allclose(obstacle, points[6:9])
                np.testing.assert_array_equal(obstacle_mask(points, model),
                    [False]*6 + [True]*5)
                np.testing.assert_array_equal(points, before)
        self.assertEqual(current_settings()['obstacle_min_height'], .15)

    def test_nonfinite_return_and_empty_scans_are_not_obstacles(self):
        points = np.array([[np.nan, 0., .4], [0., np.inf, .4], [0., 0., np.nan]])
        with np.errstate(invalid='ignore'):
            self.assertFalse(obstacle_mask(points, {'plane': [0., 0., 1., 0.]}).any())
        self.assertEqual(obstacle_mask(np.empty((0, 3)), {'plane': [0., 0., 1., 0.]}).shape, (0,))
        organized = np.array([[.15, .151], [np.nan, np.inf]], dtype='f4')
        np.testing.assert_array_equal(height_obstacle_mask(organized), [[False, True], [False, False]])

    def test_explicit_terrain_cannot_override_initial_body_height(self):
        plane = {'plane': [0., 0., 1., -2.]}
        np.testing.assert_allclose(body_seed_position([5., 8., 0.], plane, .37, ForbiddenTerrain()),
                                   [5., 8., 2.37])
        np.testing.assert_allclose(body_seed_position([5., 8., 3.], plane, .37, ForbiddenTerrain()),
                                   [5., 8., 3.])


class PointComponents(unittest.TestCase):
    def test_many_repeated_points_do_not_turn_one_cell_into_a_large_object(self):
        cells = np.array([[x, 0] for x in range(8)] + [[30, 30]]*1000)
        xy = (cells+.5)*.05
        original = xy.copy()
        keep = keep_connected(xy)
        np.testing.assert_array_equal(keep, [True]*8+[False]*1000)
        np.testing.assert_array_equal(xy, original)

    def test_exact_threshold_and_diagonal_connections(self):
        cells = [(i, i) for i in range(8)] + [(30+i, 30) for i in range(7)]
        self.assertEqual(retained_cells(cells), set(cells[:8]))
        self.assertEqual(retained_cells(cells, 8, 4), set())

    def test_chunk_inventory_and_huge_extent_preserve_same_components(self):
        cells = np.array([(10**10+i, -10**10) for i in range(8)] +
                         [(-10**10+i, 10**10) for i in range(7)])
        xy = (cells+.5)*.05
        inventory = set()
        for chunk in np.array_split(xy, 4):
            inventory.update(map(tuple, cell_keys(chunk)))
        retained = retained_cells(inventory)
        keep = keep_connected(xy)
        np.testing.assert_array_equal(keep, [True]*8+[False]*7)
        self.assertEqual(retained, set(map(tuple, cells[:8])))

    def test_sparse_decision_matches_independent_dense_component_filter(self):
        rng = np.random.default_rng(4)
        for minimum in (2, 3, 8, 12):
            for connectivity in (4, 8):
                with self.subTest(minimum=minimum, connectivity=connectivity):
                    mask = rng.random((50, 60)) < .24
                    cells = np.argwhere(mask)
                    removed, _ = small_components(mask, minimum, connectivity)
                    expected = set(map(tuple, np.argwhere(mask & ~removed)))
                    self.assertEqual(retained_cells(cells, minimum, connectivity), expected)

    def test_invalid_inputs_and_no_returns(self):
        self.assertEqual(keep_connected(np.empty((0, 2))).shape, (0,))
        self.assertFalse(keep_connected(np.array([[np.nan, 0.], [0., np.inf]])).any())
        for cells, minimum, connectivity in [([(0., 0.)], 8, 8), ([(0, 0)], 0, 8), ([(0, 0)], 8, 6)]:
            with self.assertRaises(ValueError):
                retained_cells(cells, minimum, connectivity)


if __name__ == '__main__':
    unittest.main()
