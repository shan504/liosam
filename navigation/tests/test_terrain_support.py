"""A coarse terrain plane never proves unobserved fine navigation cells free."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_go2_map import export_grid, observed_free_support
from denoise import clean_directory
from geometry import ObservedGrid
from grid_io import read_pgm
from terrain import build_terrain


def fixture(offset):
    # A 0.4m terrain cell is observed only in its lower-left corner.
    xy = np.array([[x, y] for x in (.025, .075, .125, .175)
                   for y in (.025, .075, .125, .175)]) + offset
    floor = np.column_stack((xy, np.zeros(len(xy))))
    hole = np.asarray(offset) + [.325, .325]
    obstacle_xy = np.array([[x, y] for x in (1.025, 1.075, 1.125)
                            for y in (.025, .075, .125)]) + offset
    obstacles = np.column_stack((obstacle_xy, np.full(len(obstacle_xy), .30)))
    terrain = build_terrain([floor, obstacles], np.array([[*(np.asarray(offset)+.1), 0.]]))
    return floor, hole, obstacles, terrain


def labels(directory, xy):
    nav = yaml.safe_load((directory / 'nav.yaml').read_text())
    image = read_pgm(directory / nav['image'])
    try:
        col, row = np.floor((np.asarray(xy) - nav['origin'][:2]) /
                            nav['resolution']).astype(int).T
        # An area excluded from the observed map extent remains unknown too;
        # never let NumPy negative indexing alias an outside query to a pixel.
        result = np.full(len(col), 205, dtype=np.uint8)
        inside = ((col >= 0) & (col < image.shape[1]) &
                  (row >= 0) & (row < image.shape[0]))
        result[inside] = image[image.shape[0]-1-row[inside], col[inside]]
        return result
    finally:
        image._mmap.close()


class FineTerrainSupport(unittest.TestCase):
    def test_ray_through_partial_terrain_cell_does_not_prove_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, offset in [('near_lane', (0., 0.)), ('other_lane', (8., -4.))]:
                with self.subTest(map=name):
                    floor, hole, obstacles, terrain = fixture(offset)
                    # This was the bug: the terrain plane legitimately applies
                    # to the cell but no ground endpoint exists at the ray pixel.
                    self.assertTrue(terrain.height_at(hole[None, :])[1][0])
                    grid = ObservedGrid(.05)
                    grid.free.add(tuple(np.floor(hole/.05).astype(int)))
                    output = Path(tmp) / name
                    export_grid([floor, np.vstack((floor, obstacles))], grid, output,
                                dict(obstacle_min_height=.10, obstacle_max_height=1.5),
                                dict(plane=[0., 0., 1., 0.], _terrain=terrain))
                    self.assertFalse(observed_free_support(grid, hole[None, :])[0])
                    np.testing.assert_array_equal(labels(output, hole[None, :]), 205)
                    np.testing.assert_array_equal(labels(output, floor[:, :2]), 254)
                    np.testing.assert_array_equal(labels(output, obstacles[:, :2]), 0)

    def test_denoising_cannot_turn_unsupported_single_pixel_into_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'observed_lane'
            floor, hole, obstacles, terrain = fixture((0., 0.))
            isolated = np.array([[*hole, .30]])
            grid = ObservedGrid(.05)
            export_grid([floor, np.vstack((floor, obstacles, isolated))], grid, output,
                        dict(obstacle_min_height=.10, obstacle_max_height=1.5),
                        dict(plane=[0., 0., 1., 0.], _terrain=terrain))
            np.testing.assert_array_equal(labels(output, hole[None, :]), 0)
            report = clean_directory(
                output, free_support=lambda xy: observed_free_support(grid, xy))
            self.assertGreaterEqual(report['unsupported_free_to_unknown'], 1)
            np.testing.assert_array_equal(labels(output, hole[None, :]), 205)
            np.testing.assert_array_equal(labels(output, floor[:, :2]), 254)
            # The nine-cell obstacle exceeds the original denoising threshold.
            np.testing.assert_array_equal(labels(output, obstacles[:, :2]), 0)


if __name__ == '__main__':
    unittest.main()
