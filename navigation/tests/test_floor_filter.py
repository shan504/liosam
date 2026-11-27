"""Ground must disappear from the display without becoming occupied or erasing obstacles."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gicp_localizer import map_display_points
from prepare_map import RESOLUTION, apply, is_floor, observe_rays, rigid


class MeasuredFloor(unittest.TestCase):
    def test_display_in_two_calibrated_frames(self):
        level = np.array([[0., 0., -.02], [1., 2., .035], [3., 1., .09],
                          [2., 3., .5], [1., 0., -.2]])
        for height, tilt in ((.51, .4), (.73, -.17)):
            with self.subTest(height=height, tilt=tilt):
                transform = rigid([1., -2., height], [0., math.sin(tilt/2), 0., math.cos(tilt/2)])
                raw = apply(level, transform)
                aligned = apply(raw, np.linalg.inv(transform))
                np.testing.assert_array_equal(is_floor(aligned[:, 2]), [True, True, False, False, False])
                np.testing.assert_allclose(map_display_points(aligned), level[2:], atol=1e-6)
                np.testing.assert_allclose(aligned, level, atol=1e-12)

    def test_floor_rays_and_occluded_cells(self):
        for origin_x, bearing in ((2., .01), (4., .21)):
            with self.subTest(origin=origin_x):
                origin = np.array([origin_x, 2., .52])
                free = np.zeros((200, 200), 'u1')
                def point(distance, z):
                    return [origin_x + distance*math.cos(bearing), 2.+distance*math.sin(bearing), z]
                def observed(distance):
                    x, y = np.floor(np.array(point(distance, 0.)[:2]) / RESOLUTION).astype(int)
                    return bool(free[y, x])
                floor_hits = [point(.4, .03), point(3., 0.)]
                observe_rays(free, origin, np.array(floor_hits), np.zeros(2, dtype=int))
                self.assertTrue(observed(2.))
                self.assertFalse(observed(3.5))
                free.fill(0)
                # Even a low, ambiguous return must stop clearing behind it.
                mixed = floor_hits + [point(1.8, .07), point(2., .6)]
                observe_rays(free, origin, np.array(mixed), np.zeros(2, dtype=int))
                self.assertTrue(observed(1.))
                self.assertFalse(observed(2.2))


if __name__ == '__main__':
    unittest.main()
