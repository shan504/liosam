"""Ground must disappear from the display without becoming occupied or erasing obstacles."""
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gicp_localizer import map_display_points
from obstacle_view import obstacle_mask
from prepare_map import RESOLUTION, apply, is_floor, observe_rays, rigid


class MeasuredFloor(unittest.TestCase):
    def test_display_in_two_calibrated_frames(self):
        level = np.array([[0., 0., -.02], [1., 2., .035], [3., 1., .09],
                          [2., 3., .5], [1., 0., -.2], [1., 1., 1.42], [2., 3., 2.65]])
        for height, tilt in ((.51, .4), (.73, -.17)):
            with self.subTest(height=height, tilt=tilt):
                transform = rigid([1., -2., height], [0., math.sin(tilt/2), 0., math.cos(tilt/2)])
                raw = apply(level, transform)
                aligned = apply(raw, np.linalg.inv(transform))
                np.testing.assert_array_equal(is_floor(aligned[:, 2]),
                                              [True, True, False, False, False, False, False])
                np.testing.assert_allclose(map_display_points(aligned), level[[3, 5, 6]], atol=1e-6)
                np.testing.assert_allclose(aligned, level, atol=1e-12)

    def test_display_rejects_ground_and_nonfinite_heights_without_clipping_tall_returns(self):
        heights = np.array([-np.inf, -.2, 0., .099, .10, .35, 1.50, 1.501, 4., 16., np.inf, np.nan])
        np.testing.assert_array_equal(obstacle_mask(heights),
                                      [False, False, False, False, True, True, True, True, True, True, False, False])

    def test_both_navigation_panels_show_only_filtered_pointcloud_topics(self):
        layout = json.loads((Path(__file__).resolve().parents[1] / 'foxglove.json').read_text())
        views = [panel['topics'] for name, panel in layout['configById'].items()
                 if name.startswith('3D!')]
        self.assertEqual(len(views), 2)
        for topics in views:
            for cloud in ('/map_cloud', '/navigation/cloud_filtered'):
                self.assertTrue(topics[cloud]['visible'])
                self.assertIsNone(topics[cloud].get('maxValue'))
            self.assertFalse(topics.get('/navigation/cloud', {}).get('visible', False))

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
