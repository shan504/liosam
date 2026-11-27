from pathlib import Path
import math
import sys
import unittest

import numpy as np
import yaml

NAVIGATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NAVIGATION))
from nav_sensors import scan_geometry
from dense_map import transform
from gicp_localizer import live_display_points, map_display_points
from prepare_map import apply, rigid


class Mid360Observations(unittest.TestCase):
    def test_physical_endpoints_and_ground_are_independent_of_map_origin(self):
        pitch = math.radians(30)
        for x, y, floor in [(0, 0, 0), (7, -4, 3)]:
            with self.subTest(origin=(x, y, floor)):
                body = np.array([x, y, floor+.33, 0, 0, 0, 1.])
                lidar = np.array([x+.215, y, floor+.58, 0, math.sin(pitch/2), 0, math.cos(pitch/2)])
                local = np.array([[.9, 0, .02, 1], [1., 0, .3, 2], [2., 0, .30, 3],
                                  [0, 1.5, .55, 4], [.9, 1, 1.9, 5]], dtype=np.float32)
                world = local.copy()
                world[:, :3] += [x, y, floor]
                raw, obstacles, ranges, _ = scan_geometry(world, body, lidar, .33)
                np.testing.assert_array_equal(raw[:, 3], [1, 2, 3, 4, 5])
                physical = transform(obstacles, lidar)
                np.testing.assert_allclose(physical[physical[:, 3].argsort()], world[[1, 3]], atol=1.e-5)
                self.assertEqual(np.isfinite(ranges).sum(), 2)
                self.assertAlmostEqual(float(ranges[180]), 1. - .215, places=5)

    def test_ground_only_cannot_create_an_obstacle_or_a_clearing_ray(self):
        body = np.array([0., 0., .33, 0., 0., 0., 1.])
        lidar = np.array([.2, 0., .58, 0., 0., 0., 1.])
        floor = np.array([[1., -1., 0., 1], [1., 1., .01, 2]], dtype=np.float32)
        raw, obstacles, ranges, _ = scan_geometry(floor, body, lidar, .33)
        self.assertEqual(len(raw), 2)
        self.assertEqual(obstacles.shape, (0, 4))
        self.assertTrue(np.isnan(ranges).all())

    def test_live_and_saved_obstacle_views_agree_at_two_sensor_poses(self):
        # Several hits at one bearing describe an object contour. The display
        # preserves all of them, including returns outside GICP's matching range.
        xyz = np.array([[0., 0., 0.], [.5, 0., .25], [2., 0., .55], [18., 0., 1.4],
                        [1., -1., 4.], [12., 2., -.35], [1., .2, .09]])
        for y, pitch in [(0., .52), (2.5, -.18)]:
            with self.subTest(y=y, pitch=pitch):
                xyz_map = xyz + [0., y, 0.]
                pose = rigid([.21, y, .55], [0., math.sin(pitch/2), 0., math.cos(pitch/2)])
                sensor = np.column_stack([apply(xyz_map, np.linalg.inv(pose)), np.arange(len(xyz))])
                display = live_display_points(sensor, pose)
                np.testing.assert_allclose(display[:, :3], map_display_points(xyz_map), atol=2e-6)
                np.testing.assert_array_equal(display[:, 3], [1, 2, 3, 4])
                self.assertEqual(live_display_points(sensor[[0, 5, 6]], pose).shape, (0, 4))

    def test_low_obstacle_is_still_marked_even_below_display_band(self):
        body = np.array([0., 0., .33, 0., 0., 0., 1.])
        lidar = np.array([.2, 0., .58, 0., 0., 0., 1.])
        block = np.array([[1., 0., .09, 7]], dtype=np.float32)
        raw, obstacles, ranges, _ = scan_geometry(block, body, lidar, .33)
        self.assertEqual(len(map_display_points(block[:, :3])), 0)
        self.assertEqual(len(raw), 1)
        np.testing.assert_array_equal(obstacles[:, 3], [7])
        self.assertEqual(np.isfinite(ranges).sum(), 1)

    def test_nav2_uses_sensor_frame_endpoints_and_monitors_freshness(self):
        layer = yaml.safe_load((NAVIGATION / 'nav2.yaml').read_text())[
            'local_costmap']['local_costmap']['ros__parameters']['obstacle_layer']
        source = layer[layer['observation_sources']]
        self.assertEqual((source['topic'], source['data_type']), ('/navigation/obstacles', 'PointCloud2'))
        self.assertFalse(source.get('sensor_frame'))
        self.assertTrue(source['marking'] and source['clearing'])
        self.assertGreater(source['expected_update_rate'], 0.)


if __name__ == '__main__':
    unittest.main()
