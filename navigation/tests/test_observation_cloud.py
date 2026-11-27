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
                raw, visible, obstacles, ranges, _ = scan_geometry(world, body, lidar, .33)
                np.testing.assert_array_equal(raw[:, 3], [1, 2, 3, 4, 5])
                np.testing.assert_array_equal(raw[visible, 3], [2, 3, 4, 5])
                physical = transform(obstacles, lidar)
                np.testing.assert_allclose(physical[physical[:, 3].argsort()], world[[1, 3]], atol=1.e-5)
                self.assertEqual(np.isfinite(ranges).sum(), 2)
                self.assertAlmostEqual(float(ranges[180]), 1. - .215, places=5)

    def test_ground_only_cannot_create_an_obstacle_or_a_clearing_ray(self):
        body = np.array([0., 0., .33, 0., 0., 0., 1.])
        lidar = np.array([.2, 0., .58, 0., 0., 0., 1.])
        floor = np.array([[1., -1., 0., 1], [1., 1., .01, 2]], dtype=np.float32)
        raw, visible, obstacles, ranges, _ = scan_geometry(floor, body, lidar, .33)
        self.assertEqual(len(raw), 2)
        self.assertFalse(visible.any())
        self.assertEqual(obstacles.shape, (0, 4))
        self.assertTrue(np.isnan(ranges).all())

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
