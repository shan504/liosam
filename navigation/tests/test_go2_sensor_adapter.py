from pathlib import Path
import math
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nav_sensors import body_scan, needs_mount_tf
from check_returns import POINT
from dense_map import transform
from native_cloud import place


class Go2SensorAdapter(unittest.TestCase):
    def test_deskew_into_original_mapper_frame(self):
        start, duration = 2_000_000_000, 100_000_000
        mount = np.array([.215, 0, .25, 0, math.sin(math.pi/12), 0, math.cos(math.pi/12)])
        for x, y, yaw in [(0., 0., 0.), (7., -2., .9)]:
            with self.subTest(origin=(x, y, yaw)):
                poses = [np.array([x+dx, y, .32, 0, 0, math.sin(a/2), math.cos(a/2)])
                         for dx, a in [(0., yaw), (.15, yaw + .1)]]
                # Two acquisition times observe one fixed object. A high return
                # must survive and a hit on the robot body must be removed.
                wanted = np.array([[2., .5, 2.3, 11.], [2., .5, 2.3, 12.],
                                   [0., 0., -.1, 13.]], dtype='f4')
                world = transform(wanted, poses[0])
                raw = np.zeros(3, POINT)
                for i, pose_index in enumerate([0, 1, 0]):
                    body = transform(world[i:i+1], poses[pose_index], inverse=True)
                    sensor = transform(body, mount, inverse=True)
                    raw[i]['x'], raw[i]['y'], raw[i]['z'] = sensor[0, :3]
                    raw[i]['offset_time'] = pose_index * duration
                    raw[i]['reflectivity'] = int(wanted[i, 3])
                history = [(start, poses[0]), (start+duration, poses[1])]
                deskewed, _ = place(start, raw, history, mount, .12, 40.)
                np.testing.assert_allclose(body_scan(deskewed, poses[0]), wanted[:2], atol=2e-6)

    def test_mount_tf_has_one_authority(self):
        mount = np.array([.2, 0., .25, 0., 0., 0., 1.])
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/'robot.urdf'
            file.write_text('<robot><link name="base_link"/></robot>')
            self.assertTrue(needs_mount_tf(file, 'livox_frame', mount))
            file.write_text('<robot><link name="base_link"/><link name="livox_frame"/>'
                            '<joint type="fixed"><parent link="base_link"/><child link="livox_frame"/>'
                            '<origin xyz=".2 0 .25"/></joint></robot>')
            self.assertFalse(needs_mount_tf(file, 'livox_frame', mount))
            with self.assertRaises(ValueError):
                needs_mount_tf(file, 'livox_frame', mount + [.1, 0., 0., 0., 0., 0., 0.])
            file.write_text(file.read_text().replace('type="fixed"', 'type="revolute"'))
            with self.assertRaises(ValueError):
                needs_mount_tf(file, 'livox_frame', mount)


if __name__ == '__main__':
    unittest.main()
