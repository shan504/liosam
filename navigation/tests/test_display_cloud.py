"""Display geometry and runtime wiring, independent of ROS/robot hardware."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GO2 = ROOT / 'navigation/original_go2/go2_3d'
sys.path.insert(0, str(GO2))
import display_cloud


def message(points, *, frame='map', stamp=(5, 7)):
    data = np.asarray(points, dtype='<f4').reshape(-1, 3)
    return NS(header=NS(frame_id=frame, stamp=NS(sec=stamp[0], nanosec=stamp[1])),
              height=1, width=len(data), point_step=12, row_step=len(data)*12,
              is_bigendian=False,
              fields=[NS(name=name, offset=i*4, datatype=7, count=1) for i, name in enumerate('xyz')],
              data=data.tobytes())


class DisplayCloudGeometry(unittest.TestCase):
    def test_two_selected_map_planes_keep_obstacles_and_not_ground(self):
        scenarios = [('yard_body_origin', np.array([0., 0., 1., .327])),
                     ('orchard_tilted', np.array([.05, -.08, 1., -.6]))]
        with tempfile.TemporaryDirectory() as temp:
            for name, plane in scenarios:
                with self.subTest(map=name):
                    plane /= np.linalg.norm(plane[:3])
                    folder = Path(temp) / name
                    folder.mkdir()
                    metadata = dict(frame_id='map', ground_model=dict(plane=plane.tolist()))
                    (folder/'metadata.yaml').write_text(json.dumps(metadata))
                    model = display_cloud.ground_model(metadata)
                    xy = np.array([[0., 0.], [10., -2.], [-4., 8.], [2., 1.], [12., -8.], [-2., -4.]])
                    h = np.array([0., -.01, .05, .2, 1.6, 8.])
                    z = (h-plane[3]-xy@plane[:2])/plane[2]
                    cloud = message(np.column_stack([xy, z]))
                    before = hashlib.sha256(cloud.data).hexdigest()
                    for _ in range(2):
                        # Loading each selected map repeatedly cannot reuse a
                        # previous map's floor elevation or normal.
                        loaded = display_cloud.ground_model(json.loads((folder/'metadata.yaml').read_text()))
                        result = display_cloud.display_points(cloud, loaded, block_size=2)
                        expected = np.column_stack([xy, z])[3:]
                        np.testing.assert_allclose(result, expected, atol=1e-6)
                        self.assertGreater((result@plane[:3]+plane[3]).max(), 7.9)
                        self.assertEqual(hashlib.sha256(cloud.data).hexdigest(), before)

    def test_blocked_parsing_keeps_extent_and_all_selected_points(self):
        x = np.arange(150007, dtype=float) / 10
        points = np.column_stack([x, -x, np.where(np.arange(len(x)) % 3, .25, .02)])
        cloud = message(points)
        blocks = list(display_cloud.xyz_blocks(cloud, block_size=1024))
        self.assertEqual(sum(len(block) for block in blocks), len(x))
        self.assertTrue(all(len(block) <= 1024 for block in blocks))
        result = display_cloud.display_points(cloud, {'plane':[0., 0., 1., 0.]}, block_size=1024)
        expected = points[np.arange(len(x)) % 3 != 0].astype('<f4')
        np.testing.assert_array_equal(result, expected)
        self.assertGreater(result[-1, 0], 10000.)

    def test_big_endian_organized_rows_and_field_offsets(self):
        xyz = np.array([[1, 2, 0], [3, 4, .5], [5, 6, 2],
                        [7, 8, .05], [9, 10, 5], [11, 12, np.nan]], dtype=float)
        payload = bytearray(2*88)
        for i, (x,y,z) in enumerate(xyz):
            struct.pack_into('>ddd', payload, (i//3)*88+(i%3)*24, z, x, y)
        cloud = NS(header=NS(frame_id='map'), height=2, width=3, point_step=24,
                   row_step=88, is_bigendian=True, data=bytes(payload),
                   fields=[NS(name=n, offset=o, datatype=8, count=1)
                           for n,o in [('z',0),('x',8),('y',16)]])
        result = display_cloud.display_points(cloud, {'plane':[0,0,1,0]}, block_size=2)
        np.testing.assert_array_equal(result, xyz[[1,2,4]].astype('<f4'))

    def test_empty_invalid_and_wrong_frame_inputs(self):
        model = {'plane':[0,0,1,0]}
        self.assertEqual(display_cloud.display_points(message([]), model).shape, (0,3))
        self.assertEqual(display_cloud.display_points(message([[1,2,0],[1,np.inf,2]]), model).shape, (0,3))
        with self.assertRaisesRegex(ValueError, 'registered in map'):
            display_cloud.display_points(message([[1,2,3]], frame='odom'), model)
        corrupt = message([[1,2,3]])
        corrupt.data = corrupt.data[:-1]
        with self.assertRaises(ValueError):
            display_cloud.display_points(corrupt, model)
        for metadata in ({'frame_id':'odom','ground_model':model},
                         {'frame_id':'map','ground_model':{'plane':[0,0,2,0]}},
                         {'frame_id':'map','ground_model':{'plane':[0,0,-1,0]}},
                         {'frame_id':'map','ground_model':{'plane':[0,0,1,np.nan]}}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                display_cloud.ground_model(metadata)

    def test_runtime_retains_static_map_and_publishes_newest_scan_only(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'metadata.yaml').write_text(json.dumps(
                {'frame_id':'map','ground_model':{'plane':[0,0,1,.3]}}))
            nodes = []

            class FakePublisher:
                def __init__(self, qos):
                    self.qos, self.messages = qos, []
                def publish(self, msg):
                    self.messages.append(msg)

            class FakeNode:
                def __init__(self, name):
                    self.name, self.subs, self.pubs, self.timers = name, {}, {}, []
                    nodes.append(self)
                def declare_parameter(self, name, default):
                    pass
                def get_parameter(self, name):
                    return NS(value=str(folder))
                def create_publisher(self, kind, topic, qos):
                    self.pubs[topic] = FakePublisher(qos)
                    return self.pubs[topic]
                def create_subscription(self, kind, topic, callback, qos):
                    self.subs[topic] = (callback, qos)
                def create_timer(self, period, callback):
                    self.timers.append((period, callback))
                def get_logger(self):
                    return NS(info=lambda *a,**k:None, warning=lambda *a,**k:None)
                def destroy_node(self):
                    pass

            def spin(node):
                self.assertEqual(set(node.subs), {'/map_cloud','/registered_cloud'})
                self.assertEqual(node.timers[0][0], .2)
                retained = node.pubs[display_cloud.MAP_TOPIC].qos
                self.assertEqual(retained.durability, 'transient_local')
                self.assertEqual(retained.reliability, 'reliable')
                original = message([[1,2,-.3],[3,4,.2]])
                node.subs['/map_cloud'][0](original)
                node.subs['/registered_cloud'][0](message([[5,6,.3]], stamp=(7,1)))
                latest = message([[7,8,.4]], stamp=(7,2))
                node.subs['/registered_cloud'][0](latest)
                node.timers[0][1]()
                self.assertEqual(len(node.pubs[display_cloud.MAP_TOPIC].messages), 1)
                self.assertEqual(len(node.pubs[display_cloud.SCAN_TOPIC].messages), 1)
                output = node.pubs[display_cloud.SCAN_TOPIC].messages[0]
                np.testing.assert_array_equal(output.points, [[7,8,np.float32(.4)]])
                self.assertEqual(output.header.stamp.nanosec, 2)
                self.assertEqual(latest.header.frame_id, 'map')
                node.subs['/map_cloud'][0](original)
                node.subs['/registered_cloud'][0](message([[90,90,90]], stamp=(7,1)))
                node.timers[0][1]()
                self.assertEqual(len(node.pubs[display_cloud.MAP_TOPIC].messages), 1)
                self.assertEqual(len(node.pubs[display_cloud.SCAN_TOPIC].messages), 1)
                node.subs['/registered_cloud'][0](message([[1,2,-.3]], stamp=(8,0)))
                node.timers[0][1]()
                self.assertEqual(node.pubs[display_cloud.SCAN_TOPIC].messages[-1].points.shape, (0,3))

            mods = {}
            for name in ('rclpy','rclpy.node','rclpy.qos','sensor_msgs','sensor_msgs.msg','ros_cloud'):
                mods[name] = ModuleType(name)
            mods['rclpy'].init = lambda:None
            mods['rclpy'].spin = spin
            mods['rclpy'].ok = lambda:True
            mods['rclpy'].shutdown = lambda:None
            mods['rclpy.node'].Node = FakeNode
            mods['rclpy.qos'].QoSProfile = lambda **kwargs:NS(**kwargs)
            mods['rclpy.qos'].ReliabilityPolicy = NS(RELIABLE='reliable')
            mods['rclpy.qos'].DurabilityPolicy = NS(TRANSIENT_LOCAL='transient_local')
            mods['rclpy.qos'].qos_profile_sensor_data = object()
            mods['sensor_msgs.msg'].PointCloud2 = object
            mods['ros_cloud'].xyz_message = lambda header, points:NS(header=copy.deepcopy(header), points=points.copy())
            with patch.dict(sys.modules, mods):
                display_cloud.main()
            self.assertEqual(len(nodes), 1)


if __name__ == '__main__':
    unittest.main()
