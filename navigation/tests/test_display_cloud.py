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
    def test_components_span_blocks_and_do_not_clip_large_map_extent(self):
        # Each block alone is smaller than the eight-cell group threshold;
        # the complete object survives. The isolated distant return does not.
        connected=np.array([[100000.+.02+x*.05,-100000.+.02+y*.05,.4]
                            for y in range(2) for x in range(4)])
        points=np.vstack((connected, [[-200000.,200000.,.5]]))
        result=display_cloud.display_points(message(points),{'plane':[0,0,1,0]},block_size=3)
        np.testing.assert_array_equal(result,connected.astype('<f4'))

    def test_live_display_uses_classified_obstacles_without_plane_or_road_refilter(self):
        from geometry import pose_matrix
        transform = pose_matrix([10., 4., 3.], [0., 0., np.sin(np.pi/4), np.cos(np.pi/4)])
        cloud = message([[1., 0., -.2], [0., 2., 6.], [np.nan, 0., 0.]], frame='base_link')
        np.testing.assert_allclose(display_cloud.live_display_points(cloud, transform),
                                   [[10., 5., 2.8], [8., 4., 9.]], atol=1e-6)
        self.assertEqual(cloud.header.frame_id, 'base_link')
        with self.assertRaisesRegex(ValueError, 'base_link'):
            display_cloud.live_display_points(message([[1.,2.,3.]], frame='odom'), transform)

    def test_surveyed_road_mask_is_static_only_and_uses_each_selected_map_frame(self):
        from surveyed_corridor import make_document
        with tempfile.TemporaryDirectory() as temp:
            for name, offset in [('orchard_new', [5., -3., 1.]), ('yard_new', [-2., 7., -.4])]:
                with self.subTest(map=name):
                    directory = Path(temp) / name
                    directory.mkdir()
                    source_from_map = np.eye(4)
                    source_from_map[:3, 3] = offset
                    body = source_from_map.copy()
                    row = dict(stamp_ns=1, anchor_stamp_ns=1, body_pose=body.tolist())
                    (directory / 'surveyed_corridor.json').write_text(json.dumps(make_document('saved', [row])))
                    metadata = dict(source_frame='saved', map_from_source_map=np.linalg.inv(source_from_map).tolist(),
                                    surveyed_corridor=dict(file='surveyed_corridor.json'))
                    corridor = display_cloud.load_display_corridor(directory, metadata)
                    cloud = message([[.4, .1, .4], [.4, .1, 5.], [.4, .8, .4], [2., 0., 3.]])
                    model = dict(plane=[0., 0., 1., 0.])
                    np.testing.assert_allclose(display_cloud.display_points(cloud, model, corridor=corridor, noise_min_cells=1),
                                               [[.4, .8, .4], [2., 0., 3.]])
                    # Same live scan must keep an actual person or obstacle
                    # in the road: the recorded corridor is no blind zone.
                    self.assertEqual(len(display_cloud.display_points(cloud, model, noise_min_cells=1)), 4)
                    metadata['surveyed_corridor']['file'] = '../surveyed_corridor.json'
                    with self.assertRaises(ValueError):
                        display_cloud.load_display_corridor(directory, metadata)

    def test_scan_freshness_requires_live_localization_and_source_timestamp(self):
        current = display_cloud.scan_is_current
        self.assertTrue(current(10_000_000_000, 10_500_000_000, True, 100., 100.1))
        self.assertFalse(current(10_000_000_000, 10_500_000_000, False, 100., 100.1))
        self.assertFalse(current(10_000_000_000, 10_500_000_000, True, None, 100.1))
        self.assertFalse(current(10_000_000_000, 10_500_000_000, True, 100., 101.))
        self.assertFalse(current(10_000_000_000, 12_000_000_000, True, 100., 100.1))
        self.assertFalse(current(12_000_000_000, 10_000_000_000, True, 100., 100.1))
        self.assertFalse(current(0, 0, True, 100., 100.1))

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
                        result = display_cloud.display_points(cloud, loaded, block_size=2, noise_min_cells=1)
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
        result = display_cloud.display_points(cloud, {'plane':[0., 0., 1., 0.]}, block_size=1024, noise_min_cells=1)
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
        result = display_cloud.display_points(cloud, {'plane':[0,0,1,0]}, block_size=2, noise_min_cells=1)
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
                    self.now_ns = 7_000_000_002
                    nodes.append(self)
                def get_clock(self):
                    return NS(now=lambda:NS(nanoseconds=self.now_ns, to_msg=lambda:NS(
                        sec=self.now_ns//10**9, nanosec=self.now_ns%10**9)))
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
                self.assertEqual(set(node.subs), {'/map_cloud',display_cloud.OBSTACLE_TOPIC,'/localization/valid'})
                self.assertEqual(node.timers[0][0], .2)
                node.subs['/localization/valid'][0](NS(data=True))
                retained = node.pubs[display_cloud.MAP_TOPIC].qos
                self.assertEqual(retained.durability, 'transient_local')
                self.assertEqual(retained.reliability, 'reliable')
                original = message([[1,2,-.3],[3,4,.2]])
                node.subs['/map_cloud'][0](original)
                node.subs[display_cloud.OBSTACLE_TOPIC][0](message([[5,6,.3]], frame='base_link', stamp=(7,1)))
                latest = message([[7,8,.4]], frame='base_link', stamp=(7,2))
                node.subs[display_cloud.OBSTACLE_TOPIC][0](latest)
                node.timers[0][1]()
                self.assertEqual(len(node.pubs[display_cloud.MAP_TOPIC].messages), 1)
                self.assertEqual(len(node.pubs[display_cloud.SCAN_TOPIC].messages), 1)
                output = node.pubs[display_cloud.SCAN_TOPIC].messages[0]
                np.testing.assert_array_equal(output.points, [[7,8,np.float32(.4)]])
                self.assertEqual(output.header.stamp.nanosec, 2)
                self.assertEqual(latest.header.frame_id, 'base_link')
                self.assertEqual(output.header.frame_id, 'map')
                self.assertEqual(node.buffer.calls[-1], ('map', 'base_link', 7_000_000_002))
                node.subs['/map_cloud'][0](original)
                node.subs[display_cloud.OBSTACLE_TOPIC][0](message([[90,90,90]], frame='base_link', stamp=(7,1)))
                node.timers[0][1]()
                self.assertEqual(len(node.pubs[display_cloud.MAP_TOPIC].messages), 1)
                self.assertEqual(len(node.pubs[display_cloud.SCAN_TOPIC].messages), 1)
                # An explicit invalid state removes old display points only.
                node.subs['/localization/valid'][0](NS(data=False))
                self.assertEqual(node.pubs[display_cloud.SCAN_TOPIC].messages[-1].points.shape, (0,3))
                self.assertEqual(len(node.pubs[display_cloud.MAP_TOPIC].messages), 1)
                node.subs['/localization/valid'][0](NS(data=True))
                node.now_ns = 8_000_000_000
                node.subs[display_cloud.OBSTACLE_TOPIC][0](message([], frame='base_link', stamp=(8,0)))
                node.timers[0][1]()
                self.assertEqual(node.pubs[display_cloud.SCAN_TOPIC].messages[-1].points.shape, (0,3))
                # Loss of new scans clears an old cloud even if validity is
                # still published, preserving the untouched static map.
                node.now_ns = 8_500_000_000
                node.subs[display_cloud.OBSTACLE_TOPIC][0](message([[1,2,.5]], frame='base_link', stamp=(8,500000000)))
                node.timers[0][1]()
                self.assertEqual(len(node.pubs[display_cloud.SCAN_TOPIC].messages[-1].points), 1)
                node.now_ns = 10_000_000_000
                node.timers[0][1]()
                self.assertEqual(node.pubs[display_cloud.SCAN_TOPIC].messages[-1].points.shape, (0,3))
                # Bag time rewind discards old-lap sequencing; a fresh valid
                # report and low-timestamp scan can then display normally.
                node.now_ns = 4_000_000_000
                node.timers[0][1]()
                node.subs['/localization/valid'][0](NS(data=True))
                node.subs[display_cloud.OBSTACLE_TOPIC][0](message([[1,2,.5]], frame='base_link', stamp=(4,0)))
                node.timers[0][1]()
                self.assertEqual(len(node.pubs[display_cloud.SCAN_TOPIC].messages[-1].points), 1)
                # Exact source-time TF can arrive after the cloud. The node
                # waits and retries, without using latest or relabeling XYZ.
                node.now_ns = 4_100_000_000
                node.buffer.unavailable = True
                node.subs[display_cloud.OBSTACLE_TOPIC][0](message([[3,4,.1]], frame='base_link', stamp=(4,100000000)))
                node.timers[0][1]()
                self.assertEqual(node.pubs[display_cloud.SCAN_TOPIC].messages[-1].points.shape, (0,3))
                node.buffer.unavailable = False
                node.timers[0][1]()
                np.testing.assert_allclose(node.pubs[display_cloud.SCAN_TOPIC].messages[-1].points, [[3,4,.1]])
                self.assertEqual(node.buffer.calls[-1], ('map', 'base_link', 4_100_000_000))

            mods = {}
            for name in ('rclpy','rclpy.node','rclpy.qos','sensor_msgs','sensor_msgs.msg',
                         'std_msgs','std_msgs.msg','ros_cloud','rclpy.time','tf2_ros'):
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
            class FakeTransformException(Exception):
                pass
            class FakeBuffer:
                def __init__(self):
                    self.calls, self.unavailable = [], False
                def lookup_transform(self, target, source, stamp):
                    self.calls.append((target, source, stamp))
                    if self.unavailable:
                        raise FakeTransformException('future lookup')
                    return NS(transform=NS(translation=NS(x=0.,y=0.,z=0.), rotation=NS(x=0.,y=0.,z=0.,w=1.)))
            mods['rclpy.time'].Time = NS(from_msg=lambda stamp: stamp.sec*10**9+stamp.nanosec)
            mods['tf2_ros'].Buffer = FakeBuffer
            mods['tf2_ros'].TransformListener = lambda *args: None
            mods['tf2_ros'].TransformException = FakeTransformException
            mods['sensor_msgs.msg'].PointCloud2 = object
            mods['std_msgs.msg'].Bool = object
            mods['ros_cloud'].xyz_message = lambda header, points:NS(header=copy.deepcopy(header), points=points.copy())
            with patch.dict(sys.modules, mods), patch.object(display_cloud.time, 'monotonic', return_value=100.):
                display_cloud.main()
            self.assertEqual(len(nodes), 1)


if __name__ == '__main__':
    unittest.main()
