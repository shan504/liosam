"""Fixed initial-plane policy through production live/display callbacks without ROS."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

GO2 = Path(__file__).resolve().parents[1] / 'original_go2/go2_3d'
sys.path.insert(0, str(GO2))
sys.path.insert(0, str(Path(__file__).parent))
from geometry import transform_points
from test_display_cloud import message


class Time:
    def __init__(self, ns=0):
        self.nanoseconds = ns
    @classmethod
    def from_msg(cls, stamp):
        return cls(stamp.sec*10**9+stamp.nanosec)
    def __sub__(self, other):
        return Time(self.nanoseconds-other.nanoseconds)
    def __gt__(self, other):
        return self.nanoseconds > other.nanoseconds


class TFError(Exception):
    pass


class FixedPlaneCallbacks(unittest.TestCase):
    def fake_runtime(self, directory):
        pubs, subs, queries = {}, {}, []
        current = NS(ns=100_000_000_000, ready=True, matrix=np.eye(4))
        class Node:
            def __init__(self, name):
                pass
            def declare_parameter(self, name, default):
                pass
            def get_parameter(self, name):
                return NS(value=str(directory))
            def create_publisher(self, kind, topic, qos):
                publisher = NS(messages=[], qos=qos)
                publisher.publish = publisher.messages.append
                pubs[topic] = publisher
                return publisher
            def create_subscription(self, kind, topic, callback, qos):
                subs[topic] = (callback, qos)
            def create_timer(self, period, callback):
                pass
            def get_clock(self):
                return NS(now=lambda:Time(current.ns))
            def get_logger(self):
                return NS(info=lambda *a,**k:None, warning=lambda *a,**k:None)
        class Buffer:
            def lookup_transform(self, parent, child, stamp):
                queries.append((parent,child,stamp.nanoseconds))
                if not current.ready:
                    raise TFError('not available')
                m=current.matrix
                from geometry import quaternion_from_matrix
                q=quaternion_from_matrix(m)
                return NS(header=NS(stamp=NS(sec=99,nanosec=900000000)),
                    transform=NS(translation=NS(x=m[0,3],y=m[1,3],z=m[2,3]),
                                 rotation=NS(x=q[0],y=q[1],z=q[2],w=q[3])))
        mods = {name:ModuleType(name) for name in
                ('rclpy','rclpy.node','rclpy.time','rclpy.qos','sensor_msgs','sensor_msgs.msg',
                 'std_msgs','std_msgs.msg','tf2_ros','ros_cloud','navigation_grid')}
        mods['rclpy.node'].Node=Node
        mods['rclpy.time'].Time=Time
        mods['rclpy.qos'].qos_profile_sensor_data='sensor_qos'
        mods['sensor_msgs.msg'].PointCloud2=object
        mods['std_msgs.msg'].Bool=object
        mods['std_msgs.msg'].String=NS
        mods['tf2_ros'].Buffer=Buffer
        mods['tf2_ros'].TransformListener=lambda buffer,node:None
        mods['tf2_ros'].TransformException=TFError
        mods['ros_cloud'].xyz_message=lambda header,points:NS(header=copy.deepcopy(header),points=points.copy())
        mods['navigation_grid'].current_settings=lambda:dict(obstacle_min_height=.15,obstacle_max_height=1.5,grid_resolution=.05)
        spec=importlib.util.spec_from_file_location('test_live_obstacle_module',GO2/'obstacle_cloud.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,mods):
            spec.loader.exec_module(module)
            node=module.ObstacleCloud()
        return module,node,pubs,subs,queries,current

    @staticmethod
    def clusters(plane):
        xy = np.array([[.0125+x*.05,.0125+y*.05] for y in range(2) for x in range(4)])
        bands = [0., .05, .15, .151, .4, 1.4, 8., -.2]
        rows = []
        for i,h in enumerate(bands):
            horizontal = xy + [i*2.,0.]
            z = (h-plane[3]-horizontal@plane[:2])/plane[2]
            rows.append(np.column_stack((horizontal,z)))
        return np.concatenate(rows), bands

    def test_saved_initial_plane_boundary_and_exact_time_tf_for_two_maps(self):
        import display_cloud
        with tempfile.TemporaryDirectory() as temp:
            for name, plane in [('yard',np.array([0.,0.,1.,.327])),
                                ('farm',np.array([.05,-.08,1.,-.6]))]:
                with self.subTest(map=name):
                    plane /= np.linalg.norm(plane[:3])
                    folder=Path(temp)/name
                    folder.mkdir()
                    metadata=dict(frame_id='map',ground_model=dict(plane=plane.tolist()),
                        body_height_above_ground=.327,
                        terrain_model=dict(file='must-not-load-this-missing-terrain.npz'))
                    (folder/'metadata.yaml').write_text(json.dumps(metadata))
                    module,node,pubs,subs,queries,current=self.fake_runtime(folder)
                    # Legacy terrain descriptors are ignored, including an
                    # invalid file path. New scans cannot replace the plane.
                    self.assertFalse(hasattr(module,'load_terrain'))
                    self.assertFalse(hasattr(node,'terrain'))
                    self.assertFalse(hasattr(module,'measured_ground'))
                    world,bands=self.clusters(plane)
                    current.matrix=np.array([[0.,-1.,0.,2.],[1.,0.,0.,-1.],
                                             [0.,0.,1.,.327],[0.,0.,0.,1.]])
                    body=transform_points(world,np.linalg.inv(current.matrix))
                    cloud=message(body,frame='base_link',stamp=(99,900000000))
                    original=cloud.data
                    subs['/localization/valid'][0](NS(data=True))
                    subs['/point_cloud2'][0](cloud)
                    node.process()
                    collision=pubs['/navigation/obstacle_cloud'].messages[-1]
                    display=pubs['/navigation/measured_scan_obstacles'].messages[-1]
                    # The exact 15 cm rejection boundary is common to both
                    # streams; only Nav2 omits the high scene returns.
                    np.testing.assert_allclose(collision.points,body[24:48],atol=2e-6)
                    np.testing.assert_allclose(display.points,body[24:56],atol=2e-6)
                    self.assertEqual(collision.header.frame_id,'base_link')
                    self.assertEqual(collision.header.stamp,cloud.header.stamp)
                    self.assertEqual(queries,[('map','base_link',99_900_000_000)])
                    self.assertEqual(cloud.data,original)
                    # Whole-map display and live companion use the same
                    # threshold and retain the full vertical scene.
                    result=display_cloud.display_points(message(world),metadata['ground_model'],block_size=3)
                    np.testing.assert_allclose(result,world[24:56],atol=2e-6)
                    node.report()
                    status=json.loads(pubs['/navigation/obstacle_status'].messages[-1].data)
                    self.assertEqual(status['ground_filter_mode'],'initial_plane')
                    self.assertEqual(status['floor_rejection_height'],.15)
                    self.assertEqual(status['ground_rejected_points'],32)
                    self.assertEqual(status['isolated_rejected_points'],0)
                    self.assertEqual(status['source_stamp_ns'],99_900_000_000)
                    # A new ground-only scan replaces the point cloud with an
                    # empty observation; it never asserts blind free rays.
                    node.receive(message(body[:8],frame='base_link',stamp=(99,950000000)))
                    node.process()
                    self.assertEqual(pubs['/navigation/obstacle_cloud'].messages[-1].points.shape,(0,3))

    def test_isolated_clusters_removed_from_both_outputs_without_corridor_blind_zone(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'metadata.yaml').write_text(json.dumps(dict(ground_model=dict(plane=[0,0,1,0]))))
            _,node,pubs,_,_,_=self.fake_runtime(folder)
            # This connected obstacle lies in the surveyed front metre. It
            # must remain in the real-time collision input. Duplicating one
            # distant isolated cell many times does not make a large object.
            connected=np.array([[.52+x*.05,.02+y*.05,.3] for y in range(2) for x in range(4)])
            noise=np.tile([5.,5.,.4],(40,1))
            world=np.vstack((connected,noise,connected+[0,0,5.]))
            node.on_valid(NS(data=True))
            node.receive(message(world,frame='base_link',stamp=(99,900000000)))
            node.process()
            np.testing.assert_allclose(pubs['/navigation/obstacle_cloud'].messages[-1].points,connected)
            np.testing.assert_allclose(pubs['/navigation/measured_scan_obstacles'].messages[-1].points,
                                       np.vstack((connected,connected+[0,0,5.])))
            self.assertEqual(node.counts['isolated_rejected_points'],40)
            self.assertEqual(node.counts['front_obstacle_points'],8)

    def test_missing_tf_waits_without_relabel_and_invalid_or_stale_inputs_do_not_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            (directory/'metadata.yaml').write_text(json.dumps(dict(ground_model=dict(plane=[0,0,1,0]))))
            _,node,pubs,subs,queries,current=self.fake_runtime(directory)
            node.on_valid(NS(data=True))
            current.ready=False
            cloud=message([[1,0,1]],frame='base_link',stamp=(99,900000000))
            node.receive(cloud)
            node.process()
            self.assertEqual(len(node.pending),1)
            self.assertFalse(pubs['/navigation/obstacle_cloud'].messages)
            current.ready=True
            node.process()
            self.assertEqual(len(pubs['/navigation/obstacle_cloud'].messages),1)
            node.receive(cloud)
            node.receive(message([[1,0,1]],frame='odom',stamp=(99,950000000)))
            node.receive(message([[1,0,1]],frame='base_link',stamp=(98,0)))
            node.process()
            self.assertEqual(len(pubs['/navigation/obstacle_cloud'].messages),1)
            node.receive(message([[1,0,1]],frame='base_link',stamp=(99,950000000)))
            node.on_valid(NS(data=False))
            node.process()
            self.assertFalse(node.pending)
            self.assertEqual(len(pubs['/navigation/obstacle_cloud'].messages),1)


if __name__ == '__main__':
    unittest.main()

