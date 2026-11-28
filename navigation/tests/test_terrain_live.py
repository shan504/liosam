"""Local-terrain filtering through production live/display callbacks without ROS."""
import copy
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
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


class AnalyticTerrain:
    """A saved-surface fixture; core tests separately validate model construction."""
    def __init__(self, shift, slope):
        self.shift, self.slope = np.asarray(shift), np.asarray(slope)
        self.evaluated = []
    def ground(self, xy):
        return (np.asarray(xy)-self.shift[:2])@self.slope+self.shift[2]
    def evaluate(self, xyz):
        xyz = np.asarray(xyz)
        self.evaluated.append(xyz.copy())
        supported = np.linalg.norm(xyz[:,:2]-self.shift[:2], axis=1) < 8.
        relative = xyz[:,2]-self.ground(xyz[:,:2])
        relative[~supported] = np.nan
        return relative, supported


class LiveTerrainCallbacks(unittest.TestCase):
    def fake_runtime(self, directory, terrain=None, load_saved=False):
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
        mods['tf2_ros'].Buffer=Buffer
        mods['tf2_ros'].TransformListener=lambda buffer,node:None
        mods['tf2_ros'].TransformException=TFError
        mods['ros_cloud'].xyz_message=lambda header,points:NS(header=copy.deepcopy(header),points=points.copy())
        mods['navigation_grid'].current_settings=lambda:dict(obstacle_min_height=.1,obstacle_max_height=1.5)
        spec=importlib.util.spec_from_file_location('test_live_obstacle_module',GO2/'obstacle_cloud.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,mods):
            spec.loader.exec_module(module)
            loader = contextlib.nullcontext() if load_saved else patch.object(module,'load_terrain',return_value=terrain)
            with loader:
                node=module.ObstacleCloud()
        return module,node,pubs,subs,queries,current

    def test_real_saved_terrain_load_is_shared_by_live_and_display(self):
        from terrain import TerrainModel, load_terrain
        import display_cloud
        with tempfile.TemporaryDirectory() as temp:
            for name,base,slope in [('yard',0.,0.),('orchard',1.5,.2)]:
                with self.subTest(map=name):
                    folder=Path(temp)/name
                    folder.mkdir()
                    cells=np.array([[0,0],[1,0],[2,0]])
                    centers=(cells+.5)*.4
                    planes=np.column_stack([np.full(3,slope),np.zeros(3),base+slope*centers[:,0]])
                    surface=TerrainModel(cells,planes)
                    metadata=dict(frame_id='map',ground_model=dict(plane=[0,0,1,0]),
                                  terrain_model=surface.save(folder/'terrain.npz'))
                    (folder/'metadata.yaml').write_text(json.dumps(metadata))
                    _,node,pubs,_,_,_=self.fake_runtime(folder,load_saved=True)
                    node.on_valid(NS(data=True))
                    xyz=np.array([[.1,.1,base+slope*.1],
                                  [.5,.1,base+slope*.5+.3],
                                  [.9,.1,base+slope*.9+8.],
                                  [2.,.1,base]])
                    node.receive(message(xyz,frame='base_link',stamp=(99,900000000)))
                    node.process()
                    np.testing.assert_allclose(pubs['/navigation/obstacle_cloud'].messages[-1].points,
                                               xyz[[1,3]],atol=1e-6)
                    loaded=load_terrain(folder,metadata)
                    output=display_cloud.display_points(message(xyz),metadata['ground_model'],terrain=loaded)
                    np.testing.assert_allclose(output,xyz[[1,2,3]],atol=1e-6)

    def test_two_maps_live_tf_relative_height_and_unknown_are_conservative(self):
        with tempfile.TemporaryDirectory() as temp:
            for name,shift,slope in [('yard',[0,0,0],[0,0]),
                                     ('hill',[10,-5,1.4],[.2,-.04])]:
                with self.subTest(map=name):
                    directory=Path(temp)/name
                    directory.mkdir()
                    (directory/'metadata.yaml').write_text(json.dumps(dict(ground_model=dict(plane=[0,0,1,0]))))
                    surface=AnalyticTerrain(shift,slope)
                    module,node,pubs,subs,queries,current=self.fake_runtime(directory,surface)
                    m=np.array([[0.,-1,0,shift[0]],[1,0,0,shift[1]],[0,0,1,shift[2]+.327],[0,0,0,1]])
                    current.matrix=m
                    xy=np.array([[1,0],[2,0],[3,0],[20,0],[4,0]],dtype=float)+shift[:2]
                    heights=np.array([0,.4,2.,0.,-.2])
                    world=np.column_stack([xy,surface.ground(xy)+heights])
                    body=transform_points(world,np.linalg.inv(m))
                    cloud=message(body,frame='base_link',stamp=(99,900000000))
                    original=cloud.data
                    subs['/localization/valid'][0](NS(data=True))
                    subs['/point_cloud2'][0](cloud)
                    node.process()
                    output=pubs['/navigation/obstacle_cloud'].messages[-1]
                    np.testing.assert_allclose(output.points,body[[1,3,4]],atol=1e-6)
                    np.testing.assert_allclose(surface.evaluated[-1],world,atol=1e-6)
                    self.assertEqual(output.header.frame_id,'base_link')
                    self.assertEqual(output.header.stamp,cloud.header.stamp)
                    self.assertEqual(queries,[('map','base_link',99_900_000_000)])
                    self.assertEqual(pubs['/navigation/obstacle_cloud'].qos,'sensor_qos')
                    self.assertEqual(cloud.data,original)
                    # Pure newly received ground clears the last obstacle cloud.
                    empty=message(body[[0]],frame='base_link',stamp=(99,950000000))
                    node.receive(empty)
                    node.process()
                    self.assertEqual(pubs['/navigation/obstacle_cloud'].messages[-1].points.shape,(0,3))

    def test_missing_tf_waits_without_relabel_and_invalid_or_stale_inputs_do_not_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            (directory/'metadata.yaml').write_text(json.dumps(dict(ground_model=dict(plane=[0,0,1,0]))))
            _,node,pubs,subs,queries,current=self.fake_runtime(directory,AnalyticTerrain([0,0,0],[.2,0]))
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
            node.receive(cloud)  # duplicate source stamp
            node.receive(message([[1,0,1]],frame='odom',stamp=(99,950000000)))
            node.receive(message([[1,0,1]],frame='base_link',stamp=(98,0)))
            node.process()
            self.assertEqual(len(pubs['/navigation/obstacle_cloud'].messages),1)
            node.receive(message([[1,0,1]],frame='base_link',stamp=(99,950000000)))
            node.on_valid(NS(data=False))
            node.process()
            self.assertFalse(node.pending)
            self.assertEqual(len(pubs['/navigation/obstacle_cloud'].messages),1)

    def test_display_keeps_all_heights_unknown_and_drops_while_removing_uphill_ground(self):
        import display_cloud
        for shift,slope in [([0,0,0],[0,0]),([20,-3,1.4],[.2,.05])]:
            surface=AnalyticTerrain(shift,slope)
            xy=np.array([[1,0],[2,0],[3,0],[20,0],[4,0]],dtype=float)+shift[:2]
            points=np.column_stack([xy,surface.ground(xy)+[0,.2,8.,0.,-.2]])
            cloud=message(points)
            output=display_cloud.display_points(cloud,{'plane':[0,0,1,0]},block_size=2,terrain=surface)
            np.testing.assert_allclose(output,points[[1,2,3,4]],atol=1e-6)


if __name__ == '__main__':
    unittest.main()
