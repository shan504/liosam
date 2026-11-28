import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace as S, ModuleType
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bag_topics import REPLAY_TOPICS,RECORD_TOPICS,replay_command
from replay_inputs import pose_matrix, extract_seed


class ReplayContractTests(unittest.TestCase):
    def test_record_contains_inputs_and_evidence_but_replay_is_inputs_only(self):
        self.assertTrue(set(REPLAY_TOPICS)<=set(RECORD_TOPICS))
        for forbidden in ('/tf','/tf_static','/initialpose','/goal_pose','/localization/valid','/cmd_vel'):
            self.assertNotIn(forbidden,REPLAY_TOPICS)
        self.assertFalse(any('/api/' in topic or '/control/' in topic for topic in REPLAY_TOPICS))
    def test_all_map_names_share_same_safe_player(self):
        for directory in ('/recording/orchard/bag','/recording/indoor/bag'):
            command=replay_command(directory,.5)
            self.assertEqual(command[3],directory)
            self.assertEqual(command[command.index('--topics')+1:],list(REPLAY_TOPICS))
            self.assertIn('--clock',command)
            self.assertNotIn('--loop',command)
        for rate in (0,.05,1.5):
            with self.assertRaises(ValueError):replay_command('/bag',rate)
    def test_recorded_correction_composes_full_pose_without_flattening(self):
        map_from_odom=pose_matrix([2,3,4],[0,0,np.sin(np.pi/4),np.cos(np.pi/4)])
        odom_from_body=pose_matrix([1,0,.3],[0,0,0,1])
        np.testing.assert_allclose((map_from_odom@odom_from_body)[:3,3],[2,4,4.3])
        with self.assertRaises(ValueError):pose_matrix([0,0,0],[0,0,0,0])
    def fake_seed(self, valid=True, correction_age=0):
        def stamp(ns):return S(sec=ns//10**9,nanosec=ns%10**9)
        q=S(x=0.,y=0.,z=0.,w=1.)
        transform=S(header=S(frame_id='map',stamp=stamp(5*10**9-correction_age)),
                    child_frame_id='odom',transform=S(translation=S(x=2.,y=3.,z=4.),rotation=q))
        odom=S(header=S(frame_id='odom',stamp=stamp(5*10**9)),child_frame_id='base_link',
               pose=S(pose=S(position=S(x=1.,y=0.,z=.3),orientation=q)))
        rows=[('/localization/valid',S(data=valid),5*10**9),
              ('/tf',S(transforms=[transform]),5*10**9),('/odom',odom,5*10**9)]
        class Reader:
            def open(self,*args):self.rows=iter(rows);self.remaining=len(rows)
            def get_all_topics_and_types(self):return [S(name=t,type='test') for t,_,_ in rows]
            def set_filter(self,*args):pass
            def has_next(self):return self.remaining>0
            def read_next(self):self.remaining-=1;return next(self.rows)
        bag=ModuleType('rosbag2_py');bag.SequentialReader=Reader
        bag.StorageOptions=lambda **kw:None;bag.ConverterOptions=lambda *a:None;bag.StorageFilter=lambda **kw:None
        serialization=ModuleType('rclpy.serialization');serialization.deserialize_message=lambda data,typ:data
        util=ModuleType('rosidl_runtime_py.utilities');util.get_message=lambda name:None
        with patch.dict(sys.modules,{'rosbag2_py':bag,'rclpy.serialization':serialization,
                                      'rosidl_runtime_py.utilities':util}):
            return extract_seed('/recording/bag')
    def test_seed_requires_fresh_accepted_correction_and_preserves_elevation(self):
        seed=self.fake_seed()
        self.assertEqual(seed['stamp_ns'],5*10**9)
        np.testing.assert_allclose(np.array(seed['matrix'])[:3,3],[3,3,4.3])
        self.assertIsNone(self.fake_seed(valid=False))
        self.assertIsNone(self.fake_seed(correction_age=400000000))
    def test_offline_launcher_cannot_join_live_dds_or_start_robot_adapter(self):
        shell=(ROOT/'replay.sh').read_text()
        self.assertIn('--network bridge',shell)
        self.assertIn('ROS_LOCALHOST_ONLY=1',shell)
        self.assertIn('ROS_DOMAIN_ID=77',shell)
        launch=(ROOT/'replay.launch.py').read_text()
        for forbidden in ('go2_control_adapter.py','control_bridge.py','nav_sensors.py','operator_bridge.py'):
            self.assertNotIn(forbidden,launch)
        self.assertIn("'use_sim_time':True",launch)
        self.assertIn("'use_sim_time':'true'",launch)


if __name__=='__main__':unittest.main()
