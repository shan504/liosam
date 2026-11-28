"""Exact-stamp surveyed-corridor events; synthetic geometry, no robot motion."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from corridor_recorder import CorridorRecorder, mount_from_parameter, pose_matrix
from dense_map import Archive, preview_corridor_mask
from surveyed_corridor import CorridorIndex, POLICY

MOUNT = dict(parent='base_link', child='livox_frame', xyz=[.215, 0., .25],
             xyzw=[0., math.sin(math.pi/12), 0., math.cos(math.pi/12)])


def body_pose(x, y=0., yaw=0., z=.4):
    return np.array([x,y,z,0.,0.,math.sin(yaw/2),math.cos(yaw/2)])


def lidar_pose(body):
    # Independent Z-yaw body * Y-pitch installation quaternion composition.
    yaw = 2*math.atan2(body[5],body[6])
    c, s = math.cos(yaw), math.sin(yaw)
    position = body[:3] + np.array([c*.215,s*.215,.25])
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    cp, sp = math.cos(math.pi/12), math.sin(math.pi/12)
    return np.r_[position,[-sy*sp,cy*sp,sy*cp,cy*cp]]


def offer(recorder, stamp, body, *, anchor=1_000_000_000, anchor_capture=None,
          now_ns=None, receipt=0., world_frame='odom', motion_frame='go2_native/odom'):
    now_ns = stamp if now_ns is None else now_ns
    anchor_capture = lidar_pose(body_pose(0.)) if anchor_capture is None else anchor_capture
    recorder.scan(stamp,anchor,world_frame,anchor_capture,now_ns=now_ns,receipt=receipt)
    recorder.pose('world',stamp,world_frame,'odom_mapping',lidar_pose(body),now_ns=now_ns)
    recorder.pose('motion',stamp,motion_frame,'livox_frame',lidar_pose(body),now_ns=now_ns)
    return recorder.process(now_ns=now_ns,now=receipt)


class Recorder(unittest.TestCase):
    def test_mount_lever_arm_and_exact_forward_body_motion(self):
        recorder = CorridorRecorder(MOUNT)
        self.assertEqual(offer(recorder,1_000_000_000,body_pose(0.)),[])
        events=offer(recorder,1_200_000_000,body_pose(.06))
        self.assertEqual(len(events),1)
        stamp,anchor,relative=events[0]
        self.assertEqual((stamp,anchor),(1_200_000_000,1_000_000_000))
        expected=pose_matrix(body_pose(.06))
        np.testing.assert_allclose(pose_matrix(lidar_pose(body_pose(0.)))@relative,expected,atol=1e-12)
        self.assertEqual(recorder.confirmed,1)

    def test_rotation_sideways_reverse_stationary_gap_and_duplicate_do_not_clear(self):
        cases=[(body_pose(0.),body_pose(0.,yaw=.7),.2),
               (body_pose(0.),body_pose(0.,y=.06),.2),
               (body_pose(0.),body_pose(-.06),.2),
               (body_pose(0.),body_pose(.004),.2),
               (body_pose(0.),body_pose(.06),.8)]
        for first,second,dt in cases:
            with self.subTest(second=second,dt=dt):
                recorder=CorridorRecorder(MOUNT)
                offer(recorder,1_000_000_000,first)
                self.assertEqual(offer(recorder,1_000_000_000+int(dt*1e9),second),[])
        recorder=CorridorRecorder(MOUNT)
        offer(recorder,1_000_000_000,body_pose(0.))
        self.assertEqual(len(offer(recorder,1_200_000_000,body_pose(.06))),1)
        self.assertEqual(offer(recorder,1_200_000_000,body_pose(.09)),[])
        self.assertFalse(recorder.scan(1_100_000_000,1_000_000_000,'odom',lidar_pose(body_pose(0.)),now_ns=1_200_000_000))

    def test_late_exact_messages_wait_in_order_without_latest_pose_substitution(self):
        recorder=CorridorRecorder(MOUNT)
        for stamp,x in [(1_000_000_000,0.),(1_200_000_000,.06)]:
            recorder.scan(stamp,1_000_000_000,'odom',lidar_pose(body_pose(0.)),now_ns=1_200_000_000,receipt=0.)
            recorder.pose('world',stamp,'odom','odom_mapping',lidar_pose(body_pose(x)),now_ns=1_200_000_000)
        recorder.pose('motion',1_200_000_000,'go2_native/odom','livox_frame',lidar_pose(body_pose(.06)),now_ns=1_200_000_000)
        self.assertEqual(recorder.process(now_ns=1_200_000_000,now=.1),[])
        recorder.pose('motion',1_000_000_000,'go2_native/odom','livox_frame',lidar_pose(body_pose(0.)),now_ns=1_200_000_000)
        events=recorder.process(now_ns=1_200_000_000,now=.2)
        self.assertEqual([event[0] for event in events],[1_200_000_000])

    def test_stale_missing_malformed_and_wrong_frames_produce_no_events(self):
        recorder=CorridorRecorder(MOUNT)
        offer(recorder,1_000_000_000,body_pose(0.))
        recorder.scan(1_200_000_000,1_000_000_000,'odom',lidar_pose(body_pose(0.)),now_ns=1_200_000_000,receipt=0.)
        recorder.pose('world',1_200_000_000,'odom','odom_mapping',lidar_pose(body_pose(.06)),now_ns=1_200_000_000)
        self.assertFalse(recorder.pose('motion',1_200_000_000,'wrong','livox_frame',lidar_pose(body_pose(.06)),now_ns=1_200_000_000))
        self.assertFalse(recorder.pose('motion',1_200_000_000,'go2_native/odom','wrong',lidar_pose(body_pose(.06)),now_ns=1_200_000_000))
        self.assertEqual(recorder.process(now_ns=2_300_000_000,now=1.),[])
        self.assertEqual(recorder.expired,1)
        self.assertEqual(offer(recorder,2_400_000_000,body_pose(.12)),[])
        self.assertFalse(recorder.pose('world',1_000_000_000,'odom','odom_mapping',lidar_pose(body_pose(1.)),now_ns=3_000_000_000))
        self.assertFalse(recorder.pose('world',3_100_000_000,'odom','odom_mapping',[0,0,0,0,0,0,0],now_ns=3_100_000_000))
        fresh=CorridorRecorder(MOUNT)
        self.assertEqual(offer(fresh,1_000_000_000,body_pose(0.)),[])
        fresh.scan(1_200_000_000,1_000_000_000,'odom',lidar_pose(body_pose(0.)),now_ns=1_200_000_000,receipt=0.)
        fresh.pose('world',1_200_000_000,'different_map','odom_mapping',lidar_pose(body_pose(.06)),now_ns=1_200_000_000)
        fresh.pose('motion',1_200_000_000,'go2_native/odom','livox_frame',lidar_pose(body_pose(.06)),now_ns=1_200_000_000)
        self.assertEqual(fresh.process(now_ns=1_200_000_000,now=.2),[])

    def test_same_native_or_mapping_source_requires_matching_world_frame(self):
        for frame,child in [('odom','odom_mapping'),('go2_native/odom','livox_frame')]:
            with self.subTest(frame=frame):
                recorder=CorridorRecorder(MOUNT,same_pose_source=True)
                results=[]
                for stamp,x in [(1_000_000_000,0.),(1_200_000_000,.06)]:
                    recorder.scan(stamp,1_000_000_000,frame,lidar_pose(body_pose(0.)),now_ns=stamp,receipt=0.)
                    for source in ('world','motion'):
                        recorder.pose(source,stamp,frame,child,lidar_pose(body_pose(x)),now_ns=stamp)
                    results.extend(recorder.process(now_ns=stamp,now=0.))
                self.assertEqual(len(results),1)

    def test_archive_events_follow_matching_preview_anchor_and_keep_raw_points(self):
        with tempfile.TemporaryDirectory() as temp:
            archive=Archive(Path(temp))
            archive.enable_corridor(MOUNT)
            original=lidar_pose(body_pose(0.))
            archive.path('odom',{1_000_000_000:tuple(original)})
            points=np.array([[.5,0,0,10],[2.,0,0,20]],dtype='<f4')
            archive.add(1_200_000_000,1_000_000_000,points)
            recorder=CorridorRecorder(MOUNT)
            offer(recorder,1_000_000_000,body_pose(0.))
            event=offer(recorder,1_200_000_000,body_pose(.06))[0]
            archive.add_corridor(*event)
            archive.db.commit()
            marker=json.loads(archive.db.execute("SELECT value FROM meta WHERE name='surveyed_corridor'").fetchone()[0])
            self.assertEqual(marker['policy'],POLICY)
            self.assertEqual(marker['mount'],MOUNT)
            raw=archive.db.execute('SELECT points FROM scans').fetchone()[0]
            self.assertEqual(raw,points.tobytes())
            old_events=archive.corridor_events({1_000_000_000:tuple(original)})
            np.testing.assert_allclose(old_events[0]['body_pose'],pose_matrix(body_pose(.06)),atol=1e-12)
            corrected=lidar_pose(body_pose(10.,y=20.,yaw=math.pi/2))
            archive.path('odom',{1_000_000_000:tuple(corrected)})
            old_active=archive.corridor_events({1_000_000_000:tuple(original)})
            np.testing.assert_allclose(old_active[0]['body_pose'],old_events[0]['body_pose'])
            new_events=archive.corridor_events(archive.poses)
            expected=pose_matrix(corrected)@np.linalg.inv(pose_matrix(original))@pose_matrix(body_pose(.06))
            np.testing.assert_allclose(new_events[0]['body_pose'],expected,atol=1e-12)
            self.assertTrue(CorridorIndex([new_events[0]['body_pose']]).contains([[10.,20.5,50.]])[0])
            self.assertEqual(archive.corridor_events({}),[])
            with self.assertRaisesRegex(ValueError,'exact archived scan'):
                archive.add_corridor(99,1_000_000_000,np.eye(4))
            with self.assertRaisesRegex(ValueError,'changed'):
                archive.enable_corridor(dict(MOUNT,xyz=[.3,0.,.25]))
            archive.db.close()

    def test_preview_uses_ground_horizontal_rectangle_in_tilted_source_map(self):
        ground_from_source=pose_matrix([4.,-3.,.7,0.,math.sin(math.pi/12),0.,math.cos(math.pi/12)])
        source_from_ground=np.linalg.inv(ground_from_source)
        body_ground=pose_matrix(body_pose(1.,y=2.,yaw=.6))
        body_source=source_from_ground@body_ground
        local=np.array([[.5,0.,0.],[.5,0.,9.],[1.1,0.,0.],[.5,.6,0.]])
        ground_xyz=local@body_ground[:3,:3].T+body_ground[:3,3]
        source_xyz=ground_xyz@source_from_ground[:3,:3].T+source_from_ground[:3,3]
        points=np.column_stack((source_xyz,[1.,2.,3.,4.]))
        before=points.copy()
        selected=preview_corridor_mask(points,[dict(body_pose=body_source.tolist())],ground_from_source)
        np.testing.assert_array_equal(selected,[True,True,False,False])
        np.testing.assert_array_equal(points,before)


if __name__=='__main__':
    unittest.main()
