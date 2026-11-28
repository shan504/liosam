"""Actual SQLite/PCD snapshots for newly surveyed mapping strips; no ROS needed."""
import csv
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

NOETIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NOETIC))
sys.path.insert(0, str(NOETIC.parent/'navigation'))
sys.path.insert(0, str(NOETIC.parent/'navigation/original_go2/go2_3d'))
from save_navigation_dense import export_dense
import save_navigation_dense
from corridor_export import pose_matrix, continuous_paths
from prepare_map import rigid, quaternion, apply, point_array
from surveyed_corridor import POLICY, read_document
from terrain import TerrainModel


def values(matrix):
    return [*matrix[:3,3], *quaternion(matrix[:3,:3])]


def fixture(root, name='lane', yaw=0., shift=(0.,0.,0.), marked=True, pitch=0.):
    root=Path(root)
    stage=root/name
    stage.mkdir()
    (stage/'calibration').mkdir()
    source_from_map=(rigid(shift,[0,0,math.sin(yaw/2),math.cos(yaw/2)]) @
                     rigid([0,0,0],[0,math.sin(pitch/2),0,math.cos(pitch/2)]))
    mount=rigid([.215,0,.25],[0,math.sin(math.pi/12),0,math.cos(math.pi/12)])
    (stage/'calibration/floor.json').write_text(json.dumps(dict(frame='odom',ground_frame='go2_ground',
        origin=list(shift),rotation=source_from_map[:3,:3].tolist())))
    (stage/'calibration/body_from_lidar.json').write_text(json.dumps(dict(parent='base_link',child='livox_frame',
        xyz=[.215,0,.25],xyzw=[0,math.sin(math.pi/12),0,math.cos(math.pi/12)])))
    database=root/(name+'.sqlite')
    db=sqlite3.connect(database)
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('CREATE TABLE scans(stamp INTEGER PRIMARY KEY,anchor INTEGER,points BLOB);'
        'CREATE TABLE poses(stamp INTEGER PRIMARY KEY,pose TEXT);'
        'CREATE TABLE meta(name TEXT PRIMARY KEY,value TEXT);')
    db.execute("INSERT INTO meta VALUES('frame','odom')")
    bodies=[source_from_map@rigid([x,0,.37],[0,0,0,1]) for x in (0.,2.)]
    poses={1000:bodies[0]@mount,2000:bodies[1]@mount}
    for stamp,pose in poses.items():
        db.execute('INSERT INTO poses VALUES(?,?)',(stamp,json.dumps(values(pose))))
    x,y=np.meshgrid(np.arange(-1.,5.,.05)+.025,np.arange(-1.,1.,.05)+.025)
    ground=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    # Farther scan sees these before the later body reaches x=2. They must be
    # removed retrospectively, regardless of height, after confirming that strip.
    scan0=np.vstack((ground,[[2.25,.1,8.],[2.25,-.1,-1.],[3.3,.1,8.],[2.5,.7,.5]]))
    scan1=np.array([[2.25,.1,0.],[2.75,-.1,.3],[2.5,0.,4.]])
    scan2=np.array([[4.2,.2,.5],[4.2,-.2,4.]])
    originals=[]
    for stamp,anchor,scan in [(1000,1000,scan0),(2000,2000,scan1),(2100,2000,scan2)]:
        world=np.column_stack((apply(scan,source_from_map),np.arange(len(scan))+stamp)).astype('<f4')
        local=world.copy()
        local[:,:3]=apply(world,np.linalg.inv(poses[anchor]))
        db.execute('INSERT INTO scans VALUES(?,?,?)',(stamp,anchor,local.astype('<f4').tobytes()))
        # Match float32 archive reconstruction instead of idealized input XYZ.
        reconstructed=local.copy()
        reconstructed[:,:3]=apply(local,poses[anchor])
        originals.append(reconstructed)
    if marked:
        db.execute('CREATE TABLE corridor_events(stamp INTEGER PRIMARY KEY,anchor INTEGER,body_relative TEXT)')
        db.execute("INSERT INTO meta VALUES('surveyed_corridor',?)",(json.dumps(dict(enabled=True,policy=POLICY,
            mount=dict(parent='base_link',child='livox_frame',xyz=[.215,0,.25],
                       xyzw=[0,math.sin(math.pi/12),0,math.cos(math.pi/12)]))),))
        db.execute('INSERT INTO corridor_events VALUES(?,?,?)',(2000,2000,json.dumps(np.linalg.inv(mount).tolist())))
    db.commit()
    return database,stage,db,poses,bodies,originals


def independently_inside(points, body):
    forward=body[:2,0]/np.linalg.norm(body[:2,0])
    left=np.array([-forward[1],forward[0]])
    delta=points[:,:2]-body[:2,3]
    x,y=delta@forward,delta@left
    return (x>=-1e-9)&(x<=1+1e-9)&(np.abs(y)<=.5+1e-9)


class CorridorExport(unittest.TestCase):
    def test_two_maps_filter_single_pass_without_terrain_and_preserve_index(self):
        for name,yaw,shift in [('east',0.,(0.,0.,0.)),('west',.73,(-23.,41.,.4))]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                database,stage,db,poses,bodies,raw=fixture(tmp,name,yaw,shift)
                before=db.execute('SELECT stamp,anchor,points FROM scans ORDER BY stamp').fetchall()
                with patch('terrain.build_terrain', side_effect=AssertionError('save must not fit terrain')), \
                        patch.object(TerrainModel, 'save', side_effect=AssertionError('save must not write terrain')), \
                        patch.object(save_navigation_dense, 'world_scans', wraps=save_navigation_dense.world_scans) as replay:
                    result=export_dense(database,stage/(name+'.pcd'))
                self.assertEqual(replay.call_count,1)
                self.assertFalse((stage/'survey_terrain.npz').exists())
                self.assertFalse(result['terrain_fitting'])
                document=read_document(stage/'surveyed_corridor.json',expected_frame='odom')
                self.assertEqual(len(document['poses']),1)
                np.testing.assert_allclose(document['poses'][0]['body_pose'],bodies[1],atol=1e-7)
                expected=[scan[~independently_inside(scan,bodies[1])] for scan in raw]
                actual=point_array(stage/(name+'.pcd'))
                np.testing.assert_allclose(actual,np.vstack(expected),atol=4e-6)
                self.assertEqual(result['points'],sum(map(len,expected)))
                self.assertEqual(result['source_points'],sum(map(len,raw)))
                self.assertEqual(result['cleared_points'],sum(map(len,raw))-len(actual))
                self.assertEqual(result['payload_bytes'],len(actual)*16)
                self.assertEqual(result['scans'],3)
                with (stage/'scans.csv').open() as stream:
                    rows=list(csv.DictReader(stream))
                self.assertEqual([int(row['point_count']) for row in rows],[len(expected[0]),0,len(expected[2])])
                self.assertEqual([int(row['first_point_index']) for row in rows],[0,len(expected[0]),len(expected[0])])
                self.assertEqual(db.execute('SELECT stamp,anchor,points FROM scans ORDER BY stamp').fetchall(),before)
                self.assertEqual(document['version'],2)
                self.assertNotIn('terrain',document)
                descriptor=document['calibration']
                self.assertAlmostEqual(descriptor['body_height_above_ground'],.37)
                np.testing.assert_allclose(descriptor['map_from_source_map'],np.linalg.inv(
                    rigid(shift,[0,0,math.sin(yaw/2),math.cos(yaw/2)])),atol=1e-7)
                self.assertEqual(len(list(stage.glob('*.pcd'))),1)
                db.close()

    def test_legacy_and_empty_event_sessions_do_not_invent_clearance(self):
        for marked in (False,True):
            with self.subTest(marked=marked), tempfile.TemporaryDirectory() as tmp:
                database,stage,db,poses,bodies,raw=fixture(tmp,marked=marked)
                if marked:
                    db.execute('DELETE FROM corridor_events');db.commit()
                result=export_dense(database,stage/'lane.pcd')
                np.testing.assert_allclose(point_array(stage/'lane.pcd'),np.vstack(raw),atol=2e-6)
                self.assertEqual(result['points'],sum(map(len,raw)))
                self.assertFalse((stage/'survey_terrain.npz').exists())
                if marked:
                    self.assertEqual(read_document(stage/'surveyed_corridor.json')['poses'],[])
                    self.assertEqual(result['cleared_points'],0)
                else:
                    self.assertNotIn('surveyed_corridor',result)
                    self.assertFalse((stage/'surveyed_corridor.json').exists())
                db.close()

    def test_loop_correction_moves_points_and_clearance_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            database,stage,db,poses,bodies,raw=fixture(tmp)
            correction=rigid([-.25,.15,0.],[0,0,math.sin(.1),math.cos(.1)])
            for stamp,pose in poses.items():
                db.execute('UPDATE poses SET pose=? WHERE stamp=?',(json.dumps(values(correction@pose)),stamp))
            db.commit()
            # A shared planar correction of all anchors keeps ground support.
            result=export_dense(database,stage/'lane.pcd')
            event=read_document(stage/'surveyed_corridor.json')['poses'][0]
            np.testing.assert_allclose(event['body_pose'],correction@bodies[1],atol=1e-6)
            clouds=[]
            for scan in raw:
                changed=scan.copy();changed[:,:3]=apply(scan,correction)
                clouds.append(changed[~independently_inside(changed,correction@bodies[1])])
            np.testing.assert_allclose(point_array(stage/'lane.pcd'),np.vstack(clouds),atol=3e-6)
            self.assertGreater(result['cleared_points'],0)
            db.close()

    def test_tilted_lio_world_uses_horizontal_body_strip_at_every_height(self):
        with tempfile.TemporaryDirectory() as tmp:
            database,stage,db,poses,bodies,raw=fixture(tmp,yaw=.7,shift=(5.,-3.,1.),pitch=math.pi/6)
            result=export_dense(database,stage/'lane.pcd')
            document=read_document(stage/'surveyed_corridor.json')
            ground=np.asarray(document['calibration']['map_from_source_map'])
            ground_body=ground@bodies[1]
            expected=[]
            wrong_source_removed=0
            for scan in raw:
                inside=independently_inside(apply(scan,ground),ground_body)
                wrong_source_removed+=np.count_nonzero(independently_inside(scan,bodies[1])!=inside)
                expected.append(scan[~inside])
            self.assertGreater(wrong_source_removed,0)
            np.testing.assert_allclose(point_array(stage/'lane.pcd'),np.vstack(expected),atol=4e-6)
            self.assertEqual(result['points'],sum(map(len,expected)))
            # High/low returns over the same confirmed strip are cleared; a
            # high return outside it remains despite the source-world tilt.
            original_intensities=set(np.vstack(raw)[:,3].tolist())
            kept_intensities=set(point_array(stage/'lane.pcd')[:,3].tolist())
            for intensity in (5800.,5801.):
                self.assertIn(intensity,original_intensities)
                self.assertNotIn(intensity,kept_intensities)
            self.assertIn(5802.,kept_intensities)
            db.close()

    def test_bad_anchor_transform_mount_or_missing_calibration_publishes_nothing(self):
        for problem in ('anchor','matrix','nonfinite','mount','calibration'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                database,stage,db,poses,bodies,raw=fixture(tmp)
                if problem=='anchor':
                    db.execute('UPDATE corridor_events SET anchor=1999')
                elif problem=='matrix':
                    db.execute('UPDATE corridor_events SET body_relative=?',(json.dumps(np.zeros((4,4)).tolist()),))
                elif problem=='nonfinite':
                    relative=np.eye(4);relative[0,3]=float('nan')
                    db.execute('UPDATE corridor_events SET body_relative=?',(json.dumps(relative.tolist()),))
                elif problem=='mount':
                    value=json.loads(db.execute("SELECT value FROM meta WHERE name='surveyed_corridor'").fetchone()[0])
                    value['mount']['xyz'][0]+=.1
                    db.execute("UPDATE meta SET value=? WHERE name='surveyed_corridor'",(json.dumps(value),))
                else:
                    (stage/'calibration/floor.json').unlink()
                db.commit()
                with self.assertRaises(ValueError):
                    export_dense(database,stage/'lane.pcd')
                for filename in ('lane.pcd','scans.csv','keyframes.csv','surveyed_corridor.json','survey_terrain.npz'):
                    self.assertFalse((stage/filename).exists())
                db.close()

    def test_wal_changes_after_snapshot_do_not_mix_scan_events_or_poses(self):
        with tempfile.TemporaryDirectory() as tmp:
            database,stage,db,poses,bodies,raw=fixture(tmp)
            original=save_navigation_dense.prepare_corridor
            def mutate_writer(reader,*args):
                # export_dense has already read snapshot metadata and poses.
                db.execute('UPDATE poses SET pose=? WHERE stamp=2000',
                    (json.dumps(values(rigid([100,0,.37],[0,0,0,1]))),))
                db.execute('DELETE FROM corridor_events')
                db.commit()
                return original(reader,*args)
            with patch.object(save_navigation_dense,'prepare_corridor',side_effect=mutate_writer):
                result=export_dense(database,stage/'lane.pcd')
            self.assertEqual(result['corridor_events'],1)
            np.testing.assert_allclose(read_document(stage/'surveyed_corridor.json')['poses'][0]['body_pose'],bodies[1],atol=1e-6)
            db.close()

    def test_save_connects_sparse_confirmations_along_curved_measured_keyframes(self):
        with tempfile.TemporaryDirectory() as tmp:
            database,stage,db,_,_,_=fixture(tmp)
            mount=rigid([.215,0,.25],[0,math.sin(math.pi/12),0,math.cos(math.pi/12)])
            trace=[(0.,0.,0.),(.5,0.,0.),(1.,0.,0.),(1.5,0.,0.),(2.,0.,math.pi/2),
                   (2.,.5,math.pi/2),(2.,1.,math.pi/2),(2.,1.5,math.pi/2),
                   (2.,2.,math.pi),(1.5,2.,math.pi),(1.,2.,math.pi),(.5,2.,math.pi),(0.,2.,math.pi)]
            db.execute('DELETE FROM scans');db.execute('DELETE FROM poses');db.execute('DELETE FROM corridor_events')
            cloud=np.array([[1.,0.,7.,10.],[2.,1.,-2.,11.],[1.,2.,.3,12.],
                            [1.,1.,.3,20.],[4.,4.,1.,21.]],dtype='<f4')
            for i,(x,y,yaw) in enumerate(trace):
                stamp=(i+1)*1_000_000_000
                body=rigid([x,y,.37],[0,0,math.sin(yaw/2),math.cos(yaw/2)])
                lidar=body@mount
                db.execute('INSERT INTO poses VALUES(?,?)',(stamp,json.dumps(values(lidar))))
                local=cloud.copy();local[:,:3]=apply(cloud,np.linalg.inv(lidar))
                db.execute('INSERT INTO scans VALUES(?,?,?)',(stamp,stamp,local.astype('<f4').tobytes()))
                if i in (0,len(trace)-1):
                    db.execute('INSERT INTO corridor_events VALUES(?,?,?)',
                        (stamp,stamp,json.dumps(np.linalg.inv(mount).tolist())))
            db.commit()
            result=export_dense(database,stage/'lane.pcd')
            doc=read_document(stage/'surveyed_corridor.json')
            self.assertEqual(result['corridor_events'],2)
            self.assertEqual(result['corridor_paths'],1)
            self.assertEqual(result['corridor_links'],len(trace)-1)
            self.assertEqual([event['stamp_ns'] for event in doc['paths'][0]],
                             [(i+1)*1_000_000_000 for i in range(len(trace))])
            actual=point_array(stage/'lane.pcd')
            # Three levels of ghost returns on each measured road leg vanish;
            # the U-shaped interior is not cleared by a start/end shortcut.
            self.assertEqual(set(actual[:,3]),{20.,21.})
            self.assertEqual(result['cleared_points'],3*len(trace))
            self.assertFalse((stage/'survey_terrain.npz').exists())
            db.close()

    def test_missing_scans_tracking_jumps_and_reverse_split_only_confirmed_paths(self):
        for problem in ('outage','jump','reverse'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                database,stage,db,_,_,_=fixture(tmp)
                db.execute('DELETE FROM scans')
                stamps=[1_000_000_000,2_000_000_000,3_000_000_000,4_000_000_000]
                coordinates=[0.,.5,1.,1.5]
                if problem=='outage':
                    stamps=[1_000_000_000,2_000_000_000,8_000_000_000,9_000_000_000]
                elif problem=='jump':
                    coordinates=[0.,.5,20.,20.5]
                else:
                    coordinates=[0.,.5,0.,.5]
                poses={}
                events=[]
                for stamp,x in zip(stamps,coordinates):
                    body=rigid([x,0.,.37],[0,0,0,1])
                    poses[stamp]=values(body)
                    events.append(dict(stamp_ns=stamp,anchor_stamp_ns=stamp,body_pose=body.tolist()))
                    db.execute('INSERT INTO scans VALUES(?,?,?)',(stamp,stamp,b'0'*16))
                paths=continuous_paths(db,poses,events,np.eye(4),np.eye(4))
                self.assertEqual([[e['stamp_ns'] for e in path] for path in paths],
                                 [stamps[:2],stamps[2:]])
                # Internal, unconfirmed frames after a break cannot start a
                # clearing path by themselves.
                sparse=continuous_paths(db,poses,[events[0],events[-1]],np.eye(4),np.eye(4))
                self.assertEqual([[e['stamp_ns'] for e in path] for path in sparse],
                                 [[stamps[0]],[stamps[-1]]])
                db.close()


if __name__=='__main__':
    unittest.main()
