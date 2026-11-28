"""Restore carved bundles from their exact saved scan/pose snapshot, without ROS."""
import csv
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_corridor_export import fixture, independently_inside, values
from save_navigation_dense import export_dense, pcd_header
from save_navigation_map import write_checksums
from restore_map_geometry import restore
from prepare_map import point_array, rigid


def old_bundle(tmp, name, yaw):
    tmp = Path(tmp)
    source = tmp / 'source'
    source.mkdir()
    database, stage, db, poses, bodies, raw = fixture(source, name, yaw, (7., -11., .4))
    result = export_dense(database, stage / (name+'.pcd'))
    # Reproduce the previous exporter: remove full-height measured returns from
    # the only PCD, while raw scans remain in the recorded run archive.
    kept = [scan[~independently_inside(scan, bodies[1])] for scan in raw]
    cloud = np.vstack(kept)
    (stage / (name+'.pcd')).write_bytes(pcd_header(len(cloud)) + cloud.astype('<f4').tobytes())
    with (stage/'scans.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('stamp_ns','anchor_stamp_ns','first_point_index','point_count'))
        offset = 0
        for (stamp, anchor), scan in zip([(1000,1000),(2000,2000),(2100,2000)], kept):
            writer.writerow((stamp,anchor,offset,len(scan)))
            offset += len(scan)
    result.pop('pc_geometry'); result.pop('corridor_application')
    result.update(points=len(cloud), payload_bytes=16*len(cloud),
                  cleared_points=sum(map(len,raw))-len(cloud))
    db.close()
    archive = tmp/'run'/'dense'
    archive.mkdir(parents=True)
    database.rename(archive/'scans.sqlite')
    manifest = dict(format_version=1,name=name,map_file=name+'.pcd',dense=result,
                    completed_at='2026-01-01T00:00:00+00:00',
                    source_run=dict(source_archive_dir=str(archive)),
                    session=dict(archive_dir=str(archive),run_id='original_run'))
    (stage/'manifest.json').write_text(json.dumps(manifest))
    write_checksums(stage)
    parent = tmp/'maps'/name
    parent.mkdir(parents=True)
    bundle = parent/'20260101_old'
    stage.rename(bundle)
    return tmp/'maps', bundle, archive/'scans.sqlite', raw, poses


class GeometryRestore(unittest.TestCase):
    def test_two_names_restore_deleted_returns_and_keep_original_pose_snapshot(self):
        for name,yaw in [('orchard',0.),('yard',.83)]:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                root,bundle,database,raw,poses = old_bundle(tmp,name,yaw)
                original = {str(p.relative_to(bundle)):p.read_bytes() for p in bundle.rglob('*') if p.is_file()}
                with sqlite3.connect(database) as db:
                    # A run may have continued and loop-optimized after saving.
                    db.execute('UPDATE poses SET pose=? WHERE stamp=2000',
                               (json.dumps(values(rigid([100.,50.,2.],[0.,0.,0.,1.]))),))
                    db.execute('INSERT INTO scans VALUES(3000,2000,?)',
                               (np.array([[99.,0.,0.,99999.]],dtype='<f4').tobytes(),))
                with patch('terrain.build_terrain',side_effect=AssertionError('no terrain fit')):
                    restored = restore(name,root)
                self.assertNotEqual(restored,bundle)
                np.testing.assert_allclose(point_array(restored/(name+'.pcd')),np.vstack(raw),atol=4e-6)
                self.assertEqual(len(list(restored.glob('*.pcd'))),1)
                self.assertEqual((restored/'keyframes.csv').read_bytes(),original['keyframes.csv'])
                self.assertEqual((restored/'surveyed_corridor.json').read_bytes(),original['surveyed_corridor.json'])
                for path,data in original.items():self.assertEqual((bundle/path).read_bytes(),data)
                manifest=json.loads((restored/'manifest.json').read_text())
                self.assertEqual(manifest['dense']['pc_geometry'],'full_registered_returns')
                self.assertEqual(manifest['dense']['corridor_application'],'navigation_grid_only')
                self.assertEqual(manifest['dense']['scans'],3)
                self.assertEqual(manifest['dense']['points'],sum(map(len,raw)))
                self.assertEqual(restore(name,root),restored) # latest selection and idempotence
                self.assertFalse((restored/'survey_terrain.npz').exists())

    def test_bad_provenance_or_missing_raw_never_publishes_another_map(self):
        for problem in ('missing','anchor','frame','mount','checksum','path','point_count'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root,bundle,database,raw,poses=old_bundle(tmp,'sample',0.)
                if problem=='missing': database.unlink()
                elif problem=='checksum':
                    (bundle/'keyframes.csv').write_text((bundle/'keyframes.csv').read_text()+'\n')
                elif problem=='path':
                    manifest=json.loads((bundle/'manifest.json').read_text())
                    manifest['session']['archive_dir']=str(Path(tmp)/'unrelated')
                    (bundle/'manifest.json').write_text(json.dumps(manifest));write_checksums(bundle)
                else:
                    with sqlite3.connect(database) as db:
                        if problem=='anchor':db.execute('UPDATE scans SET anchor=1999 WHERE stamp=1000')
                        elif problem=='frame':db.execute("UPDATE meta SET value='another_world' WHERE name='frame'")
                        elif problem=='point_count':db.execute('UPDATE scans SET points=? WHERE stamp=2100',(b'\0'*48,))
                        else:
                            marker=json.loads(db.execute("SELECT value FROM meta WHERE name='surveyed_corridor'").fetchone()[0])
                            marker['mount']['xyz'][0]+=.2
                            db.execute("UPDATE meta SET value=? WHERE name='surveyed_corridor'",(json.dumps(marker),))
                with self.assertRaises((ValueError,KeyError)):
                    restore('sample',root)
                complete=[p for p in (root/'sample').iterdir() if not p.name.startswith('.')]
                self.assertEqual(complete,[bundle])


if __name__=='__main__':unittest.main()
