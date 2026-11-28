"""Old Go2 grid, calibrated body start and route identity survive map conversion."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_go2_map import current_settings, export_grid, prepare
from prepare_map import apply, point_array, quaternion, rigid, transforms
from mapping_store import MappingStore
from geometry import ObservedGrid
from map_start import start_pose_from_metadata
from route_store import map_identity


def write_archive(root, name, offset, tilt):
    src = root / name / 'version'
    (src / 'calibration').mkdir(parents=True)
    ground_to_saved = rigid([1., -1., .4], [math.sin(tilt/2), 0., 0., math.cos(tilt/2)])
    mount = rigid([.21, 0, .25], [0., math.sin(.25), 0., math.cos(.25)])
    angle = .2 + offset/3.
    body = rigid([offset, .8, .37], [0., 0., math.sin(angle/2), math.cos(angle/2)])
    lidar = ground_to_saved @ body @ mount
    xy = np.mgrid[-10:11, -5:6].reshape(2,-1).T * .05
    xy += [offset + .021, .822]
    # Genuine wall group, one isolated return for legacy denoising, and a
    # high overhang retained in the GICP prior but outside the 2D height band.
    wall = np.column_stack((xy[:11], np.full(11,.50)))
    floor = np.column_stack((xy, np.zeros(len(xy))))
    level = np.vstack((wall, floor, [offset+1., 1., .5], [offset+1., 2., 3.20]))
    scan = np.column_stack((apply(level, ground_to_saved), np.arange(len(level)))).astype('<f4')
    f = src / (name+'.pcd')
    n = len(scan)*2
    f.write_bytes((f'VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n'
                   f'COUNT 1 1 1 1\nWIDTH {n}\nHEIGHT 1\nPOINTS {n}\nDATA binary\n').encode()
                  + scan.tobytes()*2)
    files = dict(name=name, map_file=f.name, scan_index='scans.csv', optimized_keyframes='poses.csv',
                 dense=dict(frame_id='odom', points=n, scans=2))
    (src/'manifest.json').write_text(json.dumps(files))
    def calib(filename, values):
        (src/'calibration'/filename).write_text(json.dumps(values))
    calib('floor.json', dict(frame='odom', ground_frame='go2_ground',
                             origin=ground_to_saved[:3,3].tolist(), rotation=ground_to_saved[:3,:3].tolist()))
    calib('body_from_lidar.json', dict(parent='base_link', child='livox_frame', xyz=mount[:3,3].tolist(),
                                        xyzw=quaternion(mount[:3,:3]).tolist()))
    with (src/'poses.csv').open('w') as stream:
        w = csv.writer(stream)
        w.writerow(['stamp_ns','x','y','z','qx','qy','qz','qw'])
        w.writerow([10, *lidar[:3,3], *quaternion(lidar[:3,:3])])
    (src/'scans.csv').write_text('stamp_ns,anchor_stamp_ns,first_point_index,point_count\n'
                                 f'10,10,0,{len(scan)}\n20,10,{len(scan)},{len(scan)}\n')
    return src, level, body, body @ mount


class Go2MapPort(unittest.TestCase):
    def test_two_names_frames_original_grid_and_route_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fingerprints = set()
            for name,offset,tilt in [('room_a',-1.1,.0), ('survey_b',2.3,.12)]:
                src, level, body, _ = write_archive(root, name, offset, tilt)
                backup = hashlib.sha256((src/(name+'.pcd')).read_bytes()).digest()
                result = prepare(src, root/'cache')
                self.assertEqual(prepare(src, root/'cache'), result)
                self.assertEqual(backup, hashlib.sha256((src/(name+'.pcd')).read_bytes()).digest())
                yaml_doc = yaml.safe_load((result/'metadata.yaml').read_text())
                np.testing.assert_allclose(start_pose_from_metadata(yaml_doc), body, atol=1e-7)
                self.assertEqual(yaml_doc['exact_origin_scans'], 1)
                cloud = point_array(result/'map.pcd')
                raw = point_array(src/(name+'.pcd'))
                ground, _, _ = transforms(src, 'odom')
                xyz = apply(raw, ground).astype('<f4')
                ref_store = MappingStore(.02, .15)
                ref_store.commit(ref_store.prepare(xyz))
                expected = ref_store.snapshot()
                ordering = np.lexsort((expected[:,2],expected[:,1],expected[:,0]))
                np.testing.assert_allclose(cloud[:, :3], expected[ordering], atol=1e-6)
                self.assertGreaterEqual(cloud[:,2].max(), 3.19)
                self.assertEqual(yaml_doc['source_point_count'],2*len(level))
                self.assertEqual(yaml_doc['point_count'],len(cloud))
                self.assertEqual(len(cloud),len(level))
                np.testing.assert_array_equal(np.sort(cloud[:,3]), np.arange(len(level)))
                self.assertGreater(yaml_doc['navigation_denoise']['removed_cells'],0)
                # Count independently labelled fixture floor endpoints in the
                # 5 cm output lattice. Neither a coarse terrain patch nor a
                # line of sight through a fine cell supplies its second hit.
                resolution = current_settings()['grid_resolution']
                fixture_floor = np.tile(level[:,2] == 0., 2)
                hit_cells = np.floor(xyz[fixture_floor,:2]/resolution).astype(int)
                floor_hits = Counter(map(tuple, hit_cells))
                wall_cells = set(map(tuple, np.floor(xyz[:11,:2]/resolution).astype(int)))
                self.assertEqual(yaml_doc['ground_filter_mode'], 'initial_plane')
                self.assertEqual(yaml_doc['floor_rejection_height'], .15)
                self.assertNotIn('terrain_model', yaml_doc)
                def grid_at(directory):
                    nav = yaml.safe_load((directory/'nav.yaml').read_text())
                    with (directory/nav['image']).open('rb') as stream:
                        self.assertEqual(stream.readline(), b'P5\n')
                        w, h = map(int, stream.readline().split())
                        self.assertEqual(stream.readline(), b'255\n')
                        pixels = np.frombuffer(stream.read(), 'u1').reshape(h, w)
                    origin = np.rint(np.array(nav['origin'][:2])/nav['resolution']).astype(int)
                    cells = np.indices((w,h)).reshape(2,-1).T + origin
                    return cells, pixels[::-1].T.reshape(-1)
                cells, labels = grid_at(result)
                actual = dict(zip(map(tuple, cells), labels))
                self.assertEqual(len(wall_cells), 11)
                for wall in wall_cells:
                    self.assertEqual(actual[wall], 0)
                self.assertTrue(np.any(labels == 254))
                np.testing.assert_array_equal(labels, [
                    0 if tuple(cell) in wall_cells else
                    254 if floor_hits[tuple(cell)] >= 2 else 205
                    for cell in cells])
                for cell in cells[labels == 254]:
                    self.assertGreaterEqual(floor_hits[tuple(cell)], 2)
                ident = map_identity(result)
                self.assertEqual(ident['label'],name)
                fingerprints.add(ident['fingerprint'])
            self.assertEqual(len(fingerprints),2)

    def test_duplicate_floor_hits_on_opposite_sides_of_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cloud = np.zeros((131076, 4), dtype='<f4')
            cloud[:, 2] = 3.0
            cloud[0, :3] = cloud[131072, :3] = [-1.02, 0., .01]
            cloud[1, :3] = [-.52, .11, .01]   # One floor hit stays unknown.
            cloud[2, :3] = [1.07, .07, .2]
            grid = ObservedGrid(.05)
            model = dict(plane=[0., 0., 1., 0.])
            export_grid((cloud[:131072], cloud[131072:]), grid, root/'ported', current_settings(), model)
            reference = ObservedGrid(.05)
            reference.export(cloud[:, :3], root/'legacy', ground_model=model)
            self.assertEqual((root/'legacy/nav.pgm').read_bytes(),(root/'ported/nav.pgm').read_bytes())
            self.assertEqual(grid.free, reference.free)

    def test_dense_prior_fuses_across_chunks_and_keeps_distant_last_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, _, _, _ = write_archive(root,'dense_room',0.,0.)
            pcd = src/'dense_room.pcd'
            raw = point_array(pcd).copy()
            rows = np.tile(raw,(550,1))
            rows[:100,3] += 1.
            rows[-1] = [12.,-1.,5.43,185.]
            h = pcd.read_bytes().split(b'DATA binary\n',1)[0]
            h = h.replace(f'{len(raw)}\n'.encode(),f'{len(rows)}\n'.encode())
            pcd.write_bytes(h+b'DATA binary\n'+rows.tobytes())
            count = len(raw)//2
            n_scans = len(rows)//count
            index = ['stamp_ns,anchor_stamp_ns,first_point_index,point_count']
            index += [f'{10+i},10,{i*count},{count}' for i in range(n_scans)]
            (src/'scans.csv').write_text('\n'.join(index)+'\n')
            manifest = json.loads((src/'manifest.json').read_text())
            manifest['dense'].update(scans=n_scans,points=len(rows))
            (src/'manifest.json').write_text(json.dumps(manifest))
            dest = prepare(src, root/'cache')
            points = point_array(dest/'map.pcd')
            self.assertEqual(len(points), count+1)
            ground, _, _ = transforms(src,'odom')
            xyz = apply(rows[-1:],ground)[0]
            np.testing.assert_allclose(points[-1,:3],xyz,atol=1e-6)
            self.assertEqual(points[-1,3],185.)
            first = apply(rows[:1],ground)[0]
            where = np.argmin(np.linalg.norm(points[:,:3]-first,axis=1))
            self.assertAlmostEqual(points[where,3],1/n_scans,places=7)
            self.assertLess(len(points),len(rows)//100)

    def test_invalid_index_is_never_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, _, _, _ = write_archive(root,'bad',0.,0.)
            index = src/'scans.csv'
            index.write_text(index.read_text().replace('10,10,0,','10,10,1,'))
            with self.assertRaisesRegex(ValueError,'索引'):
                prepare(src, root/'cache')
            self.assertEqual(list((root/'cache').glob('original-go2-*')), [])


if __name__ == '__main__':
    unittest.main()

