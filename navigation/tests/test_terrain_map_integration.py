"""Sloping farm lanes retain ground, obstacles, gaps and full 3D localization XYZ."""
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
from prepare_go2_map import prepare
from prepare_map import apply, point_array, quaternion, rigid
from terrain import load_terrain


def elevation(x, rise):
    return .5 * rise * (1 - np.cos(2 * math.pi * x / 16.))


def make_lane(root, name, offset, rise, tilt):
    source = root / name / 'version'
    (source / 'calibration').mkdir(parents=True)
    saved_from_map = rigid([.45,-.60,.32], [math.sin(tilt/2),0.,0.,math.cos(tilt/2)])
    mount = rigid([.215,0.,.25], [0.,math.sin(math.pi/12),0.,math.cos(math.pi/12)])
    indices = np.indices((320,32)).reshape(2,-1).T
    xy = (indices + .5) * .05 - [0., .8]
    # A patch without ground echoes on the side of the road.
    gap = (xy[:,0] > 5.6) & (xy[:,0] < 6.8) & (xy[:,1] > 0.)
    xy = xy[~gap]
    floor = np.column_stack((xy, elevation(xy[:,0], rise)))
    obstacles = []
    for center_x in (3.,8.,12.):
        block = (np.indices((3,3)).reshape(2,-1).T + .5) * .05 + [center_x,.2]
        obstacles.append(np.column_stack((block,elevation(block[:,0],rise) + .55)))
    # One isolated return inside the gap: denoising must leave it unknown.
    isolated = [6.225,.425,1.95]
    samples = np.vstack([floor,*obstacles,isolated])
    samples[:,0] += offset
    poses, scan_index, chunks = [], [], []
    count = 0
    for i in range(16):
        start_x = offset + i
        patch = samples[(samples[:,0] >= start_x) & (samples[:,0] < start_x+1)]
        data = np.column_stack((apply(patch, saved_from_map), np.arange(len(patch)))).astype('<f4')
        # Two scans per patch provide two distinct floor hits at each endpoint.
        foot_x = i
        body = rigid([offset+foot_x,-.2,elevation(foot_x,rise)+.37], [0,0,0,1])
        lidar = saved_from_map @ body @ mount
        stamp = (i+1)*100
        poses.append([stamp,*lidar[:3,3],*quaternion(lidar[:3,:3])])
        for scan_stamp in (stamp,stamp+10):
            scan_index.append([scan_stamp,stamp,count,len(data)])
            count += len(data)
            chunks.append(data.tobytes())
    pcd = source / (name + '.pcd')
    header = (f'VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n'
              f'COUNT 1 1 1 1\nWIDTH {count}\nHEIGHT 1\nPOINTS {count}\nDATA binary\n')
    pcd.write_bytes(header.encode() + b''.join(chunks))
    for filename, keys, rows in [('scans.csv',['stamp_ns','anchor_stamp_ns','first_point_index','point_count'],scan_index),
                                ('poses.csv',['stamp_ns','x','y','z','qx','qy','qz','qw'],poses)]:
        with (source / filename).open('w',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(keys)
            writer.writerows(rows)
    (source / 'manifest.json').write_text(json.dumps(dict(name=name, map_file=pcd.name,
        scan_index='scans.csv', optimized_keyframes='poses.csv',
        dense=dict(frame_id='odom', points=count, scans=len(scan_index)))))
    (source / 'calibration/floor.json').write_text(json.dumps(dict(frame='odom',ground_frame='go2_ground',
        origin=saved_from_map[:3,3].tolist(),rotation=saved_from_map[:3,:3].tolist())))
    (source / 'calibration/body_from_lidar.json').write_text(json.dumps(dict(parent='base_link',
        child='livox_frame',xyz=mount[:3,3].tolist(),xyzw=quaternion(mount[:3,:3]).tolist())))
    return source, samples


class SlopingMapIntegration(unittest.TestCase):
    def test_two_graded_routes_cached_grid_and_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, offset, rise, tilt in [('east_lane',0.,1.6,.12), ('west_lane',-20.,1.3,-.08)]:
                with self.subTest(name=name):
                    source, samples = make_lane(root,name,offset,rise,tilt)
                    pcd = source / (name+'.pcd')
                    original = hashlib.sha256(pcd.read_bytes()).digest()
                    bundle = prepare(source,root/'cache')
                    meta = yaml.safe_load((bundle/'metadata.yaml').read_text())
                    terrain = load_terrain(bundle,meta)
                    nav = yaml.safe_load((bundle/'nav.yaml').read_text())
                    with (bundle/nav['image']).open('rb') as pgm:
                        self.assertEqual(pgm.readline(), b'P5\n')
                        w,h = map(int,pgm.readline().split())
                        self.assertEqual(pgm.readline(), b'255\n')
                        image = np.frombuffer(pgm.read(),'u1').reshape(h,w)[::-1]
                    def labels(points):
                        col,row = np.floor((points - nav['origin'][:2])/nav['resolution']).astype(int).T
                        return image[row,col]
                    # Slope ascent, summit and descent are all continuous navigable ground.
                    road = np.column_stack((offset + np.arange(.025,16.,.05), np.full(320,-.175)))
                    z,supported = terrain.height_at(road)
                    self.assertTrue(supported.all())
                    np.testing.assert_allclose(z,elevation(road[:,0]-offset,rise),atol=.012)
                    self.assertGreater(z[160]-max(z[0],z[-1]),1.)
                    np.testing.assert_array_equal(labels(road),254)
                    obstacles = samples[-28:-1]
                    np.testing.assert_array_equal(labels(obstacles[:,:2]),0)
                    gap = np.array([[offset+6.225,.425],[offset+6.325,.475]])
                    self.assertFalse(terrain.height_at(gap)[1].any())
                    np.testing.assert_array_equal(labels(gap),205)
                    rows,cols = np.nonzero(image == 254)
                    free_xy = np.column_stack((cols+.5,rows+.5))*nav['resolution']+nav['origin'][:2]
                    self.assertTrue(terrain.height_at(free_xy)[1].all())
                    # Independent fixture truth: only road returns match the
                    # analytic hill. Each appears once in each of two scans.
                    floor = samples[np.isclose(samples[:,2], elevation(samples[:,0]-offset,rise))]
                    hits = Counter(map(tuple, np.repeat(np.floor(floor[:,:2]/nav['resolution'])
                                                        .astype(int), 2, axis=0)))
                    for cell in np.floor(free_xy/nav['resolution']).astype(int):
                        self.assertGreaterEqual(hits[tuple(cell)], 2)
                    # 3D localization prior keeps both hill relief and obstacle tops.
                    prior = point_array(bundle/'map.pcd')
                    self.assertEqual(len(prior),len(samples))
                    # Rigid alignment round-off is smaller than a millimetre.
                    actual = np.rint(np.asarray(prior[:,:3],dtype=float)*1000).astype(np.int64)
                    expected = np.rint(samples*1000).astype(np.int64)
                    np.testing.assert_array_equal(actual[np.lexsort(actual.T[::-1])],
                                                  expected[np.lexsort(expected.T[::-1])])
                    terrain_file = bundle / meta['terrain_model']['file']
                    before = terrain_file.stat().st_mtime_ns
                    self.assertEqual(prepare(source,root/'cache'),bundle)
                    self.assertEqual(terrain_file.stat().st_mtime_ns,before)
                    self.assertEqual(hashlib.sha256(pcd.read_bytes()).digest(),original)
                    # A bad model must never be served as a complete cache hit.
                    with terrain_file.open('wb') as stream:
                        np.savez(stream,version=np.int32(999))
                    with self.assertRaises(ValueError):
                        prepare(source,root/'cache')


if __name__ == '__main__':
    unittest.main()
