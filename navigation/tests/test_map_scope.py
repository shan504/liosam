"""Both map selection and derived geometry must follow the selected map version."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_map import prepare
from select_map import select


def make_bundle(root, name, center, radius):
    version = root / name / 'version'
    (version / 'calibration').mkdir(parents=True)
    angle = np.linspace(-np.pi, np.pi, 180, endpoint=False)
    xy = np.column_stack((np.cos(angle), np.sin(angle))) * radius + center
    wall = np.column_stack((xy, np.full(len(xy), .35), np.ones(len(xy))))
    floor = wall.copy()
    floor[:, :2] = (xy - center) * .5 + center
    floor[:, 2] = 0
    scan = np.concatenate((floor, wall)).astype('<f4')
    n = len(scan)
    pcd = version / (name + '.pcd')
    header = ('VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n'
              'COUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nPOINTS %d\nDATA binary\n' % (2*n, 2*n))
    pcd.write_bytes(header.encode() + scan.tobytes() * 2)
    def write(name, payload):
        (version / name).write_text(json.dumps(payload))
    write('manifest.json', dict(map_file=pcd.name, scan_index='scans.csv',
                                optimized_keyframes='keyframes.csv',
                                dense=dict(frame_id='odom', scans=2, points=2*n)))
    write('calibration/floor.json', dict(frame='odom', ground_frame='go2_ground',
                                         origin=[0, 0, 0], rotation=np.eye(3).tolist()))
    write('calibration/body_from_lidar.json', dict(parent='base_link', child='livox_frame',
                                                   xyz=[.1, 0, .2], xyzw=[0, 0, 0, 1]))
    (version/'keyframes.csv').write_text('stamp_ns,x,y,z,qx,qy,qz,qw\n'
                                         f'100,{center},0,.52,0,0,0,1\n')
    with (version/'scans.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['stamp_ns', 'anchor_stamp_ns', 'first_point_index', 'point_count'])
        writer.writerows([(100, 100, 0, n), (200, 100, n, n)])
    return version


class MapScope(unittest.TestCase):
    def test_two_maps_select_distinct_geometry_and_initial_poses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            caches = []
            for name, center, radius in [('survey_a', 0., 2.), ('survey_b', .75, 3.)]:
                version = make_bundle(root, name, center, radius)
                selected = select(name, root)
                self.assertEqual(selected, version)
                self.assertEqual(select(str(version), root), version)
                cache = prepare(selected, root/'cache')
                caches.append(cache)
                data = json.loads((cache/'metadata.json').read_text())
                np.testing.assert_allclose(data['initial_base_pose'][:3], [center-.1, 0., .32])
                self.assertEqual(prepare(selected, root/'cache'), cache)
            self.assertNotEqual(caches[0], caches[1])
            self.assertNotEqual((caches[0]/'nav.pgm').read_bytes(), (caches[1]/'nav.pgm').read_bytes())


if __name__ == '__main__':
    unittest.main()
