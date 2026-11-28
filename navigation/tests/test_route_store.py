"""Explicit grid-only route revalidation preserves coordinates and old files."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

GO2 = Path(__file__).resolve().parents[1] / 'original_go2/go2_3d'
sys.path.insert(0, str(GO2))
from route_store import (RouteStore, _canonical, map_geometry_identity, map_identity,
                         revalidate_route, validate_route)


def fixture(directory, label, origin=(0.,0.,0.), resolution=.5, pcd=b'original point cloud'):
    directory.mkdir()
    (directory/'map.pcd').write_bytes(pcd)
    nav = dict(image='nav.pgm', resolution=resolution, origin=list(origin),
               negate=0, occupied_thresh=.65, free_thresh=.196)
    (directory/'nav.yaml').write_text(yaml.safe_dump(nav))
    metadata = dict(frame_id='map', parameters=dict(map_name=label),
                    body_height_above_ground=.3,
                    map_start_pose=[0.,0.,0.,0.,0.,0.,1.],
                    map_from_source_map=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],
                    ground_model=dict(plane=[0.,0.,1.,0.], support_points=500))
    (directory/'metadata.yaml').write_text(yaml.safe_dump(metadata))
    (directory/'nav.pgm').write_bytes(b'P5\n10 10\n255\n'+bytes([254])*100)
    return nav, metadata


def route_for(directory, xy):
    return dict(schema_version=1, map=map_identity(directory),
                waypoints=[dict(position=[x,y,0.], orientation=[0.,0.,0.,1.]) for x,y in xy])


def pixel(directory, x, y, value):
    path=directory/'nav.pgm'
    data=bytearray(path.read_bytes())
    data[len(b'P5\n10 10\n255\n')+(9-y)*10+x]=value
    path.write_bytes(data)


class RouteRevalidation(unittest.TestCase):
    def test_legacy_identity_algorithm_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'old'
            nav, meta=fixture(directory,'yard')
            descriptor=dict(version=1, frame_id='map',
                pcd=hashlib.sha256((directory/'map.pcd').read_bytes()).hexdigest(),
                occupancy=hashlib.sha256((directory/'nav.pgm').read_bytes()).hexdigest(),
                grid={k:v for k,v in nav.items() if k!='image'},
                geometry={k:meta.get(k) for k in (
                    'body_height_above_ground','map_start_pose','map_from_source_map')})
            descriptor['geometry']['ground_model']={'plane':meta['ground_model']['plane']}
            expected=hashlib.sha256(json.dumps(_canonical(descriptor),sort_keys=True,
                separators=(',',':'),allow_nan=False).encode()).hexdigest()
            self.assertEqual(map_identity(directory)['fingerprint'],expected)

    def test_two_maps_revalidate_grid_only_change_without_modifying_route(self):
        for label,origin in [('yard',(0.,0.,0.)),('orchard',(100.,-50.,math.pi/2))]:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                old,new=root/'old',root/'new'
                fixture(old,label,origin,pcd=label.encode())
                fixture(new,label,origin,pcd=label.encode())
                c,s=math.cos(origin[2]),math.sin(origin[2])
                xy=[(origin[0]+c*x-s*y,origin[1]+s*x+c*y) for x,y in [(1.25,1.25),(2.25,2.25)]]
                original=route_for(old,xy)
                before=copy.deepcopy(original)
                pixel(new,0,0,0)  # Only navigation policy away from goals changed.
                self.assertNotEqual(map_identity(old),map_identity(new))
                self.assertEqual(map_geometry_identity(old),map_geometry_identity(new))
                with self.assertRaisesRegex(ValueError,'map mismatch'):
                    validate_route(original,expected_map=map_identity(new))
                updated=revalidate_route(original,old,new)
                self.assertEqual(updated['map'],map_identity(new))
                self.assertEqual(updated['waypoints'],original['waypoints'])
                self.assertEqual(original,before)
                store=RouteStore(root/'routes')
                store.save('original',original)
                saved=(root/'routes/original.json').read_bytes()
                store.save('revalidated',updated)
                self.assertEqual((root/'routes/original.json').read_bytes(),saved)
                self.assertEqual(store.load('revalidated'),updated)

    def test_missing_or_wrong_previous_map_and_coordinate_changes_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            old,new=root/'old',root/'new'
            fixture(old,'yard')
            fixture(new,'yard')
            original=route_for(old,[(1.25,1.25)])
            with self.assertRaises(FileNotFoundError):
                revalidate_route(original,root/'missing',new)
            pixel(old,0,0,0)
            with self.assertRaisesRegex(ValueError,'map mismatch'):
                revalidate_route(original,old,new)
            pixel(old,0,0,254)
            (new/'map.pcd').write_bytes(b'different point coordinates')
            with self.assertRaisesRegex(ValueError,'coordinate metadata changed'):
                revalidate_route(original,old,new)
            (new/'map.pcd').write_bytes((old/'map.pcd').read_bytes())
            meta=yaml.safe_load((new/'metadata.yaml').read_text())
            meta['map_from_source_map'][0][3]=1.0
            (new/'metadata.yaml').write_text(yaml.safe_dump(meta))
            with self.assertRaisesRegex(ValueError,'coordinate metadata changed'):
                revalidate_route(original,old,new)

    def test_new_grid_occupied_unknown_outside_and_negated_pixels_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            old,new=root/'old',root/'new'
            fixture(old,'yard')
            fixture(new,'yard')
            original=route_for(old,[(1.25,1.25),(2.25,2.25)])
            for value,state in [(0,'occupied'),(205,'unknown')]:
                pixel(new,4,4,value)
                with self.assertRaisesRegex(ValueError,'#2 .*'+state):
                    revalidate_route(original,old,new)
            pixel(new,4,4,254)
            original['waypoints'][1]['position'][0]=100.
            with self.assertRaisesRegex(ValueError,'#2 .*outside'):
                revalidate_route(original,old,new)
            original['waypoints'][1]['position'][0]=2.25
            nav=yaml.safe_load((new/'nav.yaml').read_text())
            nav['negate']=1
            (new/'nav.yaml').write_text(yaml.safe_dump(nav))
            with self.assertRaisesRegex(ValueError,'occupied'):
                revalidate_route(original,old,new)
            (new/'nav.pgm').write_bytes(b'P5\n10 10\n255\n'+bytes([0])*100)
            self.assertEqual(revalidate_route(original,old,new)['waypoints'],original['waypoints'])

    def test_corrupt_pgm_and_unsupported_grid_mode_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            old,new=root/'old',root/'new'
            fixture(old,'yard')
            fixture(new,'yard')
            original=route_for(old,[(1.25,1.25)])
            (new/'nav.pgm').write_bytes(b'P5\n10 10\n255\n'+bytes([254])*99)
            with self.assertRaisesRegex(ValueError,'pixel count'):
                revalidate_route(original,old,new)
            nav=yaml.safe_load((new/'nav.yaml').read_text())
            nav['mode']='raw'
            (new/'nav.yaml').write_text(yaml.safe_dump(nav))
            with self.assertRaisesRegex(ValueError,'trinary'):
                revalidate_route(original,old,new)


if __name__=='__main__':
    unittest.main()
