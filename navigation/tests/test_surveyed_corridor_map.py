"""Run the real SQLite -> saved corridor -> Nav2 map path without ROS."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'navigation'))
sys.path.insert(0, str(ROOT / 'noetic'))
from prepare_go2_map import prepare
from prepare_map import apply, point_array, quaternion, rigid
from save_navigation_dense import export_dense
from surveyed_corridor import POLICY, make_document
from terrain import build_terrain, load_terrain


def digest_tree(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).digest()
            for path in directory.rglob('*') if path.is_file()}


def pose_values(matrix):
    return [*matrix[:3, 3].tolist(), *quaternion(matrix[:3, :3]).tolist()]


def make_recording(root, name, *, tilted=False, marked=True, events=True):
    """Sparse confirmed rectangles with a measured straight path between them.

    Actual keyframes are 0.5m apart at one-second intervals, and intermediate
    scans remain available. Historical observations include obstacles in the
    apparent gaps, outside the strip, and far above the robot.
    """
    version = root / name / 'version'
    (version / 'calibration').mkdir(parents=True)
    frame = 'odom' if tilted else 'go2_ground'
    saved_from_map = (rigid([7.3, -4.2, -.62],
                           [math.sin(.16 / 2), 0., 0., math.cos(.16 / 2)])
                      if tilted else np.eye(4))
    yaw = .41 if tilted else -.55
    lane = rigid([-.8, .65, 0.], [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)])
    mount = rigid([.215, 0., .25], [0., math.sin(math.pi / 12), 0., math.cos(math.pi / 12)])
    physical = dict(parent='base_link', child='livox_frame', xyz=mount[:3, 3].tolist(),
                    xyzw=quaternion(mount[:3, :3]).tolist())
    (version / 'calibration/body_from_lidar.json').write_text(json.dumps(physical))
    (version / 'calibration/floor.json').write_text(json.dumps(dict(
        frame=frame, ground_frame='go2_ground', origin=saved_from_map[:3, 3].tolist(),
        rotation=saved_from_map[:3, :3].tolist())))
    ij = np.indices((120, 48)).reshape(2, -1).T
    xy = (ij + .5) * .05 - [.4, 1.2]
    floor = np.column_stack((xy, .04 * xy[:, 0]))
    blocks = []
    for x, y in [(.4, -.075), (3.4, -.075), (2., -.075), (.4, .775)]:
        bxy = (np.indices((7, 7)).reshape(2, -1).T + .5) * .025 + [x, y]
        blocks.append(np.column_stack((bxy, .04 * bxy[:, 0] + .55)))
    # Preserve geometry for GICP; only the derived static grid clears the lane.
    tall = np.array([[.6, .1, 8.], [2.2, -.1, 15.], [3.6, -.1, 11.], [4.6, .1, 13.]])
    road = np.vstack((floor, *blocks, tall))
    intensities = np.arange(1, len(road) + 1, dtype=np.float32)
    source = np.column_stack((apply(apply(road, lane), saved_from_map), intensities))
    database = root / (name + '.sqlite')
    expected_chunks, event_bodies, anchors = [], [], {}
    event_stamps = [1_240_000_000, 7_240_000_000]
    with sqlite3.connect(database) as db:
        db.executescript('CREATE TABLE meta(name TEXT PRIMARY KEY,value TEXT NOT NULL);'
                         'CREATE TABLE poses(stamp INTEGER PRIMARY KEY,pose TEXT NOT NULL);'
                         'CREATE TABLE scans(stamp INTEGER PRIMARY KEY,anchor INTEGER NOT NULL,points BLOB NOT NULL);'
                         'CREATE TABLE corridor_events(stamp INTEGER PRIMARY KEY,anchor INTEGER NOT NULL,body_relative TEXT NOT NULL);')
        db.execute('INSERT INTO meta VALUES (?,?)', ('frame', frame))
        if marked:
            db.execute('INSERT INTO meta VALUES (?,?)', ('surveyed_corridor', json.dumps(
                dict(enabled=True, policy=POLICY, mount=physical))))
        for i in range(8):
            anchor, x = 1_000_000_000 + i * 1_000_000_000, .5 * i
            body = lane @ rigid([x, 0., .37 + .04 * x], [0., 0., 0., 1.])
            lidar = saved_from_map @ body @ mount
            anchors[anchor] = lidar
            db.execute('INSERT INTO poses VALUES (?,?)', (anchor, json.dumps(pose_values(lidar))))
            local = np.column_stack((apply(source, np.linalg.inv(lidar)), source[:, 3])).astype('<f4')
            db.execute('INSERT INTO scans VALUES (?,?,?)', (anchor, anchor, local.tobytes()))
            expected_chunks.append((anchor, anchor, source.copy(), road.copy()))
            # Timestamped real scans show there was no acquisition outage.
            for step in range(1, 5):
                stamp = anchor + step * 200_000_000
                tiny_road = np.array([[x + .25, .8, .3]])
                tiny = np.column_stack((apply(apply(tiny_road, lane), saved_from_map), [64001.]))
                tiny_local = np.column_stack((apply(tiny, np.linalg.inv(lidar)), tiny[:, 3])).astype('<f4')
                db.execute('INSERT INTO scans VALUES (?,?,?)', (stamp, anchor, tiny_local.tobytes()))
                expected_chunks.append((stamp, anchor, tiny, tiny_road))
        for anchor, x in [(1_000_000_000, 0.), (7_000_000_000, 3.)]:
            lidar = anchors[anchor]
            event_body = saved_from_map @ lane @ rigid(
                [x + .12, 0., .37 + .04 * (x + .12)], [0., 0., 0., 1.])
            event_bodies.append(event_body)
            near = np.array([[x + .4, .1, 0.], [x + .5, -.1, 6.]])
            near_source = np.column_stack((apply(apply(near, lane), saved_from_map), [65001., 65002.]))
            near_local = np.column_stack((apply(near_source, np.linalg.inv(lidar)), near_source[:, 3])).astype('<f4')
            stamp = anchor + 240_000_000
            db.execute('INSERT INTO scans VALUES (?,?,?)', (stamp, anchor, near_local.tobytes()))
            expected_chunks.append((stamp, anchor, near_source, near))
            if events:
                db.execute('INSERT INTO corridor_events VALUES (?,?,?)',
                           (stamp, anchor, json.dumps((np.linalg.inv(lidar) @ event_body).tolist())))
    return dict(version=version, database=database, chunks=sorted(expected_chunks), lane=lane,
                saved_from_map=saved_from_map, mount=mount, frame=frame, events=event_bodies,
                event_stamps=event_stamps, anchors=anchors, floor=floor)


def save_recording(recording):
    version = recording['version']
    name = version.parent.name
    pcd = version / (name + '.pcd')
    # No cloud-fit terrain work may hide in a save helper or an imported alias.
    with patch('terrain.build_terrain', side_effect=AssertionError('save must not fit terrain')):
        dense = export_dense(recording['database'], pcd)
    manifest = dict(name=name, map_file=pcd.name, scan_index='scans.csv',
                    optimized_keyframes='keyframes.csv', dense=dense)
    (version / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def make_v1_saved_bundle(record):
    """Construct the preceding persisted format independently of new saving."""
    version, name = record['version'], record['version'].parent.name
    data, rows, offset = [], [], 0
    for stamp, anchor, source, road in record['chunks']:
        inside = discrete_mask(road)
        retained = source[~inside].astype('<f4')
        data.append(retained)
        rows.append((stamp, anchor, offset, len(retained)))
        offset += len(retained)
    cloud = np.vstack(data).astype('<f4')
    header = (f'VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n'
              f'COUNT 1 1 1 1\nWIDTH {len(cloud)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
              f'POINTS {len(cloud)}\nDATA binary\n')
    (version / (name + '.pcd')).write_bytes(header.encode() + cloud.tobytes())
    with (version / 'scans.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['stamp_ns', 'anchor_stamp_ns', 'first_point_index', 'point_count'])
        writer.writerows(rows)
    with (version / 'keyframes.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['stamp_ns', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        writer.writerows((stamp, *pose_values(pose)) for stamp, pose in record['anchors'].items())
    floor = apply(record['floor'], record['lane'])
    body_ground = np.array([apply(np.array([[x, 0., .04*x]]), record['lane'])[0] for x in (0., 1., 2., 3.)])
    model = build_terrain([floor, floor], body_ground)
    terrain = model.save(version / 'survey_terrain.npz')
    terrain.update(map_from_source_map=np.linalg.inv(record['saved_from_map']).tolist(),
                   body_from_lidar=record['mount'].tolist(), source_frame=record['frame'],
                   body_height_above_ground=.37)
    events = [dict(stamp_ns=stamp, anchor_stamp_ns=stamp-240_000_000, body_pose=body.tolist())
              for stamp, body in zip(record['event_stamps'], record['events'])]
    doc = make_document(record['frame'], events, terrain=terrain)
    if doc['version'] != 1:
        raise AssertionError('make_document without paths must preserve v1')
    (version / 'surveyed_corridor.json').write_text(json.dumps(doc))
    dense = dict(frame_id=record['frame'], points=len(cloud), scans=len(rows),
                 surveyed_corridor='surveyed_corridor.json')
    manifest = dict(name=name, map_file=name+'.pcd', scan_index='scans.csv',
                    optimized_keyframes='keyframes.csv', dense=dense)
    (version/'manifest.json').write_text(json.dumps(manifest))
    return manifest


def read_grid(bundle):
    config = yaml.safe_load((bundle / 'nav.yaml').read_text())
    with (bundle / config['image']).open('rb') as stream:
        assert stream.readline() == b'P5\n'
        width, height = map(int, stream.readline().split())
        assert stream.readline() == b'255\n'
        pixels = np.frombuffer(stream.read(), dtype='u1').reshape(height, width)[::-1]
    return config, pixels


def grid_labels(bundle, xy):
    config, pixels = read_grid(bundle)
    col, row = np.floor((xy - config['origin'][:2]) / config['resolution']).astype(int).T
    return pixels[row, col]


def discrete_mask(road):
    return (((road[:, 0] >= .12) & (road[:, 0] <= 1.12) |
             (road[:, 0] >= 3.12) & (road[:, 0] <= 4.12)) & (abs(road[:, 1]) <= .5))


class SurveyedCorridorMapIntegration(unittest.TestCase):
    def test_saved_hairpin_follows_measured_keyframes_not_endpoint_chord(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = make_recording(root, 'hairpin_lane', tilted=True)
            # Replace only this synthetic recording with a right-going hairpin.
            # Its endpoints have the same X but no actual travel across the
            # middle of the opening on the left of the turn.
            route = [(0., 0., 0.), (.8, 0., .3), (1.6, .4, math.pi/2),
                     (1.6, 1.2, math.pi/2), (.8, 1.6, 2.8), (0., 1.6, math.pi)]
            xy = (np.indices((100, 80)).reshape(2, -1).T + .5) * .05 - [1.5, 1.]
            floor = np.column_stack((xy, np.zeros(len(xy))))
            probes = np.array([[1.7, .6, .55], [0., .8, .55]])
            blocks = np.vstack([probe + [dx, dy, 0.] for probe in probes
                                for dx in (-.025, 0., .025) for dy in (-.025, 0., .025)])
            road = np.vstack((floor, blocks, [[1.7, .6, 15.], [0., .8, 15.]]))
            source = np.column_stack((apply(apply(road, record['lane']), record['saved_from_map']),
                                      np.arange(1, len(road)+1)))
            with sqlite3.connect(record['database']) as db:
                for table in ('scans', 'poses', 'corridor_events'):
                    db.execute('DELETE FROM '+table)
                for i, (x, y, yaw) in enumerate(route):
                    stamp = (i+1)*1_000_000_000
                    body = record['saved_from_map'] @ record['lane'] @ rigid(
                        [x, y, .37], [0., 0., math.sin(yaw/2), math.cos(yaw/2)])
                    lidar = body @ record['mount']
                    local = np.column_stack((apply(source, np.linalg.inv(lidar)), source[:, 3])).astype('<f4')
                    db.execute('INSERT INTO poses VALUES (?,?)', (stamp, json.dumps(pose_values(lidar))))
                    db.execute('INSERT INTO scans VALUES (?,?,?)', (stamp, stamp, local.tobytes()))
                    if i in (0, len(route)-1):
                        db.execute('INSERT INTO corridor_events VALUES (?,?,?)',
                                   (stamp, stamp, json.dumps(np.linalg.inv(record['mount']).tolist())))
            manifest = save_recording(record)
            actual = point_array(record['version']/manifest['map_file'])
            lane_points = apply(apply(actual, np.linalg.inv(record['saved_from_map'])),
                                np.linalg.inv(record['lane']))
            self.assertTrue(np.any(np.linalg.norm(lane_points[:, :2]-probes[0, :2], axis=1) < .04))
            self.assertTrue(np.any(np.linalg.norm(lane_points[:, :2]-probes[1, :2], axis=1) < .04))
            doc = json.loads((record['version']/'surveyed_corridor.json').read_text())
            self.assertEqual(len(doc['paths']), 1)
            self.assertEqual(len(doc['paths'][0]), len(route))
            with patch('prepare_go2_map.build_terrain', side_effect=AssertionError('no raw terrain fit')):
                bundle = prepare(record['version'], root/'cache')
            xy_probes = apply(probes, record['lane'])[:, :2]
            np.testing.assert_array_equal(grid_labels(bundle, xy_probes), [254, 0])

    def test_two_maps_keep_gicp_geometry_and_clear_navigation_gaps_without_terrain_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, tilted in [('east_farm', False), ('unrelated_hill', True)]:
                with self.subTest(map=name):
                    record = make_recording(root, name, tilted=tilted)
                    database_before = record['database'].read_bytes()
                    manifest = save_recording(record)
                    version = record['version']
                    original = digest_tree(version)
                    actual = np.asarray(point_array(version / manifest['map_file']))
                    expected = np.vstack([source for _, _, source, _ in record['chunks']])
                    self.assertEqual(len(actual), len(expected))
                    np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=2e-6, rtol=0)
                    np.testing.assert_array_equal(actual[:, 3], expected[:, 3])
                    self.assertEqual(manifest['dense']['cleared_points'], 0)
                    self.assertEqual(manifest['dense']['pc_geometry'], 'full_registered_returns')
                    with (version / 'scans.csv').open() as stream:
                        rows = list(csv.DictReader(stream))
                    counts = [int(row['point_count']) for row in rows]
                    self.assertEqual([int(row['point_count']) for row in rows
                                      if int(row['stamp_ns']) in record['event_stamps']], [2, 2])
                    np.testing.assert_array_equal([int(row['first_point_index']) for row in rows],
                                                  np.cumsum([0] + counts[:-1]))
                    document = json.loads((version / 'surveyed_corridor.json').read_text())
                    self.assertEqual(document['version'], 2)
                    self.assertNotIn('terrain', document)
                    self.assertFalse((version / 'survey_terrain.npz').exists())
                    self.assertEqual([event['stamp_ns'] for event in document['poses']], record['event_stamps'])
                    np.testing.assert_allclose([event['body_pose'] for event in document['poses']],
                                               record['events'], atol=1e-10)
                    self.assertGreater(sum(map(len, document['paths'])), len(document['poses']))
                    with patch('prepare_go2_map.build_terrain', side_effect=AssertionError('do not move cloud fit to navigation')):
                        bundle = prepare(version, root / 'cache')
                    probes = apply(np.array([[.55, 0., 0.], [2.0875, .0125, 0.], [3.55, 0., 0.]]), record['lane'])[:, :2]
                    # The disconnected rectangles and all intervening road now form one lane.
                    np.testing.assert_array_equal(grid_labels(bundle, probes), 254)
                    lateral = apply(np.array([[.4875, .8625, 0.]]), record['lane'])[:, :2]
                    np.testing.assert_array_equal(grid_labels(bundle, lateral), 0)
                    meta = yaml.safe_load((bundle / 'metadata.yaml').read_text())
                    self.assertEqual(meta['surveyed_corridor']['events'], 2)
                    self.assertGreater(meta['surveyed_corridor']['free_cells'], 0)
                    self.assertEqual(meta['pc_geometry'], 'full_registered_returns')
                    # Derived GICP map still contains tall geometry inside the
                    # same road whose static grid is free. No flattening or Z clipping.
                    localized = point_array(bundle/'map.pcd')
                    lane_geometry = apply(localized, np.linalg.inv(record['lane']))
                    self.assertTrue(np.any((lane_geometry[:, 2] > 14.) &
                                           (np.abs(lane_geometry[:, 1]) < .5)))
                    reference = load_terrain(bundle, meta)
                    heights, supported = reference.height_at(probes)
                    self.assertTrue(supported.all())
                    np.testing.assert_allclose(heights, .04 * np.array([.55, 2.0875, 3.55]), atol=.025)
                    copied = json.loads((bundle / meta['surveyed_corridor']['file']).read_text())
                    self.assertEqual(copied, document)
                    # Every full cell inside the straight rotated corridor is free;
                    # this detects seams along polygon edges and rectangular AABB clearing.
                    config, pixels = read_grid(bundle)
                    rows, cols = np.indices(pixels.shape)
                    corners = ((np.column_stack((cols.ravel(), rows.ravel()))[:, None, :] +
                                np.array([[0, 0], [1, 0], [1, 1], [0, 1]])) * config['resolution'] +
                               np.asarray(config['origin'][:2]))
                    lane_corners = (corners - record['lane'][:2, 3]) @ record['lane'][:2, :2]
                    whole = ((abs(lane_corners[:, :, 1]) <= .5) &
                             (lane_corners[:, :, 0] >= .12) & (lane_corners[:, :, 0] <= 4.12)).all(axis=1)
                    self.assertGreater(np.count_nonzero(whole), 1200)
                    self.assertEqual(np.count_nonzero(whole), meta['surveyed_corridor']['free_cells'])
                    np.testing.assert_array_equal(pixels.ravel()[whole], 254)
                    self.assertEqual(digest_tree(version), original)
                    self.assertEqual(record['database'].read_bytes(), database_before)
                    self.assertEqual(len(list(version.glob('*.pcd'))), 1)

    def test_legacy_and_stationary_bundles_do_not_gain_a_corridor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = make_recording(root, 'old_route', marked=False)
            manifest = save_recording(old)
            self.assertNotIn('surveyed_corridor', manifest['dense'])
            self.assertFalse((old['version'] / 'survey_terrain.npz').exists())
            actual = point_array(old['version'] / manifest['map_file'])
            expected = np.vstack([chunk[2] for chunk in old['chunks']])
            np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=2e-6, rtol=0)
            np.testing.assert_array_equal(actual[:, 3], expected[:, 3])
            bundle = prepare(old['version'], root / 'cache')
            blocked = apply(np.array([[.475, 0., 0.], [2.075, 0., 0.], [3.475, 0., 0.]]), old['lane'])[:, :2]
            np.testing.assert_array_equal(grid_labels(bundle, blocked), 0)
            empty = make_recording(root, 'stationary_route', events=False)
            empty_manifest = save_recording(empty)
            self.assertEqual((old['version'] / manifest['map_file']).read_bytes(),
                             (empty['version'] / empty_manifest['map_file']).read_bytes())
            with self.assertRaisesRegex(ValueError, '没有有效的前进通行带'):
                prepare(empty['version'], root / 'cache')
            self.assertEqual(json.loads((empty['version'] / 'surveyed_corridor.json').read_text())['poses'], [])

    def test_existing_v1_map_keeps_discrete_geometry_and_stored_terrain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = make_recording(root, 'old_marked_route', tilted=True)
            make_v1_saved_bundle(record)
            before = digest_tree(record['version'])
            bundle = prepare(record['version'], root / 'cache')
            probes = apply(np.array([[.55, 0., 0.], [3.55, 0., 0.]]), record['lane'])[:, :2]
            np.testing.assert_array_equal(grid_labels(bundle, probes), 254)
            gap = apply(np.array([[2.0875, .0125, 0.]]), record['lane'])[:, :2]
            np.testing.assert_array_equal(grid_labels(bundle, gap), 0)
            self.assertEqual(digest_tree(record['version']), before)
            self.assertEqual(prepare(record['version'], root / 'cache'), bundle)

    def test_cache_tracks_path_calibration_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = make_recording(root, 'cache_route')
            save_recording(record)
            source, cache = record['version'], root / 'cache'
            first = prepare(source, cache)
            before = digest_tree(first)
            self.assertEqual(prepare(source, cache), first)
            self.assertEqual(digest_tree(first), before)
            cached_doc = first / 'surveyed_corridor.json'
            original = cached_doc.read_bytes()
            cached_doc.write_text('{"version":999}')
            with self.assertRaises(ValueError):
                prepare(source, cache)
            cached_doc.write_bytes(original)
            metadata = yaml.safe_load((first / 'metadata.yaml').read_text())
            cached_reference = first / metadata['terrain_model']['file']
            reference_bytes = cached_reference.read_bytes()
            cached_reference.write_bytes(b'broken cached trajectory reference')
            with self.assertRaises((ValueError, OSError)):
                prepare(source, cache)
            cached_reference.write_bytes(reference_bytes)
            document_path = source / 'surveyed_corridor.json'
            original_doc = json.loads(document_path.read_text())
            document_path.write_text(json.dumps(original_doc, separators=(',', ':')))
            second = prepare(source, cache)
            self.assertNotEqual(second, first)
            damaged = json.loads(json.dumps(original_doc))
            damaged['paths'][0][1]['body_pose'][0][0] = 3.
            document_path.write_text(json.dumps(damaged))
            with self.assertRaises(ValueError):
                prepare(source, cache)
            damaged = json.loads(json.dumps(original_doc))
            damaged['calibration']['body_from_lidar'][0][3] += .1
            document_path.write_text(json.dumps(damaged))
            with self.assertRaises(ValueError):
                prepare(source, cache)
            document_path.write_text(json.dumps(original_doc))


if __name__ == '__main__':
    unittest.main()
