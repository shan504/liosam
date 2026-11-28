"""New-session corridor evidence survives the real archive/save/Nav2 pipeline."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'navigation'))
sys.path.insert(0, str(ROOT / 'noetic'))
from prepare_go2_map import prepare
from prepare_map import apply, point_array, quaternion, rigid
from save_navigation_dense import export_dense
from surveyed_corridor import POLICY
from terrain import load_terrain


def digest_tree(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).digest()
            for path in directory.rglob('*') if path.is_file()}


def pose_values(matrix):
    return [*matrix[:3, 3].tolist(), *quaternion(matrix[:3, :3]).tolist()]


def make_recording(root, name, *, tilted=False, marked=True, events=True, outside_obstacles=True):
    """Two measured forward events separated by an unsurveyed two-metre gap.

    All obstacles were already visible in the first scan. Each event scan only
    contains near-strip points and becomes empty after cumulative clearance.
    The anchors differ from event poses, so using keyframe origins is detectable.
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
    centers = [(.4, -.075), (3.4, -.075)]
    if outside_obstacles:
        centers.extend([(2., -.075), (.4, .775)])
    for x, y in centers:
        bxy = (np.indices((7, 7)).reshape(2, -1).T + .5) * .025 + [x, y]
        blocks.append(np.column_stack((bxy, .04 * bxy[:, 0] + .55)))
    # Height has no exemption in the requested mapping-only strip.
    tall = np.array([[.6, .1, 8.], [3.6, -.1, 11.]] +
                    ([[4.6, .1, 13.]] if outside_obstacles else []))
    road = np.vstack((floor, *blocks, tall))
    intensities = np.arange(1, len(road) + 1, dtype=np.float32)
    source = np.column_stack((apply(apply(road, lane), saved_from_map), intensities))

    database = root / (name + '.sqlite')
    expected_chunks, event_bodies = [], []
    with sqlite3.connect(database) as db:
        db.executescript('CREATE TABLE meta(name TEXT PRIMARY KEY,value TEXT NOT NULL);'
                         'CREATE TABLE poses(stamp INTEGER PRIMARY KEY,pose TEXT NOT NULL);'
                         'CREATE TABLE scans(stamp INTEGER PRIMARY KEY,anchor INTEGER NOT NULL,points BLOB NOT NULL);'
                         'CREATE TABLE corridor_events(stamp INTEGER PRIMARY KEY,anchor INTEGER NOT NULL,body_relative TEXT NOT NULL);')
        db.execute('INSERT INTO meta VALUES (?,?)', ('frame', frame))
        if marked:
            db.execute('INSERT INTO meta VALUES (?,?)', ('surveyed_corridor', json.dumps(
                dict(enabled=True, policy=POLICY, mount=physical))))
        for anchor, x in [(100, 0.), (200, 3.)]:
            body = lane @ rigid([x, 0., .37 + .04 * x], [0., 0., 0., 1.])
            lidar = saved_from_map @ body @ mount
            db.execute('INSERT INTO poses VALUES (?,?)', (anchor, json.dumps(pose_values(lidar))))
            local = np.column_stack((apply(source, np.linalg.inv(lidar)), source[:, 3])).astype('<f4')
            db.execute('INSERT INTO scans VALUES (?,?,?)', (anchor, anchor, local.tobytes()))
            expected_chunks.append((anchor, source.copy(), road.copy()))
            event_body = saved_from_map @ lane @ rigid(
                [x + .12, 0., .37 + .04 * (x + .12)], [0., 0., 0., 1.])
            event_bodies.append(event_body)
            near = np.array([[x + .4, .1, 0.], [x + .5, -.1, 6.]])
            near_source = np.column_stack((apply(apply(near, lane), saved_from_map), [65001., 65002.]))
            near_local = np.column_stack((apply(near_source, np.linalg.inv(lidar)), near_source[:, 3])).astype('<f4')
            db.execute('INSERT INTO scans VALUES (?,?,?)', (anchor + 10, anchor, near_local.tobytes()))
            expected_chunks.append((anchor + 10, near_source, near))
            if events:
                db.execute('INSERT INTO corridor_events VALUES (?,?,?)',
                           (anchor + 10, anchor, json.dumps((np.linalg.inv(lidar) @ event_body).tolist())))
    return dict(version=version, database=database, chunks=expected_chunks, lane=lane,
                saved_from_map=saved_from_map, mount=mount, frame=frame, events=event_bodies)


def save_recording(recording):
    version = recording['version']
    name = version.parent.name
    pcd = version / (name + '.pcd')
    dense = export_dense(recording['database'], pcd)
    manifest = dict(name=name, map_file=pcd.name, scan_index='scans.csv',
                    optimized_keyframes='keyframes.csv', dense=dense)
    (version / 'manifest.json').write_text(json.dumps(manifest))
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


class SurveyedCorridorMapIntegration(unittest.TestCase):
    def test_two_map_names_export_all_heights_and_free_only_recorded_corridors(self):
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
                    expected = []
                    for _, source, road in record['chunks']:
                        # Independent fixture coordinates, not production geometry.
                        in_strip = ((road[:, 0] >= .12) & (road[:, 0] <= 1.12) |
                                    (road[:, 0] >= 3.12) & (road[:, 0] <= 4.12)) & (abs(road[:, 1]) <= .5)
                        expected.append(source[~in_strip])
                    expected = np.vstack(expected)
                    self.assertEqual(len(actual), len(expected))
                    np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=2e-6, rtol=0)
                    np.testing.assert_array_equal(actual[:, 3], expected[:, 3])
                    self.assertEqual(manifest['dense']['cleared_points'],
                                     sum(len(chunk[1]) for chunk in record['chunks']) - len(expected))
                    with (version / 'scans.csv').open() as stream:
                        rows = list(csv.DictReader(stream))
                    counts = [int(row['point_count']) for row in rows]
                    self.assertEqual(counts[1::2], [0, 0])
                    np.testing.assert_array_equal([int(row['first_point_index']) for row in rows],
                                                  np.cumsum([0] + counts[:-1]))
                    document = json.loads((version / 'surveyed_corridor.json').read_text())
                    self.assertEqual([event['stamp_ns'] for event in document['poses']], [110, 210])
                    np.testing.assert_allclose([event['body_pose'] for event in document['poses']],
                                               record['events'], atol=1e-10)
                    preserved = load_terrain(version, {'terrain_model': document['terrain']})
                    probes = apply(np.array([[.55, 0., 0.], [3.55, 0., 0.]]), record['lane'])[:, :2]
                    heights, supported = preserved.height_at(probes)
                    self.assertTrue(supported.all(), 'terrain must use returns before their removal')
                    np.testing.assert_allclose(heights, [.022, .142], atol=.015)
                    bundle = prepare(version, root / 'cache')
                    # The old obstacle centres and removed floor become explicit free cells.
                    np.testing.assert_array_equal(grid_labels(bundle, probes), 254)
                    outside = apply(np.array([[2.0875, .0125, 0.], [.4875, .8625, 0.]]), record['lane'])[:, :2]
                    np.testing.assert_array_equal(grid_labels(bundle, outside), 0)
                    meta = yaml.safe_load((bundle / 'metadata.yaml').read_text())
                    self.assertEqual(meta['surveyed_corridor']['events'], 2)
                    self.assertGreater(meta['surveyed_corridor']['free_cells'], 0)
                    copied = json.loads((bundle / meta['surveyed_corridor']['file']).read_text())
                    self.assertTrue((bundle / copied['terrain']['file']).is_file())
                    load_terrain(bundle, {'terrain_model': copied['terrain']})
                    # Check every whole grid cell inside the two rotated strips,
                    # not just their centre probes or a rectangular world AABB.
                    config, pixels = read_grid(bundle)
                    rows, cols = np.indices(pixels.shape)
                    corners = ((np.column_stack((cols.ravel(), rows.ravel()))[:, None, :] +
                                np.array([[0, 0], [1, 0], [1, 1], [0, 1]])) * config['resolution'] +
                               np.asarray(config['origin'][:2]))
                    lane_corners = ((corners - record['lane'][:2, 3]) @ record['lane'][:2, :2])
                    left_right = (abs(lane_corners[:, :, 1]) <= .5).all(axis=1)
                    whole = left_right & (((lane_corners[:, :, 0] >= .12) &
                                           (lane_corners[:, :, 0] <= 1.12)).all(axis=1) |
                                          ((lane_corners[:, :, 0] >= 3.12) &
                                           (lane_corners[:, :, 0] <= 4.12)).all(axis=1))
                    self.assertEqual(np.count_nonzero(whole), meta['surveyed_corridor']['free_cells'])
                    np.testing.assert_array_equal(pixels.ravel()[whole], 254)
                    self.assertEqual(digest_tree(version), original, 'navigation must not rewrite source maps')
                    self.assertEqual(record['database'].read_bytes(), database_before)

    def test_new_survey_exports_a_free_only_navigation_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = make_recording(root, 'open_field', outside_obstacles=False)
            manifest = save_recording(record)
            self.assertGreater(manifest['dense']['cleared_points'], 0)
            bundle = prepare(record['version'], root / 'cache')
            _, pixels = read_grid(bundle)
            self.assertFalse((pixels == 0).any(), 'all original obstacle returns were in surveyed strips')
            self.assertGreater(np.count_nonzero(pixels == 254), 500)

    def test_old_archive_without_marker_preserves_points_and_obstacles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = make_recording(root, 'old_route', marked=False)
            manifest = save_recording(old)
            self.assertNotIn('surveyed_corridor', manifest['dense'])
            self.assertFalse((old['version'] / 'survey_terrain.npz').exists())
            actual = point_array(old['version'] / manifest['map_file'])
            expected = np.vstack([chunk[1] for chunk in old['chunks']])
            np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=2e-6, rtol=0)
            np.testing.assert_array_equal(actual[:, 3], expected[:, 3])
            bundle = prepare(old['version'], root / 'cache')
            blocked = apply(np.array([[.475, 0., 0.], [3.475, 0., 0.]]), old['lane'])[:, :2]
            np.testing.assert_array_equal(grid_labels(bundle, blocked), 0)

            # An enabled run with no actual forward events also clears nothing.
            empty = make_recording(root, 'stationary_route', events=False)
            empty_manifest = save_recording(empty)
            self.assertEqual((old['version'] / manifest['map_file']).read_bytes(),
                             (empty['version'] / empty_manifest['map_file']).read_bytes())
            empty_bundle = prepare(empty['version'], root / 'cache')
            np.testing.assert_array_equal(read_grid(bundle)[1], read_grid(empty_bundle)[1])
            self.assertEqual(json.loads((empty['version'] / 'surveyed_corridor.json').read_text())['poses'], [])

    def test_cache_tracks_survey_and_terrain_and_rejects_invalid_source(self):
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
            cached_original = cached_doc.read_bytes()
            cached_doc.write_text('{"version":999}')
            with self.assertRaises(ValueError):
                prepare(source, cache)
            cached_doc.write_bytes(cached_original)
            cached_model = first / 'terrain.npz'
            cached_model_bytes = cached_model.read_bytes()
            cached_model.write_bytes(b'broken cached terrain')
            with self.assertRaises((ValueError, OSError)):
                prepare(source, cache)
            cached_model.write_bytes(cached_model_bytes)

            document_path = source / 'surveyed_corridor.json'
            document = json.loads(document_path.read_text())
            # Same semantics but a changed source document must not reuse an old cache.
            document_path.write_text(json.dumps(document, separators=(',', ':')))
            second = prepare(source, cache)
            self.assertNotEqual(second, first)
            document['poses'][0]['body_pose'][0][0] = 3.
            document_path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                prepare(source, cache)
            document['poses'][0]['body_pose'] = record['events'][0].tolist()
            document_path.write_text(json.dumps(document))
            terrain_path = source / document['terrain']['file']
            original_terrain = terrain_path.read_bytes()
            terrain_path.write_bytes(b'not a terrain archive')
            with self.assertRaises((ValueError, OSError)):
                prepare(source, cache)
            terrain_path.write_bytes(original_terrain)
            terrain_path.unlink()
            with self.assertRaises((ValueError, OSError)):
                prepare(source, cache)


if __name__ == '__main__':
    unittest.main()
