"""Exact mapping rectangles, never a navigation-time collision exemption."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surveyed_corridor import (CorridorIndex, POLICY, make_document, read_document,
                                validate_document, write_document)


def pose(x=0., y=0., z=0., yaw=0., pitch=0., roll=0.):
    c, s = math.cos, math.sin
    rz = np.array([[c(yaw), -s(yaw), 0.], [s(yaw), c(yaw), 0.], [0., 0., 1.]])
    ry = np.array([[c(pitch), 0., s(pitch)], [0., 1., 0.], [-s(pitch), 0., c(pitch)]])
    rx = np.array([[1., 0., 0.], [0., c(roll), -s(roll)], [0., s(roll), c(roll)]])
    result = np.eye(4)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = [x, y, z]
    return result


def event(stamp, anchor, body):
    return dict(stamp_ns=stamp, anchor_stamp_ns=anchor, body_pose=body.tolist())


def union_cells(index, resolution, transform=None, batch_size=5):
    return set(map(tuple, np.concatenate(list(index.free_cells(resolution, transform, batch_size)), axis=0)))


class SurveyedCorridor(unittest.TestCase):
    def test_mount_lever_arm_and_hill_attitude_do_not_rotate_heading_rectangle(self):
        mount = pose(.215, 0., .25, pitch=math.pi/6)
        for name, body in [('yard', pose(2., -5., .37, yaw=math.pi/2)),
                           ('farm', pose(-24., 11., 1.8, yaw=.67, pitch=.20, roll=-.12))]:
            with self.subTest(name=name):
                world_lidar = body @ mount
                recovered_body = world_lidar @ np.linalg.inv(mount)
                index = CorridorIndex([recovered_body])
                yaw = math.atan2(body[1, 0], body[0, 0])
                f = np.array([math.cos(yaw), math.sin(yaw)])
                left = np.array([-math.sin(yaw), math.cos(yaw)])
                local = np.array([[.2, 0.], [.99, .49], [0., -.5], [1., .5],
                                  [-.01, 0.], [1.01, 0.], [.5, .51], [.5, -.51]])
                xy = body[:2, 3] + local[:, :1]*f + local[:, 1:]*left
                # All-Z means gravity-level footprint at every real point height.
                for height in (-100., -.2, 0., .5, 12., 100.):
                    cloud = np.column_stack((xy, np.full(len(xy), height), np.arange(len(xy))))
                    np.testing.assert_array_equal(index.contains(cloud), [True]*4 + [False]*4)

    def test_actual_pose_union_does_not_clear_missing_route_or_rotated_aabb_corners(self):
        index = CorridorIndex([pose(yaw=math.pi/4), pose(10., 0., yaw=math.pi/4)])
        query = np.array([[.3, .3, 10.], [10.3, .3, -8.], [5., 0., .1],
                          [1., -.3, .2], [-.3, 1., 0.]])
        np.testing.assert_array_equal(index.contains(query), [True, True, False, False, False])
        # A point captured before it became near-range is selected retrospectively.
        old_point = np.array([[2.5, .2, 1.]])
        self.assertFalse(index.contains(old_point)[0])
        later = CorridorIndex([pose(), pose(2., 0.)])
        self.assertTrue(later.contains(old_point)[0])

    def test_raster_marks_only_whole_cells_and_deduplicates_union(self):
        index = CorridorIndex([pose(), pose(), pose(2., 0.)])
        cells = union_cells(index, .1)
        expected = {(x, y) for x in list(range(10)) + list(range(20, 30)) for y in range(-5, 5)}
        self.assertEqual(cells, expected)
        batches = list(index.free_cells(.1, batch_size=7))
        self.assertTrue(all(0 < len(batch) <= 7 for batch in batches))
        self.assertEqual(sum(map(len, batches)), len(cells))
        self.assertFalse(any(10 <= x < 20 for x, y in cells))

    def test_transformed_ground_quad_contains_all_four_pixel_corners(self):
        index = CorridorIndex([pose(3., -4., 1.2, yaw=.61, pitch=.18),
                               pose(-11., 8., -.4, yaw=-.25, roll=.1)])
        transform = pose(7., 13., -.7, yaw=.2, pitch=-.11, roll=.07)
        polygons = index.polygons(transform)
        cells = union_cells(index, .05, transform)
        self.assertGreater(len(cells), 500)
        # Independent inverse 2D basis test of each entire 5cm cell.
        for cell in cells:
            pixel = (np.asarray(cell)[None, :] + [[0, 0], [1, 0], [1, 1], [0, 1]])*.05
            matched = False
            for polygon in polygons[:, :, :2]:
                basis = np.column_stack((polygon[1]-polygon[0], polygon[3]-polygon[0]))
                uv = (pixel-polygon[0]) @ np.linalg.inv(basis).T
                matched |= bool(((uv >= -1e-8) & (uv <= 1+1e-8)).all())
            self.assertTrue(matched)
        # Transforming source corner Z is material for tilted source maps.
        changed = CorridorIndex([pose(3., -4., 4.2, yaw=.61), pose(-11., 8., 2.6, yaw=-.25)])
        self.assertFalse(np.allclose(polygons[:, :, :2], changed.polygons(transform)[:, :, :2]))

    def test_sparse_large_coordinates_no_extent_limit_and_chunk_invariance(self):
        poses = [pose(15000., -22000., 120., yaw=.3), pose(-25000., 18000., -20., yaw=-.5)]
        index = CorridorIndex(poses)
        points = np.vstack([np.column_stack((np.linspace(p[0, 3]-.8, p[0, 3]+1.2, 100),
                                                np.full(100, p[1, 3]), np.full(100, p[2, 3]))) for p in poses])
        expected = index.contains(points)
        actual = np.concatenate([index.contains(points[i:i+13]) for i in range(0, len(points), 13)])
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(expected.sum(), 50)
        self.assertLess(len(index.buckets), 30)
        self.assertGreater(len(union_cells(index, .1)), 100)

    def test_document_requires_new_explicit_policy_valid_frame_and_real_event_stamps(self):
        events = [event(200, 100, pose(1., 0.)), event(100, 100, pose())]
        terrain = dict(version=1, file='survey_terrain.npz', source_frame='odom',
                       map_from_source_map=np.eye(4).tolist(), body_height_above_ground=.37)
        document = make_document('/odom', events, terrain=terrain)
        self.assertEqual(document['frame_id'], 'odom')
        self.assertEqual([row['stamp_ns'] for row in document['poses']], [100, 200])
        self.assertEqual(document['policy'], POLICY)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'surveyed_corridor.json'
            write_document(target, 'odom', events, terrain=terrain)
            self.assertEqual(read_document(target, 'odom'), document)
            with self.assertRaises(FileExistsError):
                write_document(target, 'odom', events)
            with self.assertRaisesRegex(ValueError, 'frame differs'):
                read_document(target, 'map')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                target.write_text('{"version":1,"version":1}')
                read_document(target)
        for bad in [{}, dict(document, version=2), dict(document, policy=dict(POLICY, x_max=2.)),
                    dict(document, poses=events+[events[0]]),
                    dict(document, poses=[event(50, 100, pose())]),
                    dict(document, terrain=dict(terrain, file='../survey_terrain.npz'))]:
            with self.assertRaises(ValueError):
                validate_document(bad)

    def test_empty_invalid_points_and_invalid_poses(self):
        self.assertEqual(CorridorIndex([]).contains(np.zeros((2, 3))).tolist(), [False, False])
        self.assertEqual(list(CorridorIndex([]).free_cells(.05)), [])
        index = CorridorIndex([pose()])
        self.assertEqual(index.contains(np.empty((0, 4))).shape, (0,))
        self.assertFalse(index.contains(np.array([[np.nan, 0., 0.], [.5, 0., np.inf]])).any())
        with self.assertRaises(ValueError):
            CorridorIndex([np.ones((4, 4))])
        with self.assertRaises(ValueError):
            CorridorIndex([pose(pitch=math.pi/2)])
        with self.assertRaises(ValueError):
            list(index.free_cells(0.))
        with self.assertRaises(ValueError):
            list(index.free_cells(.1, pose(roll=math.pi/2)))


if __name__ == '__main__':
    unittest.main()
