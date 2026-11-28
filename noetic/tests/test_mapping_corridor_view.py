"""Verify mapping displays clear exactly the recorded corridor, at every height."""
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corridor_recorder import pose_matrix
from dense_map import Archive
from local_detail import Window
from mapping_obstacles import CorridorView, filtered_cloud
from surveyed_corridor import make_document
from test_mapping_obstacles import connected_points, organized_cloud


IDENTITY = [0., 0., 0., 0., 0., 0., 1.]
MOUNT = dict(parent='base_link', child='livox_frame', xyz=[.215, 0., .25], xyzw=[0., 0., 0., 1.])


def document(body=None):
    return make_document('odom', [dict(stamp_ns=2_000_000_000,
        anchor_stamp_ns=1_000_000_000,
        body_pose=(np.eye(4) if body is None else body).tolist())])


class MappingCorridorView(unittest.TestCase):
    def test_full_point_records_and_timestamps_survive_rotated_frame_filter(self):
        # Source LIO coordinates can be tilted. The cleared footprint is level
        # ground XY, so a tall return must not escape via source-frame tilt.
        ground = pose_matrix([3., -2., .5, 0., math.sin(.3), 0., math.cos(.3)])
        body = pose_matrix([4., -1., .4, 0., 0., math.sin(math.pi/4), math.cos(math.pi/4)])
        source_body = np.linalg.inv(ground) @ body
        view = CorridorView()
        view.update(json.dumps(document(source_body)))
        index = view.index(ground)
        for endian in (False, True):
            points = [(4., -.5, .2), (4., -.5, 25.), (4., -.5, 0.), (-9., -9., 2.)]
            points += (connected_points(3.2, -.5, .2) + connected_points(4., .2, 4.) +
                       connected_points(4., -1.2, 2.))
            cloud, records = organized_cloud(points, endian)
            original = cloud.data
            result = filtered_cloud(cloud, index)
            self.assertEqual(result.data, b''.join(records[4:]))
            self.assertIs(result.header, cloud.header)
            self.assertEqual(result.fields, cloud.fields)
            self.assertEqual(result.is_bigendian, cloud.is_bigendian)
            self.assertEqual(cloud.data, original)

    def test_new_document_clears_old_observations_and_empty_resets(self):
        cloud, records = organized_cloud(connected_points(.5, 0., 8.) +
                                         connected_points(1.1, 0., 8.), False)
        view = CorridorView()
        self.assertEqual(filtered_cloud(cloud, view.index(np.eye(4))).data, b''.join(records))
        view.update(json.dumps(document()))
        self.assertEqual(filtered_cloud(cloud, view.index(np.eye(4))).data, b''.join(records[8:]))
        view.update(json.dumps(make_document('odom', [])))
        self.assertEqual(filtered_cloud(cloud, view.index(np.eye(4))).data, b''.join(records))

    def test_malformed_policy_keeps_prior_valid_snapshot(self):
        view = CorridorView()
        view.update(json.dumps(document()))
        broken = document()
        broken['policy']['x_max'] = 999.
        with self.assertRaises(ValueError):
            view.update(json.dumps(broken))
        self.assertEqual(view.index(np.eye(4)).contains([[.5, 0., 2.], [2., 0., 2.]]).tolist(), [True, False])

    def test_message_uses_exact_source_stamp_not_latest_transform(self):
        view = CorridorView()
        view.update(json.dumps(document()))
        stamp = SimpleNamespace(to_nsec=lambda: 2_000_000_000)
        msg = SimpleNamespace(header=SimpleNamespace(stamp=stamp))
        calls = []
        def lookup(*args):
            calls.append(args)
            return SimpleNamespace(transform=SimpleNamespace(
                translation=SimpleNamespace(x=0., y=0., z=0.),
                rotation=SimpleNamespace(x=0., y=0., z=0., w=1.)))
        index = view.for_message(msg, SimpleNamespace(lookup_transform=lookup),
                                 SimpleNamespace(Duration=lambda value:value))
        self.assertIs(calls[0][2], stamp)
        self.assertEqual(calls[0][:2], ('go2_ground', 'odom'))
        self.assertTrue(index.contains([[.5, 0., 99.]])[0])
        msg.header.stamp = SimpleNamespace(to_nsec=lambda:0)
        with self.assertRaises(ValueError):
            view.for_message(msg, None, None)


class LocalDetailCorridor(unittest.TestCase):
    def test_raw_archive_view_retroactively_clears_same_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive(Path(tmp)/'dense')
            window = None
            try:
                a, b = 1_000_000_000, 2_000_000_000
                archive.path('odom', {a: IDENTITY})
                old = np.array([(*xyz, intensity)
                                for x, z, intensity in ((.5, .2, 11.), (.5, 4., 22.), (2., .2, 33.))
                                for xyz in connected_points(x, 0., z)], '<f4')
                archive.add(a, a, old)
                archive.db.commit()
                window = Window(archive.file)
                _, before, _ = window.read(a, {a}, 'odom', tuple(IDENTITY), 0.)
                self.assertEqual(set(before[:, 3]), {11., 22., 33.})
                archive.enable_corridor(MOUNT)
                archive.add(b, a, np.array([[.5, 0., .2, 44.]], '<f4'))
                archive.add_corridor(b, a, np.eye(4))
                archive.db.commit()
                _, after, _ = window.read(b, {a, b}, 'odom', tuple(IDENTITY), 0.)
                np.testing.assert_array_equal(after, old[16:])
                # A corrected keyframe moves both raw points and its corridor.
                archive.path('odom', {a: [1., 0., 0., 0., 0., 0., 1.]})
                archive.db.commit()
                _, corrected, _ = window.read(b, {a, b}, 'odom', tuple(IDENTITY), 0.)
                expected = old[16:].copy()
                expected[:, 0] += 1.
                np.testing.assert_allclose(corrected, expected)
                self.assertEqual(archive.db.execute('SELECT count(*) FROM scans').fetchone()[0], 2)
            finally:
                if window is not None:
                    window.close()
                archive.db.close()

    def test_all_filtered_points_produce_empty_detail(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive(Path(tmp)/'dense')
            try:
                archive.path('odom', {1: IDENTITY})
                archive.enable_corridor(MOUNT)
                archive.add(1, 1, np.array([[.5, 0., 2., 1.]], '<f4'))
                archive.add_corridor(1, 1, np.eye(4))
                archive.db.commit()
                window = Window(archive.file)
                try:
                    _, points, _ = window.read(1, {1}, 'odom', tuple(IDENTITY), 0.)
                    self.assertEqual(points.shape, (0, 4))
                finally:
                    window.close()
            finally:
                archive.db.close()


if __name__ == '__main__':
    unittest.main()
