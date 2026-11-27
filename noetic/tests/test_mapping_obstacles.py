"""Verify the on-robot mapping display with real PointCloud2 byte layouts."""
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dense_map import Archive
from mapping_obstacles import filtered_cloud, main
from save_navigation_dense import export_dense


def organized_cloud(points, big_endian):
    # Ring and intensity exercise copying the entire original point record.
    record = struct.Struct(('>' if big_endian else '<') + 'HddffH')
    field_layout = [('ring', 0, 4), ('y', 2, 8), ('z', 10, 8),
                    ('x', 18, 7), ('intensity', 22, 7), ('tag', 26, 4)]
    fields = [SimpleNamespace(name=name, offset=offset, datatype=dtype, count=1)
              for name, offset, dtype in field_layout]
    records = [record.pack(i, y, z, x, 30.5 + i, 50 + i) for i, (x, y, z) in enumerate(points)]
    padding = b'row pad'
    width = 2
    assert len(records) % width == 0
    data = b''.join(b''.join(records[start:start + width]) + padding
                    for start in range(0, len(records), width))
    msg = SimpleNamespace(header=SimpleNamespace(stamp=123, frame_id='/go2_ground', seq=31),
                          height=len(records) // width, width=width, fields=fields,
                          is_bigendian=big_endian, point_step=record.size,
                          row_step=width * record.size + len(padding), data=data, is_dense=False)
    return msg, records


class MappingDisplay(unittest.TestCase):
    def test_local_scan_and_survey_points_keep_their_original_fields(self):
        for x, y, high, endian in [(0., 4., 4.5, False), (31., -12., 17., True)]:
            with self.subTest(origin=(x, y), high=high):
                points = [(x, y, 0.), (x, y, .099), (x, y, .1), (x+3, y, high),
                          (x, y, -.2), (float('nan'), y, high), (x, float('inf'), 1.),
                          (x, y, float('nan'))]
                scan, records = organized_cloud(points, endian)
                raw = scan.data
                result = filtered_cloud(scan)
                self.assertEqual((result.height, result.width), (1, 2))
                self.assertEqual(result.row_step, 2 * scan.point_step)
                self.assertEqual(result.point_step, scan.point_step)
                self.assertEqual(result.data, records[2] + records[3])
                self.assertEqual(result.fields, scan.fields)
                self.assertEqual(result.header, scan.header)
                self.assertEqual(result.is_bigendian, endian)
                self.assertEqual((scan.height, scan.width, scan.data), (4, 2, raw))

    def test_floor_only_publishes_an_empty_cloud_in_the_same_frame(self):
        floor, _ = organized_cloud([(1., 0., 0.), (2., 0., .03)], False)
        out = filtered_cloud(floor)
        self.assertEqual((out.height, out.width, out.row_step, out.data), (1, 0, 0, b''))
        self.assertEqual(out.header, floor.header)
        self.assertEqual(filtered_cloud(out).data, b'')

    def test_archived_export_still_contains_floor_and_high_points(self):
        points = np.array([[1., 2., 0., 17.], [1., 2., .07, 23.], [1., 2., 8., 42.]], '<f4')
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            archive = Archive(directory / 'dense')
            try:
                archive.path('odom', {100: [0., 0., 0., 0., 0., 0., 1.]})
                archive.add(100, 100, points)
                archive.db.commit()
                path = directory / 'survey.pcd'
                info = export_dense(archive.file, path)
                self.assertEqual(info['points'], len(points))
                header, data = path.read_bytes().split(b'DATA binary\n', 1)
                self.assertIn(b'FIELDS x y z intensity', header)
                np.testing.assert_array_equal(np.frombuffer(data, '<f4').reshape(-1, 4), points)
            finally:
                archive.db.close()

    def test_rejects_unlevelled_points_and_truncated_storage(self):
        scan, _ = organized_cloud([(1., 0., 0.), (1., 0., 2.)], False)
        scan.header.frame_id = 'livox_frame'
        with self.assertRaises(ValueError):
            filtered_cloud(scan)
        scan.header.frame_id = 'go2_ground'
        scan.data = scan.data[:-1]
        with self.assertRaises(ValueError):
            filtered_cloud(scan)

    def test_both_existing_foxglove_topics_receive_filtered_messages(self):
        publishers = {}
        subscribers = {}

        def publisher(topic, _msg_type, *, latch, queue_size):
            messages = []
            pub = SimpleNamespace(latch=latch, messages=messages, publish=messages.append)
            publishers[topic] = pub
            return pub

        def subscribe(topic, _msg_type, callback, callback_args, **options):
            subscribers[topic] = lambda msg: callback(msg, callback_args)
            return subscribers[topic]

        ros = SimpleNamespace(init_node=lambda name: None, spin=lambda: None,
                              Publisher=publisher, Subscriber=subscribe)
        with patch.dict(sys.modules, {'rospy': ros, 'sensor_msgs.msg': SimpleNamespace(PointCloud2=object)}):
            main()
        cloud, records = organized_cloud([(1., 2., 0.), (1., 2., 6.)], True)
        for name in ('dense_map', 'scan'):
            subscribers['/go2_visual/ground_full/' + name](cloud)
            pub = publishers['/go2_visual/ground/' + name]
            self.assertEqual(pub.latch, name == 'dense_map')
            self.assertEqual(pub.messages[0].data, records[1])
            self.assertEqual(pub.messages[0].header.stamp, cloud.header.stamp)


if __name__ == '__main__':
    unittest.main()
