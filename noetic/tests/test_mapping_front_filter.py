"""Synthetic checks for mapping-only, body-frame front-half filtering.

These run without ROS or robot hardware. Wire tests retain the current
LIO-SAM fork's scan-end convention; archive tests exercise real SQLite/PCD IO.
"""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from check_returns import POINT, wire_points
from dense_map import Archive, transform
from mapping_front_filter import filter_payload
import mapping_front_filter
from save_navigation_dense import export_dense


def orientation(roll, pitch, yaw):
    """Independent Rz Ry Rx oracle and matching XYZW quaternion."""
    r, p, y = np.deg2rad([roll, pitch, yaw])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    matrix = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                       [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                       [-sp, cp*sr, cp*cr]])
    cr, sr, cp, sp, cy, sy = (math.cos(r/2), math.sin(r/2), math.cos(p/2),
                             math.sin(p/2), math.cos(y/2), math.sin(y/2))
    quaternion = np.array([sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                           cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy])
    return matrix, quaternion


def packet(xyz, *, frame=b'livox_frame', stamp=(1795833870, 123456789)):
    points = np.zeros(len(xyz), dtype=POINT)
    points['offset_time'] = np.linspace(0, 100001234, len(xyz), dtype=np.uint32)
    points['x'], points['y'], points['z'] = np.asarray(xyz, dtype=np.float32).T
    points['reflectivity'] = (np.arange(len(xyz)) + 31).astype(np.uint8)
    points['tag'] = (np.arange(len(xyz)) % 4 + 16).astype(np.uint8)
    points['line'] = np.arange(len(xyz)) % 4
    header = struct.pack('<IIII', 47, *stamp, len(frame)) + frame
    meta = struct.pack('<QIB3sI', 1795833870123456789, len(points), 3, b'\x01\x02\x03', len(points))
    return header + meta + points.tobytes()


def xyz_of(points):
    return np.column_stack([points[name] for name in ('x', 'y', 'z')])


class RosHarness(types.ModuleType):
    """Exercise the real main callbacks; transport itself is a ROS-free fake."""
    class AnyMsg:
        def serialize(self, stream):
            stream.write(self._buff)

    class CustomMsg:
        _md5sum = 'synthetic-custommsg-md5'

    def __init__(self, params):
        super().__init__('rospy')
        self.params = dict(params)
        self.subscribers, self.outputs, self.logs = {}, [], []
        self.publishers = []

    def init_node(self, name):
        self.node_name = name

    def get_param(self, name, default=None):
        return self.params.get(name, default)

    def set_param(self, name, value):
        self.params[name] = value

    def Publisher(self, topic, msg_type, **kwargs):
        self.publishers.append((topic, msg_type, kwargs))

        def publish(message):
            if not isinstance(message, self.AnyMsg):
                raise AssertionError('Expected serialized AnyMsg output')
            wire = io.BytesIO()
            message.serialize(wire)
            self.outputs.append((topic, wire.getvalue()))

        return types.SimpleNamespace(publish=publish)

    def Subscriber(self, topic, msg_type, callback, **kwargs):
        self.subscribers[topic] = callback
        return object()

    def loginfo(self, *args):
        self.logs.append(args)

    loginfo_throttle = logwarn_throttle = logerr_throttle = loginfo

    def deliver(self, payload, md5=None):
        msg = self.AnyMsg()
        msg._buff = payload
        msg._connection_header = {'md5sum': self.CustomMsg._md5sum if md5 is None else md5}
        self.subscribers[mapping_front_filter.INPUT](msg)

    def run(self, mount_path, script):
        livox = types.ModuleType('livox_ros_driver2')
        livox.msg = types.ModuleType('livox_ros_driver2.msg')
        livox.msg.CustomMsg = self.CustomMsg
        tf = types.ModuleType('tf2_msgs')
        tf.msg = types.ModuleType('tf2_msgs.msg')
        tf.msg.TFMessage = type('TFMessage', (), {})
        modules = {'rospy': self, 'livox_ros_driver2': livox, 'livox_ros_driver2.msg': livox.msg,
                   'tf2_msgs': tf, 'tf2_msgs.msg': tf.msg}
        self.spin = lambda: script(self)
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {
                'GO2_BODY_FROM_LIDAR': str(mount_path), 'GO2_ROBOT_URDF': ''}):
            mapping_front_filter.main()


class FrontFilterTests(unittest.TestCase):
    def test_pitched_mount_uses_body_axis_and_full_lever_arm(self):
        rotation, quaternion = orientation(0, 30, 0)
        translation = np.array([.215, 0, .25])
        body = np.array([[.2, .5, 2.8], [-.2, -.4, -1.5],
                         [2.1, -.3, .6], [-1.8, .2, .9],
                         [-.7, .5, 2.0], [-.9, -.1, 1.0]])
        raw = (body-translation) @ rotation
        # At least one front return has negative sensor X, and vice versa.
        self.assertLess(raw[0, 0], 0)
        self.assertGreater(raw[1, 0], 0)
        payload = packet(raw)
        output, kept, rejected = filter_payload(payload, np.r_[translation, quaternion])
        _, original = wire_points(payload)
        _, filtered = wire_points(output)
        self.assertEqual((kept, rejected), (2, 4))
        np.testing.assert_array_equal(xyz_of(filtered)[[0, 2]], xyz_of(original)[[0, 2]])
        np.testing.assert_array_equal(xyz_of(filtered)[[1, 3, 4, 5]], 0.)

    def test_combined_roll_pitch_yaw_and_translation(self):
        rotation, quaternion = orientation(-17, 30, 73)
        translation = np.array([.215, -.04, .25])
        rng = np.random.default_rng(117)
        body = rng.uniform([-8, -10, -.5], [8, 10, 5], size=(1000, 3))
        body[:, 0] += np.where(body[:, 0] >= 0, .01, -.01)
        raw = (body-translation) @ rotation
        payload = packet(raw)
        output, kept, rejected = filter_payload(payload, np.r_[translation, quaternion])
        _, before = wire_points(payload)
        _, after = wire_points(output)
        expected = body[:, 0] >= 0
        self.assertEqual(kept, int(expected.sum()))
        self.assertEqual(rejected, int((~expected).sum()))
        np.testing.assert_array_equal(xyz_of(after)[expected], xyz_of(before)[expected])
        np.testing.assert_array_equal(xyz_of(after)[~expected], 0.)
        # Quaternion sign represents the same installation.
        self.assertEqual(filter_payload(payload, np.r_[translation, -quaternion]),
                         (output, kept, rejected))

    def test_boundary_invalid_and_zero_range_returns(self):
        raw = [[0, 1, 0], [1e-9, -1, 4], [-1e-9, 1, 0],
               [0, 0, 0], [np.nan, 1, 0], [1, np.inf, 0],
               [1, 0, -np.inf], [-2, 0, 1]]
        output, kept, rejected = filter_payload(packet(raw), np.array([0, 0, 0, 0, 0, 0, 1.]))
        _, points = wire_points(output)
        self.assertEqual((kept, rejected), (2, 6))
        self.assertTrue(np.isfinite(xyz_of(points)).all())
        np.testing.assert_array_equal(xyz_of(points)[2:], 0.)
        # Translation does not turn a zero-range invalid return into a real point.
        _, kept, _ = filter_payload(packet([[0, 0, 0], [-2, 0, 0]]),
                                    np.array([.215, 0, .25, 0, 0, 0, 1.]))
        self.assertEqual(kept, 0)

    def test_only_rejected_xyz_bytes_change_and_scan_timing_is_preserved(self):
        payload = packet([[2, .2, 1], [-2, .3, .8], [1, 2, 7], [-1, 0, 1], [-3, 1, 2]])
        original_hash = hashlib.sha256(payload).hexdigest()
        output, kept, rejected = filter_payload(payload, np.array([0, 0, 0, 0, 0, 0, 1.]))
        frame_size = struct.unpack_from('<I', payload, 12)[0]
        data_offset = 16 + frame_size + 20
        expected = bytearray(payload)
        for i in (1, 3, 4):
            struct.pack_into('<fff', expected, data_offset + i * POINT.itemsize + 4, 0., 0., 0.)
        self.assertEqual(output, bytes(expected))
        self.assertEqual(hashlib.sha256(payload).hexdigest(), original_hash)
        self.assertEqual((kept, rejected), (2, 3))
        old_stamp, before = wire_points(payload)
        new_stamp, after = wire_points(output)
        self.assertEqual(old_stamp, new_stamp)
        self.assertEqual(len(before), len(after))
        np.testing.assert_array_equal(before['offset_time'], after['offset_time'])
        # Fork converts point_num-1 entries; .back().time uses the penultimate.
        self.assertEqual(before[-2]['offset_time'], after[-2]['offset_time'])
        self.assertEqual(before[-1]['offset_time'], after[-1]['offset_time'])

    def test_all_rear_packet_has_no_valid_returns(self):
        payload = packet([[-1, 0, 0], [-4, 1, 5], [-2, -3, -.7]])
        output, kept, rejected = filter_payload(payload, np.array([0, 0, 0, 0, 0, 0, 1.]))
        _, points = wire_points(output)
        self.assertEqual((kept, rejected), (0, 3))
        np.testing.assert_array_equal(xyz_of(points), 0.)
        np.testing.assert_array_equal(points['offset_time'], wire_points(payload)[1]['offset_time'])

    def test_invalid_mount_and_wire_payload_are_rejected(self):
        payload = packet([[1, 0, 0], [-1, 0, 0]])
        mounts = [np.zeros(6), np.zeros(8), np.array([0, 0, 0, 0, 0, 0, 0.]),
                  np.array([0, 0, 0, 0, 0, 0, 2.]), np.array([np.nan, 0, 0, 0, 0, 0, 1.])]
        for mount in mounts:
            with self.subTest(mount=mount), self.assertRaises(ValueError):
                filter_payload(payload, mount)
        identity = np.array([0, 0, 0, 0, 0, 0, 1.])
        count_offset = 16 + len(b'livox_frame') + 8
        wrong_count = bytearray(payload)
        struct.pack_into('<I', wrong_count, count_offset, 3)
        for invalid in (b'', payload[:-1], bytes(wrong_count), packet([[1, 0, 0]])):
            with self.subTest(size=len(invalid)), self.assertRaises(ValueError):
                filter_payload(invalid, identity)

    def test_two_map_archives_export_only_front_returns_at_different_headings(self):
        # Same implementation, differently named maps, poses and geometry. The
        # only approximation here is replacing ROS/LIO registration with the
        # known rigid transform; SQLite and final PCD writer are real.
        scenarios = [('yard_synthetic', [7., -3., .4], (0, 0, 15), 1.),
                     ('orchard_synthetic', [-18., 26., 1.1], (4, -6, 137), 2.3)]
        mount_r, mount_q = orientation(0, 30, 0)
        mount_t = np.array([.215, 0, .25])
        with tempfile.TemporaryDirectory() as tmp:
            for name, body_t, body_rpy, scale in scenarios:
                with self.subTest(map=name):
                    body_points = np.array([[1.4, .2, -.2], [2., -1.2, .8],
                                            [.4, 2., 4.7], [3., .7, .3],
                                            [-1.5, .1, .4], [-2., -.3, 1.4],
                                            [-.8, .2, 1.9]])
                    body_points[:, :2] *= scale
                    raw = (body_points-mount_t) @ mount_r
                    payload = packet(raw)
                    output, kept, rejected = filter_payload(payload, np.r_[mount_t, mount_q])
                    stamp, filtered = wire_points(output)
                    # Mirror the fork's final-entry omission and positive
                    # minimum-range gate before registration.
                    points = filtered[:-1]
                    valid = np.linalg.norm(xyz_of(points), axis=1) > .1
                    points = points[valid]
                    self.assertEqual((kept, rejected, len(points)), (4, 3, 4))
                    source_xyzi = np.column_stack([xyz_of(points), points['reflectivity']]).astype('<f4')
                    body_r, body_q = orientation(*body_rpy)
                    # Compose XYZW independently from the matrix for archive
                    # anchoring by using body/world after applying the mount.
                    body_xyzi = source_xyzi.copy()
                    body_xyzi[:, :3] = source_xyzi[:, :3] @ mount_r.T + mount_t
                    world = body_xyzi.copy()
                    world[:, :3] = body_xyzi[:, :3] @ body_r.T + body_t
                    anchor = [*body_t, *body_q]
                    archive = Archive(Path(tmp) / name / 'dense')
                    try:
                        archive.path('odom', {stamp: anchor})
                        archive.add(stamp, stamp, transform(world, anchor, inverse=True))
                        archive.db.commit()
                        outputs = []
                        for version in ('first', 'reload'):
                            folder = Path(tmp) / name / version
                            folder.mkdir()
                            result = export_dense(archive.file, folder / (name+'.pcd'))
                            self.assertEqual(result['points'], 4)
                            raw_pcd = (folder / (name+'.pcd')).read_bytes()
                            cloud = np.frombuffer(raw_pcd.split(b'DATA binary\n', 1)[1], dtype='<f4').reshape(-1, 4)
                            expected_world = body_points[:4] @ body_r.T + body_t
                            np.testing.assert_allclose(cloud[:, :3], expected_world, atol=4e-6)
                            np.testing.assert_array_equal(cloud[:, 3], [31, 32, 33, 34])
                            # Retained high front return proves no accidental
                            # 1.5 m ceiling was introduced by this filter.
                            recovered = (cloud[:, :3] - body_t) @ body_r
                            self.assertGreater(recovered[:, 2].max(), 4.6)
                            self.assertTrue((recovered[:, 0] > 0).all())
                            with (folder / 'scans.csv').open() as stream:
                                rows = list(csv.DictReader(stream))
                            self.assertEqual(int(rows[0]['point_count']), 4)
                            outputs.append(raw_pcd)
                        self.assertEqual(outputs[0], outputs[1])
                    finally:
                        archive.db.close()

    def test_main_publishes_filtered_custommsg_wire_for_both_mapping_modes(self):
        payload = packet([[2, .2, 3], [-1, .3, .8], [1, -1, 7], [-2, 0, 1]])
        mount = dict(parent='base_link', child='livox_frame', xyz=[.215, 0, .25], xyzw=[0, 0, 0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mount.json'
            path.write_text(json.dumps(mount))
            for uses_lio in (True, False):
                with self.subTest(uses_lio=uses_lio):
                    params = {'~uses_lio': uses_lio}
                    if uses_lio:
                        params['/lio_sam/lidarMinRange'] = .1
                    ros = RosHarness(params)
                    ros.run(path, lambda node: node.deliver(payload))
                    self.assertEqual(len(ros.outputs), 1)
                    topic, serialized = ros.outputs[0]
                    self.assertEqual(topic, mapping_front_filter.OUTPUT)
                    expected = filter_payload(payload, np.r_[mount['xyz'], mount['xyzw']])[0]
                    self.assertEqual(serialized, expected)
                    self.assertEqual(wire_points(serialized)[0], wire_points(payload)[0])
                    self.assertIs(ros.publishers[0][1], RosHarness.CustomMsg)
                    self.assertEqual(ros.params['~policy']['frame'], 'base_link')
                    self.assertEqual(ros.params['~mount']['child'], 'livox_frame')

    def test_main_never_passes_raw_on_missing_mount_invalid_range_or_mount_change(self):
        payload = packet([[2, .2, 3], [-1, .3, .8], [1, -1, 7], [-2, 0, 1]])
        rear = packet([[-2, 0, 0], [-4, 1, 1], [-1, 2, 3]])
        doc = dict(parent='base_link', child='livox_frame', xyz=[.215, 0, .25], xyzw=[0, 0, 0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mount.json'
            ros = RosHarness({'/lio_sam/lidarMinRange': .1})

            def events(node):
                node.deliver(payload)  # Explicit mount file does not exist.
                self.assertEqual(node.outputs, [])
                path.write_text(json.dumps(doc))
                del node.params['/lio_sam/lidarMinRange']
                node.deliver(payload)  # Missing range parameter must not pass.
                for minimum in (0., -1., np.nan, np.inf):
                    node.params['/lio_sam/lidarMinRange'] = minimum
                    node.deliver(payload)
                    self.assertEqual(node.outputs, [])
                node.params['/lio_sam/lidarMinRange'] = .1
                node.deliver(payload, md5='wrong-type')
                self.assertEqual(node.outputs, [])
                node.deliver(payload)
                self.assertEqual(len(node.outputs), 1)
                node.deliver(rear)  # No real return may become a free-space ray.
                node.deliver(packet([[1, 2, 3], [2, 1, 0]], frame=b'wrong_frame'))
                self.assertEqual(len(node.outputs), 1)
                changed = dict(doc, xyz=[.8, 0, .25])
                replacement = path.with_suffix('.replacement')
                replacement.write_text(json.dumps(changed))
                replacement.replace(path)  # Inode change makes revision deterministic.
                node.deliver(payload)
                node.deliver(payload)
                self.assertEqual(len(node.outputs), 1)

            ros.run(path, events)


if __name__ == '__main__':
    unittest.main()
