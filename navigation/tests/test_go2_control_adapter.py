"""A stale or misframed MID input cannot keep the original Go2 motion gate open."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as Obj
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2_control_adapter import MidSensorInputs, UpstreamTimestamps


def header(frame, stamp):
    second, ns = divmod(stamp, 10**9)
    return Obj(frame_id=frame, stamp=Obj(sec=second, nanosec=ns))


def cloud(stamp, frame='base_link'):
    return Obj(header=header(frame, stamp), width=1, height=1, point_step=16, row_step=16,
               data=bytes(16))


def odom(stamp):
    pose = Obj(position=Obj(x=1., y=2., z=.3), orientation=Obj(x=0., y=0., z=0., w=1.))
    return Obj(header=header('odom', stamp), child_frame_id='base_link', pose=Obj(pose=pose))


class TimedSensorInputs(unittest.TestCase):
    def setUp(self):
        self.tick = 10 * 10**9
        self.inputs = MidSensorInputs()
        self.inputs.clock_map = UpstreamTimestamps()
        self.inputs.get_clock = lambda: Obj(now=lambda: Obj(nanoseconds=self.tick))
        self.inputs.record_sensor_timing = Mock()
        self.inputs.first_odom_stamp_ns = self.inputs.odom_received = self.inputs.cloud_received = None

    @patch('go2_control_adapter.time.monotonic', return_value=500.)
    def test_delayed_inputs_and_replay(self, _):
        self.inputs.accept_mid_cloud(cloud(self.tick))
        self.assertIsNone(self.inputs.cloud_received)
        self.inputs.accept_body_odom(odom(self.tick - 500_000_000))
        self.assertEqual(self.inputs.odom_received, 499.5)
        self.inputs.accept_mid_cloud(cloud(self.tick - 200_000_000))
        self.assertEqual(self.inputs.cloud_received, 499.8)
        self.inputs.accept_mid_cloud(cloud(self.tick - 200_000_000))
        self.assertEqual(self.inputs.record_sensor_timing.call_args.args[-1], 'duplicate_or_reordered')

    def test_bad_geometry_frame_clock_or_size_cannot_renew(self):
        msg = odom(self.tick)
        msg.pose.pose.orientation.w = float('nan')
        self.inputs.accept_body_odom(msg)
        self.assertIsNone(self.inputs.odom_received)
        self.inputs.accept_body_odom(odom(self.tick))
        self.inputs.accept_mid_cloud(cloud(self.tick, 'livox_frame'))
        self.inputs.accept_mid_cloud(cloud(self.tick + 1_000_000_000))
        self.tick += 2_000_000_000
        self.inputs.accept_mid_cloud(cloud(self.tick - 1_500_000_000))
        missing = cloud(self.tick)
        missing.data = b''
        self.inputs.accept_mid_cloud(missing)
        self.assertIsNone(self.inputs.cloud_received)
        self.inputs.accept_mid_cloud(cloud(self.tick))
        self.assertIsNotNone(self.inputs.cloud_received)


if __name__ == '__main__':
    unittest.main()
