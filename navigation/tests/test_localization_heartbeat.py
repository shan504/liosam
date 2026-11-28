"""Production heartbeat gates tolerate delivery gaps without extending commands."""
import ast
from collections import deque
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import tempfile
import json

ROOT = Path(__file__).resolve().parents[1]
GO2 = ROOT / 'original_go2/go2_3d'
sys.path[:0] = [str(GO2), str(ROOT/'original_go2/patches/dds'), str(Path(__file__).parent)]
from bridge_logic import CommandGate
from localization_timing import LOCALIZATION_HEARTBEAT_TIMEOUT, localization_is_fresh
from test_display_cloud import message
import test_terrain_live


def production_class(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def command(x=.3):
    return NS(linear=NS(x=x, y=0.), angular=NS(z=0.))


class HeartbeatGates(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.ns = dict(Node=object, time=NS(monotonic=lambda: self.now), math=math,
                       LOCALIZATION_HEARTBEAT_TIMEOUT=LOCALIZATION_HEARTBEAT_TIMEOUT,
                       localization_is_fresh=localization_is_fresh, Empty=NS)
        production_class(ROOT/'original_go2/patches/dds/go2_edu_dds_bridge.py',
                                'Go2EduBridge', self.ns)
        bridge = production_class(GO2/'control_bridge.py', 'LocalizedBridge', self.ns)
        self.bridge = bridge.__new__(bridge)
        b = self.bridge
        b.gate = CommandGate(max_linear=.3, max_yaw=1.2)
        b.gate.set_enabled(True)
        b.motion_allowed = b.localization_valid = True
        b.localization_received = b.odom_received = b.cloud_received = 10.
        b.cmd_received = b.cmd_nonzero = b.cmd_gate_accepted = 0
        b.cmd_dropped = dict.fromkeys(('motion_not_allowed', 'motion_disabled',
            'localization_invalid_or_stale', 'odometry_or_cloud_stale', 'nonfinite'), 0)
        b.send_move, b.send_stop = Mock(), Mock()

    def operator(self):
        kind = production_class(GO2/'operator_bridge.py', 'Operator', self.ns)
        op = kind.__new__(kind)
        op.mode = 'navigation'
        op.valid, op.valid_received = True, 10.
        op.localization_state, op.localization_state_received = 'tracking', 10.
        op.auto_resume_enabled = True
        op.auto_resume = None
        op.last_auto_resume_revocation = ''
        op.recovery_epoch = 3
        op.route_started, op.route_paused = True, False
        op.route_pause_reason = None
        op.goal_handle = object()
        op.goal_pending = op.cancelling = False
        op.cancel_requested = False
        op.waypoints = deque([(2, object())])
        op.inflight_waypoint = (1, object())
        op.goal_inputs = deque()
        op.goal_generation = op.motion_generation = 1
        op.completed_waypoints = 0
        op.enable_query = None
        op.motion_requests = []
        op.pause_disable_future = None
        op.waypoint_edit = None
        op.report = Mock()
        op.process_goal_inputs = Mock()
        op.try_automatic_resume = Mock()
        op.check_motion_state = Mock()
        op.disable_paused_bridge, op.cancel_active_goal = Mock(), Mock()
        op.recover_localization = NS(publish=Mock())
        op.discard_goal_inputs = Mock()
        op.publish_waypoints = Mock()
        op.parameters = NS(service_is_ready=lambda: False)
        return kind, op

    def test_shared_finite_receipt_deadline_includes_no_future_time(self):
        self.assertEqual(LOCALIZATION_HEARTBEAT_TIMEOUT, 1.)
        for received, now, expected in [(10., 10.75, True), (10., 10.999, True),
            (10., 11., False), (10., 9.9, False), (None, 10., False),
            (float('nan'), 10., False), (10., float('inf'), False)]:
            self.assertEqual(localization_is_fresh(True, received, now), expected)
        self.assertFalse(localization_is_fresh(False, 10., 10.01))

    def test_short_heartbeat_gap_keeps_forwarding_current_commands(self):
        self.now = 10.75
        b = self.bridge
        b.odom_received = b.cloud_received = self.now
        b.on_cmd(command())
        b.control_tick()
        b.send_move.assert_called_once_with((.3, 0., 0.))
        b.send_stop.assert_not_called()
        self.assertEqual(b.cmd_gate_accepted, 1)

    def test_explicit_false_stops_and_discards_old_commands_without_grace(self):
        b = self.bridge
        b.on_cmd(command())
        b.control_tick()
        self.now = 10.01
        b.on_validity(NS(data=False))
        b.on_cmd(command())
        b.control_tick()
        b.send_stop.assert_called_once()
        self.assertIsNone(b.gate.command)
        self.assertIsNone(b.gate.received_at)
        self.assertEqual(b.cmd_dropped['localization_invalid_or_stale'], 1)
        b.on_validity(NS(data=True))
        b.control_tick()
        self.assertEqual(b.send_move.call_count, 1)  # No replay on heartbeat recovery.
        b.on_cmd(command())
        b.control_tick()
        self.assertEqual(b.send_move.call_count, 2)

    def test_heartbeat_deadline_stops_even_with_fresh_sensors_and_commands(self):
        b = self.bridge
        b.on_cmd(command())
        b.control_tick()
        self.now = 10.99
        b.odom_received = b.cloud_received = self.now
        b.on_cmd(command())
        self.now = 11.
        b.control_tick()
        b.send_stop.assert_called_once()
        self.assertIsNone(b.gate.command)
        self.assertIsNone(b.gate.received_at)

    def test_command_and_sensor_watchdogs_are_not_extended(self):
        b = self.bridge
        b.on_cmd(command())
        b.control_tick()
        self.now = 10.51
        b.on_validity(NS(data=True))
        b.odom_received = b.cloud_received = self.now
        b.control_tick()
        b.send_stop.assert_called_once()
        self.assertEqual(b.gate.timeout, .5)
        b.on_cmd(command())
        b.control_tick()
        self.now = 11.51
        b.on_validity(NS(data=True))
        b.on_cmd(command())
        b.control_tick()
        self.assertEqual(b.send_stop.call_count, 2)
        self.assertIsNone(b.gate.command)

    def test_operator_short_gap_retains_same_running_action_and_queue(self):
        _, op = self.operator()
        active = op.goal_handle
        inflight = op.inflight_waypoint
        self.now = 10.75
        op.watchdog()
        self.assertIs(op.goal_handle, active)
        self.assertEqual(op.inflight_waypoint, inflight)
        self.assertEqual([p[0] for p in op.waypoints], [2])
        self.assertTrue(op.route_started)
        self.assertFalse(op.route_paused)
        op.cancel_active_goal.assert_not_called()
        op.disable_paused_bridge.assert_not_called()

    def test_operator_false_and_deadline_pause_without_dropping_waypoints(self):
        for explicit_false in (True, False):
            with self.subTest(explicit_false=explicit_false):
                self.now = 10.
                _, op = self.operator()
                self.now = 10.01 if explicit_false else 11.
                if explicit_false:
                    op.on_validity(NS(data=False))
                op.watchdog()
                self.assertTrue(op.route_paused)
                self.assertFalse(op.route_started)
                self.assertEqual([p[0] for p in op.waypoints], [1, 2])
                self.assertIsNotNone(op.auto_resume)
                op.cancel_active_goal.assert_called_once()
                op.disable_paused_bridge.assert_called_once()
                op.watchdog()
                op.cancel_active_goal.assert_called_once()

    def test_manual_stop_revokes_resume_even_after_fresh_validity(self):
        kind, op = self.operator()
        self.now = 10.01
        op.on_validity(NS(data=False))
        op.watchdog()
        self.assertIsNotNone(op.auto_resume)
        op.on_enable(NS(data=False))
        self.assertIsNone(op.auto_resume)
        self.assertFalse(op.route_started)
        self.assertFalse(op.waypoints)
        op.on_validity(NS(data=True))
        # Invoke the production resume method rather than the watchdog's IO stub.
        kind.try_automatic_resume(op)
        self.assertIsNone(op.auto_resume)
        self.assertFalse(op.route_started)

    def test_nav2_startup_uses_same_heartbeat_but_still_requires_fresh_tf(self):
        class Time:
            def __init__(self, ns=0): self.nanoseconds = ns
            @classmethod
            def from_msg(cls, stamp): return cls(stamp.sec*10**9+stamp.nanosec)
            def __sub__(self, other): return Time(self.nanoseconds-other.nanoseconds)
        self.ns.update(Time=Time, ManageLifecycleNodes=NS(Request=type('Request', (), {'STARTUP':0})),
                       TransformException=RuntimeError)
        kind = production_class(GO2/'nav2_ready.py', 'Nav2Ready', self.ns)
        for valid, elapsed, tf_nsec, should_start in [(True,.75,900000000,True),
                (True,1.,900000000,False), (False,.01,900000000,False),
                (True,.75,500000000,False)]:
            with self.subTest(valid=valid, elapsed=elapsed, tf_nsec=tf_nsec):
                self.now=10.
                node=kind.__new__(kind)
                node.attempted=False
                node.valid,node.received=True,10.
                node.on_valid(NS(data=valid))
                node.ready_since=9.
                node.localization_status=node.status_received=None
                self.now=10.+elapsed
                node.get_clock=lambda:NS(now=lambda:Time(100_000_000_000))
                node.buffer=NS(lookup_transform=lambda *a:NS(header=NS(stamp=NS(sec=99,nanosec=tf_nsec))))
                node.client=NS(service_is_ready=lambda:True,call_async=Mock(return_value=NS(add_done_callback=Mock())))
                node.waiting=node.report=Mock()
                node.tick()
                self.assertEqual(node.attempted,should_start)
                self.now=10.

    def test_diagnostics_does_not_report_short_gap_as_heartbeat_failure(self):
        from navigation_diagnostics import ControlHistory
        history=ControlHistory()
        history.status('localization_valid',True,10.)
        history.status('localization_state','tracking',10.)
        for elapsed,expected in [(.75,True),(1.,False)]:
            snapshot=history.snapshot(10.+elapsed)
            self.assertEqual(snapshot['latest']['localization_valid']['fresh'],expected)
            self.assertEqual(snapshot['latest']['localization_state']['fresh'],expected)

    def test_obstacle_stream_and_display_share_gap_and_explicit_false_rules(self):
        import display_cloud
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'metadata.yaml').write_text(json.dumps(dict(ground_model=dict(plane=[0,0,1,0]))))
            fixture = test_terrain_live.FixedPlaneCallbacks()
            module, node, pubs, _, _, _ = fixture.fake_runtime(folder)
            with patch.object(module.time, 'monotonic', side_effect=lambda: self.now):
                node.on_valid(NS(data=True))
                self.now = 10.75
                node.receive(message([[1,0,1]], frame='base_link', stamp=(99,900000000)))
                node.process()
                self.assertEqual(len(pubs['/navigation/obstacle_cloud'].messages), 1)
                self.assertTrue(display_cloud.scan_is_current(99_900_000_000,100_000_000_000,True,10.,self.now))
                node.on_valid(NS(data=False))
                node.receive(message([[1,0,1]], frame='base_link', stamp=(99,950000000)))
                node.process()
                self.assertEqual(len(pubs['/navigation/obstacle_cloud'].messages), 1)
                self.assertFalse(display_cloud.scan_is_current(99_900_000_000,100_000_000_000,False,10.75,self.now))
                node.valid, node.valid_received = True, 10.
                self.now = 11.
                node.receive(message([[1,0,1]], frame='base_link', stamp=(99,950000000)))
                node.process()
                self.assertEqual(len(pubs['/navigation/obstacle_cloud'].messages), 1)


if __name__ == '__main__':
    unittest.main()
