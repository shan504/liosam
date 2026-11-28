"""Source-stamped recording of an operator-confirmed mapping corridor.

No ROS imports, control commands, pose interpolation, or invented scan origins.
Both lidar poses and a captured scan/anchor must share the exact source stamp.
"""
from collections import OrderedDict
from threading import Lock
import math
import time

import numpy as np


def pose_matrix(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError('Corridor needs a finite XYZ/XYZW pose')
    x, y, z, w = values[3:]
    norm = x*x+y*y+z*z+w*w
    if not math.isclose(norm, 1., rel_tol=0., abs_tol=.001):
        raise ValueError('Corridor pose quaternion must be unit length')
    s = 2./norm
    result = np.eye(4)
    result[:3, :3] = [[1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
                      [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
                      [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)]]
    result[:3, 3] = values[:3]
    return result


def mount_from_parameter(value):
    if not isinstance(value, dict) or value.get('parent') != 'base_link':
        raise ValueError('Corridor needs the physical base_link mounting calibration')
    child = value.get('child')
    if not isinstance(child, str) or not child or child.lstrip('/') in ('base_link', 'go2_native/base_link'):
        raise ValueError('Corridor mounting child must be the physical lidar frame')
    pose = list(value['xyz']) + list(value['xyzw'])
    pose_matrix(pose)
    return dict(parent='base_link', child=child.lstrip('/'),
                xyz=[float(v) for v in pose[:3]], xyzw=[float(v) for v in pose[3:]])


class CorridorRecorder:
    """Join bounded transport queues, then record only measured forward motion."""
    MAX_AGE_NS = 1_000_000_000
    FUTURE_NS = 100_000_000

    def __init__(self, mount, *, motion_frame='go2_native/odom', same_pose_source=False,
                 maximum_wait=.75, minimum_forward=.005, maximum_step_time=.5):
        self.mount = mount_from_parameter(mount)
        self.inverse_mount = np.linalg.inv(pose_matrix(self.mount['xyz']+self.mount['xyzw']))
        self.motion_frame = motion_frame.lstrip('/') if motion_frame else None
        self.same_pose_source = bool(same_pose_source)
        self.maximum_wait = maximum_wait
        self.minimum_forward = minimum_forward
        self.maximum_step_time = maximum_step_time
        self.guard = Lock()
        self.poses = {name: OrderedDict() for name in ('world', 'motion')}
        self.pending = OrderedDict()
        self.child_frames = {}
        self.last_stamp = 0
        self.previous = None
        self.rejected = self.expired = self.confirmed = 0

    @classmethod
    def fresh(cls, stamp, now_ns):
        return type(stamp) is int and stamp > 0 and -cls.FUTURE_NS <= now_ns-stamp <= cls.MAX_AGE_NS

    def pose(self, source, stamp, frame, child, values, *, now_ns):
        if source not in self.poses:
            raise ValueError('Unknown corridor pose source')
        try:
            matrix = pose_matrix(values)
            frame, child = frame.lstrip('/'), child.lstrip('/')
            if not frame or not child or not self.fresh(stamp, now_ns):
                raise ValueError('Invalid or stale corridor pose header')
            if source == 'motion' and not self.same_pose_source:
                if frame != self.motion_frame or child != self.mount['child']:
                    raise ValueError('Native corridor motion pose frame mismatch')
            if child in ('base_link', 'go2_native/base_link'):
                raise ValueError('Corridor inputs must be lidar poses, not body poses')
        except (TypeError, ValueError, AttributeError):
            with self.guard:
                self.rejected += 1
            return False
        with self.guard:
            if stamp <= self.last_stamp or source in self.child_frames and self.child_frames[source] != child:
                self.rejected += 1
                return False
            self.child_frames.setdefault(source, child)
            buffer = self.poses[source]
            if stamp in buffer:
                return False
            buffer[stamp] = (frame, matrix)
            while len(buffer) > 256:
                buffer.popitem(last=False)
        return True

    def scan(self, stamp, anchor, frame, anchor_pose, *, now_ns, receipt=None):
        """Only call after this exact scan has been appended to the raw archive."""
        try:
            matrix = pose_matrix(anchor_pose)
            frame = frame.lstrip('/')
            if not frame or type(anchor) is not int or anchor <= 0 or not self.fresh(stamp, now_ns):
                raise ValueError('Invalid or stale corridor scan header')
        except (TypeError, ValueError, AttributeError):
            with self.guard:
                self.rejected += 1
            return False
        with self.guard:
            if stamp <= self.last_stamp or stamp in self.pending:
                return False
            self.pending[stamp] = (anchor, frame, matrix, time.monotonic() if receipt is None else receipt)
            while len(self.pending) > 128:
                self.pending.popitem(last=False)
                self.previous = None
                self.expired += 1
        return True

    def process(self, *, now_ns, now=None):
        """Return (stamp, anchor, T_anchor_body) events; missing data creates none."""
        now = time.monotonic() if now is None else now
        events = []
        with self.guard:
            for stamp in sorted(self.pending):
                anchor, frame, anchor_pose, receipt = self.pending[stamp]
                world = self.poses['world'].get(stamp)
                motion = self.poses['motion'].get(stamp)
                if (not self.fresh(stamp, now_ns) or now-receipt > self.maximum_wait):
                    self.pending.pop(stamp)
                    self.last_stamp = max(self.last_stamp, stamp)
                    self.previous = None
                    self.expired += 1
                    continue
                if world is None or motion is None:
                    break  # Preserve chronology when exact-stamp messages arrive late.
                self.pending.pop(stamp)
                self.last_stamp = max(self.last_stamp, stamp)
                expected_motion = frame if self.same_pose_source else self.motion_frame
                if world[0] != frame or motion[0] != expected_motion:
                    self.rejected += 1
                    self.previous = None
                    continue
                world_body, motion_body = world[1] @ self.inverse_mount, motion[1] @ self.inverse_mount
                previous, self.previous = self.previous, (stamp, motion_body)
                if previous is not None:
                    old_stamp, old_body = previous
                    dt = (stamp-old_stamp)/1e9
                    forward = old_body[:2, 0]
                    length = float(np.linalg.norm(forward))
                    if length > 1e-6 and 0 < dt <= self.maximum_step_time:
                        distance = float((motion_body[:2, 3]-old_body[:2, 3]) @ (forward/length))
                        if distance > self.minimum_forward:
                            events.append((stamp, anchor, np.linalg.inv(anchor_pose) @ world_body))
                            self.confirmed += 1
            for buffer in self.poses.values():
                for stamp in list(buffer):
                    if stamp <= self.last_stamp:
                        buffer.pop(stamp)
        return events
