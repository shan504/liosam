"""Bound planar dead reckoning by the last accepted scan, never a rejection."""
import math
import numpy as np


def planar_sample(odom):
    """Validate native pose before taking XY/yaw; a vertical heading is undefined."""
    pose = np.asarray(odom, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
            not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-7) or
            not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-5) or
            np.linalg.norm(pose[:2, 0]) < 1e-6):
        raise ValueError('Invalid native odometry pose or undefined planar heading')
    return pose, math.atan2(pose[1, 0], pose[0, 0])


class OdometryBudget:
    def __init__(self):
        self.reset(None, None)

    def reset(self, odom, stamp):
        if odom is not None:
            odom, _ = planar_sample(odom)
            if stamp is None or not np.isfinite(stamp):
                raise ValueError('Invalid native odometry timestamp')
        self.previous = None if odom is None else odom.copy()
        self.stamp = stamp
        self.distance = self.angle = 0.0

    def observe(self, odom, stamp):
        odom, yaw = planar_sample(odom)
        if stamp is None or not np.isfinite(stamp):
            raise ValueError('Invalid native odometry timestamp')
        if self.previous is None or stamp < self.stamp:
            return False
        if stamp > self.stamp:
            # Navigation progress is native-odom XY and heading. Vertical
            # walking bounce and roll/pitch must not consume turning budget.
            # Sum observed forward/backward and left/right increments:
            # returning to the accepted pose does not replenish allowance.
            self.distance += float(np.linalg.norm(odom[:2, 3]-self.previous[:2, 3]))
            previous_yaw = math.atan2(self.previous[1, 0], self.previous[0, 0])
            self.angle += abs(math.atan2(math.sin(yaw-previous_yaw), math.cos(yaw-previous_yaw)))
            self.previous, self.stamp = odom.copy(), stamp
        return True

    def exceeded(self, distance, angle):
        # Accumulate travel, including reversals; returning to the starting
        # pose cannot replenish the dead-reckoning budget.
        if self.distance > distance:
            return 'odometry_distance_limit'
        if self.angle > angle:
            return 'odometry_angle_limit'
        return ''
