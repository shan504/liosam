"""Check localization orientation and radius selection independently of the map name."""
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
with patch.dict(sys.modules, {'scipy': None, 'open3d': None}):
    import gicp_localizer as loc


class LocalizationGeometry(unittest.TestCase):
    def test_yaw_wrap_uses_shortest_arc(self):
        start = np.eye(4)
        start[:3, :3] = loc.rotation_z(math.radians(179.))
        end = start.copy()
        end[:3, :3] = loc.rotation_z(math.radians(-179.))
        np.testing.assert_allclose(loc.difference(start, end), [0., 2.], atol=1e-8)
        midpoint = loc.blend_at_body(start, end, .5)
        self.assertAlmostEqual(abs(loc.yaw(midpoint)), math.pi, places=8)

    def test_blend_near_half_turn_in_arbitrary_body_frame(self):
        # An oblique axis and a rotated base distinguish body- from world-frame blending.
        axis = np.array([1., -2., 3.]) / math.sqrt(14.)
        start = loc.matrix([102., -47., 3.], [0., .5, 0., math.sqrt(.75)])
        for angle in (0., 1e-6, math.pi - 1e-5):
            with self.subTest(angle=angle):
                def step(fraction):
                    half_angle = angle * fraction * .5
                    q = np.r_[axis * math.sin(half_angle), math.cos(half_angle)]
                    return loc.matrix([0., 0., 0.], q)
                end = start @ step(1.)
                end[:3, 3] += [2., -2., 1.]
                expected = start @ step(.4)
                expected[:3, 3] += [.8, -.8, .4]
                np.testing.assert_allclose(loc.blend_at_body(start, end, .4), expected, atol=1e-8)
                np.testing.assert_allclose(loc.difference(start, end),
                                           [3., math.degrees(angle)], atol=1e-7)

    def test_quaternion_sign_and_known_basis(self):
        q = np.array([0., 0., math.sqrt(.5), math.sqrt(.5)])
        for sign in (-1., 1.):
            with self.subTest(sign=sign):
                transform = loc.matrix([1., 2., 3.], sign*q)
                np.testing.assert_allclose(transform @ [1., 0., 0., 1.], [1., 3., 3., 1.], atol=1e-12)
        with self.assertRaises(ValueError):
            loc.matrix([0., 0., 0.], [0., 0., 0., 0.])

    def test_open3d_submaps_use_3d_radius_and_refresh_after_motion(self):
        try:
            import open3d  # noqa: F401
        except ImportError as error:
            self.skipTest(f'Open3D integration requires the navigation image: {error}')
        rng = np.random.default_rng(82)
        inside = rng.uniform(-2., 2., size=(150, 3))
        second = rng.uniform(-2., 2., size=(150, 3)) + [40., 0., 0.]
        # This point lies inside the radius-aligned bounding cube but outside the sphere.
        outside = np.array([[12., 12., 12.]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            np.save(path/'target.npy', np.vstack((inside, second, outside)).astype('f4'))
            matcher = loc.Matcher(path)
            first = matcher.submap(np.array([0., 0., 0.]))
            first_points = np.asarray(first.points)
            self.assertGreater(len(first_points), 100)
            self.assertLessEqual(np.max(abs(first_points)), 2.)
            self.assertIs(first, matcher.submap(np.array([.25, 0., 0.])))
            moved = np.asarray(matcher.submap(np.array([40., 0., 0.])).points)
            self.assertGreater(len(moved), 100)
            self.assertGreaterEqual(moved[:, 0].min(), 38.)
            with self.assertRaises(ValueError):
                matcher.submap(np.array([100., 0., 0.]))


if __name__ == '__main__':
    unittest.main()
