"""Native Open3D regression checks for bounded map localization.

The two synthetic structures and coordinate conventions exercise the same
solver entry points used for any map. Timing is printed for this host only;
it is not a Jetson/Go2 real-time guarantee. ROS commit/freshness tests live in
the separate mapper tests.
"""
import importlib.util
from pathlib import Path
import sys
import time
import unittest

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


ROOT = Path(__file__).resolve().parents[1]/'original_go2/go2_3d'


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if o3d is not None:
    geometry = _load('geometry', 'geometry.py')
    registration = _load('native_gicp_registration', 'registration.py')
    target_module = _load('native_gicp_localization_target', 'localization_target.py')


def pose(xyz, roll=0., pitch=0., yaw=0.):
    roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                      [-sp, cp*sr, cp*cr]]
    result[:3, 3] = xyz
    return result


def plane(rng, axis, coordinate, first, second, count):
    result = np.empty((count, 3))
    result[:, axis] = coordinate
    other = [index for index in range(3) if index != axis]
    result[:, other[0]] = rng.uniform(*first, count)
    result[:, other[1]] = rng.uniform(*second, count)
    return result


def box(rng, lower, upper, count=250):
    result = []
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        for side in (lower[axis], upper[axis]):
            result.append(plane(rng, axis, side,
                                (lower[other[0]], upper[other[0]]),
                                (lower[other[1]], upper[other[1]]), count))
    return result


def synthetic_maps():
    rng = np.random.default_rng(765)
    room = [plane(rng, 2, 0., (-5., 6.), (-4., 5.), 5000),
            plane(rng, 0, -5., (-4., 5.), (0., 3.), 2200),
            plane(rng, 0, 6., (-4., 5.), (0., 3.), 2200),
            plane(rng, 1, -4., (-5., 6.), (0., 3.), 2200),
            plane(rng, 1, 5., (-5., 6.), (0., 3.), 2200)]
    room += box(rng, [-2.7, 1.2, 0.], [-1.4, 2.3, 1.7])
    room += box(rng, [2.2, -1.6, 0.], [3.1, -.2, .9])
    farm = [plane(rng, 2, 0., (-9., 9.), (-4., 4.), 5500),
            plane(rng, 0, 8., (-4., 4.), (0., 4.), 2500),
            plane(rng, 1, -3.5, (-9., 8.), (0., 2.4), 2500)]
    for x, y, height in [(-6., 3.2, 3.5), (-2.5, 2.8, 2.2),
                         (1.3, 3.1, 3.7), (5.8, 2.9, 2.9)]:
        farm += box(rng, [x-.25, y-.25, 0.], [x+.25, y+.25, height], 180)
    farm += box(rng, [3.5, -.8, 0.], [4.4, .4, 1.1], 300)
    return [('synthetic_room', np.vstack(room)), ('synthetic_farm', np.vstack(farm))]


@unittest.skipIf(o3d is None, 'native Open3D is not installed')
class NativeLocalizationGICP(unittest.TestCase):
    def test_two_maps_two_frames_capture_then_reuse(self):
        rng = np.random.default_rng(82)
        for name, local_map in synthetic_maps():
            for frame_name, frame in (
                    ('raw_body_origin', pose([0., 0., -.31])),
                    ('ground_aligned_offset', pose([19., -13., 0.], yaw=31.))):
                with self.subTest(map=name, frame=frame_name):
                    started = time.perf_counter()
                    full_map = geometry.transform_points(local_map, frame)
                    preserved = full_map.copy()
                    full_map.setflags(write=False)
                    actual = frame @ pose([.3, -.2, .31], roll=1.4, pitch=-2., yaw=8.)
                    scan = geometry.transform_points(full_map, np.linalg.inv(actual))
                    scan = scan[np.linalg.norm(scan, axis=1) < 12.]
                    scan = scan[rng.random(len(scan)) > .15]
                    scan += rng.normal(0., .006, scan.shape)
                    guess = actual @ pose([.5, 0., 0.], yaw=15.)
                    # The production mapper builds this index from its
                    # already voxelized localization map, not the raw PCD.
                    map_input = np.asarray(registration.cloud(full_map, .15).points)
                    target = target_module.LocalizationTarget(map_input)
                    pair = target.get(guess[:3, 3])
                    cold_elapsed = time.perf_counter()-started
                    solve_started = time.perf_counter()
                    coarse_source = target_module.bounded_cloud(scan, .30, 3000)
                    coarse = registration.align(coarse_source, pair.coarse, guess,
                        correspondence=1.2, min_fitness=.20, max_rmse=.60,
                        max_translation=3., max_rotation=np.deg2rad(90.), iterations=20,
                        relative_fitness=1e-4, relative_rmse=1e-4)
                    self.assertTrue(coarse.accepted, coarse.reason)
                    fine_source = target_module.bounded_cloud(scan, .15, 6000)
                    fine = registration.align(fine_source, pair.fine, coarse.transform,
                        correspondence=.70, min_fitness=.20, max_rmse=.20,
                        max_translation=3., max_rotation=np.deg2rad(90.), iterations=20,
                        innovation_guess=guess, relative_fitness=1e-4, relative_rmse=1e-4)
                    self.assertTrue(fine.accepted, fine.reason)
                    error = np.linalg.inv(actual) @ fine.transform
                    distance = np.linalg.norm(error[:3, 3])
                    angle = np.degrees(geometry.rotation_angle(error))
                    self.assertLess(distance, .06)
                    self.assertLess(angle, 1.)
                    solve_elapsed = time.perf_counter()-solve_started
                    self.assertIs(target.get(actual[:3, 3]), pair)
                    self.assertEqual(target.cache_hits, 1)
                    self.assertEqual(target.cache_misses, 1)
                    np.testing.assert_array_equal(full_map, preserved)
                    self.assertLessEqual(len(coarse_source.points), 3000)
                    self.assertLessEqual(len(fine_source.points), 6000)
                    self.assertLessEqual(len(pair.coarse.points), 8000)
                    self.assertLessEqual(len(pair.fine.points), 20000)
                    print(f'Open3D {o3d.__version__} {name}/{frame_name}: '
                          f'cold={cold_elapsed:.3f}s solve={solve_elapsed:.3f}s '
                          f'error={distance:.4f}m/{angle:.3f}deg '
                          f'source={len(fine_source.points)} target={len(pair.fine.points)}')

    def test_dense_solver_input_hits_budget_without_modifying_input(self):
        points = np.random.default_rng(906).uniform([-8., -6., 0.], [8., 6., 4.], (150000, 3))
        before = points.copy()
        self.assertGreater(len(registration.cloud(points, .15).points), 20000)
        started = time.perf_counter()
        bounded = target_module.bounded_cloud(points, .15, 6000)
        self.assertGreater(len(bounded.points), 100)
        self.assertLessEqual(len(bounded.points), 6000)
        self.assertEqual(len(bounded.covariances), len(bounded.points))
        np.testing.assert_array_equal(points, before)
        np.testing.assert_array_less(.8*np.ptp(points, axis=0), np.ptp(bounded.points, axis=0))
        print(f'Open3D {o3d.__version__} dense 150000 -> {len(bounded.points)} points '
              f'in {time.perf_counter()-started:.3f}s')

    def test_nonoverlap_stays_rejected(self):
        _, points = synthetic_maps()[0]
        target = target_module.LocalizationTarget(points).get([0., 0., 0.])
        source = target_module.bounded_cloud(points+[70., 0., 0.], .15, 6000)
        result = registration.align(source, target.fine, np.eye(4),
            correspondence=.70, min_fitness=.20, max_rmse=.20,
            max_translation=3., max_rotation=np.deg2rad(90.), iterations=20)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, 'overlap/residual rejected')


if __name__ == '__main__':
    unittest.main()
