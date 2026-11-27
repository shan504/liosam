"""Solver-budget/cache geometry tests with a numerical Open3D API substitute.

These test indexing, geometry ownership and covariance algebra, not GICP
accuracy or timing on the robot's Open3D/Jetson installation.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np


class Cloud:
    normal_calls = 0

    def __init__(self):
        self.points = np.empty((0, 3))
        self.normals = np.empty((0, 3))
        self.covariances = np.empty((0, 3, 3))

    def voxel_down_sample(self, voxel):
        result = Cloud()
        if not len(self.points):
            return result
        cells = np.floor((self.points-self.points.min(axis=0))/voxel).astype(np.int64)
        _, inverse = np.unique(cells, axis=0, return_inverse=True)
        values = np.zeros((int(inverse.max())+1, 3))
        np.add.at(values, inverse, self.points)
        result.points = values/np.bincount(inverse)[:, None]
        return result

    def estimate_normals(self, search):
        Cloud.normal_calls += 1
        self.normals = np.tile([0., 0., 1.], (len(self.points), 1))


class Index:
    def __init__(self, owner):
        self.owner = owner

    def search_radius_vector_3d(self, center, radius):
        distance = np.sum((self.owner.points-center)**2, axis=1)
        indices = np.flatnonzero(distance < radius**2)
        return len(indices), indices.tolist(), distance[indices].tolist()


def load_module():
    substitute = types.SimpleNamespace(
        geometry=types.SimpleNamespace(PointCloud=Cloud, KDTreeFlann=Index,
                                       KDTreeSearchParamKNN=lambda count: count),
        utility=types.SimpleNamespace(Vector3dVector=lambda points: np.array(points, copy=True),
                                     Matrix3dVector=lambda values: np.array(values, copy=True)))
    path = Path(__file__).resolve().parents[1]/'original_go2/go2_3d/localization_target.py'
    spec = importlib.util.spec_from_file_location('tested_localization_target', path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {'open3d': substitute, spec.name: module}):
        spec.loader.exec_module(module)
    return module


target_module = load_module()


class LocalizationTargetTests(unittest.TestCase):
    def test_bounded_solver_inputs_preserve_original_map(self):
        generator = np.random.default_rng(72)
        for name, points in (
                ('small_room', generator.uniform([-4, -3, 0], [4, 3, 3], (12000, 3))),
                ('long_corridor', generator.uniform([80, -1, -0.3], [120, 1, 2.7], (22000, 3)))):
            with self.subTest(name=name):
                original = points.copy()
                points.setflags(write=False)
                result = target_module.bounded_cloud(points, .05, 600)
                self.assertGreater(len(result.points), 100)
                self.assertLessEqual(len(result.points), 600)
                np.testing.assert_array_equal(points, original)
                self.assertEqual(result.covariances.shape, (len(result.points), 3, 3))
                # Voxel budgets retain coverage on every spatial axis.
                np.testing.assert_array_less(.75*np.ptp(points, axis=0),
                                             np.ptp(result.points, axis=0))

    def test_cache_reuse_recenter_and_coordinate_independence(self):
        rng = np.random.default_rng(97)
        geometry = rng.uniform([-12, -8, -1], [12, 8, 3], (6000, 3))
        for offset in (np.zeros(3), np.array([91., -48., .32])):
            with self.subTest(offset=offset.tolist()):
                points = geometry + offset
                original = points.copy()
                target = target_module.LocalizationTarget(points, radius=10., recenter_distance=2.,
                    coarse_limit=300, fine_limit=800)
                initial = target.get(offset)
                normals_before = Cloud.normal_calls
                self.assertIs(target.get(offset+[1.9, 0, 0]), initial)
                self.assertEqual(Cloud.normal_calls, normals_before)
                self.assertEqual(initial.radius, 12.)
                self.assertIsNot(target.get(offset+[2.01, 0, 0]), initial)
                self.assertEqual(target.cache_hits, 1)
                self.assertEqual(target.cache_misses, 2)
                self.assertLessEqual(target.diagnostics()['target_coarse'], 300)
                self.assertLessEqual(target.diagnostics()['target_fine'], 800)
                np.testing.assert_array_equal(points, original)
                # Empty space returns an empty target; it never falls back to
                # unrelated full-map points to manufacture a match.
                empty = target.get(offset+[200, 0, 0])
                self.assertEqual(len(empty.fine.points), 0)

    def test_normal_sign_and_covariance_eigenvalues(self):
        result = Cloud()
        result.points = np.zeros((6, 3))
        normals = np.array([[1., 0, 0], [-1., 0, 0], [0, 1, 0], [0, -1., 0],
                            [1., 2, 3], [-1., -2, -3]])
        result.estimate_normals = lambda search: setattr(result, 'normals', normals)
        target_module.prepare_gicp_covariances(result)
        for pair in range(0, len(normals), 2):
            np.testing.assert_allclose(result.covariances[pair], result.covariances[pair+1])
        np.testing.assert_allclose(np.linalg.eigvalsh(result.covariances),
                                   np.tile([.001, 1., 1.], (6, 1)), atol=1e-14)

    def test_exact_fallback_budget_is_reproducible(self):
        points = np.random.default_rng(7).uniform(-10, 10, (501, 3))
        # Model an input whose voxel reduction does not converge within the
        # bounded retry budget. Exact fallback must still cover the extent.
        with mock.patch.object(Cloud, 'voxel_down_sample', lambda self, voxel: self):
            first = target_module.bounded_cloud(points, .01, 100)
            reversed_input = target_module.bounded_cloud(points[::-1], .01, 100)
        self.assertEqual(len(first.points), 100)
        np.testing.assert_array_equal(first.points, reversed_input.points)
        self.assertLessEqual(first.points[:, 0].min(), points[:, 0].min())
        self.assertGreaterEqual(first.points[:, 0].max(), points[:, 0].max())

    def test_invalid_inputs_fail_before_solver(self):
        points = np.zeros((100, 3))
        for voxel, limit in ((0, 100), (.1, 0), (.1, 100.5), (.1, float('inf'))):
            with self.subTest(voxel=voxel, limit=limit), self.assertRaises(ValueError):
                target_module.bounded_cloud(points, voxel, limit)
        for radius in (-1, float('nan')):
            with self.assertRaises(ValueError):
                target_module.LocalizationTarget(points, radius=radius)
        with self.assertRaises(ValueError):
            target_module.LocalizationTarget(np.full((100, 3), float('nan')))


if __name__ == '__main__':
    unittest.main()
