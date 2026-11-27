"""Bounded, cached GICP inputs for any immutable saved localization map.

Only solver inputs are downsampled. The caller's complete map and the map used
for saving/display are never changed. This uses the legacy Open3D API shipped
with Humble, including its ability to consume precomputed GICP covariances.
"""
from dataclasses import dataclass

import numpy as np
import open3d as o3d


def _coordinates(points):
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError('localization points must have shape (N, 3)')
    if not np.isfinite(coordinates).all():
        raise ValueError('localization points must be finite')
    return coordinates


def _cloud(points):
    result = o3d.geometry.PointCloud()
    # Open3D 0.14 requires writable input even though Vector3dVector copies it.
    result.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64, copy=True))
    return result


def _budget(voxel, limit):
    if not np.isfinite(voxel) or voxel <= 0:
        raise ValueError('localization voxel must be finite and positive')
    if (isinstance(limit, bool) or not np.isfinite(limit) or
            int(limit) != limit or limit < 100):
        raise ValueError('localization point budget must be an integer >= 100')


def prepare_gicp_covariances(result, epsilon=0.001):
    """Cache regularized plane covariances used by Open3D's GICP fallback.

    Raw PCA covariances are deliberately not substituted: the normal direction
    gets eigenvalue epsilon and the two tangential directions get one. Normal
    signs therefore do not matter. GICP copies these covariances internally,
    avoiding repeated target normal/neighbourhood estimation on every scan.
    """
    if not np.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError('GICP covariance epsilon must be in (0, 1]')
    if not len(result.points):
        return result
    result.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    normals = np.asarray(result.normals, dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or np.any(lengths <= 1e-12):
        raise ValueError('cannot prepare GICP covariances from invalid normals')
    normals = normals / lengths[:, None]
    covariances = (np.eye(3)[None, :, :] +
                   (epsilon - 1.0) * normals[:, :, None] * normals[:, None, :])
    result.covariances = o3d.utility.Matrix3dVector(covariances)
    return result


def bounded_cloud(points, voxel, limit, prepare_covariances=True):
    """Keep spatial voxel coverage while bounding work independently of map size."""
    _budget(voxel, limit)
    coordinates = _coordinates(points)
    original = _cloud(coordinates)
    result = original.voxel_down_sample(float(voxel))
    # Revoxelize the original input, not already averaged centroids. This
    # avoids a cascade of biased centroids and preserves broad spatial cover.
    for _ in range(16):
        count = len(result.points)
        if count <= limit:
            break
        voxel *= max(1.25, (count / limit) ** (1.0 / 3.0))
        result = original.voxel_down_sample(float(voxel))
    if len(result.points) > limit:
        # Unusual extreme-coordinate clouds still have an exact CPU bound.
        # Canonical spatial ordering makes this fallback reproducible even if
        # Open3D changes the iteration order of its voxel hash table.
        values = np.asarray(result.points)
        order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
        selected = np.linspace(0, len(order)-1, int(limit), dtype=np.int64)
        result = _cloud(values[order[selected]])
    return prepare_gicp_covariances(result) if prepare_covariances else result


@dataclass(frozen=True)
class TargetClouds:
    coarse: object
    fine: object
    center: np.ndarray
    radius: float


class LocalizationTarget:
    """Index once; reuse bounded target geometry until the robot moves enough."""
    def __init__(self, points, radius=25.0, recenter_distance=2.0,
                 coarse_voxel=0.30, coarse_limit=8000,
                 fine_voxel=0.15, fine_limit=20000):
        for name, value in (('radius', radius), ('recenter_distance', recenter_distance)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'localization {name} must be finite and positive')
        _budget(coarse_voxel, coarse_limit)
        _budget(fine_voxel, fine_limit)
        coordinates = _coordinates(points)
        if len(coordinates) < 100:
            raise ValueError('localization map needs at least 100 points')
        self.radius, self.recenter_distance = float(radius), float(recenter_distance)
        self.coarse_voxel, self.coarse_limit = float(coarse_voxel), int(coarse_limit)
        self.fine_voxel, self.fine_limit = float(fine_voxel), int(fine_limit)
        # Keep the owner alive: legacy KDTreeFlann can reference cloud storage.
        self._map = _cloud(coordinates)
        self._points = np.asarray(self._map.points)
        self._points.setflags(write=False)
        self._index = o3d.geometry.KDTreeFlann(self._map)
        self._cached = None
        self.cache_hits = self.cache_misses = 0

    def get(self, center):
        center = np.asarray(center, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('localization target center must be a finite XYZ vector')
        if (self._cached is not None and
                np.linalg.norm(center-self._cached.center) <= self.recenter_distance):
            self.cache_hits += 1
            return self._cached
        # Query extra support once so every cache hit still contains the full
        # requested radius around its current pose, including at the boundary.
        support = self.radius + self.recenter_distance
        _, indices, _ = self._index.search_radius_vector_3d(center, support)
        nearby = self._points[np.sort(np.asarray(indices, dtype=np.int64))]
        coarse = bounded_cloud(nearby, self.coarse_voxel, self.coarse_limit)
        fine = bounded_cloud(nearby, self.fine_voxel, self.fine_limit)
        saved_center = center.copy()
        saved_center.setflags(write=False)
        cached = TargetClouds(coarse, fine, saved_center, support)
        self._cached = cached
        self.cache_misses += 1
        return cached

    def diagnostics(self):
        cached = self._cached
        return dict(cache_hits=self.cache_hits, cache_misses=self.cache_misses,
                    target_coarse=0 if cached is None else len(cached.coarse.points),
                    target_fine=0 if cached is None else len(cached.fine.points))
