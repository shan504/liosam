"""Cheap road-height reference from an explicitly surveyed body trajectory.

This does not fit point clouds and does not claim measured ground support.
The operator-confirmed corridor supplies the occupancy policy; body altitude
minus the saved standing height supplies a local live-obstacle height origin.
Only whole corridor cells and the surveyed robot footprint receive a height
reference. The latter does not grant free-space occupancy without real floor
returns; it prevents the robot's own support surface becoming a live obstacle.
"""
import math

import numpy as np

from terrain import TerrainModel
from surveyed_corridor import ContinuousCorridorIndex


def trajectory_reference(index, standing_height, resolution, cells, start_pose=None):
    if (not math.isfinite(standing_height) or not .1 < standing_height < 1. or
            not math.isfinite(resolution) or resolution <= 0.):
        raise ValueError('Invalid surveyed road reference calibration')
    paths = [np.asarray(path, dtype=float) for path in index.paths if len(path)]
    if not paths:
        raise ValueError('Surveyed road reference requires a recorded path')
    # A 1m square centered on the measured body contains the actual Nav2
    # footprint (+/-0.40m by +/-0.22m, with 0.01m padding). This supports
    # ground classification under the rear feet at the beginning of a route.
    # It is height reference only: callers do not add these cells to the
    # operator-confirmed free-cell set.
    footprint_paths = []
    for path in paths:
        shifted = path.copy()
        forward = shifted[:, :2, 0]
        forward = forward / np.linalg.norm(forward, axis=1)[:, None]
        shifted[:, :2, 3] -= .5 * forward
        footprint_paths.append(shifted)
    if start_pose is not None:
        pose = np.asarray(start_pose, dtype=float)
        shifted = pose.copy()
        forward = pose[:2, 0] / np.linalg.norm(pose[:2, 0])
        shifted[:2, 3] -= .5 * forward
        footprint_paths.append(np.asarray([shifted]))
        paths.append(np.asarray([pose]))
    body_footprints = ContinuousCorridorIndex([], footprint_paths)
    reference_cells = set(cells)
    for batch in body_footprints.free_cells(resolution):
        reference_cells.update(map(tuple, batch))
    cells = np.asarray(sorted(reference_cells), dtype=np.int64).reshape(-1, 2)
    segments = []
    for path in paths:
        positions = path[:, :3, 3]
        if len(path) == 1:
            segments.append((positions[0], positions[0]))
        else:
            segments.extend(zip(positions[:-1], positions[1:]))
    starts = np.asarray([a for a, _ in segments])
    ends = np.asarray([b for _, b in segments])
    delta = ends - starts
    lengths2 = np.sum(delta[:, :2] ** 2, axis=1)

    # A 1m forward, 0.5m half-width footprint lies within sqrt(1.25)m
    # of its body center. Index path segments locally rather than allocating
    # a map-cell by trajectory-point distance matrix.
    radius = math.sqrt(1.25) + 2 * resolution
    buckets = {}
    for n, (a, b) in enumerate(zip(starts, ends)):
        lower = np.floor(np.minimum(a[:2], b[:2]) - radius).astype(np.int64)
        upper = np.floor(np.maximum(a[:2], b[:2]) + radius).astype(np.int64)
        for x in range(int(lower[0]), int(upper[0]) + 1):
            for y in range(int(lower[1]), int(upper[1]) + 1):
                buckets.setdefault((x, y), []).append(n)
    centers = (cells.astype(float) + .5) * resolution
    bins = np.floor(centers).astype(np.int64)
    order = np.lexsort((bins[:, 1], bins[:, 0]))
    ordered = bins[order]
    breaks = np.flatnonzero(np.r_[True, np.any(ordered[1:] != ordered[:-1], axis=1)]) if len(order) else []
    heights = np.full(len(cells), np.nan)
    ambiguous = np.zeros(len(cells), dtype=bool)
    for begin, end in zip(breaks, np.r_[breaks[1:], len(order)]):
        rows = order[begin:end]
        xy = centers[rows]
        best = np.full(len(rows), np.inf)
        level = np.full(len(rows), np.nan)
        conflict = np.zeros(len(rows), dtype=bool)
        for candidate in buckets.get(tuple(ordered[begin]), ()):
            if lengths2[candidate] > 1e-12:
                fraction = np.clip((xy - starts[candidate, :2]) @ delta[candidate, :2] /
                                   lengths2[candidate], 0., 1.)
            else:
                fraction = np.zeros(len(rows))
            closest = starts[candidate, :2] + fraction[:, None] * delta[candidate, :2]
            distance = np.sum((xy - closest) ** 2, axis=1)
            z = starts[candidate, 2] + fraction * delta[candidate, 2] - standing_height
            better = distance < best - 1e-10
            conflict[better] = False
            level[better] = z[better]
            best[better] = distance[better]
            # Distinct levels crossing at the same XY cannot be represented
            # by a single 2D road surface. Leave their reference unsupported.
            tied = np.abs(distance - best) <= 1e-10
            conflict |= tied & (np.abs(z - level) > .10)
        heights[rows] = level
        ambiguous[rows] = conflict
    good = np.isfinite(heights) & ~ambiguous
    # Five-centimetre constant cells avoid fitting any surfaces or extrapolating
    # a sparse body tilt into the surrounding field. Actual XYZ stays unchanged.
    planes = np.column_stack((np.zeros((np.count_nonzero(good), 2)), heights[good]))
    statistics = dict(candidate_cells=len(cells), supported_cells=int(np.count_nonzero(good)),
                      discontinuity_cells=int(np.count_nonzero(ambiguous)),
                      evidence='operator_trajectory_reference', path_segments=len(segments),
                      footprint_reference_only=True)
    return TerrainModel(cells[good], planes, parameters=dict(resolution=resolution),
                        statistics=statistics)
