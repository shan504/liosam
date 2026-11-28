"""Measured local ground for navigation, independent of saved corridor clearing.

Fit only a nearby part of the current, timestamp-aligned scan. The fit radius
limits estimation work, not sensor output or the obstacle detection range.
Unobserved or discontinuous surfaces keep their saved-map classification.
"""
from collections import deque
import math

import numpy as np

from terrain import TerrainModel, _candidate, _cells, _options


def measured_ground(points, body_position, standing_height, radius=4.0):
    """Return supported local planes or None; never invent free-space returns."""
    points = np.asarray(points, dtype=float)
    body = np.asarray(body_position, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or body.shape != (3,) or
            not np.isfinite(body).all() or not math.isfinite(standing_height) or
            not .1 < standing_height < 1. or not math.isfinite(radius) or radius <= 0.):
        raise ValueError('Invalid live terrain geometry')
    p = _options()
    nearby = np.isfinite(points).all(axis=1) & (np.linalg.norm(points[:, :2]-body[:2], axis=1) <= radius)
    source = points[nearby]
    if len(source) < p['min_points']:
        return None
    # Recenter XY for stable statistics on maps far from their origin. Retain
    # the map's grid lattice so a given physical cell has one meaning.
    keys = np.column_stack((_cells(source[:, :2], p['resolution']),
                            _cells(source[:, 2:3], p['vertical_bin'])[:, 0]))
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys, source = keys[order], source[order]
    local = source.copy()
    local[:, :2] -= (keys[:, :2]+.5)*p['resolution']
    x,y,z = local.T
    moments = np.column_stack((np.ones(len(local)),x,y,z,x*x,x*y,x*z,y*y,y*z,z*z))
    begin = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
    sums = np.add.reduceat(moments, begin, axis=0)
    lower, upper = np.minimum.reduceat(local[:, :2], begin), np.maximum.reduceat(local[:, :2], begin)
    bins = keys[begin]
    stats = np.column_stack((bins[:, 2], sums, lower[:, 0],upper[:, 0],lower[:, 1],upper[:, 1]))
    columns = np.flatnonzero(np.r_[True,np.any(bins[1:, :2] != bins[:-1, :2], axis=1)])
    candidates = {}
    for first,last in zip(columns,np.r_[columns[1:],len(bins)]):
        plane = _candidate(stats[first:last],p)
        if plane is not None:
            candidates[tuple(bins[first,:2])] = plane
    anchor = body.copy()
    anchor[2] -= standing_height
    seeds = set()
    for cell,plane in candidates.items():
        delta = anchor[:2]-(np.asarray(cell)+.5)*p['resolution']
        if (np.linalg.norm(delta) <= p['anchor_radius'] and
                abs(plane[2]+plane[:2]@delta-anchor[2]) <= p['anchor_height_tolerance']):
            seeds.add(cell)
    if not seeds:
        return None
    def neighbors(cell):
        x,y = cell
        return ((x+1,y),(x-1,y),(x,y+1),(x,y-1))
    def agrees(a,b):
        delta = (np.asarray(b)-a)*p['resolution']
        first,second = candidates[a],candidates[b]
        return max(abs(first[2]+first[:2]@delta-second[2]),
                   abs(second[2]-second[:2]@delta-first[2])) <= p['continuity_tolerance']
    accepted,queue = set(seeds),deque(sorted(seeds))
    while queue:
        cell = queue.popleft()
        for neighbor in neighbors(cell):
            if neighbor in candidates and neighbor not in accepted and agrees(cell,neighbor):
                accepted.add(neighbor)
                queue.append(neighbor)
    # Leave step/drop boundaries unsupported; do not reinterpret them as ramps.
    discontinuities = set()
    for cell in accepted:
        for neighbor in neighbors(cell):
            if neighbor in candidates and not agrees(cell,neighbor):
                discontinuities.add(cell)
                if neighbor in accepted:
                    discontinuities.add(neighbor)
    accepted -= discontinuities
    if not accepted:
        return None
    cells = sorted(accepted)
    return TerrainModel(cells,[candidates[cell] for cell in cells],p,
                        dict(evidence='current_scan_measured_ground',observations=len(source),
                             candidate_cells=len(candidates),supported_cells=len(cells),
                             anchor_cells=len(seeds),discontinuity_cells=len(discontinuities)))


def relative_heights(points, plane, saved=None, measured=None):
    """Prefer directly measured local surface; preserve all unsupported flags."""
    points = np.asarray(points)
    if saved is None:
        normal = np.asarray(plane['plane'])
        heights = points@normal[:3]+normal[3]
        support = np.isfinite(points).all(axis=1)
    else:
        heights,support = saved.evaluate(points)
    heights,support = heights.copy(),support.copy()
    live_support = np.zeros(len(points),dtype=bool)
    if measured is not None:
        actual,live_support = measured.evaluate(points)
        heights[live_support],support[live_support] = actual[live_support],True
    return heights,support,live_support
