"""Sparse XY component filtering shared by obstacle clouds and their displays.

Groups count occupied grid cells, not repeated returns. The existing navigation
filter policy removes groups with fewer than eight 5 cm cells, including diagonal
connections. There is no allocation proportional to the map's bounding rectangle.
This intentionally can also discard small real objects; it is not object recognition.
"""
import math

import numpy as np

from denoise import settings


def cell_keys(points_xy, resolution=0.05):
    """Quantize finite XY rows on the common floor(x/resolution) lattice."""
    points_xy = np.asarray(points_xy)
    if (points_xy.ndim != 2 or points_xy.shape[1] != 2 or
            not np.isfinite(points_xy).all() or not math.isfinite(resolution) or resolution <= 0):
        raise ValueError('Invalid point-noise XY geometry')
    scaled = np.floor(points_xy.astype(np.float64) / resolution)
    if np.any(np.abs(scaled) >= np.iinfo(np.int64).max):
        raise ValueError('Point-noise cell coordinates cannot be represented')
    return scaled.astype(np.int64)


def retained_cells(cells, min_cells=None, connectivity=None):
    """Return the cells belonging to sufficiently large whole-map components.

    Callers processing static point-cloud blocks first collect the complete cell
    inventory, then call this once. Filtering each block separately would split
    real objects at serialization boundaries.
    """
    policy = settings() if min_cells is None or connectivity is None else {}
    minimum = policy.get('minimal_group_size') if min_cells is None else min_cells
    connectivity = policy.get('group_connectivity_type') if connectivity is None else connectivity
    if type(minimum) is not int or minimum < 1 or connectivity not in (4, 8):
        raise ValueError('Invalid point-noise component policy')
    remaining = set()
    for cell in cells:
        if len(cell) != 2 or any(not isinstance(value, (int, np.integer)) for value in cell):
            raise ValueError('Point-noise cells must be integer XY pairs')
        remaining.add((int(cell[0]), int(cell[1])))
    if minimum == 1:
        return remaining
    offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               if (dx or dy) and (connectivity == 8 or abs(dx) + abs(dy) == 1)]
    retained = set()
    while remaining:
        first = remaining.pop()
        pending, small = [first], [first]
        large = False
        while pending:
            x, y = pending.pop()
            for dx, dy in offsets:
                neighbor = (x + dx, y + dy)
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                pending.append(neighbor)
                if large:
                    retained.add(neighbor)
                else:
                    small.append(neighbor)
                    if len(small) >= minimum:
                        retained.update(small)
                        small.clear()
                        large = True
    return retained


def keep_connected(points_xy, resolution=0.05, min_cells=None, connectivity=None):
    """Return a row mask without changing point order, XYZ or other fields."""
    points_xy = np.asarray(points_xy)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError('Point-noise filter needs XY rows')
    finite = np.isfinite(points_xy).all(axis=1)
    result = np.zeros(len(points_xy), dtype=bool)
    cells = cell_keys(points_xy[finite], resolution)
    accepted = retained_cells(np.unique(cells, axis=0), min_cells, connectivity)
    result[finite] = np.fromiter((tuple(cell) in accepted for cell in cells),
                                 dtype=bool, count=len(cells))
    return result
