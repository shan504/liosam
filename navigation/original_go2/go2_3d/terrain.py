"""Observed, trajectory-anchored local ground for height-varying maps.

This is a small NumPy implementation of a sparse 2.5-D terrain model, not
Patchwork++ or a replacement for Nav2 collision checking. It keeps only local
low clusters with two-dimensional support, a gentle fitted slope, a nearby
physical foot-height anchor, or a continuous connection to such an anchor.
Unobserved cells, sharp height discontinuities and unsupported surfaces remain
unknown. No global map area, height or point-count limit is imposed.
"""
from collections import deque
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

VERSION = 1
DEFAULTS = dict(resolution=0.4, vertical_bin=0.02, min_points=6,
                max_slope_deg=25.0, max_fit_rmse=0.025,
                continuity_tolerance=0.06, anchor_radius=0.8,
                anchor_height_tolerance=0.08)


def _options(parameters=None):
    result = dict(DEFAULTS)
    if parameters:
        if set(parameters) - set(result):
            raise ValueError('Unknown terrain parameter')
        result.update(parameters)
    if (type(result['min_points']) is not int or result['min_points'] < 3 or
            any(not math.isfinite(float(v)) or float(v) <= 0 for v in result.values()) or
            not 0 < result['max_slope_deg'] < 45):
        raise ValueError('Invalid terrain fit parameters')
    return result


def _points(values, dimensions):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < dimensions:
        raise ValueError(f'Terrain expects Nx{dimensions} coordinates')
    return values[:, :dimensions]


def _cells(xy, resolution):
    indices = np.floor(xy / resolution)
    if not np.isfinite(indices).all() or (np.abs(indices) >= np.iinfo(np.int64).max).any():
        raise ValueError('Terrain coordinates exceed representable integer cell indices')
    return indices.astype(np.int64)


def _structured(cells):
    result = np.empty(len(cells), dtype=[('x', '<i8'), ('y', '<i8')])
    result['x'], result['y'] = cells[:, 0], cells[:, 1]
    return result


class TerrainModel:
    def __init__(self, cells, planes, parameters=None, statistics=None):
        self.parameters = _options(parameters)
        self.resolution = float(self.parameters['resolution'])
        self.cells = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
        self.planes = np.asarray(planes, dtype=np.float64).reshape(-1, 3)
        if (len(self.cells) != len(self.planes) or not np.isfinite(self.planes).all() or
                (np.linalg.norm(self.planes[:, :2], axis=1) >
                 math.tan(math.radians(self.parameters['max_slope_deg'])) + 1e-8).any()):
            raise ValueError('Invalid local terrain planes')
        order = np.lexsort((self.cells[:, 1], self.cells[:, 0]))
        self.cells, self.planes = self.cells[order], self.planes[order]
        self.keys = _structured(self.cells)
        if len(self.keys) > 1 and np.any(self.keys[1:] == self.keys[:-1]):
            raise ValueError('Duplicate terrain cells')
        self.statistics = dict(statistics or {})

    def height_at(self, xy):
        """Return (ground_z, local_model_support), never across missing cells.

        A supported 0.4m cell is a fitted local surface, not evidence that all
        its smaller occupancy pixels are free. Callers must retain their actual
        observed-return/ray evidence policy instead of filling supported cells.
        """
        xy = _points(xy, 2)
        z = np.full(len(xy), np.nan)
        supported = np.zeros(len(xy), dtype=bool)
        finite = np.isfinite(xy).all(axis=1)
        if not finite.any() or not len(self.keys):
            return z, supported
        rows = np.flatnonzero(finite)
        query_cells = _cells(xy[finite], self.resolution)
        keys = _structured(query_cells)
        positions = np.searchsorted(self.keys, keys)
        inside = positions < len(self.keys)
        matches = np.zeros(len(keys), dtype=bool)
        matches[inside] = self.keys[positions[inside]] == keys[inside]
        rows = rows[matches]
        positions = positions[matches]
        if len(rows):
            center = (self.cells[positions].astype(np.float64) + .5) * self.resolution
            plane = self.planes[positions]
            z[rows] = plane[:, 2] + np.sum(plane[:, :2] * (xy[rows] - center), axis=1)
            supported[rows] = True
        return z, supported

    def evaluate(self, points):
        """Return signed vertical point height and support for each map point."""
        points = _points(points, 3)
        z, supported = self.height_at(points[:, :2])
        supported &= np.isfinite(points[:, 2])
        height = points[:, 2] - z
        height[~supported] = np.nan
        return height, supported

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.terrain-', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez_compressed(stream, version=np.array(VERSION, dtype=np.int32),
                                    cells=self.cells, planes=self.planes,
                                    parameters=np.array(json.dumps(self.parameters, sort_keys=True)),
                                    statistics=np.array(json.dumps(self.statistics, sort_keys=True)))
                stream.flush()
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)
        return dict(version=VERSION, file=path.name, **self.statistics,
                    parameters=dict(self.parameters))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'version', 'cells', 'planes', 'parameters', 'statistics'}:
                raise ValueError('Malformed terrain model archive')
            if data['version'].shape != () or int(data['version']) != VERSION:
                raise ValueError('Unsupported terrain model version')
            cells = data['cells']
            if cells.dtype.kind not in 'iu' or cells.ndim != 2 or cells.shape[1] != 2:
                raise ValueError('Invalid terrain cell indices')
            return cls(cells, data['planes'], json.loads(str(data['parameters'])),
                       json.loads(str(data['statistics'])))


def load_terrain(directory, metadata):
    """Load an explicit terrain contract; absent legacy metadata returns None."""
    descriptor = metadata.get('terrain_model')
    if descriptor is None:
        return None
    if not isinstance(descriptor, dict) or descriptor.get('version') != VERSION:
        raise ValueError('Invalid terrain model descriptor')
    filename = descriptor.get('file')
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError('Terrain model file must be a local bundle filename')
    directory = Path(directory).resolve()
    path = (directory / filename).resolve(strict=True)
    if path.parent != directory:
        raise ValueError('Terrain model file escapes its map bundle')
    return TerrainModel.load(path)


def _accumulate(db, points, p):
    # All observations contribute to on-disk statistics. Arrays are bounded by
    # the caller's chunk, not total map size; no reservoir or point truncation.
    points = _points(points, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return 0
    xy = _cells(points[:, :2], p['resolution'])
    z = _cells(points[:, 2:3], p['vertical_bin'])[:, 0]
    keys = np.column_stack((xy, z))
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys, points = keys[order], points[order]
    local = points.copy()
    local[:, :2] -= (keys[:, :2].astype(np.float64) + .5) * p['resolution']
    x, y, z = local.T
    moments = np.column_stack((np.ones(len(local)), x, y, z,
                              x*x, x*y, x*z, y*y, y*z, z*z))
    starts = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
    sums = np.add.reduceat(moments, starts, axis=0)
    low = np.minimum.reduceat(local[:, :2], starts, axis=0)
    high = np.maximum.reduceat(local[:, :2], starts, axis=0)
    rows = [(*map(int, key), *map(float, total), float(lo[0]), float(hi[0]),
             float(lo[1]), float(hi[1]))
            for key, total, lo, hi in zip(keys[starts], sums, low, high)]
    names = ('n', 'sx', 'sy', 'sz', 'sxx', 'sxy', 'sxz', 'syy', 'syz', 'szz')
    updates = ','.join(f'{n}={n}+excluded.{n}' for n in names)
    db.executemany('INSERT INTO bins VALUES (' + ','.join(['?']*17) + ') '
                   'ON CONFLICT(ix,iy,iz) DO UPDATE SET ' + updates +
                   ',xmin=min(xmin,excluded.xmin),xmax=max(xmax,excluded.xmax),'
                   'ymin=min(ymin,excluded.ymin),ymax=max(ymax,excluded.ymax)', rows)
    return len(points)


def _fit(stats, p):
    n, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz = stats[:10]
    xmin, xmax, ymin, ymax = stats[10:]
    if (n < p['min_points'] or min(xmax-xmin, ymax-ymin) < p['resolution']*.12):
        return None
    covariance = np.array([[sxx-sx*sx/n, sxy-sx*sy/n],
                           [sxy-sx*sy/n, syy-sy*sy/n]]) / n
    if np.linalg.eigvalsh(covariance)[0] < (p['resolution']*.035)**2:
        return None
    slope = np.linalg.solve(covariance, np.array([sxz-sx*sz/n, syz-sy*sz/n])/n)
    if np.linalg.norm(slope) > math.tan(math.radians(p['max_slope_deg'])):
        return None
    intercept = (sz - slope @ [sx, sy])/n
    abc = np.r_[slope, intercept]
    gram = np.array([[sxx,sxy,sx],[sxy,syy,sy],[sx,sy,n]])
    error = max(0., szz - 2*abc @ [sxz,syz,sz] + abc @ gram @ abc)
    if math.sqrt(error/n) > p['max_fit_rmse']:
        return None
    return abc


def _candidate(rows, p):
    # Each XY column contains vertically grouped sufficient statistics. The
    # lowest *supported, planar* cluster wins; isolated low noise cannot seed it.
    cluster = None
    first = previous = None
    max_band = p['resolution']*math.sqrt(2)*math.tan(math.radians(p['max_slope_deg'])) + .04
    for row in rows:
        zbin, values = row[0], np.asarray(row[1:], dtype=float)
        if (previous is None or zbin-previous > 2 or
                (zbin-first)*p['vertical_bin'] > max_band):
            cluster, first = values.copy(), zbin
        else:
            cluster[:10] += values[:10]
            cluster[10], cluster[11] = min(cluster[10],values[10]), max(cluster[11],values[11])
            cluster[12], cluster[13] = min(cluster[12],values[12]), max(cluster[13],values[13])
        previous = zbin
        fit = _fit(cluster, p)
        if fit is not None:
            return fit
    return None


def build_terrain(chunks, anchors, work_directory=None, parameters=None, progress=None):
    """Build ground from observed map XYZ and measured trajectory foot anchors.

    Anchors are Nx3 expected ground points computed from optimized body poses
    and the saved standing height. They select supported surfaces; they never
    create free cells or synthetic ground samples. Four-neighbor plane agreement
    follows slopes while preventing jumps onto elevated tables or step tops.
    """
    p = _options(parameters)
    anchors = _points(anchors, 3)
    if not len(anchors) or not np.isfinite(anchors).all():
        raise ValueError('Terrain needs finite measured trajectory foot-height anchors')
    candidates = {}
    count = 0
    with tempfile.TemporaryDirectory(dir=work_directory, prefix='terrain-') as temporary:
        with closing(sqlite3.connect(Path(temporary)/'statistics.sqlite')) as db:
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('PRAGMA synchronous=OFF')
            db.execute('PRAGMA cache_size=-16384')
            db.execute('CREATE TABLE bins (ix INTEGER,iy INTEGER,iz INTEGER,'
                       'n REAL,sx REAL,sy REAL,sz REAL,sxx REAL,sxy REAL,sxz REAL,syy REAL,syz REAL,szz REAL,'
                       'xmin REAL,xmax REAL,ymin REAL,ymax REAL,PRIMARY KEY(ix,iy,iz)) WITHOUT ROWID')
            for chunk in chunks:
                count += _accumulate(db, chunk, p)
                if progress:
                    progress('observations', count)
            previous, rows, processed_cells = None, [], 0
            for row in db.execute('SELECT * FROM bins ORDER BY ix,iy,iz'):
                cell = row[:2]
                if previous is not None and cell != previous:
                    candidate = _candidate(rows, p)
                    if candidate is not None:
                        candidates[previous] = candidate
                    rows = []
                    processed_cells += 1
                    if progress and processed_cells % 10000 == 0:
                        progress('candidate_columns', processed_cells)
                rows.append(row[2:])
                previous = cell
            if previous is not None:
                candidate = _candidate(rows, p)
                if candidate is not None:
                    candidates[previous] = candidate
    anchor_cells = {}
    for cell, point in zip(_cells(anchors[:, :2], p['resolution']), anchors):
        anchor_cells.setdefault(tuple(cell), []).append(point)
    radius = math.ceil(p['anchor_radius']/p['resolution'])
    seeds = set()
    for checked, (cell, plane) in enumerate(candidates.items(), 1):
        if progress and checked % 10000 == 0:
            progress('anchor_checks', checked)
        center = (np.asarray(cell)+.5)*p['resolution']
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nearby = anchor_cells.get((cell[0]+dx,cell[1]+dy), ())
                for point in nearby:
                    delta = point[:2]-center
                    if (np.linalg.norm(delta) <= p['anchor_radius'] and
                            abs(plane[2]+plane[:2] @ delta-point[2]) <= p['anchor_height_tolerance']):
                        seeds.add(cell)
                        break
                if cell in seeds:
                    break
            if cell in seeds:
                break
    def adjacent(cell):
        return ((cell[0]+1,cell[1]),(cell[0]-1,cell[1]),
                (cell[0],cell[1]+1),(cell[0],cell[1]-1))
    def agrees(a, b):
        delta = (np.asarray(b)-a)*p['resolution']
        first, second = candidates[a], candidates[b]
        # Compare both local plane extrapolations at the shared boundary, and
        # their centers; a flat 15cm step is not a gentle 0.4m-wide ramp.
        return max(abs(first[2]+first[:2] @ delta-second[2]),
                   abs(second[2]-second[:2] @ delta-first[2])) <= p['continuity_tolerance']
    accepted, queue = set(seeds), deque(sorted(seeds))
    while queue:
        cell = queue.popleft()
        for neighbor in adjacent(cell):
            if neighbor in candidates and neighbor not in accepted and agrees(cell,neighbor):
                accepted.add(neighbor)
                queue.append(neighbor)
    discontinuities = set()
    for cell in accepted:
        for neighbor in adjacent(cell):
            if neighbor in candidates and not agrees(cell,neighbor):
                discontinuities.add(cell)
                if neighbor in accepted:
                    discontinuities.add(neighbor)
    accepted -= discontinuities
    cells = sorted(accepted)
    statistics = dict(observations=count, candidate_cells=len(candidates), anchor_cells=len(seeds),
                      supported_cells=len(cells), discontinuity_cells=len(discontinuities))
    if progress:
        progress('supported_cells', len(cells))
    if not cells:
        raise ValueError('No observed local terrain is consistent with the trajectory foot heights')
    return TerrainModel(cells, [candidates[cell] for cell in cells], p, statistics)
