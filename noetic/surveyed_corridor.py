"""Mapping-only, operator-confirmed free rectangles and their exact geometry.

Rectangles follow each recorded body's gravity-projected heading. Version 1
keeps discrete rectangles; explicit version 2 paths also clear their continuous
sweeps. Separate paths are never joined. Neither is a live obstacle exemption.
Callers transform recorded poses and query points into calibrated go2_ground
before indexing; the recorded document retains source-map poses for tracing.
All point heights are selected. Sparse buckets accelerate exact rectangle
tests on bounded chunks without a map-area ceiling.
"""
import json
import math
from pathlib import Path

import numpy as np

VERSION = 1
CONTINUOUS_VERSION = 2
POLICY = dict(frame='gravity_aligned_body_heading', x_min=0.0, x_max=1.0,
              y_min=-0.5, y_max=0.5, z='all',
              provenance='operator_confirmed_free', mapping_only=True)
_CORNERS = np.array([[0., -.5], [1., -.5], [1., .5], [0., .5]])
_EPS = 1e-9


def rigid_matrix(value):
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-9, rtol=0.) or
            not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0.) or
            not math.isclose(float(np.linalg.det(matrix[:3, :3])), 1., abs_tol=1e-6)):
        raise ValueError('Survey corridor requires a finite rigid body transform')
    return matrix.copy()


def _frame(value):
    if not isinstance(value, str) or not value.strip('/') or any(c.isspace() for c in value):
        raise ValueError('Survey corridor requires a nonempty frame_id without whitespace')
    return value.strip('/')


def _json_value(value):
    """Reject NaN/object payloads while copying ordinary metadata."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError('Survey metadata contains a nonfinite number')
        return value
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _json_value(v) for k, v in value.items()}
    raise ValueError('Survey metadata must contain ordinary JSON values')


def _event(event):
    if not isinstance(event, dict) or set(event) != {'stamp_ns', 'anchor_stamp_ns', 'body_pose'}:
        raise ValueError('Malformed surveyed corridor pose event')
    stamp, anchor = event['stamp_ns'], event['anchor_stamp_ns']
    if type(stamp) is not int or type(anchor) is not int or not 0 < anchor <= stamp:
        raise ValueError('Surveyed corridor events need positive source stamps and valid anchors')
    matrix = rigid_matrix(event['body_pose'])
    return dict(stamp_ns=stamp, anchor_stamp_ns=anchor, body_pose=matrix.tolist())


def _calibration(value, frame):
    fields = {'map_from_source_map', 'body_from_lidar', 'source_frame', 'body_height_above_ground'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('Continuous corridor needs its saved physical calibration')
    result = dict(value)
    for key in ('map_from_source_map', 'body_from_lidar'):
        result[key] = rigid_matrix(result[key]).tolist()
    result['source_frame'] = _frame(result['source_frame'])
    if result['source_frame'] != frame:
        raise ValueError('Continuous corridor calibration frame differs from its map frame')
    height = result['body_height_above_ground']
    if isinstance(height, bool) or not isinstance(height, (int, float)) or not math.isfinite(height) or not .1 < height < 1.:
        raise ValueError('Invalid surveyed standing height')
    result['body_height_above_ground'] = float(height)
    return result


def validate_document(document, expected_frame=None):
    required = {'version', 'frame_id', 'policy', 'poses'}
    if not isinstance(document, dict) or not required <= set(document):
        raise ValueError('Malformed surveyed corridor document')
    version = document['version']
    if type(version) is not int or version not in (VERSION, CONTINUOUS_VERSION):
        raise ValueError('Unsupported surveyed corridor version')
    allowed = required | {'terrain'}
    if version == CONTINUOUS_VERSION:
        allowed |= {'paths', 'calibration'}
        if not {'paths', 'calibration'} <= set(document):
            raise ValueError('Continuous corridor requires explicit paths and calibration')
    if set(document) - allowed:
        raise ValueError('Malformed surveyed corridor document')
    if document['policy'] != POLICY or not isinstance(document['policy'], dict):
        raise ValueError('Surveyed corridor policy does not match the authorized mapping rectangle')
    frame = _frame(document['frame_id'])
    if expected_frame is not None and frame != _frame(expected_frame):
        raise ValueError('Surveyed corridor frame differs from its map frame')
    if not isinstance(document['poses'], list):
        raise ValueError('Surveyed corridor poses must be a list')
    events = [_event(event) for event in document['poses']]
    if len({row['stamp_ns'] for row in events}) != len(events):
        raise ValueError('Surveyed corridor events need unique source stamps')
    result = dict(version=version, frame_id=frame, policy=dict(POLICY),
                  poses=sorted(events, key=lambda event: event['stamp_ns']))
    if version == CONTINUOUS_VERSION:
        paths = document['paths']
        if not isinstance(paths, list):
            raise ValueError('Continuous corridor paths must be a list')
        result['calibration'] = _calibration(document['calibration'], frame)
        result['paths'] = []
        last_end = 0
        event_by_stamp = {event['stamp_ns']: event for event in events}
        for path in paths:
            if not isinstance(path, list) or not path:
                raise ValueError('Continuous corridor paths must be nonempty measured pose lists')
            rows = [_event(row) for row in path]
            stamps = [row['stamp_ns'] for row in rows]
            if any(a >= b for a, b in zip(stamps, stamps[1:])) or stamps[0] <= last_end:
                raise ValueError('Continuous corridor path stamps must be strictly chronological')
            last_end = stamps[-1]
            # A connected interval must be explicitly bounded by real forward
            # confirmations; keyframes may only fill its interior, not extend it.
            if rows[0] != event_by_stamp.get(stamps[0]) or rows[-1] != event_by_stamp.get(stamps[-1]):
                raise ValueError('Continuous corridor path endpoints need matching forward confirmations')
            for row in rows:
                event = event_by_stamp.get(row['stamp_ns'])
                if event is not None and row != event:
                    raise ValueError('Continuous corridor path conflicts with a recorded forward pose')
            result['paths'].append(rows)
    if 'terrain' in document:
        terrain = _json_value(document['terrain'])
        if not isinstance(terrain, dict):
            raise ValueError('Surveyed corridor terrain metadata must be a mapping')
        if 'file' in terrain:
            name = terrain['file']
            if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
                raise ValueError('Survey terrain file must be local to the map bundle')
        if 'version' in terrain and (type(terrain['version']) is not int or terrain['version'] != 1):
            raise ValueError('Unsupported surveyed terrain model version')
        if 'map_from_source_map' in terrain:
            terrain['map_from_source_map'] = rigid_matrix(terrain['map_from_source_map']).tolist()
        if 'body_height_above_ground' in terrain:
            height = terrain['body_height_above_ground']
            if isinstance(height, bool) or not isinstance(height, (int, float)) or height <= 0:
                raise ValueError('Invalid surveyed standing height')
        if 'source_frame' in terrain:
            terrain['source_frame'] = _frame(terrain['source_frame'])
        result['terrain'] = terrain
    return result


def make_document(frame_id, events, terrain=None, *, paths=None, calibration=None):
    value = dict(version=VERSION if paths is None else CONTINUOUS_VERSION,
                 frame_id=frame_id, policy=dict(POLICY), poses=list(events))
    if terrain is not None:
        value['terrain'] = terrain
    if paths is not None:
        value.update(paths=paths, calibration=calibration)
    elif calibration is not None:
        raise ValueError('Calibration requires an explicit continuous corridor document')
    return validate_document(value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate surveyed corridor JSON key: ' + key)
        result[key] = value
    return result


def read_document(path, expected_frame=None):
    with Path(path).open(encoding='utf-8') as stream:
        return validate_document(json.load(stream, object_pairs_hook=_unique_object), expected_frame)


def write_document(path, frame_id, events, terrain=None, *, paths=None, calibration=None):
    """Create a new staged artifact, never replace an existing map's record."""
    value = make_document(frame_id, events, terrain, paths=paths, calibration=calibration)
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        stream.write('\n')
    return value


def _indices(xy, size):
    cells = np.floor(np.asarray(xy, dtype=np.float64) / size)
    if not np.isfinite(cells).all() or (np.abs(cells) >= np.iinfo(np.int64).max).any():
        raise ValueError('Survey coordinates exceed integer grid representation')
    return cells.astype(np.int64)


class CorridorIndex:
    def __init__(self, poses, bin_size=1.0):
        if not math.isfinite(bin_size) or bin_size <= 0:
            raise ValueError('Survey index bin size must be finite and positive')
        self.bin_size = float(bin_size)
        self.poses = np.asarray([rigid_matrix(p) for p in poses], dtype=np.float64).reshape(-1, 4, 4)
        forward = self.poses[:, :2, 0]
        norms = np.linalg.norm(forward, axis=1)
        if (norms <= 1e-8).any():
            raise ValueError('A vertical body forward axis has no ground heading')
        self.forward = forward / norms[:, None]
        self.left = np.column_stack((-self.forward[:, 1], self.forward[:, 0]))
        self.origin = self.poses[:, :2, 3]
        self._corners = (self.origin[:, None, :] +
                         _CORNERS[None, :, :1] * self.forward[:, None, :] +
                         _CORNERS[None, :, 1:] * self.left[:, None, :])
        self.buckets = {}
        for index, corners in enumerate(self._corners):
            low, high = _indices(np.array([corners.min(axis=0), corners.max(axis=0)]), self.bin_size)
            for x in range(int(low[0]), int(high[0])+1):
                for y in range(int(low[1]), int(high[1])+1):
                    self.buckets.setdefault((x, y), []).append(index)

    def contains(self, points):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError('Survey point selection requires Nx3 or wider coordinates')
        result = np.zeros(len(points), dtype=bool)
        rows = np.flatnonzero(np.isfinite(points[:, :3]).all(axis=1))
        if not len(rows) or not self.buckets:
            return result
        xy = points[rows, :2].astype(np.float64)
        bins = _indices(xy, self.bin_size)
        order = np.lexsort((bins[:, 1], bins[:, 0]))
        ordered = bins[order]
        starts = np.flatnonzero(np.r_[True, np.any(ordered[1:] != ordered[:-1], axis=1)])
        ends = np.r_[starts[1:], len(order)]
        for start, end in zip(starts, ends):
            candidates = self.buckets.get(tuple(ordered[start]), ())
            if not candidates:
                continue
            selected = order[start:end]
            local_xy = xy[selected]
            inside = np.zeros(len(selected), dtype=bool)
            for candidate in candidates:
                delta = local_xy - self.origin[candidate]
                x, y = delta @ self.forward[candidate], delta @ self.left[candidate]
                inside |= ((x >= -_EPS) & (x <= 1.+_EPS) & (y >= -.5-_EPS) & (y <= .5+_EPS))
                if inside.all():
                    break
            result[rows[selected]] = inside
        return result

    def polygons(self, transform=None):
        """Actual sampled rectangles, at each body Z, optionally in another map."""
        polygons = np.empty((len(self.poses), 4, 3), dtype=np.float64)
        polygons[:, :, :2] = self._corners
        polygons[:, :, 2] = self.poses[:, None, 2, 3]
        if transform is not None:
            matrix = rigid_matrix(transform)
            polygons = polygons @ matrix[:3, :3].T + matrix[:3, 3]
        return polygons

    def free_cells(self, resolution, transform=None, batch_size=65536):
        """Yield unique XY cells wholly inside at least one recorded rectangle.

        A transformed rectangle may project to a parallelogram. Test all four
        pixel corners against its edges; its bounding box only bounds the work.
        Missing trajectory segments are never joined or filled.
        """
        if not math.isfinite(resolution) or resolution <= 0 or type(batch_size) is not int or batch_size <= 0:
            raise ValueError('Survey raster resolution and batch size must be positive')
        seen, batch = set(), []
        pixel_offsets = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]) * resolution
        for vertices in self.polygons(transform)[:, :, :2]:
            edges = np.roll(vertices, -1, axis=0) - vertices
            # Translate before the shoelace sum to avoid cancellation far from
            # the world origin; this does not change the polygon's footprint.
            local_vertices = vertices - vertices[0]
            area = float(np.sum(local_vertices[:, 0]*np.roll(local_vertices[:, 1], -1) -
                                local_vertices[:, 1]*np.roll(local_vertices[:, 0], -1))) / 2.
            if abs(area) <= 1e-10:
                raise ValueError('Survey rectangle has no finite ground-plane footprint')
            sign = 1. if area > 0 else -1.
            low, high = _indices(np.array([vertices.min(axis=0), vertices.max(axis=0)]), resolution)
            # A rectangle is only 1m x 1m; process one bounded row at a time.
            for x in range(int(low[0]), int(high[0])+1):
                for y0 in range(int(low[1]), int(high[1])+1, batch_size):
                    ys = np.arange(y0, min(y0+batch_size, int(high[1])+1), dtype=np.int64)
                    cells = np.column_stack((np.full(len(ys), x, dtype=np.int64), ys))
                    corners = cells[:, None, :].astype(np.float64)*resolution + pixel_offsets
                    delta = corners[:, :, None, :] - vertices[None, None, :, :]
                    cross = edges[None, None, :, 0]*delta[:, :, :, 1] - edges[None, None, :, 1]*delta[:, :, :, 0]
                    supported = np.all(sign*cross >= -_EPS, axis=(1, 2))
                    for cell in map(tuple, cells[supported]):
                        if cell not in seen:
                            seen.add(cell)
                            batch.append(cell)
                            if len(batch) >= batch_size:
                                yield np.asarray(batch, dtype=np.int64)
                                batch = []
        if batch:
            yield np.asarray(batch, dtype=np.int64)


def _convex_hull(points):
    """Convex XY hull of measured/interpolated footprint vertices, retaining Z."""
    points = np.asarray(points, dtype=np.float64)
    order = sorted(range(len(points)), key=lambda i: (points[i, 0], points[i, 1]))
    unique = []
    for index in order:
        if not unique or not np.array_equal(points[index, :2], points[unique[-1], :2]):
            unique.append(index)
    def cross(a, b, c):
        u, v = points[b, :2]-points[a, :2], points[c, :2]-points[a, :2]
        return u[0]*v[1]-u[1]*v[0]
    lower, upper = [], []
    for index in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], index) <= 1e-12:
            lower.pop()
        lower.append(index)
    for index in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], index) <= 1e-12:
            upper.pop()
        upper.append(index)
    result = points[lower[:-1]+upper[:-1]]
    if len(result) < 3:
        raise ValueError('Continuous corridor footprint has no finite area')
    return result


def _footprint(origin, yaw):
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-forward[1], forward[0]])
    polygon = np.empty((4, 3), dtype=np.float64)
    polygon[:, :2] = origin[:2] + _CORNERS[:, :1]*forward + _CORNERS[:, 1:]*left
    polygon[:, 2] = origin[2]
    return polygon


class ContinuousCorridorIndex:
    """Swept footprint union along explicit measured route polylines.

    Straight fixed-heading translations have exact convex sweeps. For rotating
    footprints, each step is at most 0.1m and 1 degree, preserving the route's
    intervening vertices rather than replacing bends with one long chord.
    The same polygons select historical PCD points and navigation grid cells.
    No connection is inferred between separate paths.
    """
    def __init__(self, poses, paths, bin_size=1.0):
        discrete = CorridorIndex(poses, bin_size)
        self.poses = discrete.poses
        self.paths = [np.asarray([rigid_matrix(p) for p in path]).reshape(-1, 4, 4)
                      for path in paths]
        self.bin_size = discrete.bin_size
        self._polygons = [polygon.copy() for polygon in discrete.polygons()]
        for path in self.paths:
            if not len(path):
                raise ValueError('Continuous corridor path must not be empty')
            # Validates the physical heading, including degenerate vertical axes.
            heading = CorridorIndex(path)
            yaws = np.arctan2(heading.forward[:, 1], heading.forward[:, 0])
            if len(path) == 1:
                self._polygons.extend(heading.polygons())
            for i in range(len(path)-1):
                first, last = path[i, :3, 3], path[i+1, :3, 3]
                delta_yaw = math.atan2(math.sin(yaws[i+1]-yaws[i]), math.cos(yaws[i+1]-yaws[i]))
                if abs(delta_yaw) <= 1e-10:
                    self._polygons.append(_convex_hull(np.vstack((
                        _footprint(first, yaws[i]), _footprint(last, yaws[i+1])))))
                    continue
                steps = max(1, int(math.ceil(abs(delta_yaw)/math.radians(1.))),
                            int(math.ceil(np.linalg.norm(last[:2]-first[:2])/.1)))
                previous = _footprint(first, yaws[i])
                for step in range(1, steps+1):
                    fraction = step/steps
                    current = _footprint(first+(last-first)*fraction, yaws[i]+delta_yaw*fraction)
                    self._polygons.append(_convex_hull(np.vstack((previous, current))))
                    previous = current
        self.buckets, self._edges = {}, []
        for index, polygon in enumerate(self._polygons):
            xy = polygon[:, :2]
            edges = np.roll(xy, -1, axis=0)-xy
            normals = np.column_stack((-edges[:, 1], edges[:, 0]))
            self._edges.append((normals, np.einsum('ij,ij->i', normals, xy)))
            low, high = _indices(np.array([xy.min(axis=0), xy.max(axis=0)]), self.bin_size)
            for x in range(int(low[0]), int(high[0])+1):
                for y in range(int(low[1]), int(high[1])+1):
                    self.buckets.setdefault((x, y), []).append(index)

    def contains(self, points):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError('Survey point selection requires Nx3 or wider coordinates')
        result = np.zeros(len(points), dtype=bool)
        rows = np.flatnonzero(np.isfinite(points[:, :3]).all(axis=1))
        if not len(rows) or not self.buckets:
            return result
        xy = points[rows, :2].astype(np.float64)
        bins = _indices(xy, self.bin_size)
        order = np.lexsort((bins[:, 1], bins[:, 0]))
        ordered = bins[order]
        starts = np.flatnonzero(np.r_[True, np.any(ordered[1:] != ordered[:-1], axis=1)])
        ends = np.r_[starts[1:], len(order)]
        for start, end in zip(starts, ends):
            candidates = self.buckets.get(tuple(ordered[start]), ())
            selected = order[start:end]
            local_xy = xy[selected]
            inside = np.zeros(len(selected), dtype=bool)
            for candidate in candidates:
                normals, offsets = self._edges[candidate]
                pending = np.flatnonzero(~inside)
                if not len(pending):
                    break
                inside[pending] |= np.all(local_xy[pending] @ normals.T >= offsets-_EPS, axis=1)
            result[rows[selected]] = inside
        return result

    def polygons(self, transform=None):
        if transform is None:
            return self._polygons
        matrix = rigid_matrix(transform)
        return [polygon @ matrix[:3, :3].T+matrix[:3, 3] for polygon in self._polygons]

    def free_cells(self, resolution, transform=None, batch_size=65536):
        """Unique wholly covered grid cells; overlapping sweeps avoid frame seams."""
        if not math.isfinite(resolution) or resolution <= 0 or type(batch_size) is not int or batch_size <= 0:
            raise ValueError('Survey raster resolution and batch size must be positive')
        seen, batch = set(), []
        pixel_offsets = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])*resolution
        for polygon in self.polygons(transform):
            vertices = polygon[:, :2]
            edges = np.roll(vertices, -1, axis=0)-vertices
            local = vertices-vertices[0]
            area = np.sum(local[:, 0]*np.roll(local[:, 1], -1)-local[:, 1]*np.roll(local[:, 0], -1))*.5
            if abs(area) <= 1e-10:
                raise ValueError('Survey corridor has no finite ground-plane footprint')
            sign = 1. if area > 0 else -1.
            low, high = _indices(np.array([vertices.min(axis=0), vertices.max(axis=0)]), resolution)
            for x in range(int(low[0]), int(high[0])+1):
                for y0 in range(int(low[1]), int(high[1])+1, batch_size):
                    ys = np.arange(y0, min(y0+batch_size, int(high[1])+1), dtype=np.int64)
                    cells = np.column_stack((np.full(len(ys), x, dtype=np.int64), ys))
                    corners = cells[:, None, :].astype(np.float64)*resolution+pixel_offsets
                    delta = corners[:, :, None, :]-vertices[None, None, :, :]
                    cross = edges[None, None, :, 0]*delta[:, :, :, 1]-edges[None, None, :, 1]*delta[:, :, :, 0]
                    supported = np.all(sign*cross >= -_EPS, axis=(1, 2))
                    for cell in map(tuple, cells[supported]):
                        if cell not in seen:
                            seen.add(cell)
                            batch.append(cell)
                            if len(batch) >= batch_size:
                                yield np.asarray(batch, dtype=np.int64)
                                batch = []
        if batch:
            yield np.asarray(batch, dtype=np.int64)


def index_from_document(document, ground):
    """Build one canonical, ground-frame index without upgrading legacy maps."""
    document = validate_document(document)
    ground = rigid_matrix(ground)
    poses = [ground @ np.asarray(row['body_pose']) for row in document['poses']]
    if document['version'] == VERSION:
        return CorridorIndex(poses)
    expected = np.asarray(document['calibration']['map_from_source_map'])
    if not np.allclose(ground, expected, atol=1e-7, rtol=0):
        raise ValueError('Continuous corridor calibration differs from its requested ground frame')
    paths = [[ground @ np.asarray(row['body_pose']) for row in path] for path in document['paths']]
    return ContinuousCorridorIndex(poses, paths)
