"""Mapping-only, operator-confirmed free rectangles and their exact geometry.

Rectangles follow each recorded body's gravity-projected heading. They do not
interpolate missing trajectory segments and are not a live obstacle exemption.
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


def validate_document(document, expected_frame=None):
    required = {'version', 'frame_id', 'policy', 'poses'}
    if not isinstance(document, dict) or not required <= set(document) or set(document) - required - {'terrain'}:
        raise ValueError('Malformed surveyed corridor document')
    if type(document['version']) is not int or document['version'] != VERSION:
        raise ValueError('Unsupported surveyed corridor version')
    if document['policy'] != POLICY or not isinstance(document['policy'], dict):
        raise ValueError('Surveyed corridor policy does not match the authorized mapping rectangle')
    frame = _frame(document['frame_id'])
    if expected_frame is not None and frame != _frame(expected_frame):
        raise ValueError('Surveyed corridor frame differs from its map frame')
    if not isinstance(document['poses'], list):
        raise ValueError('Surveyed corridor poses must be a list')
    events, stamps = [], set()
    for event in document['poses']:
        if not isinstance(event, dict) or set(event) != {'stamp_ns', 'anchor_stamp_ns', 'body_pose'}:
            raise ValueError('Malformed surveyed corridor pose event')
        stamp, anchor = event['stamp_ns'], event['anchor_stamp_ns']
        if type(stamp) is not int or type(anchor) is not int or not 0 < anchor <= stamp or stamp in stamps:
            raise ValueError('Surveyed corridor events need unique positive source stamps and valid anchors')
        stamps.add(stamp)
        matrix = rigid_matrix(event['body_pose'])
        events.append(dict(stamp_ns=stamp, anchor_stamp_ns=anchor, body_pose=matrix.tolist()))
    result = dict(version=VERSION, frame_id=frame, policy=dict(POLICY),
                  poses=sorted(events, key=lambda event: event['stamp_ns']))
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


def make_document(frame_id, events, terrain=None):
    value = dict(version=VERSION, frame_id=frame_id, policy=dict(POLICY), poses=list(events))
    if terrain is not None:
        value['terrain'] = terrain
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


def write_document(path, frame_id, events, terrain=None):
    """Create a new staged artifact, never replace an existing map's record."""
    value = make_document(frame_id, events, terrain)
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
