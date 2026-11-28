"""ROS-independent SE(3), cloud filtering and observed-space grid export."""
import math
import sys
from pathlib import Path
import numpy as np
import yaml
from grid_io import cell_chunks, coordinate_dtype, create_pgm, write_cells_npz


def pose_matrix(position, quaternion):
    q = np.asarray(quaternion, dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Invalid quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    matrix[:3, 3] = position
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite pose")
    return matrix


def quaternion_from_matrix(matrix):
    # Stable eigendecomposition, including rotations near pi.
    r = matrix[:3, :3]
    k = np.array([
        [r[0,0]-r[1,1]-r[2,2], r[1,0]+r[0,1], r[2,0]+r[0,2], r[2,1]-r[1,2]],
        [r[1,0]+r[0,1], r[1,1]-r[0,0]-r[2,2], r[2,1]+r[1,2], r[0,2]-r[2,0]],
        [r[2,0]+r[0,2], r[2,1]+r[1,2], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
        [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], r.trace()],
    ]) / 3.0
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def planar_pose(matrix):
    """Project a measured body pose onto the navigation grid's map XY plane."""
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return pose_matrix([matrix[0, 3], matrix[1, 3], 0.0],
                       [0.0, 0.0, math.sin(yaw/2), math.cos(yaw/2)])


def first_map_pose(odom_body):
    """Put first body position/heading at zero without tilting the odom Z axis."""
    pose = np.eye(4)
    pose[:3,:3] = planar_pose(odom_body)[:3,:3].T@odom_body[:3,:3]
    return pose


def grid_cell(grid, xy):
    """Convert an XY position in the grid frame, including rotated origins."""
    p, q = grid.info.origin.position, grid.info.origin.orientation
    origin = pose_matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
    local = np.linalg.inv(origin) @ [xy[0], xy[1], p.z, 1.0]
    x, y = np.floor(local[:2]/grid.info.resolution).astype(int)
    return (int(x), int(y)) if 0 <= x < grid.info.width and 0 <= y < grid.info.height else None


def rotation_angle(matrix):
    return math.acos(float(np.clip((np.trace(matrix[:3, :3])-1)/2, -1, 1)))


def filter_points(points, min_range=0.35, max_range=20.0):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    distances = np.linalg.norm(points, axis=1)
    return points[np.isfinite(points).all(axis=1) &
                  (distances >= min_range) & (distances <= max_range)]


def cloud_xyz(message):
    """Respect PointCloud2 field offsets, datatype, endianness and row padding."""
    types = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    fields = {field.name: field for field in message.fields}
    endian = '>' if message.is_bigendian else '<'
    dtype = np.dtype({'names': ['x', 'y', 'z'],
                      'formats': [endian+types[fields[k].datatype] for k in ('x','y','z')],
                      'offsets': [fields[k].offset for k in ('x','y','z')],
                      'itemsize': message.point_step})
    data = np.ndarray((message.height, message.width), dtype=dtype,
                      buffer=message.data, strides=(message.row_step, message.point_step))
    return np.column_stack([data[k].reshape(-1) for k in ('x','y','z')])


def line_cells(start, end):
    x, y = start
    ex, ey = end
    dx, dy = abs(ex-x), -abs(ey-y)
    sx, sy = (1 if x < ex else -1), (1 if y < ey else -1)
    error = dx + dy
    while True:
        yield x, y
        if x == ex and y == ey:
            break
        twice = 2*error
        if twice >= dy:
            error += dy
            x += sx
        if twice <= dx:
            error += dx
            y += sy


class ObservedGrid:
    """Unknown stays unknown; occupied voxels override any observed free rays."""
    def __init__(self, resolution=0.10, max_cells=None):
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be finite and positive")
        self.resolution = resolution
        # Retain the legacy argument/property for callers, without a map-size ceiling.
        self.max_cells = None
        self.free = set()

    def prepare_observation(self, points, origin):
        """Return new observed free cells without changing the live grid.

        Preparation never changes the live grid. Callers can finish other
        fallible frame calculations before committing these cells with
        ``free.update``. Only previously unseen cells are retained, so this
        does not copy the entire accumulated map on each observation.
        """
        start = tuple(np.floor(np.asarray(origin)[:2]/self.resolution).astype(int))
        delta = points[:, :2] - np.asarray(origin)[:2]
        distances = np.linalg.norm(delta, axis=1)
        # Only the nearest return per angular bin clears space; never a ray
        # through a nearer obstacle to a farther ceiling or floor return.
        bins = np.floor((np.arctan2(delta[:,1], delta[:,0])+np.pi)/0.01).astype(int)
        selected = {}
        for index in np.argsort(distances):
            if distances[index] < 0.2 or distances[index] > 20:
                continue
            selected.setdefault(int(bins[index]), int(index))
        additions = set()
        for index in selected.values():
            end = tuple(np.floor(points[index,:2]/self.resolution).astype(int))
            cells = list(line_cells(start, end))
            additions.update(cell for cell in cells[:-1] if cell not in self.free)
        return additions

    def observe(self, points, origin):
        self.free.update(self.prepare_observation(points, origin))

    def export(self, points, directory, floor_z=-0.30, min_height=0.10, max_height=1.5,
               ground_model=None):
        if ground_model is None:
            obstacle = points[(points[:,2] >= floor_z+min_height) &
                              (points[:,2] <= floor_z+max_height)]
        else:
            from ground import floor_and_obstacles
            floor,obstacle = floor_and_obstacles(points,ground_model,min_height,max_height)
            # Actual observed floor endpoints provide free-space evidence.
            # A near floor hit must not hide every farther floor hit in the
            # same azimuth, as happened with nearest-return-only clearing.
            cells,counts = np.unique(np.floor(floor[:,:2]/self.resolution).astype(int),
                                     axis=0,return_counts=True)
            self.free.update(map(tuple,cells[counts>=2]))
        occupied = set()
        for offset in range(0, len(obstacle), 65536):
            occupied.update(map(tuple, np.floor(
                obstacle[offset:offset + 65536, :2] / self.resolution).astype(np.int64)))
        if not occupied or not self.free:
            raise ValueError("No occupied or observed-free cells; check height slice and sensor data")
        # Find sparse extents without duplicating both sets into their union and
        # a full coordinate array. Preserve the original two-cell unknown border.
        lower = upper = None
        for cells in (self.free, occupied):
            for batch in cell_chunks(cells):
                lo, hi = batch.min(axis=0), batch.max(axis=0)
                lower = lo if lower is None else np.minimum(lower, lo)
                upper = hi if upper is None else np.maximum(upper, hi)
        lower = tuple(int(value) - 2 for value in lower)
        upper = tuple(int(value) + 2 for value in upper)
        width, height = (upper[i] - lower[i] + 1 for i in (0, 1))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        print(f'导航栅格：{width} × {height} = {width * height} 格，'
              f'分辨率 {self.resolution:g} m；按完整范围分块写入。',
              file=sys.stderr, flush=True)
        image = create_pgm(directory / 'nav.pgm', width, height)
        try:
            for cells, value in ((self.free, 254), (occupied, 0)):
                for batch in cell_chunks(cells):
                    x = batch[:, 0] - lower[0]
                    y = height - 1 - (batch[:, 1] - lower[1])
                    image[y, x] = value
            image.flush()
        finally:
            image._mmap.close()
        metadata = dict(image='nav.pgm', mode='trinary', resolution=self.resolution,
                        origin=[float(lower[0]*self.resolution), float(lower[1]*self.resolution), 0.0],
                        negate=0, occupied_thresh=0.65, free_thresh=0.196)
        (directory/'nav.yaml').write_text(yaml.safe_dump(metadata))
        write_cells_npz(directory / 'observed_free.npz', self.free,
                        coordinate_dtype(lower, upper))
        return metadata
