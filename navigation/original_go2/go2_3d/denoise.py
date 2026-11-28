"""Navigation-grid component filtering; never modifies the localization PCD."""
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import yaml


def settings():
    config = yaml.safe_load(Path(__file__).with_name('navigation_filter.yaml').read_text())
    if (type(config.get('minimal_group_size')) is not int or
            not 1 <= config['minimal_group_size'] <= 100 or
            config.get('group_connectivity_type') not in (4, 8)):
        raise ValueError('Invalid navigation_filter.yaml: group size 1..100, connectivity 4 or 8')
    return config


def _neighbor_counts(mask, offsets):
    counts = np.zeros(mask.shape, dtype=np.uint8)
    height, width = mask.shape
    for dy, dx in offsets:
        rows = slice(max(0, -dy), min(height, height-dy))
        cols = slice(max(0, -dx), min(width, width-dx))
        counts[rows, cols] += mask[max(0, dy):min(height, height+dy),
                                   max(0, dx):min(width, width+dx)]
    return counts


def small_components(mask, minimum=3, connectivity=8, *, _core=None):
    """Return a mask of occupied groups smaller than minimum, with no dilation."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or type(minimum) is not int or minimum < 1 or connectivity not in (4, 8):
        raise ValueError('Invalid component-filter input')
    remove = np.zeros_like(mask)
    if minimum == 1:
        return remove, 0
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy or dx) and (connectivity == 8 or abs(dy)+abs(dx) == 1)]
    core = (slice(None), slice(None)) if _core is None else _core
    # The shared policy removes groups of one/two cells. Their graph degree
    # identifies them exactly without a Python tuple/set for every obstacle.
    if minimum in (2, 3):
        degree = _neighbor_counts(mask, offsets)
        isolated = mask & (degree == 0)
        remove |= isolated
        groups = int(np.count_nonzero(isolated[core]))
        if minimum == 3:
            singles = mask & (degree == 1)
            pairs = singles & (_neighbor_counts(singles, offsets) != 0)
            remove |= pairs
            earlier = [(dy, dx) for dy, dx in offsets if dy < 0 or (dy == 0 and dx < 0)]
            representatives = pairs & (_neighbor_counts(pairs, earlier) == 0)
            groups += int(np.count_nonzero(representatives[core]))
        return remove, groups
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = 0
    while remaining:
        first = remaining.pop()
        stack, group = [first], [first]
        while stack:
            row, col = stack.pop()
            for dy, dx in offsets:
                cell = row+dy, col+dx
                if cell in remaining:
                    remaining.remove(cell)
                    stack.append(cell)
                    group.append(cell)
        if len(group) < minimum:
            rows, cols = np.asarray(group).T
            remove[rows, cols] = True
            representative = min(group)
            if (_core is None or
                    (_core[0].start <= representative[0] < _core[0].stop and
                     _core[1].start <= representative[1] < _core[1].stop)):
                groups += 1
    return remove, groups


def clean_image(image, config=None):
    """Only selected occupied pixels become free; unknown pixels are untouched."""
    config = settings() if config is None else config
    removed, groups = small_components(np.asarray(image) <= 10,
                                      config['minimal_group_size'],
                                      config['group_connectivity_type'])
    result = np.asarray(image).copy()
    result[removed] = 254
    return result, dict(removed_cells=int(removed.sum()), removed_groups=groups,
                       retained_occupied_cells=int(np.count_nonzero(result <= 10)),
                       **config)


def clean_directory(directory, config=None):
    """Filter fixed-size tiles against an immutable disk-backed source image.

    A component with at least ``minimum`` cells has at least that many cells
    within ``minimum-1`` neighbour steps of any member. That halo therefore
    preserves the exact whole-image decision, including at tile seams. Count
    each removed group only in the core containing its first pixel.
    """
    from grid_io import create_pgm, read_pgm, write_observed_free

    directory = Path(directory)
    config = settings() if config is None else config
    minimum = config['minimal_group_size']
    connectivity = config['group_connectivity_type']
    if type(minimum) is not int or minimum < 1 or connectivity not in (4, 8):
        raise ValueError('Invalid component-filter input')
    nav = yaml.safe_load((directory/'nav.yaml').read_text())
    path = (directory/nav['image']).resolve()
    if path.parent != directory.resolve():
        raise ValueError('Navigation image must be inside its map directory')
    if abs(nav['origin'][2]) > 1e-8:
        raise ValueError('Rotated grid origin unsupported')
    image = read_pgm(path)
    height, width = image.shape
    lower = np.rint(np.asarray(nav['origin'][:2])/nav['resolution']).astype(np.int64)
    halo = minimum-1
    removed_cells = removed_groups = retained = 0
    last_report = time.monotonic()
    clean = None
    try:
        with tempfile.TemporaryDirectory(prefix='.denoise-', dir=directory) as tmp:
            staged = Path(tmp)
            clean = create_pgm(staged/'nav.pgm', width, height)
            tile = 512
            for y0 in range(0, height, tile):
                y1 = min(height, y0+tile)
                for x0 in range(0, width, tile):
                    x1 = min(width, x0+tile)
                    ey0, ey1 = max(0, y0-halo), min(height, y1+halo)
                    ex0, ex1 = max(0, x0-halo), min(width, x1+halo)
                    core = (slice(y0-ey0, y1-ey0), slice(x0-ex0, x1-ex0))
                    remove, groups = small_components(image[ey0:ey1, ex0:ex1] <= 10,
                                                       minimum, connectivity, _core=core)
                    block = np.array(image[y0:y1, x0:x1], copy=True)
                    block[remove[core]] = 254
                    clean[y0:y1, x0:x1] = block
                    removed_cells += int(np.count_nonzero(remove[core]))
                    removed_groups += groups
                    retained += int(np.count_nonzero(block <= 10))
                if time.monotonic()-last_report >= 10.:
                    print(f'导航栅格去噪：{y1}/{height} 行，保留完整地图范围',
                          file=sys.stderr, flush=True)
                    last_report = time.monotonic()
            clean.flush()
            # Avoid materializing all free pixel coordinates and PGM bytes in
            # RAM. Both outputs are staged before replacing the runtime copy.
            write_observed_free(clean, staged/'observed_free.npz', lower)
            clean._mmap.close()
            clean = None
            image._mmap.close()
            (staged/'observed_free.npz').replace(directory/'observed_free.npz')
            (staged/'nav.pgm').replace(path)
    finally:
        if clean is not None:
            clean._mmap.close()
        image._mmap.close()
    return dict(removed_cells=removed_cells, removed_groups=removed_groups,
                retained_occupied_cells=retained, **config)
