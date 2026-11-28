"""Stream navigation grids and observed cells without a fixed map-size ceiling."""
from itertools import islice
import operator
from pathlib import Path
import tempfile

import numpy as np


# These are working-buffer sizes, not limits on map dimensions or cell counts.
_BATCH_CELLS = 65536
_IMAGE_CHUNK = 1024 * 1024


def _dimensions(width, height):
    width, height = operator.index(width), operator.index(height)
    if width <= 0 or height <= 0:
        raise ValueError('PGM dimensions must be positive integers')
    return width, height


def create_pgm(path, width, height, fill=205):
    """Create a complete P5 image on disk and return its writable pixel mapping."""
    width, height = _dimensions(width, height)
    fill = operator.index(fill)
    if not 0 <= fill <= 255:
        raise ValueError('PGM fill must be an unsigned byte')
    path = Path(path)
    header = f'P5\n{width} {height}\n255\n'.encode('ascii')
    block = bytes([fill]) * min(_IMAGE_CHUNK, width * height)
    with path.open('wb') as stream:
        stream.write(header)
        remaining = width * height
        while remaining:
            size = min(len(block), remaining)
            stream.write(block[:size])
            remaining -= size
    return np.memmap(path, dtype=np.uint8, mode='r+', offset=len(header),
                     shape=(height, width))


def read_pgm(path, mode='r'):
    """Validate the project's strict three-line P5 format without reading pixels."""
    path = Path(path)
    if mode not in ('r', 'r+', 'c'):
        raise ValueError('PGM mapping mode must be r, r+ or c')
    with path.open('rb') as stream:
        magic = stream.readline(4096)
        dimensions = stream.readline(4096)
        maximum = stream.readline(4096)
        if magic != b'P5\n' or maximum != b'255\n' or not dimensions.endswith(b'\n'):
            raise ValueError('Unsupported navigation PGM')
        try:
            width, height = map(int, dimensions.split())
            width, height = _dimensions(width, height)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('Invalid navigation PGM dimensions') from error
        offset = stream.tell()
        if path.stat().st_size != offset + width * height:
            raise ValueError('Navigation PGM pixel count does not match dimensions')
    return np.memmap(path, dtype=np.uint8, mode=mode, offset=offset,
                     shape=(height, width))


def cell_chunks(cells, batch_size=_BATCH_CELLS):
    """Yield bounded arrays from a reusable set/sequence of integer XY cells."""
    iterator = iter(cells)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        values = np.asarray(batch)
        if values.ndim != 2 or values.shape[1] != 2 or values.dtype.kind not in 'iub':
            raise ValueError('Grid cells must contain integer XY pairs')
        if values.dtype.kind == 'u' and (values > np.iinfo(np.int64).max).any():
            raise ValueError('Grid cell coordinates exceed int64 representation')
        yield values.astype(np.int64, copy=False)


def coordinate_dtype(lower, upper):
    """Keep the existing int32 archive format where its coordinates fit."""
    lo, hi = np.iinfo(np.int32).min, np.iinfo(np.int32).max
    return np.int32 if all(lo <= int(x) <= hi for x in (*lower, *upper)) else np.int64


def write_cells_npz(path, cells, dtype=np.int64):
    """Write sparse observed cells through a disk-backed array, in bounded batches."""
    path = Path(path)
    count = len(cells)
    if count == 0:
        np.savez_compressed(path, cells=np.empty((0, 2), dtype=dtype))
        return
    with tempfile.TemporaryDirectory(prefix='.observed-cells-', dir=path.parent) as tmp:
        values = np.lib.format.open_memmap(Path(tmp) / 'cells.npy', mode='w+',
                                          dtype=dtype, shape=(count, 2))
        try:
            offset = 0
            limits = np.iinfo(dtype)
            for batch in cell_chunks(cells):
                if (batch < limits.min).any() or (batch > limits.max).any():
                    raise ValueError('Observed cell coordinates exceed their integer representation')
                values[offset:offset + len(batch)] = batch
                offset += len(batch)
            if offset != count:
                raise ValueError('Observed cell collection changed during export')
            values.flush()
            np.savez_compressed(path, cells=values)
        finally:
            values._mmap.close()


def write_observed_free(image, path, lower):
    """Stream visible free pixels into the standard compressed XY-cell archive."""
    pixels = np.asarray(image)
    if pixels.ndim != 2:
        raise ValueError('Observed-free image must be two-dimensional')
    height, width = pixels.shape
    width, height = _dimensions(width, height)
    lower = tuple(operator.index(value) for value in lower)
    if len(lower) != 2:
        raise ValueError('Observed-free origin must contain two integer coordinates')
    upper = (lower[0] + width - 1, lower[1] + height - 1)
    dtype = coordinate_dtype(lower, upper)
    limits = np.iinfo(np.int64)
    if not all(limits.min <= value <= limits.max for value in (*lower, *upper)):
        raise ValueError('Observed cell coordinates exceed int64 representation')
    flat = pixels.reshape(-1)
    count = sum(int(np.count_nonzero(flat[start:start + _IMAGE_CHUNK] >= 250))
                for start in range(0, flat.size, _IMAGE_CHUNK))
    path = Path(path)
    if count == 0:
        np.savez_compressed(path, cells=np.empty((0, 2), dtype=dtype))
        return
    with tempfile.TemporaryDirectory(prefix='.observed-cells-', dir=path.parent) as tmp:
        values = np.lib.format.open_memmap(Path(tmp) / 'cells.npy', mode='w+',
                                          dtype=dtype, shape=(count, 2))
        try:
            offset = 0
            for start in range(0, flat.size, _IMAGE_CHUNK):
                indices = np.flatnonzero(flat[start:start + _IMAGE_CHUNK] >= 250) + start
                end = offset + len(indices)
                values[offset:end, 0] = indices % width + lower[0]
                values[offset:end, 1] = height - 1 - indices // width + lower[1]
                offset = end
            if offset != count:
                raise ValueError('Navigation image changed during observed-free export')
            values.flush()
            np.savez_compressed(path, cells=values)
        finally:
            values._mmap.close()
