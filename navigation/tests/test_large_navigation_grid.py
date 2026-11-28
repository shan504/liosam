"""Large navigation maps retain their extent, resolution and obstacle semantics."""
import csv
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_go2_map as port
import prepare_map as legacy
from prepare_map import apply, point_array, quaternion, rigid, transforms
from denoise import clean_directory, clean_image
from free_space import resample_free
from geometry import ObservedGrid
from map_start import start_pose_from_metadata


def archive(root, name, lower, extent, aligned):
    """Few real returns spanning a large area; no huge input fixture needed."""
    source = root / name / 'version'
    (source / 'calibration').mkdir(parents=True)
    lower, extent = np.asarray(lower, dtype=float), np.asarray(extent, dtype=float)
    corners = lower + np.array([[0., 0.], [extent[0], 0.],
                                [0., extent[1]], extent])
    floors = []
    obstacles = []
    for corner in corners:
        xy = np.mgrid[0:7, 0:7].reshape(2, -1).T * .05 + corner + .025
        floors.append(np.column_stack((xy, np.zeros(len(xy)))))
        wall_xy = corner + .025 + np.mgrid[0:3, 0:3].reshape(2, -1).T * .05
        obstacles.append(np.column_stack((wall_xy, np.full(9, .50))))
    level = np.vstack((*floors, *obstacles,
                       [lower[0] + 1., lower[1] + 1., 3.20]))
    tilt = 0. if aligned else .12
    ground_to_saved = (np.eye(4) if aligned else
                       rigid([0., 0., -.37],
                             [math.sin(tilt / 2), 0., 0., math.cos(tilt / 2)]))
    body = rigid([lower[0] + 1., lower[1] + 1., .37],
                 [0., 0., math.sin(.19), math.cos(.19)])
    mount = rigid([.215, 0., .25], [0., math.sin(.25), 0., math.cos(.25)])
    lidar = ground_to_saved @ body @ mount
    scan = np.column_stack((apply(level, ground_to_saved),
                            np.arange(len(level)))).astype('<f4')
    points = np.tile(scan, (2, 1))
    path = source / (name + '.pcd')
    path.write_bytes((f'VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n'
                      f'TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {len(points)}\n'
                      f'HEIGHT 1\nPOINTS {len(points)}\nDATA binary\n').encode()
                     + points.tobytes())
    frame = 'go2_ground' if aligned else 'odom'
    manifest = dict(name=name, map_file=path.name, scan_index='scans.csv',
                    optimized_keyframes='poses.csv',
                    dense=dict(frame_id=frame, points=len(points), scans=2))
    (source / 'manifest.json').write_text(json.dumps(manifest))
    (source / 'calibration/floor.json').write_text(json.dumps(dict(
        frame=frame, ground_frame='go2_ground',
        origin=ground_to_saved[:3, 3].tolist(),
        rotation=ground_to_saved[:3, :3].tolist())))
    (source / 'calibration/body_from_lidar.json').write_text(json.dumps(dict(
        parent='base_link', child='livox_frame', xyz=mount[:3, 3].tolist(),
        xyzw=quaternion(mount[:3, :3]).tolist())))
    with (source / 'poses.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['stamp_ns', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        writer.writerow([10, *lidar[:3, 3], *quaternion(lidar[:3, :3])])
    (source / 'scans.csv').write_text(
        'stamp_ns,anchor_stamp_ns,first_point_index,point_count\n'
        f'10,10,0,{len(scan)}\n20,10,{len(scan)},{len(scan)}\n')
    return source, level, body


def read_grid(directory):
    """Read actual exported bytes independently of the production PGM helper."""
    nav = yaml.safe_load((directory / 'nav.yaml').read_text())
    with (directory / nav['image']).open('rb') as stream:
        if stream.readline() != b'P5\n':
            raise AssertionError('Expected binary PGM')
        width, height = map(int, stream.readline().split())
        if stream.readline() != b'255\n':
            raise AssertionError('Expected 8-bit PGM')
        offset = stream.tell()
    image = np.memmap(directory / nav['image'], mode='r', dtype='u1',
                      offset=offset, shape=(height, width))
    return nav, image


def reference_clean(image, minimum, connectivity):
    """Independent union-find oracle for thresholded component sizes."""
    coordinates = set(map(tuple, np.argwhere(image <= 10)))
    parent = {cell: cell for cell in coordinates}

    def root(cell):
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    neighbours = [(-1, 0), (0, -1)]
    if connectivity == 8:
        neighbours += [(-1, -1), (-1, 1)]
    for row, col in coordinates:
        for dy, dx in neighbours:
            other = row + dy, col + dx
            if other in coordinates:
                parent[root(other)] = root((row, col))
    groups = {}
    for cell in coordinates:
        groups.setdefault(root(cell), []).append(cell)
    small = [group for group in groups.values() if len(group) < minimum]
    result = image.copy()
    for group in small:
        rows, cols = np.asarray(group).T
        result[rows, cols] = 254
    return result, dict(removed_cells=sum(map(len, small)),
                        removed_groups=len(small),
                        retained_occupied_cells=len(coordinates) - sum(map(len, small)),
                        minimal_group_size=minimum,
                        group_connectivity_type=connectivity)


class LargeNavigationGrid(unittest.TestCase):
    def test_progress_keeps_cache_path_stdout_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, _ = archive(root, 'progress_test', (-2., -3.), (4., 5.), True)
            output, progress = io.StringIO(), io.StringIO()
            with redirect_stdout(output), redirect_stderr(progress):
                prepared = port.prepare(source, root/'cache')
            self.assertTrue((prepared/'nav.pgm').is_file())
            self.assertEqual(output.getvalue(), '')
            for stage in ('原始回波', '扫描帧', '定位体素', '栅格', '去除导航栅格孤立噪点'):
                self.assertIn(stage, progress.getvalue())

    def test_historical_converter_has_no_separate_area_ceiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, _ = archive(root, 'legacy_large', (-110., -95.), (220., 190.), True)
            result = legacy.prepare(source, root/'cache')
            nav, image = read_grid(result)
            self.assertGreater(image.size, 16_000_000)
            self.assertEqual(nav['resolution'], .05)
            self.assertTrue((image == 0).any())
            self.assertTrue((image == 254).any())
            del image

    def test_legacy_capacity_argument_cannot_truncate_observation_or_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            grid = ObservedGrid(.05, max_cells=1)
            obstacle = np.array([[1.025, .025, .5]])
            grid.observe(obstacle, np.zeros(3))
            self.assertGreater(len(grid.free), 1)
            grid.export(obstacle, directory, floor_z=0.)
            nav, image = read_grid(directory)
            rows, cols = np.nonzero(image >= 250)
            lower = np.rint(np.asarray(nav['origin'][:2]) / .05).astype(int)
            expected = set(zip(cols + lower[0], image.shape[0] - 1 - rows + lower[1]))
            transferred = resample_free(directory, {'plane': [0., 0., 1., 0.]},
                                        np.eye(4), .05, max_cells=1)
            self.assertEqual(transferred, expected)
            self.assertGreater(len(transferred), 1)
            del image

    def test_two_large_map_shapes_full_pipeline_and_cached_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = []
            for name, lower, extent, aligned in (
                    ('wide_survey', (-120., -45.), (150., 90.), False),
                    ('long_track', (2., 3.), (300., 38.), True)):
                with self.subTest(name=name):
                    source, level, body = archive(root, name, lower, extent, aligned)
                    pcd = source / (name + '.pcd')
                    original_hash = hashlib.sha256(pcd.read_bytes()).digest()
                    result = port.prepare(source, root / 'cache')
                    results.append(result)
                    nav, image = read_grid(result)
                    self.assertGreater(image.size, 4_000_000)
                    self.assertEqual(nav['resolution'], .05)
                    # Sparse large maps retain their unobserved interior.
                    self.assertEqual(int(image[image.shape[0] // 2,
                                               image.shape[1] // 2]), 205)
                    metadata = yaml.safe_load((result / 'metadata.yaml').read_text())
                    np.testing.assert_allclose(start_pose_from_metadata(metadata), body, atol=1e-7)
                    np.testing.assert_allclose(metadata['ground_model']['plane'], [0, 0, 1, 0])
                    self.assertEqual(metadata['source_point_count'], 2 * len(level))
                    cloud = point_array(result / 'map.pcd')
                    self.assertEqual(len(cloud), len(level))
                    # Unique intensity identifies every original surface sample,
                    # including the distant final point and high overhang.
                    order = np.argsort(cloud[:, 3])
                    np.testing.assert_array_equal(cloud[order, 3], np.arange(len(level)))
                    ground, _, _ = transforms(source, metadata['source_frame'])
                    expected = apply(point_array(pcd)[:len(level)], ground)
                    np.testing.assert_allclose(cloud[order, :3], expected, atol=2e-5)
                    # All four distant wall groups survive denoising.
                    for xyz in level[-37:-1]:
                        col, row_from_bottom = np.floor(
                            (xyz[:2] - np.asarray(nav['origin'][:2])) / nav['resolution']).astype(int)
                        self.assertEqual(int(image[image.shape[0] - 1 - row_from_bottom, col]), 0)
                    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
                              for p in result.iterdir() if p.is_file()}
                    self.assertEqual(port.prepare(source, root / 'cache'), result)
                    self.assertEqual(before, {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
                                              for p in result.iterdir() if p.is_file()})
                    self.assertEqual(original_hash, hashlib.sha256(pcd.read_bytes()).digest())
                    del cloud, image
            self.assertNotEqual(results[0], results[1])

    def test_tiled_denoise_matches_independent_components_and_preserves_unknown(self):
        rng = np.random.default_rng(58231)
        image = np.full((535, 547), 205, dtype='u1')
        image[rng.random(image.shape) < .14] = 254
        image[rng.random(image.shape) < .008] = 0
        image[100:105, 200:207] = 100
        # Large and tiny components cross both axes of the tile boundary;
        # diagonal chains distinguish four- from eight-neighbour connectivity.
        image[506:518, 506:518] = 205
        image[511, 508:517] = 0
        image[508:517, 511] = 0
        image[250:262, 250:262] = 205
        for index in range(253, 260):
            image[index, index] = 0
        image[511:513, 30] = 0
        image[30, 511:513] = 10
        image[0, 0] = 0
        image[-1, -1] = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for minimum in (1, 3, 7, 8):
                for connectivity in (4, 8):
                    with self.subTest(minimum=minimum, connectivity=connectivity):
                        directory = root / f'case_{minimum}_{connectivity}'
                        directory.mkdir()
                        (directory / 'nav.pgm').write_bytes(
                            f'P5\n{image.shape[1]} {image.shape[0]}\n255\n'.encode() + image.tobytes())
                        nav = dict(image='nav.pgm', resolution=.05, origin=[-12.5, 4.0, 0.0])
                        (directory / 'nav.yaml').write_text(yaml.safe_dump(nav))
                        config = dict(minimal_group_size=minimum,
                                      group_connectivity_type=connectivity)
                        expected, expected_report = reference_clean(image, minimum, connectivity)
                        pure, pure_report = clean_image(image, config)
                        np.testing.assert_array_equal(pure, expected)
                        self.assertEqual(pure_report, expected_report)
                        report = clean_directory(directory, config)
                        _, actual = read_grid(directory)
                        np.testing.assert_array_equal(actual, expected)
                        np.testing.assert_array_equal(actual[image == 205], image[image == 205])
                        self.assertEqual(report, expected_report)
                        rows, cols = np.nonzero(expected >= 250)
                        lower = np.rint(np.array(nav['origin'][:2]) / nav['resolution']).astype(int)
                        expected_cells = np.column_stack((cols + lower[0],
                                                          image.shape[0] - 1 - rows + lower[1]))
                        with np.load(directory / 'observed_free.npz') as saved:
                            np.testing.assert_array_equal(saved['cells'], expected_cells)
                        del actual

    def test_failed_conversion_never_publishes_partial_cache_or_edits_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, _ = archive(root, 'failed_conversion', (-1., -1.), (2., 3.), True)
            before = {p.relative_to(source): hashlib.sha256(p.read_bytes()).digest()
                      for p in source.rglob('*') if p.is_file()}
            with mock.patch.object(port, 'clean_directory', side_effect=OSError('simulated full disk')):
                with self.assertRaisesRegex(OSError, 'simulated full disk'):
                    port.prepare(source, root / 'cache')
            self.assertEqual(list((root / 'cache').glob('original-go2-*')), [])
            self.assertEqual(list((root / 'cache').glob('.preparing-go2-*')), [])
            self.assertEqual(before, {p.relative_to(source): hashlib.sha256(p.read_bytes()).digest()
                                      for p in source.rglob('*') if p.is_file()})

    def test_malformed_grid_is_rejected_without_changing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / 'nav.pgm'
            bad = b'P5\n10 10\n255\n' + bytes(99)
            path.write_bytes(bad)
            (directory / 'nav.yaml').write_text(yaml.safe_dump(
                dict(image='nav.pgm', resolution=.05, origin=[0., 0., 0.])))
            with self.assertRaises(ValueError):
                clean_directory(directory)
            self.assertEqual(path.read_bytes(), bad)


if __name__ == '__main__':
    unittest.main()
