#!/usr/bin/env python3
"""Offline tests of the optional measured, aligned half-voxel preview."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dense_map import Archive, Display, MapPreview, transform

ORIGIN = (0., 0., 0., 0., 0., 0., 1.)


def contours():
    # Two measured 6-mm-separated rims occupy opposing halves of one 15-mm
    # parent in x. An additional, less central return competes in the left
    # half. Intensity is a source identifier so invented points are detectable.
    y = np.arange(-64, 65) * .015 + .007
    pts = np.concatenate([
        np.column_stack((np.full(len(y), x), y, np.full(len(y), .010),
                         label + np.arange(len(y))))
        for x, label in ((.004, 1000), (.011, 2000), (.006, 3000))
    ]).astype('<f4')
    fine = MapPreview(.015, 450000, .02, .15, 5., .0075)
    base = Display(.015, 450000)
    for t, batch in enumerate(np.array_split(pts, 8)):
        base.add(batch)
        fine.add(t, 0, batch, ORIGIN, batch, ORIGIN)
    rendered = fine.active.display_points()
    actual = set(map(tuple, rendered))
    assert set(map(tuple, base.points)) <= actual <= set(map(tuple, pts))
    assert len(actual) == len(rendered) == 2 * len(y)
    assert len(base.points) == len(y)
    assert np.unique(rendered[:, 0]).size >= 2
    print('Fine rims:', len(base.points), '->', len(rendered), 'measured rows')

    with TemporaryDirectory() as tmp:
        store = Archive(Path(tmp))
        store.path('odom', {0: ORIGIN})
        for t, batch in enumerate(np.array_split(pts, 8)):
            store.add(t, 0, batch)
        corrected = (.052, .001, 0., 0., 0., 0., 1.)
        store.path('odom', {0: corrected})
        assert fine.maybe_begin(store, 10.)
        for i in range(8):
            assert fine.advance(store, 11.+i/10, max_rows=1) == (i == 7)
        assert fine.fine_voxel == .0075
        assert set(map(tuple, fine.active.display_points())) <= set(map(tuple, transform(pts, corrected)))
        store.db.close()


def long_route():
    rng = np.random.default_rng(102)
    limit = 450000
    n = 65
    poses = {i: (3.5*i, 0., 0., 0., 0., 0., 1.) for i in range(n + 1)}
    fine = MapPreview(.015, limit, .02, .15, 5., .0075)
    reference = MapPreview(.015, limit, .02, .15, 5.)
    with TemporaryDirectory() as tmp:
        store = Archive(Path(tmp))
        store.path('odom', dict(list(poses.items())[:n]))
        switched = False
        for i in range(n):
            local = rng.uniform([0, -1.5, 0, 0], [3.49, 1.5, 1.8, 180],
                                size=(10500, 4)).astype('<f4')
            world = transform(local, poses[i])
            store.add(i, i, local)
            fine.add(i, i, local, poses[i], world, poses[i])
            reference.add(i, i, local, poses[i], world, poses[i])
            size = len(fine.active.fine.points if fine.fine_voxel else fine.active.points)
            assert size <= limit
            if switched:
                assert fine.fine_voxel == 0.
            elif fine.fine_voxel == 0.:
                switched = True
        assert switched
        np.testing.assert_array_equal(fine.active.display_points(), reference.active.points)
        assert fine.active.points[:, 0].min() < 2 and fine.active.points[:, 0].max() > 225

        poses[7] = (3.5*7+.35, .12, .01, 0., 0., 0., 1.)
        store.path('odom', poses)
        for preview in (fine, reference):
            assert preview.maybe_begin(store, 100.)
            assert not preview.advance(store, 100., max_rows=1)
            assert preview.building.fine is None
        store.add(n, n, local)
        for preview in (fine, reference):
            preview.add(n, n, local, poses[n], transform(local, poses[n]), poses[n])
            for i in range(1, n):
                assert preview.advance(store, 100.+i*.1, max_rows=1) == (i == n-1)
                assert len(preview.active.points) <= limit
        np.testing.assert_array_equal(fine.active.display_points(), reference.active.points)
        assert len(fine.active.anchors) == n + 1
        assert fine.active.anchors[7] == poses[7]
        assert fine.active.display_points()[:, 0].max() > poses[n][0]
        assert store.db.execute('SELECT COUNT(*) FROM scans').fetchone()[0] == n+1
        print('Long route:', (n+1)*len(local), 'measured rows,', len(fine.active.points),
              'visible, cap', limit, '; after live loop replay bit-for-bit baseline')
        store.db.close()


if __name__ == '__main__':
    contours()
    long_route()
