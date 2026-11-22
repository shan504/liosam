#!/usr/bin/env python3
"""Offline regressions, no ROS imports. Run with Python 3 and NumPy."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from dense_map import Archive, Display, MapPreview, cell_ids, transform


def representatives():
    rng = np.random.RandomState(381)
    # Repeated observations of each 15-mm cell, including negative X and Y.
    p = np.column_stack((rng.uniform(-.035,.035,12000), rng.uniform(-.035,.035,12000),
                         rng.uniform(-.005,.005,12000), np.arange(12000))).astype('<f4')
    ds = Display(.015, 450000)
    for rows in np.array_split(p, 16):
        ds.add(rows)
    xyz = p[:, :3].astype(np.float64) / .015
    cells = np.floor(xyz)
    score = ((xyz - cells - .5)**2).sum(axis=1).astype(np.float32)
    keys = cell_ids(cells)
    expected = {int(k):p[np.where(keys == k)[0][np.argmin(score[keys == k])]]
                for k in np.unique(keys)}
    np.testing.assert_array_equal(ds.points, np.array([expected[k] for k in ds.keys]))
    print('Nearest measured return and intensity preserved:', len(p), '->', len(ds.points))


def cap_and_contour():
    rng = np.random.RandomState(72)
    ds = Display(.015, 6000)
    # Flat floor meets narrow vertical wall at y=0. No generated return should
    # appear in the gap between the two surfaces, including the budgeted route.
    for x in np.linspace(0,200,120):
        u = rng.uniform(0.,3.,3000)
        v = rng.uniform(0.,2.,3000)
        ground = rng.rand(3000) > .5
        samples = np.column_stack((x+u, np.where(ground,v,0),
                                   np.where(ground,0,v), ground)).astype('<f4')
        ds.add(samples)
        assert len(ds.points) <= ds.limit
    assert np.all((ds.points[:,1] == 0) | (ds.points[:,2] == 0))
    assert ds.points[:,0].min() < 3. and ds.points[:,0].max() > 200.
    assert np.all(np.diff(ds.keys) > 0)
    print('Budget/coverage and two measured surfaces:', len(ds.keys), 'of', ds.limit, 'points')


def corrected_poses():
    pose0 = (0,0,0,0,0,0,1)
    pose1 = (4,0,0,0,0,0,1)
    pose2 = (8,0,0,0,0,0,1)
    adjusted = (5,0,0,0,0,0,1)
    pts = np.array([[0,0,0,11],[0,0,1,20]],dtype='<f4')
    with TemporaryDirectory() as tmp:
        archive = Archive(Path(tmp))
        archive.path('odom',{100:pose0,110:pose1})
        preview = MapPreview(.015,450000,.02,.15,5.)
        for t,pose in [(100,pose0),(110,pose1),(115,pose1)]:
            anchor = 100 if t == 100 else 110
            world = transform(pts,pose)
            archive.add(t,anchor,pts)
            preview.add(t,anchor,pts,pose,world,pose)
        archive.path('odom',{100:pose0,110:adjusted,120:pose2})
        assert preview.maybe_begin(archive,5.)
        assert not preview.advance(archive,5.,max_rows=1)
        archive.add(120,120,pts)
        preview.add(120,120,pts,pose2,transform(pts,pose2),pose2)
        assert set(preview.active.points[:,0]) == {0,4,8}
        assert not preview.advance(archive,5.1,max_rows=1)
        assert preview.advance(archive,5.2,max_rows=1)
        assert set(preview.active.points[:,0]) == {0,5,8}
        # The new frame is retained across the swap, but the stale pose is gone.
        assert len(preview.active.points) == 6
        assert archive.db.execute('SELECT COUNT(*) FROM scans').fetchone() == (4,)
        archive.db.close()
    print('Finite reproject during new scans: old map available; corrected map kept latest observations')


if __name__ == '__main__':
    representatives()
    cap_and_contour()
    corrected_poses()
