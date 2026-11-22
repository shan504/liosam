#!/usr/bin/env python3
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as S
import sqlite3
import json
import time

import numpy as np
import local_detail as d

ID = (0., 0., 0., 0., 0., 0., 1.)
SECOND = 10**9


def archive_check(root):
    f = Path(root)/'scans.sqlite'
    db = sqlite3.connect(str(f))
    db.executescript("PRAGMA journal_mode=WAL; CREATE TABLE scans(stamp INTEGER PRIMARY KEY, anchor INTEGER,points BLOB); "
                     "CREATE TABLE poses(stamp INTEGER PRIMARY KEY, pose TEXT);CREATE TABLE meta(name TEXT PRIMARY KEY,value TEXT);"
                     "INSERT INTO meta VALUES('frame','odom');")
    # Old and corrected walls, two real 6 mm-separated rims, and below-plane returns.
    pts = np.array([[1.001, 0, .1, 46], [1.007, 0, .1, 52], [1., 0., -.02, 200],
                   [1., 0., .045, 125], [12., 0., 1., 8]], '<f4')
    # .045 is stored slightly above nominal decimal on some NumPy versions.
    pts[3, 2] = .04
    for idx in range(55):
        anchor = 1 if idx < 54 else 2
        stamp = SECOND*200+idx*100000000
        db.execute('INSERT INTO scans VALUES(?,?,?)', (stamp, anchor, pts.tobytes()))
    for idx in [1, 2]:
        db.execute('INSERT INTO poses VALUES(?,?)', (idx, json.dumps(ID)))
    db.commit()
    w = d.Window(f)
    try:
        for live, frame in [({stamp-1}, 'odom'), ({stamp}, 'map')]:
            try:
                w.read(stamp, live, frame, ID, .045)
            except ValueError:
                pass
            else:
                raise AssertionError('wrong live association or frame passed')
        when, a, (frames, n, vox) = w.read(stamp, {stamp}, 'odom', ID, .045)
        assert when == stamp and frames == 30 and n == 60 and vox == 2 and len(a) == 2
        np.testing.assert_allclose(a[:, 0], [1.001, 1.007], atol=1e-6)
        np.testing.assert_array_equal(a[:, 3], [46, 52])
        # Optimize both anchors so that no old wall should survive. Archive
        # geometry is local: rebuilding still works across map translation.
        new = (1.4, -.51, .09, 0., 0., 0., 1.)
        db.execute('UPDATE poses SET pose=?', (json.dumps(new),))
        db.commit()
        _, b, _ = w.read(stamp, {stamp}, 'odom', ID, .045)
        # The two formerly below-band returns now lie above ground: valid behavior.
        assert np.all(b[:, 0] > 2.39)
        kept = b[np.isin(b[:, 3], [46, 52])]
        np.testing.assert_allclose(kept[:, 0], [2.401, 2.407], atol=1e-6)
        # Move the output ground coordinate transform, leave source archive intact.
        gt = (0., 0., .5, 0., 0., 0., 1.)
        _, c, _ = w.read(stamp, {stamp}, 'odom', gt, .045)
        assert np.all(c[:, 2] > .55)
        assert db.execute('SELECT COUNT(*) FROM scans').fetchone() == (55,)
    finally:
        w.close()
        db.close()
    print("PASS. Exact stamp/frame, 30-frame bound, 6-mm double rim, original intensity, corrected whole wall and ground TF.")


def performance():
    rng = np.random.RandomState(74)
    clouds = []
    for i in range(30):
        a = rng.uniform(-6, 6, (19968, 4)).astype('<f4')
        a[:, 2] = abs(a[:, 2])*.6
        a[:9500, 0] = 1.3
        a[9500:13000, 1] = 2.4
        a[:, 3] = i*20000+np.arange(len(a))
        clouds.append(a)
    t = time.monotonic()
    o, total, vox = d.local_cloud(clouds, np.zeros(3), .045)
    elapsed = (time.monotonic()-t)*1000
    assert len(o) == d.CAP and total >= vox >= len(o)
    potential = np.stack([clouds[int(label)//20000][int(label) % 20000] for label in o[::150, 3]])
    np.testing.assert_array_equal(potential, o[::150])
    print('PASS. %d measured returns -> %d fine voxels -> %d real rows, %.0f ms here.' %
          (total, vox, len(o), elapsed))


def fallback():
    x = np.array([[1., 3., -.1, 9.], [0., 0., 0., 10.], [.4, 2., .04, 11.], [.01, .02, .06, 12.]], '>f4')
    msg = S(header=S(frame_id='/go2_ground'), is_bigendian=True, point_step=16, row_step=36,
            fields=[S(name=name, offset=i*4, datatype=7, count=1)
                    for i, name in enumerate(['x', 'y', 'z', 'intensity'])],
            height=2, width=2, data=x[:2].tobytes()+b'    '+x[2:].tobytes()+b'    ')
    np.testing.assert_array_equal(d.single_xyzi(msg, .045), x[-1:])
    print("PASS. Single-scan fallback respects the same floor threshold and retains the measured XYZI.")


if __name__ == '__main__':
    with TemporaryDirectory() as root:
        archive_check(root)
    performance()
    fallback()
