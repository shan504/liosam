"""Stream a consistent live Go2 archive snapshot into one world-frame PCD."""
from contextlib import closing, ExitStack
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np

from dense_map import geometry
from corridor_export import prepare_corridor, world_scans


def export_dense(database, output):
    """Write one binary XYZI PCD, plus its scan index and optimized keyframe poses."""
    database, output = Path(database), Path(output)
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        # Pin one committed view of the live WAL. Incoming scans and loop updates
        # remain writable, and cannot change the count or poses midway through export.
        db.execute('BEGIN')
        frame = db.execute("SELECT value FROM meta WHERE name='frame'").fetchone()
        count, size, first, last = db.execute(
            'SELECT count(*), sum(length(points)), min(stamp), max(stamp) FROM scans'
        ).fetchone()
        if not frame or not frame[0] or not count or not size:
            raise ValueError('稠密归档缺少地图坐标系或有效扫描')
        source_points = size // 16
        if size % 16 or source_points > 0xffffffff:
            raise ValueError('稠密归档点数或 XYZI 格式不适用于单个 PCD 文件')
        poses = {stamp: json.loads(pose) for stamp, pose in db.execute('SELECT stamp,pose FROM poses ORDER BY stamp')}
        # The caller creates a private staging directory. Never overwrite a previous map.
        created = []
        try:
            corridor, selection = prepare_corridor(db, poses, frame[0], output.parent, created)
            points = source_points
            if corridor is not None and corridor['poses']:
                points = sum(int(np.count_nonzero(~selection.contains(world)))
                             for _, _, world in world_scans(db, poses))
                if not points:
                    raise ValueError('全部点云都落在人工清理范围内，缺少用于定位的地图回波；未发布地图')
            size = points * 16
            header = (
                '# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\n'
                'SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
                f'WIDTH {points}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
                f'POINTS {points}\nDATA binary\n'
            ).encode('ascii')
            with ExitStack() as files:
                def new_file(path, mode, **kwargs):
                    stream = files.enter_context(path.open(mode, **kwargs))
                    created.append(path)
                    return stream

                handle = new_file(output, 'xb')
                keyframes = csv.writer(new_file(output.parent / 'keyframes.csv', 'x', newline=''))
                keyframes.writerow(('stamp_ns', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'))
                for stamp, pose in poses.items():
                    geometry(pose)
                    keyframes.writerow((stamp, *pose))
                index = csv.writer(new_file(output.parent / 'scans.csv', 'x', newline=''))
                index.writerow(('stamp_ns', 'anchor_stamp_ns', 'first_point_index', 'point_count'))
                handle.write(header)
                point_offset = 0
                for stamp, anchor, world in world_scans(db, poses):
                    # Apply the complete corrected corridor to every historical
                    # scan, including points first seen outside the near strip.
                    if corridor is not None and corridor['poses']:
                        world = world[~selection.contains(world)]
                    if len(world):
                        handle.write(memoryview(world).cast('B'))
                    index.writerow((stamp, anchor, point_offset, len(world)))
                    point_offset += len(world)
                if point_offset != points or handle.tell() != len(header) + size:
                    raise OSError('导出 PCD 的有效载荷字节数与归档不一致')
                if corridor is not None:
                    from surveyed_corridor import validate_document
                    corridor = validate_document(corridor, expected_frame=frame[0])
                    stream = new_file(output.parent/'surveyed_corridor.json', 'x', encoding='utf-8')
                    json.dump(corridor, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write('\n')
        except BaseException:
            for path in created:
                path.unlink()
            raise
    result = dict(frame_id=frame[0], scans=count, points=points, anchors=len(poses),
                  first_stamp_ns=first, last_stamp_ns=last, payload_bytes=size)
    if corridor is not None:
        result.update(surveyed_corridor='surveyed_corridor.json', source_points=source_points,
                      cleared_points=source_points-points, corridor_events=len(corridor['poses']))
    return result
