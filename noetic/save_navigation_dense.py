"""Stream a consistent live Go2 archive snapshot into one world-frame PCD."""
from contextlib import closing, ExitStack
import csv
import json
from pathlib import Path
import sqlite3
import time

from dense_map import geometry
from corridor_export import prepare_corridor, world_scans


def pcd_header(points):
    """Fixed-width decimal counts let the one-pass writer patch its own header."""
    return (
        '# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\n'
        'SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
        f'WIDTH {points:010d}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
        f'POINTS {points:010d}\nDATA binary\n'
    ).encode('ascii')


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
            header = pcd_header(0)
            with ExitStack() as files:
                def new_file(path, mode, **kwargs):
                    stream = files.enter_context(path.open(mode, **kwargs))
                    created.append(path)
                    return stream

                handle = new_file(output, 'x+b')
                keyframes = csv.writer(new_file(output.parent / 'keyframes.csv', 'x', newline=''))
                keyframes.writerow(('stamp_ns', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'))
                for stamp, pose in poses.items():
                    geometry(pose)
                    keyframes.writerow((stamp, *pose))
                index = csv.writer(new_file(output.parent / 'scans.csv', 'x', newline=''))
                index.writerow(('stamp_ns', 'anchor_stamp_ns', 'first_point_index', 'point_count'))
                handle.write(header)
                point_offset = source_seen = 0
                last_progress = time.monotonic()
                for scan_number, (stamp, anchor, world) in enumerate(world_scans(db, poses), 1):
                    source_seen += len(world)
                    # Apply the complete corrected corridor to every historical
                    # scan, including points first seen outside the near strip.
                    if corridor is not None and corridor['poses']:
                        world = world[~selection.contains(world)]
                    if len(world):
                        handle.write(memoryview(world).cast('B'))
                    index.writerow((stamp, anchor, point_offset, len(world)))
                    point_offset += len(world)
                    now = time.monotonic()
                    if now-last_progress >= 5. or scan_number == count:
                        print(f'写入地图：{scan_number:,}/{count:,} 帧，保留 {point_offset:,} 点，'
                              f'清理 {source_seen-point_offset:,} 点。', flush=True)
                        last_progress = now
                points, size = point_offset, point_offset*16
                if not points:
                    raise ValueError('全部点云都落在人工清理范围内，缺少用于定位的地图回波；未发布地图')
                if source_seen != source_points or handle.tell() != len(header) + size:
                    raise OSError('导出 PCD 的有效载荷字节数与归档不一致')
                # No count pass, spool copy, or second permanent cloud. The
                # caller publishes this private staging directory only later.
                handle.seek(0)
                final_header = pcd_header(points)
                if len(final_header) != len(header):
                    raise ValueError('PCD 点数超出固定宽度头部')
                handle.write(final_header)
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
                      cleared_points=source_points-points, corridor_events=len(corridor['poses']),
                      corridor_paths=len(corridor.get('paths', [])),
                      corridor_splits=max(0, len(corridor.get('paths', []))-1),
                      corridor_links=sum(max(0, len(path)-1) for path in corridor.get('paths', [])),
                      terrain_fitting=False)
    return result
