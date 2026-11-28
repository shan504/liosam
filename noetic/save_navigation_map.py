#!/usr/bin/env python3
"""Save a versioned navigation dataset from the running Noetic Go2 mapping session."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile

from save_navigation_dense import export_dense
from save_navigation_ros import capture
from save_navigation_support import snapshot_run

ARCHIVE_PARAM = '/go2_l1_style_dense/archive_dir'


def live_session(rospy):
    master = rospy.get_master()
    nodes = {}
    for node in ('/go2_l1_style_dense', '/go2_native_odom_bridge'):
        code, _, nodes[node] = master.lookupNode(node)
        if code != 1:
            raise RuntimeError('请保留 --go2 建图窗口，等待节点 ' + node)
    return dict(nodes=nodes, run_id=rospy.get_param('/run_id'),
                archive_dir=rospy.get_param(ARCHIVE_PARAM))


def check_space(source, destination):
    with closing(sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        payload = db.execute('SELECT coalesce(sum(length(points)),0) FROM scans').fetchone()[0]
    if not payload:
        raise ValueError('本轮还没有已归档的点云，请等待几帧')
    free = shutil.disk_usage(destination).free
    print('主地图数据预计 %.2f GiB，当前可用 %.2f GiB。' %
          (payload / 2**30, free / 2**30))
    if payload >= free:
        raise OSError('磁盘空间不足以保存点云')


def write_checksums(directory):
    entries = []
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            entries.append('%s  %s\n' % (digest.hexdigest(), path.relative_to(directory)))
    (directory / 'SHA256SUMS').write_text(''.join(entries), encoding='utf-8')


def save(rospy, name, home):
    session = live_session(rospy)
    archive = Path(session['archive_dir']).expanduser().resolve(strict=True)
    source = archive / 'scans.sqlite'
    parent = home / 'go2_maps' / name
    parent.mkdir(parents=True, exist_ok=True)
    check_space(source, parent)
    stamp = datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')
    stage = Path(tempfile.mkdtemp(prefix='.saving-' + stamp + '_', dir=str(parent)))
    final = parent / stage.name[len('.saving-'):]
    try:
        print('正在读取本轮轨迹、TF、里程计和标定……')
        live = capture(stage)
        if '/lio_sam/mapping/odometry' not in live['topics_received']:
            raise RuntimeError('采样期间没有收到当前 LIO 建图里程计')
        run = snapshot_run(archive, stage)
        if live_session(rospy) != session:
            raise RuntimeError('采样期间建图会话发生变化，未发布该版本')
        print('正在按优化关键帧位姿写入 ' + name + '.pcd，逐帧索引和轨迹另存为 CSV……')
        dense = export_dense(source, stage / (name + '.pcd'))
        manifest = dict(format_version=1, name=name, completed_at=datetime.now(timezone.utc).isoformat(),
                        map_file=name + '.pcd', dense=dense, source_run=run, live=live, session=session,
                        scan_index='scans.csv', optimized_keyframes='keyframes.csv',
                        limitations=[
                            'PCD 和逐帧索引使用同一数据库快照；ROS 话题快照的采样窗见 live。',
                            '点云来自本次归档的配准回波；锚点不等于每帧精确的雷达光心。',
                            '当前导出为三维先验地图；二维导航栅格可在规划阶段按所需高度和分辨率构建。',
                            '原始 LiDAR 包、IMU、RGB 和高频本体里程计历史不在地图归档内。'])
        if dense.get('surveyed_corridor'):
            manifest['limitations'].append(
                '本轮采用人工确认前行通行带：前方 0–1m、左右各 0.5m，全部高度。'
                '主 PCD 已按优化位姿清理历史回波；surveyed_corridor.json 记录范围，'
                '清理前局部地形另存，原扫描 SQLite 保留在本轮运行归档。')
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n',
                                             encoding='utf-8')
        print('正在生成文件校验清单……')
        write_checksums(stage)
        stage.rename(final)
    except BaseException:
        print('此次保存未完成，临时数据在 ' + str(stage), file=sys.stderr)
        raise
    print('地图保存完成：' + str(final / (name + '.pcd')))
    print('地图坐标系 %s；%d 帧；%d 点。' % (dense['frame_id'], dense['scans'], dense['points']))
    if dense.get('surveyed_corridor'):
        print('本轮记录 %d 段前行通行带，清除 %d 个历史回波。旧地图未修改。' %
              (dense['corridor_events'], dense['cleared_points']))
        if not dense['corridor_events']:
            print('本轮没有确认的前进记录，未清理点云或强制设置通行格；请检查 F_dense.log 的前进计数。')
    for warning in run['warnings']:
        print('标定/来源备注：' + warning)
    if not live['ground_tf_available']:
        print('本次未读到 go2_ground ← odom，地面变换可用性见标定和清单。')
    return final


def main():
    import rospy
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('name', help='地图名称')
    name = ap.parse_args().name
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        ap.error('地图名须为 1–64 个英文字母、数字、下划线或连字符，并以字母或数字开头')
    home = Path.home()
    root = home / 'go2_maps'
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.save.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('另一个地图保存任务正在运行')
        rospy.init_node('go2_save_navigation_map', anonymous=True)
        save(rospy, name, home)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        sys.exit('保存未完成：%s' % (exc or '中断'))

