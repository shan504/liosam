#!/usr/bin/env python3
"""Prepare the selected archived map, then run only offline navigation components."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from bag_topics import replay_command


def main(rate):
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1' or os.environ.get('ROS_DOMAIN_ID') != '77':
        raise SystemExit('回放必须在独立 domain 77 / localhost-only 容器内启动')
    subprocess.run(['/usr/bin/python3', '/opt/go2_project/go2_3d/prepare.py', '--mode', 'mapping'], check=True)
    cache = subprocess.check_output(['/usr/bin/python3', '-u', '/repo/navigation/prepare_go2_map.py',
                                    '/map_source', '/cache'], text=True).strip()
    source = Path(cache).resolve()
    if source.parent != Path('/cache') or not (source/'metadata.yaml').is_file():
        raise SystemExit('回放地图缓存无效')
    destination=Path('/runtime/navigation_map')
    if destination.exists(): destination.rmdir()
    destination.symlink_to(source, target_is_directory=True)
    from replay_inputs import extract_seed
    seed = extract_seed(Path('/recording/bag'))
    Path('/runtime/replay_seed.json').write_text(json.dumps(seed)+'\n')
    print('回放起点：' + ('采用录包中第一帧有效定位位姿' if seed else
          '没有可用历史定位，请在 Foxglove 指定初始位姿；不会假定当前位置等于建图起点'), flush=True)
    children=[]
    stopping=False
    def stop(signum, frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        children.append(subprocess.Popen(['ros2','launch','/repo/navigation/replay.launch.py'],start_new_session=True))
        # Waiting for real subscriptions avoids losing the beginning while a large map loads.
        import rclpy
        rclpy.init(args=[])
        node=rclpy.create_node('go2_replay_startup')
        end=time.monotonic()+300.
        while not stopping and time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.2)
            nodes={name for name,namespace in node.get_node_names_and_namespaces()}
            if {'go2_gicp','go2_replay_inputs','go2_ground_obstacles'}<=nodes and node.count_subscribers('/point_cloud2')>=2:
                break
            if children[0].poll() is not None:raise RuntimeError('离线处理节点启动失败')
        else:
            if not stopping:raise RuntimeError('等待离线点云处理节点超时')
        node.destroy_node();rclpy.shutdown()
        if stopping:return 130
        children.append(subprocess.Popen(replay_command('/recording/bag',rate),start_new_session=True))
        while not stopping:
            if children[0].poll() is not None:raise RuntimeError('离线处理节点退出')
            if children[1].poll() is not None:
                print('本包回放结束。轨迹固定，回放不会执行新目标的实际运动；重新执行命令可重测。',flush=True)
                return children[1].returncode
            time.sleep(.2)
        return 130
    finally:
        for p in reversed(children):
            if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
        for p in children:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM)


if __name__=='__main__':
    sys.exit(main(float(sys.argv[1])))
