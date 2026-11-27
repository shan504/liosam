"""Supervise one Humble navigation session; stop the motion bridge first on exit."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time


def stop(children, grace):
    for process in children:
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
    limit = time.monotonic() + grace
    for process in children:
        try:
            process.wait(timeout=max(0., limit - time.monotonic()))
        except subprocess.TimeoutExpired:
            # Each component has its own group (including nodes launched by ros2).
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def run():
    here = Path(__file__).resolve().parent
    py = sys.executable
    commands = {
        'Go2/MID-360S': [py, '-u', here/'nav_sensors.py', '--go2-cloud',
                         '--scan-dir', '/run/go2_scan', '--map-dir', '/map_source'],
        'Go2 模型': [py, '-u', here/'robot_model.py'],
        '原 Go2 导航': [
            'ros2', 'launch', '/opt/go2_project/go2_3d/stack.launch.py',
            'mode:=navigation', 'auto_initialize:=true', 'initial_from_map_start:=true',
            'gicp_params:=/opt/go2_project/go2_3d/gicp_dense.yaml', 'port:=8766',
        ],
    }
    lease = Path('/run/go2_scan/running')
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)
    processes = {}
    try:
        for name, command in commands.items():
            if stopping or not lease.is_file():
                return 0
            processes[name] = subprocess.Popen(command, start_new_session=True)
        while not stopping and lease.is_file():
            for name, process in processes.items():
                if process.poll() is not None:
                    print('%s 已退出 (code=%s)，结束本轮导航' % (name, process.returncode), file=sys.stderr)
                    return 1
            time.sleep(.15)
        return 0
    finally:
        navigation = processes.pop('原 Go2 导航', None)
        if navigation is not None:
            stop([navigation], 12)
        stop(list(processes.values()), 5)


if __name__ == '__main__':
    sys.exit(run())
