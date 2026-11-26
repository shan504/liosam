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


def run(cache):
    here = Path(__file__).resolve().parent
    py = sys.executable
    commands = {
        'Go2/MID-360S': [py, '-u', here/'nav_sensors.py', '--scan-dir', '/run/go2_scan', '--map-dir', '/map_source',
                           '--geometry', cache/'metadata.json'],
        'GICP': [py, '-u', here/'gicp_localizer.py', cache],
        'Nav2': ['ros2', 'launch', here/'nav2.launch.py', 'map:=' + str(cache/'nav.yaml')],
        'Foxglove': ['ros2', 'launch', here/'foxglove.launch.xml'],
        '航点与运动控制': [py, '-u', here/'mission.py'],
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
        mission = processes.pop('航点与运动控制', None)
        if mission is not None:
            stop([mission], 3)
        stop(list(processes.values()), 5)


if __name__ == '__main__':
    sys.exit(run(Path(sys.argv[1])))
