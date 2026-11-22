#!/usr/bin/env bash
# Add or reuse the current-run geometry detail view on the running Noetic graph.
set -eo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
if [[ ${GO2_DETAIL_CLEAN_ENV:-} != 1 ]]; then
    exec env -i HOME="$HOME" USER="${USER:-$(id -un)}" TERM="${TERM:-xterm}" \
        LANG=C.UTF-8 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 GO2_DETAIL_CLEAN_ENV=1 \
        /bin/bash "$here/live_detail.sh" "$@"
fi
for setup in /opt/ros/noetic/setup.bash "$HOME/mid360s_nk_noetic_ws/devel/setup.bash"; do
    [[ -r $setup ]] || { echo "缺少 Noetic 环境：$setup" >&2; exit 1; }
    source "$setup" --extend
done
export ROS_MASTER_URI=http://127.0.0.1:11321 ROS_IP=127.0.0.1
unset ROS_HOSTNAME
case ${1:-} in
    --check-still) shift; exec /usr/bin/python3 -u "$here/check_alignment.py" "$@" ;;
    --check-stream) shift; exec /usr/bin/python3 -u "$here/check_stream.py" "$@" ;;
    --check-returns) shift; exec /usr/bin/python3 -u "$here/check_returns.py" "$@" ;;
    --capture-still) shift; exec /usr/bin/python3 -u "$here/capture_still.py" "$@" ;;
esac
[[ -r $here/local_detail.py ]] || { echo "缺少 $here/local_detail.py" >&2; exit 1; }
mkdir -p "$HOME/.ros"
exec 8>"$HOME/.ros/go2_detail_11321.lock"
owns_lock=1
flock -n 8 || owns_lock=0
exec /usr/bin/python3 -u - "$here/local_detail.py" "$owns_lock" "${1:-}" <<'PY'
import signal
import socket
import subprocess
import sys
import time
from xmlrpc.client import Error, ServerProxy

socket.setdefaulttimeout(1.5)
caller = "/go2_detail_start_check"
master = ServerProxy("http://127.0.0.1:11321")
name = "/go2_detail_local"
outputs = ("/go2_visual/detail/local",)

def signature():
    code, message, graph = master.getSystemState(caller)
    if code != 1:
        raise RuntimeError(message)
    code, message, master_pid = master.getPid(caller)
    if code != 1:
        raise RuntimeError(message)
    registered = {node for group in graph for _, nodes in group for node in nodes}
    publishers = dict(graph[0])
    candidates = {name}
    candidates.update(node for topic in outputs for node in publishers.get(topic, []))
    alive = set()
    for node in sorted(candidates & registered):
        try:
            code, _, uri = master.lookupNode(caller, node)
            if code == 1 and ServerProxy(uri).getPid(caller)[0] == 1:
                alive.add(node)
        except (OSError, ValueError, Error):
            pass
    owners = {topic: set(publishers.get(topic, [])) & alive for topic in outputs}
    competing = set().union(*owners.values()) - {name}
    if competing:
        raise RuntimeError("显示输出已有其他发布者：" + ", ".join(sorted(competing)))
    if name in alive:
        return ("reuse" if all(value == {name} for value in owners.values()) else "pending"), master_pid
    return "start", master_pid

try:
    state, master_pid = signature()
    if sys.argv[2] == "0":
        for _ in range(20):
            if state == "reuse":
                break
            time.sleep(.25)
            state, master_pid = signature()
        if state != "reuse":
            raise RuntimeError("另一个显示启动进程持有锁，尚未提供完整输出，请查看其窗口")
    if state == "pending":
        raise RuntimeError("同名细节点运行中，但几何输出尚未注册，请查看其窗口")
except (OSError, ValueError, Error, RuntimeError) as exc:
    sys.exit("无法连接或复用 ROS master 11321 上的显示节点：" + str(exc))

if sys.argv[3] == "--check":
    print(state)
    sys.exit(0)
if sys.argv[3]:
    sys.exit("用法：bash live_detail.sh [--check-still | --capture-still | --check-returns | --check-stream]")
if state == "reuse":
    print("已复用 ROS master 11321 上的 go2_detail_local，当前轮几何细节输出已注册。")
    sys.exit(0)

child = None
stopped = 0
def stop(signum, frame):
    global stopped
    stopped = signum
for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(signum, stop)

try:
    print("启动当前轮几何细节，等待已有建图扫描与地面变换。保留此窗口，Ctrl+C 结束本次显示。")
    child = subprocess.Popen([sys.executable, "-u", sys.argv[1]])
    next_check = time.monotonic() + 2
    while child.poll() is None and not stopped:
        time.sleep(.2)
        if time.monotonic() >= next_check:
            try:
                code, _, current_pid = master.getPid(caller)
                same_master = code == 1 and current_pid == master_pid
            except (OSError, ValueError, Error):
                same_master = False
            if not same_master:
                print("本轮 ROS master 已结束，正在关闭本次显示。")
                stopped = signal.SIGINT
            next_check = time.monotonic() + 2
finally:
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=6)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
sys.exit(128 + stopped if stopped else child.returncode)
PY
