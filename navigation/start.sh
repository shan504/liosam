#!/usr/bin/env bash
# Start one navigation session for any versioned map exported by noetic/save_map.sh.
set -eo pipefail
if [[ $# != 1 ]]; then
    echo '用法：bash navigation/start.sh 地图名（或指定保存版本目录）' >&2
    exit 2
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
version=$(/usr/bin/python3 "$repo/navigation/select_map.py" "$1")
dds_file=${GO2_DDS_FILE:-$HOME/cyclonedds_ws/cyclonedds.xml}
[[ -r $dds_file ]] || { echo "缺少 Go2 DDS 配置 $dds_file" >&2; exit 1; }
mkdir -p "$HOME/.ros" "$HOME/.cache/go2_lio_navigation"
exec 8>>"$HOME/.ros/go2_lio_navigation.lock"
flock -n 8 || { echo '已有导航窗口正在运行。' >&2; exit 1; }
flock -n "$HOME/.ros/go2_one_command_start.lock" true || {
    echo '请先保存地图，在原建图窗口按 Ctrl+C；等该窗口退出，再启动导航。' >&2; exit 1;
}
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then sudo -v; docker_cmd=(sudo -n docker); fi
image=go2-3d:edu
container=go2-lio-nav
"${docker_cmd[@]}" image inspect "$image" >/dev/null || { echo "未找到原 Go2 的 $image 镜像。" >&2; exit 1; }
for active in $("${docker_cmd[@]}" ps --format '{{.Names}}'); do
    case "$active" in
        go2-lio-nav|go2-3d|go2-sdk)
            echo "容器 $active 正在运行，请在其原窗口结束该会话后启动导航。" >&2
            exit 1 ;;
    esac
done
cache="$HOME/.cache/go2_lio_navigation"
original="$repo/navigation/original_go2"
runtime=$(mktemp -d "${TMPDIR:-/tmp}/go2-nav-config.XXXXXXXX")
trap 'rm -rf -- "$runtime"' EXIT
mkdir -p "$runtime/navigation_map"
echo "原 Go2 导航 · 地图版本 $version"
echo '检查旧版 Nav2 插件、生成 SDK 配置，并从本次地图存档准备原 Go2 使用的地图。'
prepared=$("${docker_cmd[@]}" run --rm --init --network none --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$repo:/repo:ro" -v "$original:/opt/go2_project:ro" -v "$runtime:/runtime:rw" \
    -v "$version:/map_source:ro" -v "$cache:/cache:rw" --entrypoint /bin/bash "$image" -c '
        set -e
        source /opt/ros/humble/setup.bash
        source /ros2_ws/install/setup.bash
        cd /opt/go2_project/go2_3d
        for pkg in nav2_bringup pointcloud_to_laserscan robot_state_publisher foxglove_bridge; do
            ros2 pkg prefix "$pkg" >/dev/null
        done
        /usr/bin/python3 check_nav2_plugins.py >&2
        /usr/bin/python3 prepare.py --mode mapping >&2
        /usr/bin/python3 -c "import mapper, operator_bridge, obstacle_cloud, control_bridge, navigation_diagnostics" >&2
        exec /usr/bin/python3 -u /repo/navigation/prepare_go2_map.py /map_source /cache')
[[ $prepared =~ ^/cache/(original-go2-[a-f0-9]{24})$ ]] || { echo "导航地图准备失败: $prepared" >&2; exit 1; }
map_cache="$cache/${BASH_REMATCH[1]}"
for file in map.pcd metadata.yaml nav.yaml nav.pgm; do
    [[ -s "$map_cache/$file" ]] || { echo "缺少导航地图文件: $map_cache/$file" >&2; exit 1; }
done
/usr/bin/python3 - <<'PY_PORT'
import socket
with socket.socket() as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(('0.0.0.0', 8766))
    except OSError as error:
        raise SystemExit('Foxglove 端口 8766 已被占用，请先结束旧桥接会话：' + str(error))
PY_PORT
share=$(mktemp -d /dev/shm/go2_nav_XXXXXXXX)
touch "$share/running"
lidar_pid=
docker_pid=
cleanup() {
    trap - EXIT INT TERM HUP
    # The lease initiates an orderly motion stop in the container even after the sudo ticket expires.
    rm -f -- "$share/running"
    if [[ -n $docker_pid ]]; then wait "$docker_pid" 2>/dev/null || :; fi
    if [[ -n $lidar_pid ]]; then
        kill -TERM "$lidar_pid" 2>/dev/null || :
        wait "$lidar_pid" 2>/dev/null || :
    fi
    rm -rf -- "$share" "$runtime"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
# Keep the host driver's sudo ticket usable for any required LiDAR interface setup.
sudo -v
/bin/bash "$repo/navigation/lidar_host.sh" "$share" 8>&- &
lidar_pid=$!
for ((step=0; step<240; step++)); do
    if ! kill -0 "$lidar_pid" 2>/dev/null; then wait "$lidar_pid" || :; echo '雷达驱动已退出。' >&2; exit 1; fi
    [[ -s $share/scan.bin ]] && break
    sleep 0.25
done
[[ -s $share/scan.bin ]] || { echo '60 秒内未收到 MID‑360S 扫描，请看上方驱动输出。' >&2; exit 1; }
wifi=$(ip -o -4 addr show dev wlan0 2>/dev/null | awk 'NR==1 {split($4, a, "/"); print a[1]}')
[[ -z $wifi ]] || echo "Foxglove WebSocket：ws://$wifi:8766"
echo '初始定位采用所选地图的建图起点及朝向；在其他位置可用 Foxglove 的“估计位姿”指定。运动初始关闭。'
diag_root="$HOME/.local/state/go2-navigation/diagnostics"
session=$(/usr/bin/python3 - "$original/go2_3d" "$diag_root" "$version" <<'PY_SESSION'
import os
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from fault_store import create_session
root = Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
print(create_session(root, {"map_source": sys.argv[3], "capture_pid": os.getppid()}))
PY_SESSION
)
diagnostics_mounts=(-v "$session:/diagnostics:rw" -e GO2_DIAGNOSTICS_SESSION=/diagnostics
                   -e GO2_DIAGNOSTICS_UID="$(id -u)" -e GO2_DIAGNOSTICS_GID="$(id -g)")
echo "导航事件目录 $session"
"${docker_cmd[@]}" run --rm --init --network host --name "$container" \
    -e ROS_DOMAIN_ID=0 -e ROS_LOCALHOST_ONLY=0 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=file:///etc/cyclonedds.xml -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=2 \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -v "$dds_file:/etc/cyclonedds.xml:ro" -v "$repo:/repo:ro" -v "$original:/opt/go2_project:ro" \
    -v "$version:/map_source:ro" -v "$runtime:/runtime:ro" -v "$map_cache:/runtime/navigation_map:ro" \
    "${diagnostics_mounts[@]}" -v "$share:/run/go2_scan:ro" --entrypoint /bin/bash "$image" \
    -c 'set -e; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; exec /usr/bin/python3 -u /repo/navigation/run_stack.py' 8>&- &
docker_pid=$!
# Both processes are critical. Cleanup revokes the lease before stopping the host sensor.
set +e
wait -n
code=$?
set -e
exit "$code"
