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
exec 8>"$HOME/.ros/go2_lio_navigation.lock"
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
echo "导航使用地图版本 $version"
echo '正在检查 Humble 环境并准备 GICP 目标和 Nav2 栅格；后续运行复用本版本的有效缓存。'
prepared=$("${docker_cmd[@]}" run --rm --init --network none --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$repo:/repo:ro" -v "$version:/map_source:ro" -v "$cache:/cache" --entrypoint /bin/bash "$image" -c '
        set -e
        source /opt/ros/humble/setup.bash
        source /ros2_ws/install/setup.bash
        cd /repo/navigation
        /usr/bin/python3 - >&2 <<PY_RUNTIME
import open3d as o3d
import gicp_localizer
import mission
import nav2_ready

o3d.geometry.KDTreeFlann().search_radius_vector_3d
registration = o3d.pipelines.registration
registration.registration_generalized_icp
registration.TransformationEstimationForGeneralizedICP()
PY_RUNTIME
        for pkg in nav2_map_server nav2_planner nav2_controller nav2_behaviors nav2_velocity_smoother nav2_bt_navigator nav2_lifecycle_manager nav2_rotation_shim_controller foxglove_bridge; do ros2 pkg prefix "$pkg" >/dev/null; done
        exec /usr/bin/python3 -u /repo/navigation/prepare_map.py /map_source /cache')
[[ $prepared == /cache/* && -s "$cache/${prepared#/cache/}/nav.yaml" ]] || {
    echo "地图准备未返回有效栅格：$prepared" >&2; exit 1;
}
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
    rm -rf -- "$share"
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
"${docker_cmd[@]}" run --rm --init --network host --name "$container" \
    -e ROS_DOMAIN_ID=0 -e ROS_LOCALHOST_ONLY=0 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=file:///etc/cyclonedds.xml -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=2 \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -v "$dds_file:/etc/cyclonedds.xml:ro" -v "$repo:/repo:ro" -v "$version:/map_source:ro" \
    -v "$cache:/cache:ro" -v "$share:/run/go2_scan:ro" --entrypoint /bin/bash "$image" \
    -c 'set -e; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; exec /usr/bin/python3 -u /repo/navigation/run_stack.py "$1"' bash "$prepared" 8>&- &
docker_pid=$!
# Both processes are critical. Cleanup revokes the lease before stopping the host sensor.
set +e
wait -n
code=$?
set -e
exit "$code"
