#!/usr/bin/env bash
# Offline nav regression: Docker bridge + localhost DDS, never live robot network.
set -euo pipefail
[[ $# -le 2 ]] || { echo '用法：bash navigation/replay.sh [录包目录或 latest] [速度 0.5..1.0]' >&2; exit 2; }
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
argument=${1:-latest}
if [[ $argument == latest ]]; then argument=$(cat "${GO2_BAG_ROOT:-$HOME/go2_bags}/LATEST"); fi
session=$(realpath -- "$argument")
rate=${2:-1.0}
/usr/bin/python3 - "$session" "$rate" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); rate=float(sys.argv[2])
if not .5<=rate<=1:raise SystemExit('回放速度必须在 0.5 到 1.0 之间')
for name in ('bag/metadata.yaml','map_source/manifest.json','recording_report.json'):
 if not (p/name).is_file():raise SystemExit('缺少完整录包文件：'+str(p/name))
if not json.loads((p/'recording_report.json').read_text()).get('required_inputs_complete'):
 raise SystemExit('本包没有完整点云和里程计输入')
PY
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then sudo -v; docker_cmd=(sudo -n docker); fi
runtime=$(mktemp -d "${TMPDIR:-/tmp}/go2-offline.XXXXXXXX")
trap 'rm -rf -- "$runtime"' EXIT
cache=${GO2_REPLAY_CACHE:-$HOME/.cache/go2_lio_replay}
mkdir -p "$cache" "$runtime/navigation_map"
echo '离线回放：重新计算定位、障碍物和路径；没有本体控制节点，不会驱动机器狗。'
echo 'Foxglove：ws://本机IP:8766；导入同一 Go2.json。重复试验请退出后重新执行本命令。'
# No --network host, host sensor devices, DDS configuration or actuator bridge mounts.
"${docker_cmd[@]}" run --rm --init --name go2-nav-replay --network bridge -p 8766:8766 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e ROS_DOMAIN_ID=77 -e ROS_LOCALHOST_ONLY=1 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=2 \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -v "$repo:/repo:ro" -v "$repo/navigation/original_go2:/opt/go2_project:ro" \
  -v "$session:/recording:ro" -v "$session/map_source:/map_source:ro" \
  -v "$runtime:/runtime:rw" -v "$cache:/cache:rw" --entrypoint /bin/bash go2-3d:edu \
  -c 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && exec /usr/bin/python3 -u /repo/navigation/replay_session.py "$1"' _ "$rate"
