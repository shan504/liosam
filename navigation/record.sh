#!/usr/bin/env bash
# Run in a second robot terminal while navigation/start.sh is running.
set -euo pipefail
[[ $# -le 1 ]] || { echo '用法：bash navigation/record.sh [本次记录名称]' >&2; exit 2; }
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
label=${1:-field}
[[ $label =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]] || { echo '记录名称只用字母、数字、下划线或连字符。' >&2; exit 2; }
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then sudo -v; docker_cmd=(sudo -n docker); fi
inspection=$("${docker_cmd[@]}" inspect go2-lio-nav)
base=${GO2_BAG_ROOT:-$HOME/go2_bags}
mkdir -p "$base"
session=$(mktemp -d "$base/${label}_$(date +%Y%m%d_%H%M%S)_XXXX")
printf '%s' "$inspection" > "$session/container.json"
/usr/bin/python3 - "$session" "$repo" <<'PY'
import json, shutil, subprocess, sys
from pathlib import Path
out,repo=map(Path,sys.argv[1:])
info=json.loads((out/'container.json').read_text())[0]
if not info['State']['Running']:raise SystemExit('导航容器未运行；请先启动导航。')
mounts={m['Destination']:Path(m['Source']) for m in info['Mounts']}
source=mounts.get('/map_source')
if source is None or not (source/'manifest.json').is_file():raise SystemExit('找不到本次导航的完整源地图。')
# One self-contained source bundle, copied once per experiment; no live map mutation.
shutil.copytree(source,out/'map_source')
config=mounts.get('/runtime')
if config and (config/'config').is_dir():shutil.copytree(config/'config',out/'recorded_config')
revision=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],text=True,capture_output=True).stdout.strip()
(out/'session.json').write_text(json.dumps({'format':1,'map_source':str(source),
 'code_revision':revision,'capture':'normalized native odometry and deskewed base_link cloud',
 'note':'Navigation regression inputs; raw Livox packet/IMU for LIO mapping is not recorded.'},indent=2)+'\n')
PY
dds_file=${GO2_DDS_FILE:-$HOME/cyclonedds_ws/cyclonedds.xml}
[[ -r $dds_file ]] || { echo "缺少 DDS 配置 $dds_file" >&2; exit 1; }
name=go2-nav-record-$$
echo "录包目录：$session"
echo '保持静止约 10 秒，再按正常方式操控/导航；复现问题后在本录包窗口按 Ctrl+C。'
echo '记录点云、本体里程计、相机、TF 和诊断消息；不改变导航状态。'
trap ' : ' INT
set +e
"${docker_cmd[@]}" run --rm --init --name "$name" --network host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e ROS_DOMAIN_ID=0 -e ROS_LOCALHOST_ONLY=0 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e CYCLONEDDS_URI=file:///etc/cyclonedds.xml \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$dds_file:/etc/cyclonedds.xml:ro" \
  -v "$repo:/repo:ro" -v "$session:/recording:rw" --entrypoint /bin/bash go2-3d:edu \
  -c 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && exec /usr/bin/python3 -u /repo/navigation/record_session.py'
code=$?
set -e
trap - INT
if [[ -s $session/bag/metadata.yaml ]]; then
  printf '%s\n' "$session" > "$base/LATEST"
  echo "已保留录包：$session"
  echo "离线回放：bash $repo/navigation/replay.sh '$session'"
else
  echo "录包未正常完成，请检查 $session；不要删除未恢复的 .db3 文件。" >&2
fi
exit "$code"
