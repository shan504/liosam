#!/usr/bin/env bash
# Reuse shan504/go2 patches/dds/run.sh's proven ROS2 runtime, not host Foxy.
# This container runs only the read-only odometry subscriber, never the control stack.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
dds_file="$HOME/cyclonedds_ws/cyclonedds.xml"
image=go2-ros2-sdk:edu
[[ -r $dds_file ]] || { echo "缺少原仓库使用的 DDS 配置：$dds_file" >&2; exit 1; }
[[ -r $here/check_native.py ]] || { echo '缺少 check_native.py' >&2; exit 1; }
command -v docker >/dev/null || { echo '未找到原系统的 Docker。未改用宿主机 Foxy。' >&2; exit 1; }
sudo -v
echo '=== 本机网卡和地址（不修改） ==='
ip -br addr || true
ip -4 route show || true
echo '=== 原系统 DDS 网络配置（不修改） ==='
/usr/bin/python3 - "$dds_file" <<'PY'
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
network_tags = {'Domain', 'NetworkInterfaceAddress', 'NetworkInterface',
                'AllowMulticast', 'Peer', 'ParticipantIndex', 'ExternalNetworkAddress',
                'EnableMulticastLoopback', 'Transport', 'MaxAutoParticipantIndex'}
for item in root.iter():
    tag = item.tag.rsplit('}', 1)[-1]
    if tag in network_tags:
        print(tag, dict(item.attrib), (item.text or '').strip())
PY
echo '=== 正在运行的容器 ==='
sudo docker ps --format '{{.Names}} | {{.Image}} | {{.Status}}'
if sudo docker inspect go2-sdk >/dev/null 2>&1; then
    sudo docker inspect go2-sdk | /usr/bin/python3 -c '
import json, sys
c = json.load(sys.stdin)[0]
print("go2-sdk:", c["State"]["Status"], "network=" + c["HostConfig"]["NetworkMode"])
allowed = {"ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "CYCLONEDDS_URI", "ROS_LOCALHOST_ONLY"}
for value in c["Config"].get("Env", []):
    if value.split("=", 1)[0] in allowed:
        print(value)
for mount in c.get("Mounts", []):
    if "cyclonedds" in mount.get("Destination", "").lower():
        print("DDS mount:", mount.get("Source"), "->", mount.get("Destination"))
'
fi
if ! sudo docker image inspect "$image" >/dev/null 2>&1; then
    echo "未找到原系统本地镜像 $image。没有下载镜像或启动其他节点。" >&2
    echo '可用镜像：' >&2
    sudo docker image ls --format '{{.Repository}}:{{.Tag}}'
    exit 1
fi
echo "复用原 go2 运行环境：$image；Humble + /ros2_ws；DDS domain 0。"
echo "使用现有配置 $dds_file。保持实机静止；只订阅里程计，不启动建图或控制容器。"
exec sudo docker run --rm --init --network host \
    -e ROS_DOMAIN_ID=0 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=file:///etc/cyclonedds.xml \
    -e ROS_LOCALHOST_ONLY=0 -e PYTHONUNBUFFERED=1 \
    -v "$dds_file:/etc/cyclonedds.xml:ro" \
    -v "$here/check_native.py:/opt/go2_native_check.py:ro" \
    --entrypoint /bin/bash "$image" -c \
    'set -e; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; exec /usr/bin/python3 -u /opt/go2_native_check.py'
