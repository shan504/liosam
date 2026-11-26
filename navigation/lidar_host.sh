#!/usr/bin/env bash
# Run the installed Noetic driver in its normal ROS environment.
set -eo pipefail
dir=${1:?pass the private shared scan directory}
[[ -d "$dir" && -w "$dir" ]] || { echo "无法写入雷达共享目录：$dir" >&2; exit 1; }
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source /opt/ros/noetic/setup.bash
source "$HOME/mid360s_nk_noetic_ws/devel/setup.bash" --extend
export ROS_MASTER_URI=http://127.0.0.1:11321 ROS_IP=127.0.0.1
unset ROS_HOSTNAME
mkdir -p "$HOME/.ros"
exec 9>"$HOME/.ros/go2_one_command_start.lock"
flock -n 9 || { echo "请在原窗口停止上一轮建图或导航，然后重新启动。" >&2; exit 1; }
if pgrep -af '(^|/)livox_ros_driver2_node([[:space:]]|$)'; then
    echo "已有 Livox 驱动正在使用雷达，请在原启动窗口结束它。" >&2; exit 1
fi
livox_share=$(rospack find livox_ros_driver2)
pair=$(/usr/bin/python3 - "$livox_share/config/MID360s_config.json" <<'PY'
import ipaddress, json, sys
with open(sys.argv[1]) as f:
    config=json.load(f)
sections=[v for k,v in config.items() if k.lower()=='mid360s']
if len(sections)!=1 or len(config['lidar_configs'])!=1:
    sys.exit('需要当前实机的单雷达 MID360s 配置')
ips={row['host_ip'] for row in sections[0]['host_net_info']}
if len(ips)!=1:
    sys.exit('雷达配置含多个接收 IPv4 地址')
host=ipaddress.IPv4Address(ips.pop())
lidar=ipaddress.IPv4Address(config['lidar_configs'][0]['ip'])
if host==lidar or any(a.is_unspecified or a.is_loopback or a.is_multicast or a.is_reserved for a in (host,lidar)):
    sys.exit('雷达配置的 IPv4 地址无效')
print(host,lidar)
PY
)
read -r host_ip lidar_ip <<< "$pair"
if ! ip -4 -o addr show dev eth0 | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "$host_ip"; then
    sudo ip addr add "$host_ip/32" dev eth0
fi
route=" $(ip -4 route show exact "$lidar_ip/32") "
if [[ $route != *' dev eth0 '* || $route != *" src $host_ip "* || $route == *' via '* ]]; then
    sudo ip route replace "$lidar_ip/32" dev eth0 src "$host_ip"
fi
pids=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || :; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || :; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
roslaunch -p 11321 livox_ros_driver2 msg_MID360s.launch &
pids+=($!)
/usr/bin/python3 -u "$here/livox_transport.py" "$dir" &
pids+=($!)
wait -n || exit $?
echo "导航激光雷达或扫描转发进程已结束。" >&2
exit 1
