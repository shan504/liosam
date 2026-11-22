#!/usr/bin/env bash
# Start the proven Noetic MID360s mapping commands in one supervised terminal.
set -eo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
if [[ ${GO2_START_CLEAN_ENV:-} != 1 ]]; then
    exec env -i HOME="$HOME" USER="${USER:-$(id -un)}" TERM="${TERM:-xterm}" \
        LANG=C.UTF-8 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 GO2_START_CLEAN_ENV=1 \
        /bin/bash "$here/start.sh" "$@"
fi

for setup in /opt/ros/noetic/setup.bash "$HOME/mid360s_nk_noetic_ws/devel/setup.bash" \
             "$HOME/mid360s_imu_noetic_ws/devel/setup.bash"; do
    [[ -r $setup ]] || { echo "缺少 Noetic 环境：$setup" >&2; exit 1; }
    source "$setup" --extend
done
export ROS_MASTER_URI=http://127.0.0.1:11321 ROS_IP=127.0.0.1
unset ROS_HOSTNAME
for file in "$here/dense_map.py" "$HOME/go2_farm_ground/start_ground.sh" \
            "$HOME/go2_model_live.pyz"; do
    [[ -r $file ]] || { echo "缺少已验证的显示文件：$file" >&2; exit 1; }
done
lio_share=$(rospack find lio_sam)
livox_share=$(rospack find livox_ros_driver2)
[[ -r $lio_share/config/paramsLivoxIMU.yaml ]] || {
    echo "缺少 $lio_share/config/paramsLivoxIMU.yaml" >&2; exit 1;
}
mkdir -p "$HOME/.ros"
exec 9>"$HOME/.ros/go2_one_command_start.lock"
flock -n 9 || { echo '本启动脚本已经运行，请查看原启动窗口。' >&2; exit 1; }

expected_nodes=$(roslaunch --nodes livox_ros_driver2 msg_MID360s.launch)
expected_nodes+=$'\n'$(roslaunch --nodes lio_sam module_loam.launch)
state=$(/usr/bin/python3 - "$expected_nodes" "$livox_share/config/MID360s_config.json" <<'PY'
from ipaddress import IPv4Address
import json
import socket
import sys
from xmlrpc.client import Error, ServerProxy

try:
    with open(sys.argv[2], encoding="utf-8") as stream:
        config = json.load(stream)
    sections = [value for key, value in config.items() if key.lower() == "mid360s"]
    if len(sections) != 1 or len(config["lidar_configs"]) != 1:
        raise ValueError("需要一套 Mid360s 配置和一个雷达 IP")
    addresses = {row["host_ip"] for row in sections[0]["host_net_info"]}
    if len(addresses) != 1:
        raise ValueError("host_net_info 必须使用唯一主机 IP")
    host_ip = IPv4Address(addresses.pop())
    lidar_ip = IPv4Address(config["lidar_configs"][0]["ip"])
    if host_ip == lidar_ip or any(address.is_unspecified or address.is_multicast
                                  or address.is_loopback or address.is_reserved
                                  for address in (host_ip, lidar_ip)):
        raise ValueError("主机和雷达必须使用不同的 IPv4 单播地址")
except (OSError, KeyError, TypeError, ValueError) as exc:
    sys.exit("无法读取已安装雷达的网络配置 " + sys.argv[2] + "：" + str(exc))

socket.setdefaulttimeout(2)
caller = "/go2_start_check"
master = ServerProxy("http://127.0.0.1:11321")
try:
    code, message, graph = master.getSystemState(caller)
    if code != 1:
        sys.exit("ROS master 检查失败：" + message)
except ConnectionRefusedError:
    graph = None
except (OSError, ValueError, Error) as exc:
    sys.exit("11321 端口无法作为 ROS master 使用：" + str(exc))

if graph is not None:
    nodes = {node for group in graph for _, group_nodes in group for node in group_nodes}
    expected = set(sys.argv[1].split()) | {
        "/mid360s_imu_units", "/mid360s_attitude", "/go2_l1_style_dense",
        "/go2_ground_view", "/go2_live_model_view", "/go2_visual_model",
    }
    topics = {
        "/livox/lidar", "/livox/imu", "/mid360s/imu_si", "/mid360s/imu_filtered",
        "/lio_sam/mapping/odometry", "/lio_sam/mapping/cloud_registered_raw",
        "/go2_visual/dense_map", "/go2_visual/ground/dense_map",
        "/go2_visual/ground/scan", "/go2_visual/live/model_mesh",
    }
    expected.update(node for topic, publishers in graph[0] if topic in topics for node in publishers)
    expected.discard("/rosout")
    running = []
    for node in sorted(expected & nodes):
        try:
            code, _, uri = master.lookupNode(caller, node)
            if code == 1 and ServerProxy(uri).getPid(caller)[0] == 1:
                running.append(node)
        except (OSError, ValueError, Error):
            pass
    if running:
        sys.exit("这些旧建图或显示节点仍在运行。请在其原窗口 Ctrl+C 后再启动：\n" + "\n".join(running))

# A pre-existing Foxglove connection is reusable only after the protocol responds.
try:
    connection = socket.create_connection(("127.0.0.1", 8766), timeout=2)
except ConnectionRefusedError:
    bridge = "start"
except OSError as exc:
    sys.exit("无法检查 Foxglove 8766 端口：" + str(exc))
else:
    with connection:
        request = ("GET / HTTP/1.1\r\nHost: 127.0.0.1:8766\r\nUpgrade: websocket\r\n"
                   "Connection: Upgrade\r\nSec-WebSocket-Key: MDEyMzQ1Njc4OWFiY2RlZg==\r\n"
                   "Sec-WebSocket-Version: 13\r\nSec-WebSocket-Protocol: foxglove.websocket.v1\r\n\r\n")
        connection.sendall(request.encode("ascii"))
        response = b""
        try:
            while b"\r\n\r\n" not in response and len(response) < 8192:
                part = connection.recv(8192 - len(response))
                if not part:
                    break
                response += part
        except OSError as exc:
            sys.exit("8766 端口没有返回完整 Foxglove 协议响应：" + str(exc))
        response = response.lower()
    if b" 101 " not in response.split(b"\r\n", 1)[0] or b"foxglove.websocket.v1" not in response:
        sys.exit("8766 端口已占用，但没有响应 Foxglove 协议。请先处理该端口的旧进程。")
    associated = False
    if graph is not None:
        bridge_names = {"foxglove_bridge", "foxglove_nodelet_manager", "foxglove_bridge_nodelet_manager"}
        for node in sorted(nodes):
            if node.rsplit("/", 1)[-1] not in bridge_names:
                continue
            try:
                code, _, uri = master.lookupNode(caller, node)
                if code == 1 and ServerProxy(uri).getPid(caller)[0] == 1:
                    associated = True
                    break
            except (OSError, ValueError, Error):
                pass
    if not associated:
        sys.exit("8766 上的 bridge 未连接本次 ROS master 11321。请在其原窗口 Ctrl+C 后重启。")
    bridge = "reuse"
print("start" if graph is None else "reuse", bridge, host_ip, lidar_ip)
PY
)
read -r master_state bridge_state host_ip lidar_ip <<< "$state"
if pgrep -af '(^|/)livox_ros_driver2_node([[:space:]]|$)'; then
    echo '机器上还有 Livox 驱动进程，请先在原驱动窗口 Ctrl+C。' >&2
    exit 1
fi

ip link show dev eth0 >/dev/null || { echo '找不到雷达网卡 eth0。' >&2; exit 1; }
if ! ip -4 -o addr show dev eth0 | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "$host_ip"; then
    sudo ip addr add "$host_ip/32" dev eth0
fi
host_route=" $(ip -4 route show exact "$lidar_ip/32") "
if [[ $host_route != *' dev eth0 '* || $host_route != *" src $host_ip "* || $host_route == *' via '* ]]; then
    sudo ip route replace "$lidar_ip/32" dev eth0 src "$host_ip"
fi

mkdir -p "$HOME/go2_mapping_runs"
run_dir=$(mktemp -d "$HOME/go2_mapping_runs/run_$(date +%Y%m%d_%H%M%S)_XXXX")
mkdir "$run_dir/dense"
printf '本次归档：%s\n稠密点云：%s/dense\n' "$run_dir" "$run_dir"
pids=()
names=()
master_pid=
start() {
    local name=$1
    shift
    echo "启动 $name，日志 $run_dir/$name.log"
    setsid "$@" >"$run_dir/$name.log" 2>&1 &
    child_pid=$!
    pids+=("$child_pid")
    names+=("$name")
}
shutdown() {
    local code=$? pid step active
    trap - EXIT INT TERM HUP
    echo "正在停止本次建图并关闭点云归档……"
    for pid in "${pids[@]}"; do
        [[ $pid == "$master_pid" ]] || kill -INT -- "-$pid" 2>/dev/null || true
    done
    for ((step=0; step<20; step++)); do
        active=0
        for pid in "${pids[@]}"; do
            [[ $pid == "$master_pid" ]] || ! kill -0 -- "-$pid" 2>/dev/null || active=1
        done
        (( active )) || break
        sleep 0.5
    done
    for pid in "${pids[@]}"; do
        [[ $pid == "$master_pid" ]] || kill -TERM -- "-$pid" 2>/dev/null || true
    done
    for ((step=0; step<10; step++)); do
        active=0
        for pid in "${pids[@]}"; do
            [[ $pid == "$master_pid" ]] || ! kill -0 -- "-$pid" 2>/dev/null || active=1
        done
        (( active )) || break
        sleep 0.5
    done
    for pid in "${pids[@]}"; do
        if [[ $pid != "$master_pid" ]]; then
            kill -KILL -- "-$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    if [[ -n $master_pid ]]; then
        kill -INT -- "-$master_pid" 2>/dev/null || true
        for ((step=0; step<20; step++)); do
            kill -0 -- "-$master_pid" 2>/dev/null || break
            sleep 0.5
        done
        kill -TERM -- "-$master_pid" 2>/dev/null || true
        sleep 0.2
        kill -KILL -- "-$master_pid" 2>/dev/null || true
        wait "$master_pid" 2>/dev/null || true
    fi
    echo "本次数据保留在 $run_dir"
    exit "$code"
}
trap shutdown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
check_alive() {
    local i
    for i in "${!pids[@]}"; do
        if ! kill -0 "${pids[i]}" 2>/dev/null; then
            echo "${names[i]} 已退出，最近日志：" >&2
            tail -n 35 "$run_dir/${names[i]}.log" >&2
            exit 1
        fi
    done
}
declare -A progress_lines=([E_ground]=0 [F_dense]=0 [G_model]=0)
declare -A progress_seen=()
show_progress() {
    local name count line
    for name in E_ground F_dense G_model; do
        [[ -f $run_dir/$name.log ]] || continue
        count=$(wc -l <"$run_dir/$name.log")
        if (( count > progress_lines[$name] )); then
            while IFS= read -r line; do
                case "$name:$line" in
                    E_ground:*地面已锁定*|G_model:*已发布贴地\ Go2\ 模型*)
                        if [[ -z ${progress_seen[$name]:-} ]]; then
                            printf '%s\n' "$line"
                            progress_seen[$name]=1
                        fi
                        ;;
                    F_dense:*Dense:*scans\ saved*) printf '%s\n' "$line" ;;
                esac
            done < <(sed -n "$((progress_lines[$name]+1)),${count}p" "$run_dir/$name.log")
            progress_lines[$name]=$count
        fi
    done
}
wait_topic() {
    local topic=$1
    until timeout 15s rostopic echo -n 1 "$topic" > /dev/null 2>&1; do
        check_alive
        echo "仍在等待 $topic，节点仍运行，继续等待……"
    done
    check_alive
    echo "已收到 $topic"
}

if [[ $master_state == start ]]; then
    start master roscore -p 11321
    master_pid=$child_pid
    until timeout 3s rosparam list >/dev/null 2>&1; do check_alive; sleep 1; done
else
    echo '复用现有 ROS master 11321。'
fi
if [[ $master_state == reuse ]] && [[ $(rosparam get /use_sim_time 2>/dev/null || true) == true ]]; then
    echo '现有 ROS master 使用仿真时钟。请在原会话结束后重新启动实机建图。' >&2
    exit 1
fi
start A_lidar roslaunch -p 11321 livox_ros_driver2 msg_MID360s.launch
wait_topic /livox/lidar/point_num
wait_topic /livox/imu/header

cat >"$run_dir/imu_units.py" <<'PY'
import rospy
from sensor_msgs.msg import Imu

GRAVITY = 9.8035
rospy.init_node("mid360s_imu_units")
publisher = rospy.Publisher("/mid360s/imu_si", Imu, queue_size=200)

def receive(message):
    message.linear_acceleration.x *= GRAVITY
    message.linear_acceleration.y *= GRAVITY
    message.linear_acceleration.z *= GRAVITY
    if message.linear_acceleration_covariance[0] >= 0:
        message.linear_acceleration_covariance = [
            value * GRAVITY * GRAVITY for value in message.linear_acceleration_covariance
        ]
    publisher.publish(message)

subscriber = rospy.Subscriber("/livox/imu", Imu, receive, queue_size=200, tcp_nodelay=True)
rospy.spin()
PY
start B_imu_units /usr/bin/python3 -u "$run_dir/imu_units.py"
start B_attitude rosrun imu_filter_madgwick imu_filter_node __name:=mid360s_attitude \
    _use_mag:=false _world_frame:=enu _publish_tf:=false _remove_gravity_vector:=false \
    imu/data_raw:=/mid360s/imu_si imu/data:=/mid360s/imu_filtered
wait_topic /mid360s/imu_filtered/header

rosparam load "$lio_share/config/paramsLivoxIMU.yaml"
rosparam set /lio_sam/imuTopic /mid360s/imu_filtered
rosparam set /lio_sam/imuType 1
rosparam set /lio_sam/extrinsicRPY '[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]'
rosparam set /lio_sam/useImuHeadingInitialization false
rosparam set /lio_sam/savePCD false
start C_mapping roslaunch -p 11321 lio_sam module_loam.launch
wait_topic /lio_sam/mapping/odometry/header

if [[ $bridge_state == start ]]; then
    bridge_args=(port:=8766)
    for option in "topic_whitelist:=['^(?!/lio_sam/mapping/loop_closure_constraints$).*']" \
                  send_buffer_limit:=25165824; do
        if roslaunch -p 11321 --dump-params foxglove_bridge foxglove_bridge.launch \
                "${bridge_args[@]}" "$option" >"$run_dir/D_bridge_params.yaml" 2>"$run_dir/D_bridge_options.log"; then
            bridge_args+=("$option")
        else
            echo "当前 bridge launch 未接受 $option，使用其已验证参数。"
        fi
    done
    start D_bridge roslaunch -p 11321 foxglove_bridge foxglove_bridge.launch "${bridge_args[@]}"
else
    echo '复用现有 Foxglove bridge 8766。'
fi
start E_ground /bin/bash "$HOME/go2_farm_ground/start_ground.sh" --height 0.50
start F_dense /usr/bin/python3 -u "$here/dense_map.py" \
    "_archive_dir:=$run_dir/dense" _voxel:=0.015 _max_points:=450000 _scan_period:=0.09 _period:=3.5
start G_model /usr/bin/python3 -u "$HOME/go2_model_live.pyz"
echo "已启动。保持机器狗在平整地面静止，等待本窗口提示“地面已锁定”。"
echo 'Foxglove 连接端口 8766。保留本窗口，Ctrl+C 停止本次进程并保留归档。'
while :; do sleep 2; check_alive; show_progress; done
