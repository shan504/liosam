#!/usr/bin/env bash
# Observe the mapping master, the original bridge master and the old DDS network.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work=$(mktemp -d) || exit 1
dds_file=${GO2_DDS_FILE:-$HOME/cyclonedds_ws/cyclonedds.xml}
image=go2-3d:edu
docker_cmd=(docker)
pids=()
cleanup() {
    local pid cid
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || :; done
    if [[ -s $work/cid ]]; then
        read -r cid <"$work/cid" || :
        if [[ $cid =~ ^[0-9a-f]{64}$ ]]; then
            timeout 4 "${docker_cmd[@]}" rm -f "$cid" >/dev/null 2>&1 || :
        fi
    fi
    rm -rf -- "$work"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

for port in 11321 11311; do
    uri="http://127.0.0.1:$port"
    echo "检查 ROS1 Noetic $uri"
    timeout 15s env -i HOME="$HOME" USER="${USER:-unitree}" PATH=/usr/bin:/bin LANG=C.UTF-8 \
      ROS_MASTER_URI="$uri" ROS_IP=127.0.0.1 /bin/bash -c '
        source /opt/ros/noetic/setup.bash || exit 1
        exec /usr/bin/python3 -u "$1" ros1
      ' bash "$here/native_inspect.py" >"$work/$port" 2>&1 &
    pids+=("$!")
done

can_docker=0
if [[ -r $dds_file ]] && command -v docker >/dev/null; then
    if ! timeout 3s docker image inspect "$image" >/dev/null 2>&1; then
        docker_cmd=(sudo -n docker)
        if [[ -t 0 ]] && ! sudo -n true 2>/dev/null; then sudo -v || :; fi
    fi
    if timeout 4s "${docker_cmd[@]}" image inspect "$image" >/dev/null 2>&1; then can_docker=1; fi
fi
if ((can_docker)); then
    echo "检查 ROS2 Humble（$image，DDS domain 0，$dds_file）"
    timeout 22s "${docker_cmd[@]}" run --rm --init --network host --cidfile "$work/cid" \
      -e ROS_DOMAIN_ID=0 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
      -e CYCLONEDDS_URI=file:///etc/cyclonedds.xml -e ROS_LOCALHOST_ONLY=0 \
      -v "$dds_file:/etc/cyclonedds.xml:ro" \
      -v "$here/native_inspect.py:/opt/native_inspect.py:ro" \
      -v "$here/native_inspect_tf.py:/opt/native_inspect_tf.py:ro" \
      --entrypoint /bin/bash "$image" -c '
        set -e; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash
        exec timeout 15s /usr/bin/python3 -u /opt/native_inspect.py ros2
      ' >"$work/DDS" 2>&1 &
    pids+=("$!")
else
    echo "ROS2 未检查：需要已有 $image 镜像、可读 $dds_file 和 Docker 权限。" >"$work/DDS"
fi
labels=(11321 11311 DDS)
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then echo "检查未完成（master、运行环境或超时）" >>"$work/${labels[$i]}"; fi
done
pids=()
for label in "${labels[@]}"; do
    echo "=== $label ==="
    cat "$work/$label"
done
