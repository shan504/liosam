#!/usr/bin/env bash
# The Noetic master and ROS environment are supplied by the supervising launcher.
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --inspect ]]; then exec /bin/bash "$here/native_inspect.sh"; fi
source_mode=${GO2_NATIVE_SOURCE:-auto}
dds_file=${GO2_DDS_FILE:-$HOME/cyclonedds_ws/cyclonedds.xml}
case "$source_mode" in
    ros1) exec /usr/bin/python3 -u "$here/native_odom.py" ros1 ;;
    auto|ros2) ;;
    *) echo "GO2_NATIVE_SOURCE supports auto, ros1 or ros2" >&2; exit 2 ;;
esac
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then docker_cmd=(sudo -n docker); fi
if [[ ! -r $dds_file ]] || ! "${docker_cmd[@]}" image inspect go2-3d:edu >/dev/null 2>&1; then
    if [[ $source_mode == auto ]]; then
        echo "DDS XML、Docker 权限或旧镜像不可用，等待 ROS1 原生输入。" >&2
        exec /usr/bin/python3 -u "$here/native_odom.py" ros1
    fi
    echo "读取 ROS2 原生数据需要 $dds_file 及可访问的 go2-3d:edu 镜像。" >&2
    exit 1
fi
if [[ $source_mode == auto ]]; then consumer=auto; else consumer=pipe; fi
work=$(mktemp -d)
mkfifo "$work/odom"
touch "$work/running"
finish() {
    rm -f -- "$work/running"
    if [[ -s $work/id ]]; then
        local cid
        read -r cid < "$work/id" || :
        if [[ $cid =~ ^[0-9a-f]{64}$ ]]; then "${docker_cmd[@]}" rm -f "$cid" >/dev/null 2>&1 || :; fi
    fi
    rm -rf -- "$work"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
camera_args=()
if [[ -n ${GO2_CAMERA_DIR:-} ]]; then
    camera_args=(-v "$GO2_CAMERA_DIR:/run/go2_camera:rw" -v "$here/d435_camera.py:/opt/d435_camera.py:ro")
fi
"${docker_cmd[@]}" run "${camera_args[@]}" --rm --init --network host --cidfile "$work/id" \
    -e ROS_DOMAIN_ID=0 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=file:///etc/cyclonedds.xml -e ROS_LOCALHOST_ONLY=0 \
    -v "$work:/run/go2_native:ro" \
    -v "$dds_file:/etc/cyclonedds.xml:ro" \
    -v "$here/status_input.py:/opt/status_input.py:ro" \
    -v "$here/native_mount.py:/opt/native_mount.py:ro" \
    -v "$here/native_odom.py:/opt/native_odom.py:ro" \
    --entrypoint /bin/bash go2-3d:edu -c \
    'set -e; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; if [[ -d /run/go2_camera ]]; then /usr/bin/python3 -u /opt/d435_camera.py ros2 /run/go2_camera/dds >&2 & fi
    exec /usr/bin/python3 -u /opt/native_odom.py dds' >"$work/odom" &
/usr/bin/python3 -u "$here/native_odom.py" "$consumer" <"$work/odom"
