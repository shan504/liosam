#!/usr/bin/env bash
# Read only. Keep ROS2 separate from the running Noetic mapping environment.
set -eo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
if [[ ${GO2_NATIVE_CHECK_CLEAN:-} != 1 ]]; then
    settings=()
    for key in ROS_DOMAIN_ID RMW_IMPLEMENTATION CYCLONEDDS_URI FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY; do
        if [[ -v $key ]]; then settings+=("$key=${!key}"); fi
    done
    exec env -i HOME="$HOME" USER="${USER:-$(id -un)}" TERM="${TERM:-xterm}" \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin LANG=C.UTF-8 \
        GO2_NATIVE_CHECK_CLEAN=1 "${settings[@]}" /bin/bash "$here/check_native.sh"
fi
setup=
for distro in foxy humble; do
    if [[ -r /opt/ros/$distro/setup.bash ]]; then setup=/opt/ros/$distro/setup.bash; break; fi
done
[[ -n $setup ]] || { echo '未发现 Foxy/Humble；当前尚不能读取旧系统的原生里程计。'; exit 1; }
source "$setup"
/usr/bin/python3 -u "$here/check_native.py"
