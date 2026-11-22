#!/usr/bin/env bash
set -eo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
if [[ ${GO2_DISPLAY_CLEAN_ENV:-} != 1 ]]; then
    exec env -i HOME="$HOME" USER="${USER:-$(id -un)}" TERM="${TERM:-xterm}" \
        LANG=C.UTF-8 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        GO2_DISPLAY_CLEAN_ENV=1 /bin/bash "$here/repair_display.sh" "$@"
fi
source /opt/ros/noetic/setup.bash
source "$HOME/mid360s_nk_noetic_ws/devel/setup.bash" --extend
export ROS_MASTER_URI=http://127.0.0.1:11321 ROS_IP=127.0.0.1
unset ROS_HOSTNAME
mkdir -p "$HOME/.ros"
exec 8>"$HOME/.ros/go2_display_reload.lock"
flock -n 8 || { echo '另一个显示修复正在运行。'; exit 1; }
/usr/bin/python3 -u "$here/repair_display.py"
