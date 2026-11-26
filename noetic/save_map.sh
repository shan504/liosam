#!/usr/bin/env bash
# Run against the same Noetic master and Python installation as start.sh --go2.
set -eo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source /opt/ros/noetic/setup.bash
source "$HOME/mid360s_nk_noetic_ws/devel/setup.bash" --extend
export ROS_MASTER_URI=http://127.0.0.1:11321 ROS_IP=127.0.0.1 OPENBLAS_NUM_THREADS=1
unset ROS_HOSTNAME
exec /usr/bin/python3 -u "$here/save_navigation_map.py" "$@"
