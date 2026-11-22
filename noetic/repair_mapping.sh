#!/usr/bin/env bash
set -eo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
if [[ ${GO2_REPAIR_CLEAN:-} != 1 ]]; then
    exec env -i HOME="$HOME" USER="${USER:-$(id -un)}" TERM="${TERM:-xterm}"         PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin         LANG=C.UTF-8 GO2_REPAIR_CLEAN=1         bash "$here/repair_mapping.sh"
fi
source /opt/ros/noetic/setup.bash
source "$HOME/mid360s_nk_noetic_ws/devel/setup.bash"
export ROS_MASTER_URI=http://127.0.0.1:11321 ROS_IP=127.0.0.1
/usr/bin/python3 - <<'PY'
import socket
from xmlrpc.client import ServerProxy
socket.setdefaulttimeout(3)
master = ServerProxy('http://127.0.0.1:11321')
try:
    code, message, graph = master.getSystemState('/go2_repair')
except ConnectionRefusedError:
    graph = [[], [], []]
else:
    if code != 1:
        raise SystemExit(message)
active = sorted({node for group in graph for _, nodes in group for node in nodes
                 if any(word in node for word in ('lio_sam', 'livox_lidar', 'go2_ground',
                                                   'go2_live_model', 'go2_detail', 'go2_l1_style'))})
if active:
    raise SystemExit('请先在原一键建图窗口按 Ctrl+C 等它退出，再执行本命令。仍注册：'+', '.join(active))
PY
package=$(rospack find lio_sam)
/usr/bin/python3 "$here/fix_preintegration.py" "$package" "$HOME/mid360s_nk_noetic_ws"
exec bash "$here/start.sh"
