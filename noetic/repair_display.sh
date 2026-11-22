#!/usr/bin/env bash
# Compatibility entry point: never unload a running Foxglove nodelet.
set -eo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
echo '已取消显示桥热卸载。请先在原启动窗口 Ctrl+C，等待退出。'
echo '现在进入正常启动流程；检测到旧建图节点时会停止，旧归档保留。'
exec /bin/bash "$here/start.sh" "$@"
