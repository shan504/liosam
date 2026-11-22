#!/usr/bin/env python3
"""Reload only the existing Foxglove nodelet with a narrow display topic policy."""
import json
from pathlib import Path
import socket
import time

NAME = '/foxglove_bridge'
MANAGER = '/foxglove_nodelet_manager'
PARAM = NAME + '/topic_whitelist'


def reload_bridge(rospy, list_nodes, unload, load, request, desired):
    if NAME not in list_nodes().nodelets:
        raise RuntimeError('现有管理器中没有 /foxglove_bridge；未更改任何节点。')
    old = rospy.get_param(PARAM)
    # The loader reads private parameters on load; rosparam set alone is not enough.
    rospy.set_param(PARAM, desired)
    try:
        if not unload(NAME).success:
            raise RuntimeError('桥接模块拒绝卸载')
    except Exception:
        rospy.set_param(PARAM, old)
        raise
    try:
        if not load(request).success:
            raise RuntimeError('桥接模块重新加载失败')
        if NAME not in list_nodes().nodelets:
            raise RuntimeError('重新加载后找不到桥接模块')
    except Exception as exc:
        rospy.set_param(PARAM, old)
        # Only restore an absent bridge. Never unload another newly loaded instance.
        try:
            if NAME in list_nodes().nodelets:
                raise RuntimeError('桥接模块仍在管理器中，未重复加载')
            if not load(request).success:
                raise RuntimeError('旧桥接配置也未能加载')
        except Exception as rollback:
            raise RuntimeError('显示加载失败：%s；恢复情况：%s。建图未重启。' % (exc, rollback)) from exc
        raise RuntimeError('新显示配置加载失败，已恢复旧桥接配置：%s' % exc) from exc


def main():
    import rospy
    import rosgraph
    from nodelet.srv import NodeletList, NodeletUnload, NodeletLoad, NodeletLoadRequest
    socket.setdefaulttimeout(10)
    rospy.init_node('go2_repair_display', anonymous=True, disable_signals=True)
    desired = json.loads(Path(__file__).with_name('bridge_topics.json').read_text())
    proxies = []
    for suffix, cls in [('list', NodeletList), ('unload_nodelet', NodeletUnload),
                        ('load_nodelet', NodeletLoad)]:
        service = MANAGER + '/' + suffix
        rospy.wait_for_service(service, timeout=5)
        proxies.append(rospy.ServiceProxy(service, cls))
    request = NodeletLoadRequest(name=NAME, type='foxglove_bridge/foxglove_bridge_nodelet',
                                 remap_source_args=[], remap_target_args=[], my_argv=[], bond_id='')
    print('只重载 Foxglove 桥接模块；连接将短暂断开。', flush=True)
    reload_bridge(rospy, *proxies, request, desired)
    # No bond: the existing manager owns its lifetime, including original startup shutdown.
    print('已重载显示桥。点云密度、地图归档、里程计、地面和模型节点未修改。', flush=True)
    print('保持 Foxglove 原布局连接 ws://192.168.1.230:8766，等待自动重连。', flush=True)
    master = rosgraph.Master(rospy.get_name())
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        _, subs, _ = master.getSystemState()
        if any(t.startswith('/go2_visual/ground/') and
               any('foxglove' in n for n in names) for t, names in subs):
            print('已检测到 Foxglove 重新订阅贴地显示。')
            import subprocess
            raise SystemExit(subprocess.call(['/usr/bin/python3', '-u',
                             str(Path(__file__).with_name('check_stream.py'))]))
        time.sleep(.5)
    print('桥接已加载；尚未检测到显示订阅。请在 Foxglove 点击重新连接。')


if __name__ == '__main__':
    main()
