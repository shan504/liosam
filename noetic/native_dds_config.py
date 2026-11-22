#!/usr/bin/env python3
"""Select the observed robot-side address in a process-only XML copy."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

ROBOT_ADDRESS = '192.168.123.18'


def pin_config(source, interfaces):
    addresses = {entry.get('local') for iface in interfaces
                 if iface.get('ifname') == 'eth0'
                 for entry in iface.get('addr_info', [])
                 if entry.get('family') == 'inet'}
    if ROBOT_ADDRESS not in addresses:
        raise ValueError('eth0 没有 192.168.123.18，拒绝猜测或修改网卡地址')
    root = ET.fromstring(source)
    ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ''
    if ns:
        ET.register_namespace('', ns[1:-1])
    domains = root.findall(ns + 'Domain')
    if len(domains) != 1 or domains[0].get('Id', 'any') not in ('any', '0'):
        raise ValueError('DDS Domain 配置不是单个 any/0，保留原配置并停止')
    general = domains[0].find(ns + 'General')
    if general is None:
        raise ValueError('DDS 缺少 General，保留原配置并停止')
    if general.find(ns + 'NetworkInterfaceAddress') is not None:
        raise ValueError('检测到旧版 NetworkInterfaceAddress；不自动混用两种语法')
    selected = general.find(ns + 'Interfaces')
    if selected is None or len(selected) != 1:
        raise ValueError('DDS 不是单网卡配置，保留原配置并停止')
    entry = selected[0]
    if entry.tag != ns + 'NetworkInterface' or entry.get('name') != 'eth0':
        raise ValueError('DDS 原配置未选择 eth0，保留原配置并停止')
    entry.attrib.pop('name', None)
    entry.attrib.pop('autodetermine', None)
    entry.set('address', ROBOT_ADDRESS)
    return ET.tostring(root, encoding='unicode')


def main():
    from pathlib import Path
    interfaces = json.loads(subprocess.check_output(
        ['ip', '-j', '-4', 'addr', 'show', 'dev', 'eth0'], text=True))
    print(pin_config(Path(sys.argv[1]).read_text(), interfaces))


if __name__ == '__main__':
    main()
