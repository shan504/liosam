"""Save and reuse ordered, map-bound routes without network access or ROS on host."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent / 'original_go2/go2_3d'))
from route_store import RouteStore, route_name, validate_route


RESULT_PREFIX = 'GO2_ROUTE_RESULT '
CONTAINER = 'go2-lio-nav'
CLIENT_COMMAND = ('set -e; source /opt/ros/humble/setup.bash; '
                  'source /ros2_ws/install/setup.bash; '
                  'exec /usr/bin/python3 -u /repo/navigation/route_client.py')


def docker_prefix():
    try:
        result = subprocess.run(['docker', 'info'], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=5, check=False)
        if result.returncode == 0:
            return ['docker']
    except subprocess.TimeoutExpired:
        pass
    # Password entry remains at the host terminal; no password enters JSON/stdin.
    subprocess.run(['sudo', '-v'], check=True)
    return ['sudo', '-n', 'docker']


def request_route(command, route=None):
    request = dict(request_id=str(uuid4()), command=command)
    if route is not None:
        request['route'] = validate_route(route)
    prefix = docker_prefix()
    payload = json.dumps(request, separators=(',', ':'), allow_nan=False)
    try:
        response = subprocess.run(
            prefix + ['exec', '-i', CONTAINER, 'bash', '-c', CLIENT_COMMAND],
            input=payload, text=True, capture_output=True, timeout=20, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('未收到路线命令回执；结果未确认。请先查看 Foxglove 队列和 /operator/status，'
                           '不要把超时当作没有执行后直接重复 run。') from error
    results = []
    for line in response.stdout.splitlines():
        if not line.startswith(RESULT_PREFIX):
            continue
        try:
            result = json.loads(line[len(RESULT_PREFIX):])
        except ValueError:
            continue
        if isinstance(result, dict) and result.get('request_id') == request['request_id']:
            results.append(result)
    if not results:
        details = response.stderr.strip() or response.stdout.strip()
        raise RuntimeError('未收到路线命令回执。请先运行 navigation/start.sh 并保持导航窗口；'
                           '检查 Foxglove /operator/status 后再操作。' + ('\n' + details if details else ''))
    result = results[-1]
    if result.get('success') is not True:
        raise ValueError(result.get('message', '路线命令被拒绝'))
    if response.returncode != 0:
        raise RuntimeError('路线命令已返回回执，但客户端异常退出；请检查 /operator/status 后再操作。')
    return result


def parser():
    result = argparse.ArgumentParser(description='离线保存/加载 Foxglove 多点队列；路线与原地图版本绑定。')
    result.add_argument('--directory', type=Path,
                        default=Path.home() / '.local/share/go2-navigation/routes',
                        help='路线保存目录，默认 ~/.local/share/go2-navigation/routes')
    commands = result.add_subparsers(dest='command', required=True)
    save = commands.add_parser('save', help='保存本轮完整点位顺序（含已完成点位），不改变运动')
    save.add_argument('name', type=route_name)
    save.add_argument('--overwrite', action='store_true', help='明确覆盖同名路线文件')
    for name, help_text in [('load', '加载到空队列；不启用运动，随后点“启用运动”'),
                            ('run', '加载到空队列并明确请求启用运动；保留定位和碰撞检查'),
                            ('show', '查看本机路线文件，无需导航进程')]:
        item = commands.add_parser(name, help=help_text)
        item.add_argument('name', type=route_name)
    commands.add_parser('list', help='查看已保存路线，无需导航进程')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    store = RouteStore(args.directory)
    try:
        if args.command == 'list':
            routes = store.list_routes()
            if not routes:
                print('尚未保存路线：' + str(store.directory))
            for route in routes:
                print(f'{route["name"]}\t{route["waypoints"]} 个点\t地图 {route["map"]}')
        elif args.command == 'show':
            print(json.dumps(store.load(args.name), ensure_ascii=False, indent=2))
        elif args.command == 'save':
            snapshot = request_route('snapshot')
            route = validate_route(snapshot.get('route'))
            path = store.save(args.name, route, overwrite=args.overwrite)
            print(f'点位已保存：{path}\n共 {len(route["waypoints"])} 个点，按设置顺序保存；'
                  '包含本轮已完成点位，运动状态不变。')
        else:
            route = store.load(args.name)
            result = request_route(args.command, route)
            if args.command == 'load':
                print(f'已加载 {len(route["waypoints"])} 个点到队列。确认定位有效后，点击 Foxglove“启用运动”。')
            else:
                print(f'已加载 {len(route["waypoints"])} 个点并请求启用运动；'
                      '是否开始及执行结果请看 /operator/status，这不是到达确认。')
            print(result.get('message', ''))
        return 0
    except FileExistsError:
        print('保存失败：同名路线已存在。换一个路线名，或使用 save 路线名 --overwrite 明确覆盖。', file=sys.stderr)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print('路线操作失败：' + str(error), file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
