"""Save and reuse ordered, map-bound routes without network access or ROS on host."""
import argparse
import json
import math
import os
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
REVALIDATE_PREFIX = 'GO2_ROUTE_REVALIDATED '


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


def request_route(command, route=None, number=None):
    request = dict(request_id=str(uuid4()), command=command)
    if route is not None:
        request['route'] = validate_route(route)
    if number is not None:
        if type(number) is not int or number < 1:
            raise ValueError('航点编号必须为正整数')
        request['number'] = number
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


def revalidate_saved_route(route):
    """Run map verification in an isolated image with read-only map mounts, never ROS."""
    route = validate_route(route)
    prefix = docker_prefix()
    inspected = subprocess.run(prefix + ['inspect', CONTAINER], text=True,
                               capture_output=True, timeout=10, check=True)
    containers = json.loads(inspected.stdout)
    if not isinstance(containers, list) or len(containers) != 1:
        raise RuntimeError('无法确定当前导航容器，请先启动所需地图的导航。')
    container = containers[0]
    if container.get('State', {}).get('Running') is not True:
        raise RuntimeError('导航容器未运行，请先启动所需地图的导航。')
    mounts = [item for item in container.get('Mounts', [])
              if item.get('Destination') == '/runtime/navigation_map' and item.get('Type') == 'bind']
    if len(mounts) != 1 or not isinstance(mounts[0].get('Source'), str):
        raise RuntimeError('未找到当前导航地图的只读缓存挂载，无法重新校验路线。')
    current = Path(mounts[0]['Source']).resolve(strict=True)
    if not current.is_dir() or not current.name.startswith('original-go2-'):
        raise RuntimeError('当前地图不是本项目已准备的导航缓存，无法确定旧缓存来源。')
    image = container.get('Image')
    if not isinstance(image, str) or not image.startswith('sha256:'):
        raise RuntimeError('未找到当前导航使用的本机镜像 ID。')
    repo = Path(__file__).resolve().parents[1]
    command = prefix + ['run', '--rm', '-i', '--network', 'none',
        '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'PYTHONDONTWRITEBYTECODE=1',
        '-v', f'{repo}:/repo:ro', '-v', f'{current.parent}:/map_cache:ro',
        '-v', f'{current}:/current_map:ro', '--entrypoint', '/usr/bin/python3',
        image, '/repo/navigation/route_revalidate.py']
    print('正在核对旧路线地图、当前地图几何和各目标格；不发送运动命令……', flush=True)
    # Hashing large farm maps can take longer than a ROS command timeout.
    # This offline process has no deadline; stderr reports per-cache progress.
    response = subprocess.run(command, input=json.dumps(route, allow_nan=False),
                              text=True, stdout=subprocess.PIPE, check=True)
    results = [line[len(REVALIDATE_PREFIX):] for line in response.stdout.splitlines()
               if line.startswith(REVALIDATE_PREFIX)]
    if len(results) != 1:
        raise RuntimeError('地图重新校验没有返回唯一结果，未保存新路线。')
    updated = validate_route(json.loads(results[0]))
    if updated['waypoints'] != route['waypoints']:
        raise RuntimeError('重新校验改变了点位坐标或顺序，拒绝保存。')
    return updated


def waypoint_number(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError('航点编号必须为正整数') from error
    if number < 1:
        raise argparse.ArgumentTypeError('航点编号必须为正整数')
    return number


def queue_report(result):
    """Show only what the live operator acknowledged, never assume missing data is free."""
    queue = result.get('queue')
    if not isinstance(queue, dict) or not isinstance(queue.get('entries'), list):
        raise RuntimeError('队列回执缺少明细，未完成检查；请核对更新后的导航进程。')
    for name in ('completed', 'queued'):
        if type(queue.get(name)) is not int or queue[name] < 0:
            raise RuntimeError('队列回执计数无效，未完成检查。')
    for name in ('started', 'paused'):
        if type(queue.get(name)) is not bool:
            raise RuntimeError('队列回执状态无效，未完成检查。')
    for name in ('current', 'editing'):
        if name not in queue or (queue[name] is not None and
                                (type(queue[name]) is not int or queue[name] < 1)):
            raise RuntimeError('队列回执航点编号无效，未完成检查。')
    labels = {'free':'目标格空闲', 'occupied':'障碍格', 'clearance':'障碍安全边界',
              'unknown':'未知格', 'outside':'地图范围外', 'unavailable':'地图未收到',
              'invalid':'地图或坐标无效'}
    lines = [f'当前执行：{queue["current"] if queue["current"] is not None else "无"}；'
             f'待执行：{queue["queued"]}；已完成：{queue["completed"]}；'
             f'已开始：{queue["started"]}；已暂停：{queue["paused"]}']
    if queue['editing'] is not None:
        lines.append(f'等待替换 #{queue["editing"]}：下一次 Foxglove 目标点击将替换此点，不追加。')
    blocked = []
    for entry in queue['entries']:
        if (not isinstance(entry, dict) or type(entry.get('number')) is not int or
                entry['number'] < 1 or type(entry.get('active')) is not bool or
                entry.get('status') not in labels):
            raise RuntimeError('队列回执航点明细无效，未完成检查。')
        xyz = [entry.get(axis) for axis in ('x', 'y', 'z')]
        if entry['status'] == 'invalid' and xyz == [None, None, None]:
            coordinates = '坐标无效'
        else:
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in xyz):
                raise RuntimeError('队列回执包含无效坐标，未完成检查。')
            coordinates = f'x={xyz[0]:.3f} y={xyz[1]:.3f} z={xyz[2]:.3f}'
        state = '执行中' if entry['active'] else '待执行'
        value = entry.get('value')
        lines.append(f'#{entry["number"]} {state}  {coordinates}  '
                     f'{labels[entry["status"]]}  value={value if value is not None else "未知"}')
        if entry.get('message'):
            lines.append('  ' + str(entry['message']))
        if entry['status'] != 'free':
            blocked.append(entry['number'])
    if blocked:
        lines.append('需要检查的点：' + ', '.join('#'+str(n) for n in blocked) +
                     '；可用 routes.sh replace 编号 原位重设，其他点保留。')
    elif queue['entries']:
        lines.append('上述目标格目前均空闲；这不代表整条路径可达，执行仍检查定位、代价地图和碰撞。')
    else:
        lines.append('当前没有执行中或待执行点位。')
    return '\n'.join(lines)


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
    commands.add_parser('check', help='检查当前队列每个点的编号、坐标和静态地图状态，不改变运动')
    replace = commands.add_parser('replace', help='让下一次 Foxglove 目标点击原位替换指定待执行点；不启用运动')
    replace.add_argument('number', type=waypoint_number, help='check 中显示的待执行航点编号')
    revalidate = commands.add_parser('revalidate', help='只读核对旧/新地图后另存路线；不加载队列或启用运动')
    revalidate.add_argument('name', type=route_name)
    revalidate.add_argument('--as', dest='new_name', type=route_name, required=True,
                            help='新路线名，禁止覆盖原文件或已有文件')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    store = RouteStore(args.directory)
    try:
        if args.command == 'revalidate':
            destination = store.directory / (args.new_name + '.json')
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(str(destination))
            updated = revalidate_saved_route(store.load(args.name))
            path = store.save(args.new_name, updated, overwrite=False)
            print(f'地图核对通过，路线另存为：{path}\n'
                  f'全部 {len(updated["waypoints"])} 个点的坐标、朝向和顺序保留；原路线未覆盖。\n'
                  '尚未加载队列或启用运动，后续使用 load 新路线名。')
        elif args.command == 'check':
            print(queue_report(request_route('check')))
        elif args.command == 'replace':
            result = request_route('replace', number=args.number)
            if type(result.get('editing')) is not int or result['editing'] != args.number:
                raise RuntimeError('替换回执编号不匹配，操作未确认；请运行 routes.sh check 查看当前状态。')
            print(f'已进入 #{args.number} 的重设模式。\n'
                  '在 Foxglove 导航视图设置一次新的空闲目标位置，用于替换此点；其余队列保留。\n'
                  '替换后运行 routes.sh check 核对；这条命令没有启用运动。')
            print(result.get('message', ''))
        elif args.command == 'list':
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
        if args.command == 'revalidate':
            print('新路线名已存在：请换一个 --as 名称；重新校验不覆盖任何旧文件。', file=sys.stderr)
        else:
            print('保存失败：同名路线已存在。换一个路线名，或使用 save 路线名 --overwrite 明确覆盖。', file=sys.stderr)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print('路线操作失败：' + str(error), file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
