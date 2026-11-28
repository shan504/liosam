"""Offline route/map verification in a network-isolated, read-only container."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'original_go2/go2_3d'))
from route_store import map_identity, revalidate_route, validate_route

RESULT_PREFIX = 'GO2_ROUTE_REVALIDATED '


def revalidate_from_cache(route, cache_root, current_directory):
    route = validate_route(route)
    directories = sorted(path for path in Path(cache_root).glob('original-go2-*') if path.is_dir())
    if not directories:
        raise ValueError('未找到旧地图缓存，无法确认旧路线坐标；请在当前地图重新设置点位。')
    previous = None
    for index, directory in enumerate(directories, 1):
        print(f'核对旧地图缓存 {index}/{len(directories)}：{directory.name}', file=sys.stderr, flush=True)
        try:
            identity = map_identity(directory)
        except (OSError, ValueError, TypeError, KeyError) as error:
            print(f'缓存不可验证，跳过：{error}', file=sys.stderr, flush=True)
            continue
        if identity['fingerprint'] == route['map']['fingerprint']:
            previous = directory
            break
    if previous is None:
        raise ValueError('未找到与旧路线指纹完全一致的地图缓存；不能仅按地图名复用坐标，请重新设置点位。')
    print('已找到旧地图，正在核对当前 PCD、坐标参数和全部目标格……', file=sys.stderr, flush=True)
    return revalidate_route(route, previous, current_directory)


def main():
    try:
        raw = sys.stdin.read(131073)
        if len(raw.encode('utf-8')) > 131072:
            raise ValueError('Route request exceeds 128 KiB')
        route = revalidate_from_cache(json.loads(raw), '/map_cache', '/current_map')
        print(RESULT_PREFIX + json.dumps(route, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print('路线地图校验失败：' + str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
