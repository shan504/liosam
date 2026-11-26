"""Select a complete, named map bundle, or validate an explicit version directory."""
from datetime import datetime
import json
import re
import sys
from pathlib import Path


def select(argument, root):
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', argument):
        directory = root / argument
        versions = [p for p in directory.glob('*') if p.is_dir() and
                    not p.name.startswith('.') and (p / 'manifest.json').is_file()]
        if not versions:
            raise ValueError('该名称下没有已保存的地图版本：' + str(directory))
        def completed(path):
            manifest = path / 'manifest.json'
            record = json.loads(manifest.read_text(encoding='utf-8'))
            stamp = record.get('completed_at')
            return datetime.fromisoformat(stamp).timestamp() if stamp else manifest.stat().st_mtime
        version = max(versions, key=lambda v: (completed(v), v.name))
    else:
        version = Path(argument).expanduser().resolve()
    manifest = json.loads((version / 'manifest.json').read_text(encoding='utf-8'))
    filename = manifest.get('map_file', '')
    if not filename or Path(filename).name != filename or not (version / filename).is_file():
        raise ValueError('所选目录缺少 save_map.sh 导出的源点云：' + str(version))
    if not all((version / key).is_file() for key in ('keyframes.csv', 'scans.csv')):
        raise ValueError('该版本缺少扫描索引或建图轨迹：' + str(version))
    return version.resolve()


if __name__ == '__main__':
    try:
        print(select(sys.argv[1], Path.home() / 'go2_maps'))
    except (OSError, ValueError) as error:
        sys.exit('地图选择失败：' + str(error))
