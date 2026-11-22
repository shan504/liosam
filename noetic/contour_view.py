#!/usr/bin/env python3
"""Fixed-scale orthographic crops of measured returns during a stationary capture."""
import argparse
from pathlib import Path
import struct
import zlib

import numpy as np

from liorf_compare import bag_index, read_selected
from stationary_surfaces import read_xyz


def choose_center(points, width):
    # Prefer a populated local region with depth variation in three directions.
    # This selects a region to inspect, not a classified object or a fitted edge.
    cells = np.floor(points / (width / 2)).astype(np.int32)
    _, group, count = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    group = group.reshape(-1)
    best, best_score = None, -1.
    for i in np.argsort(-count)[:64]:
        center = np.median(points[group == i], axis=0)
        delta = points - center
        patch = delta[np.max(np.abs(delta), axis=1) < width / 2]
        if len(patch) < 35:
            continue
        # Suppress isolated far outliers before scoring spatial variation.
        lo, hi = np.percentile(patch, [5, 95], axis=0)
        patch = patch[((patch >= lo) & (patch <= hi)).all(axis=1)]
        if len(patch) < 16:
            continue
        eigen = np.linalg.eigvalsh(np.cov(patch.T)).clip(0)
        score = float(eigen[0] * np.sqrt(eigen[1]) * np.log1p(len(patch)))
        if score > best_score:
            best, best_score = center, score
    return best


# Minimal built-in 3x5 glyphs keep ROS Noetic independent of GUI/font packages.
GLYPH = {
    'A':('010','101','111','101','101'), 'C':('111','100','100','100','111'),
    'E':('111','100','110','100','111'), 'F':('111','100','110','100','100'),
    'I':('111','010','010','010','111'), 'K':('101','110','100','110','101'),
    'L':('100','100','100','100','111'), 'M':('101','111','111','101','101'),
    'O':('111','101','101','101','111'), 'R':('110','101','110','101','101'),
    'S':('111','100','111','001','111'), 'T':('111','010','010','010','010'),
    'X':('101','101','010','101','101'), 'Y':('101','101','010','010','010'),
    'Z':('111','001','010','100','111'), '-':('000','000','111','000','000'),
    '.':('000','000','000','000','010'), '+':('000','010','111','010','000'),
    '0':('111','101','101','101','111'), '1':('010','110','010','010','111'),
    '2':('110','001','010','100','111'), '3':('110','001','010','001','110'),
    '4':('101','101','111','001','001'), '5':('111','100','110','001','110'),
    '6':('011','100','111','101','111'), '7':('111','001','010','010','010'),
    '8':('111','101','111','101','111'), '9':('111','101','111','001','110'),
}


def label(canvas, x, y, text):
    for i, char in enumerate(text):
        for dy, row in enumerate(GLYPH.get(char, ())):
            for dx, bit in enumerate(row):
                if bit == '1':
                    xx, yy = x+8*i+2*dx, y+2*dy
                    canvas[yy:yy+2, xx:xx+2] = 215


def annotate(canvas, side, gap, width, count):
    h,w = canvas.shape[:2]
    top,left,bottom = 24,26,28
    full = np.full((h+top+bottom, w+left, 3), 25, dtype=np.uint8)
    full[top:top+h,left:] = canvas
    for k in range(2*count):
        text = ('LIO-SAM' if k < 2 else 'LIORF') + (' 1 FRAME' if k % 2 == 0 else ' STACK')
        label(full, left+k*(side+gap)+(side-8*len(text))//2, 7, text)
    for k, text in enumerate(('XY', 'XZ', 'YZ')):
        label(full, 4, top+k*(side+gap)+side//2-5, text)
    power = 10**np.floor(np.log10(width/4))
    metres = max(v*power for v in (1,2,5) if v*power <= width/4 * (1+1e-9))
    length = max(1, int(round(metres/width*side)))
    base = top+h+12
    full[base:base+2,left:left+length+1] = 235
    for x in (left,left+length):
        full[base-4:base+6,x:x+2] = 235
    label(full, left+length+10, base-4, '%.3G M' % metres)
    return full


def png_bytes(image, legend):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind+data))
    h, w = image.shape[:2]
    raw = b''.join(b'\0'+row.tobytes() for row in image)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', w, h, 8, 2, 0, 0, 0))
            + chunk(b'tEXt', b'Description\x00'+legend.encode('ascii'))
            + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def draw_crop(groups, stamps, center, width):
    # Each pixel is one occupied bin at fixed physical scale. The blank gaps
    # separate projections; there are no interpolated points or fitted lines.
    side, gap = 320, 5
    col = 2 * len(groups)
    canvas = np.zeros((3 * side + 2 * gap, col * side + (col-1) * gap, 3), np.uint8)
    for row in (1, 2):
        canvas[row*(side+gap)-gap:row*(side+gap)] = 45
    for k in range(1, col):
        canvas[:, k*(side+gap)-gap:k*(side+gap)] = 45
    counts = []
    for j, data in enumerate(groups):
        first = stamps[0]
        r, offset = data[first][2]
        group_counts = []
        for i, t in enumerate(stamps):
            p = (data[t][1] - offset) @ r.T - center
            p = p[(np.abs(p) < width/2).all(axis=1)]
            group_counts.append(len(p))
            bins = np.floor((p/width + .5)*side).astype(np.int64).clip(0, side-1)
            for row, (u, v) in enumerate(((0, 1), (0, 2), (1, 2))):
                y = row*(side+gap) + side-1 - bins[:, v]
                x = j*2*(side+gap) + bins[:, u]
                if i == 0:
                    canvas[y, x] = (235, 235, 235)
                canvas[y, x+side+gap] = (235, 235, 235)
        counts.append(group_counts)
    return annotate(canvas, side, gap, width, len(groups)), counts


def full_crop(bag, namespace, first_samples, stamps, center, width):
    from sensor_msgs.msg import PointCloud2
    rot, offset = first_samples[stamps[0]][2]
    frames = {}
    for _, row, _ in bag.read_messages(topics=[f'/{namespace}/mapping/cloud_registered_raw'], raw=True):
        sec, ns = struct.unpack_from('<II', row[1], 4)
        t = sec*10**9 + ns
        if t not in first_samples:
            continue
        msg = PointCloud2().deserialize(row[1])
        total = msg.height * msg.width
        inside = []
        for start in range(0, total, 16384):
            pts = read_xyz(msg, np.arange(start, min(start+16384, total)))
            finite = pts[np.isfinite(pts).all(axis=1)]
            local = (finite-offset) @ rot.T - center
            inside.append(finite[(np.abs(local) < width/2).all(axis=1)])
        frames[t] = (None, np.concatenate(inside) if inside else np.empty((0,3)), (rot, offset))
    return frames


def render_capture(baseline_path, output_path, comparison_path=None, center=None, width=1.2):
    """Render actual recorded samples. Returns False if the crop lacks repeated scans."""
    import rosbag
    from contextlib import ExitStack
    if not np.isfinite(width) or width <= 0:
        raise ValueError('裁剪边长须为正且有限')
    with ExitStack() as stack:
        inputs = [(baseline_path, 'lio_sam')]
        if comparison_path is not None:
            inputs.append((comparison_path, 'liorf'))
        bags = [(stack.enter_context(rosbag.Bag(str(p))), ns) for p, ns in inputs]
        ids = [bag_index(b, ns) for b, ns in bags]
        common = sorted(set.intersection(*(m[k] for m in ids for k in ('sensor', 'world'))))
        if len(common) < 4:
            print('共同配准扫描不足 4 帧，无法绘制静止叠帧轮廓。')
            return False
        take = np.linspace(0, len(common)-1, min(12, len(common)), dtype=int)
        stamps = [common[i] for i in take]
        runs = [read_selected(b, ns, set(), set(stamps)) for b, ns in bags]
        if len({d[2]['sensor'] for d in runs}) != 1:
            raise ValueError('两路传感器参考系不同，不能共用裁剪框')
        groups = [d[1] for d in runs]
        if center is None:
            center = choose_center(groups[0][stamps[0]][0], width)
        if center is None:
            print('首帧缺少有足够实测点的局部区域。')
            return False
        center = np.asarray(center, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('裁剪中心需要三个有限坐标')
        groups = [full_crop(b, ns, data, stamps, center, width) for (b, ns), data in zip(bags, groups)]
    image, counts = draw_crop(groups, stamps, center, width)
    if any(sum(x > 0 for x in c) < 2 or c[0] == 0 for c in counts):
        print('裁剪框中缺少首帧或后续回波。')
        return False
    legend = ('Rows top to bottom: XY, XZ, YZ. Column pairs per run: first scan, registered stack. '
              + 'Runs: '+', '.join(v[1] for v in inputs)+'. Frame: '+runs[0][2]['sensor'].encode('ascii','replace').decode()
              +'. Center %.3f %.3f %.3f m, side %.3f m, %d frames.' % (*center, width, len(stamps)))
    with Path(output_path).open('xb') as f:
        f.write(png_bytes(image, legend))
    print('轮廓实测图 %s。行从上到下 XY、XZ、YZ；每组左列首帧、右列同一物理区域的配准叠帧。' % output_path)
    print('首帧传感器系 %s，中心 (%.3f, %.3f, %.3f) m，边长 %.2f m。%d 帧，横纵每像素 %.1f mm。'
          % (runs[0][2]['sensor'], *center, width, len(stamps), 1000*width/320))
    for (_, ns), c in zip(inputs, counts):
        print('%s 裁剪框首帧 %d 个实测点，全部选中帧共 %d 个。' % (ns, c[0], sum(c)))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--liorf', type=Path)
    parser.add_argument('--center', type=float, nargs=3, metavar=('X','Y','Z'))
    parser.add_argument('--width', type=float, default=1.2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    dest = args.output or args.baseline.with_name('contour_ab.png' if args.liorf else 'contour.png')
    try:
        ok = render_capture(args.baseline, dest, args.liorf, args.center, args.width)
    except (OSError, ValueError, ImportError, np.linalg.LinAlgError) as exc:
        parser.exit(2, '绘图未完成：%s\n' % exc)
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
