"""Inspect observed /tf_static mounting chains in one ROS graph."""
import json
import math
from threading import Lock


IDENTITY = ((0., 0., 0.), (0., 0., 0., 1.))


def multiply(a, b):
    x, y, z, w = a
    u, v, s, t = b
    return (w*u+x*t+y*s-z*v, w*v-x*s+y*t+z*u,
            w*s+x*v-y*u+z*t, w*t-x*u-y*v-z*s)


def conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def rotate(q, p):
    return multiply(multiply(q, p + (0.,)), conjugate(q))[:3]


def compose(a, b):
    p, q = a
    delta = rotate(q, b[0])
    return (tuple(i+j for i, j in zip(p, delta)), multiply(q, b[1]))


def invert(transform):
    p, q = transform
    r = conjugate(q)
    return rotate(r, tuple(-n for n in p)), r


def same(a, b):
    # Quaternions q and -q describe the same physical rotation.
    return (max(abs(i-j) for i, j in zip(a[0], b[0])) < 1e-6
            and abs(sum(i*j for i, j in zip(a[1], b[1]))) > 1. - 1e-12)


def part_frame(name):
    name = name.lower()
    leaf = name.split('/')[-1]
    return (leaf not in ('world', 'odom', 'map', 'odom_mapping', 'go2_ground', 'go2_ground_focus')
            and not name.startswith(('go2_visual/', 'go2_native/')))


def origins(frame, edges, bad):
    transforms = {frame: IDENTITY}
    trail = [frame]
    while frame in edges:
        if frame in bad:
            raise ValueError('有冲突或无效的静态变换，子帧 ' + frame)
        parent, transform = edges[frame]
        if not part_frame(parent):
            break
        if parent in transforms:
            raise ValueError('静态 TF 环路经过 ' + frame)
        transforms[parent] = compose(transform, transforms[frame])
        trail.append(parent)
        frame = parent
    if frame in bad:
        raise ValueError('有冲突或无效的静态变换，子帧 ' + frame)
    return trail, transforms


def resolve(parent, child, edges, bad):
    up, a = origins(parent, edges, bad)
    down, b = origins(child, edges, bad)
    ancestor = next((f for f in up if f in b), None)
    if ancestor is None:
        return None
    path = up[:up.index(ancestor) + 1] + list(reversed(down[:down.index(ancestor)]))
    return path, compose(invert(a[ancestor]), b[ancestor])


class StaticTF:
    def __init__(self):
        self.lock = Lock()
        self.edges = {}
        self.bad = set()

    def add(self, tf_message):
        with self.lock:
            for item in tf_message.transforms:
                parent = item.header.frame_id.lstrip('/')
                child = item.child_frame_id.lstrip('/')
                if not parent or not child:
                    continue
                tr = item.transform.translation
                rot = item.transform.rotation
                p, q = (tr.x, tr.y, tr.z), (rot.x, rot.y, rot.z, rot.w)
                norm = math.sqrt(sum(n*n for n in q))
                if (not all(math.isfinite(n) for n in p + q)
                        or abs(norm - 1.) > .01 or child == parent):
                    self.bad.add(child)
                    continue
                value = p, tuple(n / norm for n in q)
                old = self.edges.get(child)
                if old and (old[0] != parent or not same(old[1], value)):
                    self.bad.add(child)
                else:
                    self.edges[child] = parent, value

    def report(self, body_frames_from_odom=(), lidar_frames_from_cloud=(),
               camera_frames_from_camera_info=()):
        with self.lock:
            edges, bad = self.edges.copy(), self.bad.copy()
        if not edges and not bad:
            print('本域未收到 /tf_static 安装变换')
            return
        frames = set(edges) | {parent for parent, _ in edges.values()}
        def names(values):
            return {f.lstrip('/') for f in values if f and part_frame(f.lstrip('/'))}
        bodies = names(body_frames_from_odom) | {f for f in frames if part_frame(f)
                    and f.split('/')[-1] in ('base_link', 'base', 'trunk', 'body', 'base_footprint')}
        cameras = names(camera_frames_from_camera_info) | {f for f in frames if part_frame(f)
                    and any(tag in f.lower() for tag in ('camera', 'd435', 'realsense'))}
        lidars = names(lidar_frames_from_cloud) | {f for f in frames if part_frame(f)
                   and any(tag in f.lower() for tag in ('livox', 'mid360', 'mid_360', 'lidar'))}
        print('静态 TF 连接数 %d；冲突子帧 %s' % (len(edges), ', '.join(sorted(bad)) or '无'))
        for role, selected in [('机身候选', bodies), ('相机候选', cameras), ('雷达候选', lidars)]:
            print('%s帧 %s' % (role, ', '.join(sorted(selected)) or '未发现'))
        results = 0
        for body in sorted(bodies):
            for target in sorted((cameras | lidars) - {body}):
                try:
                    result = resolve(body, target, edges, bad)
                except ValueError as error:
                    print('跳过 %s ← %s：%s' % (body, target, error))
                    continue
                if result is None:
                    continue
                chain, (p, q) = result
                desc = dict(parent=body, child=target,
                            xyz=[round(v, 8) for v in p], xyzw=[round(v, 10) for v in q])
                role = '雷达'
                if target in cameras:
                    role = '相机光学帧' if 'optical' in target.lower() else '相机非光学命名帧'
                print('%s 静态链 [%s]，将子帧坐标转换到父帧，平移单位 m：%s' %
                      (role, ' / '.join(chain), json.dumps(desc, ensure_ascii=False)))
                results += 1
        if not results:
            print('本域未观测到连接上述机身与传感器的完整静态 TF 链')
