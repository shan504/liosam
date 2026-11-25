#!/usr/bin/env python3
"""Use the physical Go2 + MID-360 URDF or enter measured body-to-sensor mounting."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from xml.etree import ElementTree

from native_inspect_tf import multiply, part_frame, resolve


def triple(text):
    values = tuple(float(s) for s in text.split())
    if len(values) != 3 or not all(map(math.isfinite, values)):
        raise ValueError("请输入三个有限数值，以空格分隔")
    return values


def quaternion(rpy):
    roll, pitch, yaw = (angle / 2 for angle in rpy)
    rx = (math.sin(roll), 0, 0, math.cos(roll))
    ry = (0, math.sin(pitch), 0, math.cos(pitch))
    rz = (0, 0, math.sin(yaw), math.cos(yaw))
    return multiply(multiply(rz, ry), rx)


def urdf_mount(urdf_xml, child, parent="base_link"):
    """Return [body XYZ, body-from-sensor XYZW] from an expanded fixed joint chain."""
    if not urdf_xml:
        return None
    child, parent = child.lstrip("/"), parent.lstrip("/")
    if not child or child == parent or not part_frame(child) or not part_frame(parent):
        return None
    try:
        robot = ElementTree.fromstring(urdf_xml)
    except ElementTree.ParseError as error:
        raise ValueError("URDF XML 解析失败: " + str(error)) from error
    if robot.tag != "robot":
        return None
    links = {link.get("name") for link in robot.findall("link")}
    if parent not in links or child not in links:
        return None
    edges, bad, seen = {}, set(), set()
    for joint in robot.findall("joint"):
        end, start = joint.find("child"), joint.find("parent")
        if end is None or start is None:
            continue
        target, source = end.get("link"), start.get("link")
        if not target or not source:
            continue
        if target in seen or target == source:
            bad.add(target)
        seen.add(target)
        if joint.get("type") != "fixed" or source not in links or target not in links:
            continue
        origin = joint.find("origin")
        try:
            attrs = origin.attrib if origin is not None else {}
            xyz = triple(attrs.get("xyz", "0 0 0"))
            q = quaternion(triple(attrs.get("rpy", "0 0 0")))
        except ValueError:
            bad.add(target)
            continue
        edges[target] = (source, (xyz, q))
    result = resolve(parent, child, edges, bad)
    if result is None:
        return None
    _, (xyz, q) = result
    return list(xyz + q)


def candidates(home, child):
    # Existing local files are suggestions. Ask the operator which URDF matches this bracket.
    roots = (home/"ros2_ws/src", home/"go2_lio_ws/src", home/"go2", home/"go2-detail",
             home/"mid360s_nk_noetic_ws/src", home/"cyclonedds_ws/src")
    seen = set()
    prune = {".git", ".venv", "build", "log", "meshes", "node_modules"}
    for root in roots:
        for dirname, dirs, names in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in prune and not d.startswith("."))
            for name in sorted(names):
                path = Path(dirname) / name
                if path.suffix.lower() != ".urdf":
                    continue
                try:
                    if path.stat().st_size > 8 * 1024 * 1024:
                        continue
                    pose = urdf_mount(path.read_bytes(), child)
                except (ValueError, OSError):
                    continue
                if pose is not None and tuple(pose) not in seen:
                    seen.add(tuple(pose))
                    yield path, pose


def save(path, desc, replace):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(desc, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            if replace:
                os.replace(temp, path)
            else:
                os.link(temp, path)
        finally:
            if temp.exists():
                temp.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", default="livox_frame", help="MID-360 原始点云的 header.frame_id")
    parser.add_argument("--urdf", type=Path, help="含有实机支架的已展开 URDF")
    parser.add_argument("--xyz", nargs=3, type=float, metavar=("X", "Y", "Z"), help="雷达中心相对 base_link 的三轴平移（米）")
    parser.add_argument("--rpy-deg", nargs=3, type=float, metavar=("ROLL", "PITCH", "YAW"), help="安装 RPY，按 URDF 约定（度）")
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("GO2_BODY_FROM_LIDAR") or
                        "~/.config/go2/body_from_lidar.json"))
    parser.add_argument("--replace", action="store_true", help="用指定安装标定替换原文件")
    args = parser.parse_args()
    path = args.output.expanduser()
    if path.exists() and not args.replace:
        print("已有安装标定 " + str(path))
        print(path.read_text())
        return
    child = args.child.lstrip("/")
    if not child or child == "base_link" or not part_frame(child):
        parser.error("请填写雷达的物理坐标系名")
    if (args.xyz is None) != (args.rpy_deg is None) or (args.urdf and args.xyz is not None):
        parser.error("选用 --urdf 或同时给出 --xyz 和 --rpy-deg")
    pose = None
    if args.urdf:
        pose = urdf_mount(args.urdf.expanduser().read_bytes(), child)
        if pose is None:
            parser.error("该 URDF 没有连接 base_link 和 " + child + " 的物理固定关节链")
        source = str(args.urdf.expanduser())
    elif args.xyz is None:
        print("需要 base_link ← %s 的安装标定。正在已安装的工作区中查找 URDF。" % child)
        picks = list(candidates(Path.home(), child))
        for i, (candidate, value) in enumerate(picks, 1):
            print("%d. %s  xyz=%s，xyzw=%s" % (i, candidate, value[:3], value[3:]))
        if picks:
            selected = input("输入与本机支架相符的 URDF 编号，直接回车可手动填写：").strip()
            if selected:
                if not selected.isdigit() or not 1 <= int(selected) <= len(picks):
                    parser.error("无效的 URDF 编号")
                model, pose = picks[int(selected)-1]
                source = str(model)
    if pose is None:
        print("base_link 采用前、左、上为正。平移以机身原点为起点。0.50 m 是雷达距地面的高度。")
        xyz = args.xyz if args.xyz is not None else triple(input("雷达中心相对机身的 X Y Z（米）："))
        rpy = args.rpy_deg if args.rpy_deg is not None else triple(input("雷达坐标轴相对机身的 roll pitch yaw（度）："))
        if not all(map(math.isfinite, list(xyz) + list(rpy))):
            parser.error("需要有限数值")
        pose = list(xyz) + list(quaternion(map(math.radians, rpy)))
        source = "operator measurement"
    save(path, dict(parent="base_link", child=child, xyz=pose[:3], xyzw=pose[3:], source=source),
         args.replace)
    print("已保存安装标定 " + str(path))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, EOFError) as error:
        sys.exit("安装标定失败：" + str(error))
