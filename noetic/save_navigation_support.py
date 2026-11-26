"""Save calibration, logs and provenance for one active mapping run."""
import json
import os
from pathlib import Path
import re
import subprocess


def copy_file_prefix(source, target):
    # Loggers may append while we save. Freeze the readable length for this copy.
    with source.open("rb") as src, target.open("xb") as dst:
        remaining = os.fstat(src.fileno()).st_size
        while remaining:
            chunk = src.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise OSError("文件复制时被截短: " + str(source))
            dst.write(chunk)
            remaining -= len(chunk)


def last_path_in_log(log, pattern):
    if not log.is_file():
        return None
    latest = None
    with log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            cleaned = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
            match = pattern.search(cleaned)
            if match:
                latest = match.group(1)
    return latest


def snapshot_run(archive_dir, output_dir):
    """archive_dir must be the active dense node's ~archive_dir, never a 'latest' link."""
    dense = Path(archive_dir).expanduser().resolve(strict=True)
    output = Path(output_dir)
    run = dense.parent
    result = dict(source_archive_dir=str(dense), calibration_sources={}, warnings=[])

    logs = output / "run_logs"
    logs.mkdir()
    result["run_files"] = []
    for path in sorted(run.iterdir()):
        if path.is_file() and (path.suffix == ".log" or
                              path.name in ("D_bridge_params.yaml", "imu_units.py")):
            copy_file_prefix(path, logs / path.name)
            result["run_files"].append(path.name)

    calib = output / "calibration"
    calib.mkdir(exist_ok=True)
    paths = (
        ("floor.json", "E_ground.log",
         re.compile(r"(?:参数文件|本次标定)\s*[:：]?\s*(/.*?/floor\.json)$")),
        ("body_from_lidar.json", "C_native_seed.log",
         re.compile(r"建图预测采用 .* 安装外参 \((/.*)\)$")),
    )
    for filename, log, pattern in paths:
        reported = last_path_in_log(logs / log, pattern)
        if reported and Path(reported).is_file():
            target = calib / filename
            copy_file_prefix(Path(reported), target)
            with target.open(encoding="utf-8") as stream:
                if not isinstance(json.load(stream), dict):
                    raise ValueError("无效的标定 JSON: " + str(target))
            result["calibration_sources"][filename] = reported
        else:
            result["warnings"].append("当前会话未提供可读取的标定文件 " + filename)

    repo = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL, timeout=5, text=True)
        result["checkout_commit_at_save"] = commit.strip()
    except (OSError, subprocess.SubprocessError):
        result["warnings"].append("无法读取保存时工作区 Git 提交号")
    return result
