#!/usr/bin/env python3
"""Collect one stationary sample for both the live alignment check and offline replay."""
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

INPUTS = ("/livox/lidar", "/mid360s/imu_filtered")
BASELINE = ("/lio_sam/mapping/odometry", "/lio_sam/deskew/cloud_deskewed",
            "/lio_sam/mapping/cloud_registered_raw")


def stop_recording(recorder):
    if recorder.poll() is None:
        recorder.send_signal(signal.SIGINT)
    try:
        return recorder.wait(timeout=25)
    except subprocess.TimeoutExpired:
        recorder.terminate()
        return recorder.wait(timeout=5)


def main():
    if len(sys.argv) != 1:
        sys.exit("用法：bash noetic/live_detail.sh --capture-still")
    import rosbag

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, interrupted)

    directory = Path.home() / "go2_alignment_runs"
    directory.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=time.strftime("still_%Y%m%d_%H%M%S_"), dir=str(directory)))
    config = run / "lio_sam.yaml"
    subprocess.run(["rosparam", "dump", str(config), "/lio_sam"], timeout=8, check=True)
    bag_file = run / "baseline.bag"
    topics = INPUTS + BASELINE
    print("本次样本目录 %s。保持四脚静止，约 15 秒。" % run, flush=True)
    recorder = subprocess.Popen(
        ["/opt/ros/noetic/lib/rosbag/record", "--duration=40", "-O", str(bag_file), *topics],
        start_new_session=True)
    try:
        # Wait for the recorder to open the file before measuring the stationary window.
        deadline = time.monotonic() + 8
        while not (run / "baseline.bag.active").exists():
            if recorder.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("录包未正常启动，请查看本终端的 record 输出。")
            time.sleep(0.1)
        time.sleep(1.0)
        probe = Path(__file__).with_name("check_alignment.py")
        report = run / "alignment.txt"
        with report.open("w", encoding="utf-8") as output, subprocess.Popen(
                [sys.executable, "-u", str(probe)], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True) as child:
            for line in child.stdout:
                output.write(line)
                print(line, end="", flush=True)
            status = child.wait()
    finally:
        stop_recording(recorder)
    missing = []
    with rosbag.Bag(str(bag_file), "r") as bag:
        info = bag.get_type_and_topic_info().topics
        for topic in topics:
            count = info[topic].message_count if topic in info else 0
            print("%s  %d 条" % (topic, count))
            if count == 0:
                missing.append(topic)
    print("实机样本 %s，诊断报告 %s" % (bag_file, report))
    if not any(topic in missing for topic in BASELINE[1:]):
        from contour_view import render_capture
        preview = run / "contours.png"
        try:
            ready = render_capture(bag_file, preview)
        except (ValueError, OSError, rosbag.ROSBagException) as exc:
            print("未能生成轮廓图：%s；本次原始录包和报告仍可读取。" % exc, file=sys.stderr)
            ready = False
        if ready:
            print("实测轮廓图 %s" % preview)
            print('Windows PowerShell 查看本次原图：')
            print('scp "unitree@192.168.1.230:%s" "$env:TEMP\\go2-contours.png"; if ($LASTEXITCODE -eq 0) { Invoke-Item "$env:TEMP\\go2-contours.png" }' % preview)
    if missing:
        print("这些话题缺数据，本次不足以做完整对比：" + "、".join(missing), file=sys.stderr)
    return status or int(bool(missing))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        sys.exit("静止采样未完成：%s" % exc)
