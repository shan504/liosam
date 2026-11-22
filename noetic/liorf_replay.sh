#!/usr/bin/env bash
# Run a converted static comparison capture against an already-built ROS1 LIORF workspace.
set -eo pipefail
if [[ ${1:-} == --help || $# != 2 ]]; then
    echo 'Usage: bash liorf_replay.sh CAPTURE_DIRECTORY /path/to/built/liorf_workspace'
    [[ ${1:-} == --help ]] && exit 0
    exit 2
fi
if [[ ${LIORF_REPLAY_ENV:-} != 1 ]]; then
    exec env -i HOME="$HOME" USER="${USER:-$(id -un)}" PATH=/usr/local/bin:/usr/bin:/bin \
        LIORF_REPLAY_ENV=1 bash "$0" "$@"
fi
source /opt/ros/noetic/setup.bash
source "$2/devel/setup.bash"
[[ ${ROS_DISTRO:-} == noetic ]] || { echo 'LIORF workspace must be ROS1 Noetic.' >&2; exit 1; }
export ROS_IP=127.0.0.1
export PYTHONUNBUFFERED=1
package=$(rospack find liorf)
here=$(cd "$(dirname "$0")" && pwd)
exec /usr/bin/python3 - "$1" "$package" "$here" <<'PY'
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import xmlrpc.client
from pathlib import Path

import numpy as np
import rosbag
import yaml

capture, pkg, here = (Path(v).resolve() for v in sys.argv[1:])
baseline = capture / "baseline.bag"
snapshot = capture / "lio_sam.yaml"
if not baseline.is_file():
    sys.exit("Missing original capture: " + str(baseline))
lidar = "/liorf_ab/lidar"
imu = "/liorf_ab/imu"
outputs = ["/liorf/mapping/odometry", "/liorf/mapping/cloud_registered_raw",
           "/liorf/deskew/cloud_deskewed"]
module = pkg / "launch/include/module_loam.launch"
if not module.is_file():
    sys.exit("Missing upstream LIORF launch: " + str(module))
with snapshot.open() as f:
    src = yaml.safe_load(f)
if isinstance(src, dict):
    src = src.get("lio_sam", src)
if not isinstance(src, dict) or src.get("sensor") != "livox" or src.get("imuType") != 1:
    sys.exit("Expected a /lio_sam parameter dump with sensor=livox and imuType=1 (filtered SI IMU).")
if str(src.get("imuTopic", "")).strip("/") != "mid360s/imu_filtered":
    sys.exit("Capture parameters must describe /mid360s/imu_filtered.")
with (pkg / "config/lio_sam_livox.yaml").open() as f:
    stock = yaml.safe_load(f)["liorf"]
required = ("extrinsicRot", "extrinsicRPY", "extrinsicTrans", "N_SCAN", "Horizon_SCAN",
            "imuGravity", "imuGyrNoise", "imuAccNoise", "imuGyrBiasN", "imuAccBiasN",
            "downsampleRate", "lidarMinRange", "lidarMaxRange", "mappingSurfLeafSize",
            "lidarFrame", "mappingProcessInterval")
missing = [k for k in required if k not in src]
if missing:
    sys.exit("Incomplete capture parameters: " + ", ".join(missing))
for k in ["extrinsicRot", "extrinsicRPY"]:
    a = np.asarray(src[k], dtype=float).reshape(3, 3)
    if not np.allclose(a.T @ a, np.eye(3), atol=1e-3) or abs(np.linalg.det(a)-1) > 1e-3:
        sys.exit("Invalid rotation: " + k)
if len(src["extrinsicTrans"]) != 3 or not np.isfinite(src["extrinsicTrans"]).all():
    sys.exit("Invalid extrinsicTrans")
out = Path(tempfile.mkdtemp(prefix="liorf_ab_", dir=str(capture)))
bag = out / "converted.bag"
print("本次离线对比结果目录", out)
subprocess.run([sys.executable, str(here / "liorf_bag.py"), str(baseline), str(bag)], check=True)
with rosbag.Bag(str(bag)) as b:
    info = b.get_type_and_topic_info().topics
    for name, typ in [(lidar, "sensor_msgs/PointCloud2"), (imu, "sensor_msgs/Imu")]:
        if name not in info or info[name].msg_type != typ:
            sys.exit("Expected converted input %s (%s)" % (name, typ))
    imu_times = []
    max_points = 0
    for topic, msg, _ in b.read_messages(topics=[lidar, imu]):
        if topic == imu:
            imu_times.append(msg.header.stamp.to_nsec())
        else:
            if not max_points:
                fields = {f.name: (f.datatype, f.count) for f in msg.fields}
                wanted = {"x": 7, "y": 7, "z": 7, "intensity": 7, "ring": 4, "time": 7}
                if any(fields.get(k) != (n, 1) for k, n in wanted.items()):
                    sys.exit("LIORF expects float32 x,y,z,intensity,time and uint16 ring.")
            max_points = max(max_points, msg.width * msg.height)
    first_time = b.get_start_time()
    duration = b.get_end_time() - first_time
dts = [b-a for a, b in zip(imu_times, imu_times[1:])]
if not dts or min(dts) <= 0 or max_points == 0:
    sys.exit("Missing LiDAR returns or IMU timestamps are insufficient/out of order.")
rate = 1e9/statistics.median(dts)
# Both branches use extRot * acceleration/gyro, q * inverse(extrinsicRPY) in mode 1.
# Keep matching ranges, surface voxel and processing interval from the captured fork.
p = {**stock, **{k: v for k, v in src.items() if k in stock}}
nscan = int(src["N_SCAN"])
if nscan <= 0 or int(src["Horizon_SCAN"]) <= 0:
    sys.exit("Capture has invalid N_SCAN/Horizon_SCAN.")
p.update(pointCloudTopic=lidar, imuTopic=imu, imuRate=rate, point_filter_num=1,
         Horizon_SCAN=max(int(src["Horizon_SCAN"]), math.ceil(max_points / nscan)),
         imuType=1, savePCD=False, odomTopic="/liorf_ab/imu_odom",
         gpsTopic="/liorf_ab/unused_gps", baselinkFrame=src["lidarFrame"],
         surroundingKeyframeMapLeafSize=src["mappingSurfLeafSize"],
         loopClosureICPSurfLeafSize=src["mappingSurfLeafSize"])
# LIORF uses preallocated N_SCAN*Horizon_SCAN vectors. Capacity may need to grow
# because its frontend visits all points without range-image column truncation.
with (out / "liorf.yaml").open("w") as f:
    yaml.safe_dump({"liorf": p}, f)
print("结果目录", out, "| IMU %.1f Hz | 最大单帧 %d 点" % (rate, max_points))
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
if port in (11311, 11321):
    sys.exit("Reserved ROS master port.")
uri = "http://127.0.0.1:%d" % port
os.environ.update(ROS_MASTER_URI=uri, ROS_LOG_DIR=str(out / "roslog"))
socket.setdefaulttimeout(0.75)
master = xmlrpc.client.ServerProxy(uri)
procs = []
logs = []

def launch(name, argv):
    f = (out / (name + ".log")).open("w")
    logs.append(f)
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=f,
                             stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(child)
    return child

def stop(child):
    if child.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 12), (signal.SIGTERM, 4), (signal.SIGKILL, 2)):
        try:
            os.killpg(child.pid, sig)
            child.wait(timeout=timeout)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            pass

def is_our_master(pid, root):
    while pid > 1:
        if pid == root:
            return True
        try:
            pid = int(Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()[1])
        except (FileNotFoundError, ValueError, IndexError):
            return False
    return False

def wait_for(check, processes, desc, secs=30):
    deadline = time.monotonic()+secs
    while time.monotonic() < deadline:
        if any(c.poll() is not None for c in processes):
            raise RuntimeError("Process exited: %s; logs in %s" % (desc, out))
        try:
            if check():
                return
        except (OSError, xmlrpc.client.Error):
            pass
        time.sleep(0.1)
    raise RuntimeError("Timeout waiting for %s; logs in %s" % (desc, out))

def connected():
    pubs, subs, _ = master.getSystemState("/liorf_replay")[2]
    pubs, subs = dict(pubs), dict(subs)
    return (all("/liorf_recorder" in subs.get(t, []) and pubs.get(t) for t in outputs)
            and "/liorf_imageProjection" in subs.get(lidar, [])
            and "/liorf_imuPreintegration" in subs.get(imu, [])
            and "/liorf_imageProjection" in subs.get(imu, []))

def interrupted(signum, frame):
    raise KeyboardInterrupt

for sig in (signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, interrupted)
try:
    core = launch("master", ["roscore", "-p", str(port)])
    wait_for(lambda: is_our_master(master.getPid("/liorf_replay")[2], core.pid), [core],
             "private master " + uri)
    subprocess.run(["rosparam", "load", str(out / "liorf.yaml")], check=True)
    # Use ROS wall time for process scheduling. LIORF integrates header stamps;
    # the offline result must be compared by header stamp, never record-arrival time.
    nodes = launch("nodes", ["roslaunch", "--wait", "-p", str(port), str(module)])
    result = out / "liorf.bag"
    rec = launch("rosbag_record", ["/opt/ros/noetic/lib/rosbag/record", "__name:=liorf_recorder",
                                   "-O", str(result)] + outputs)
    wait_for(connected, [core, nodes, rec], "LIORF inputs and three recorded outputs", 45)
    player = launch("rosbag_play", ["/opt/ros/noetic/lib/rosbag/play", "--rate", "0.5",
                                    "--delay", "1.0", "--wait-for-subscribers",
                                    "--topics", lidar, imu, "--bags", str(bag)])
    player.wait(timeout=max(45, duration*2 + 35))
    if player.returncode:
        raise RuntimeError("rosbag play failed; see " + str(out / "rosbag_play.log"))
    time.sleep(3)
    stop(rec)
    if rec.returncode not in (0, -signal.SIGINT):
        raise RuntimeError("rosbag record failed; see " + str(out / "rosbag_record.log"))
    with rosbag.Bag(str(result)) as b:
        counts = [b.get_message_count(t) for t in outputs]
    if min(counts) == 0:
        raise RuntimeError("No comparable result: %s=%s; logs %s" % (outputs, counts, out))
    print("LIORF 里程计/配准点云/去畸变点云帧数", counts)
    print(result)
    if "process has died" in (out / "nodes.log").read_text(errors="replace"):
        raise RuntimeError("LIORF node restarted during replay; inspect " + str(out / "nodes.log"))
finally:
    for child in reversed(procs):
        stop(child)
    for f in logs:
        f.close()
check = subprocess.run([sys.executable, str(here / "liorf_compare.py"), str(baseline), str(result)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(out / "comparison.txt").write_text(check.stdout)
print(check.stdout, end="")
print("对比报告", out / "comparison.txt")
preview = subprocess.run([sys.executable, str(here / "contour_view.py"), str(baseline),
                          "--liorf", str(result), "--output", str(out / "contour_ab.png")])
sys.exit(check.returncode or preview.returncode)
PY
