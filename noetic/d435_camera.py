#!/usr/bin/env python3
"""Relay the D435i color stream into the existing Noetic Foxglove image panel."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

OUTPUT = '/camera/image/compressed'
PERIOD = 1 / 8
MAX_BYTES = 8 * 1024 * 1024


def color_topic(topic, msg_type):
    kind = msg_type.replace('/msg/', '/')
    return ((topic.endswith('/color/image_raw') and kind == 'sensor_msgs/Image') or
            (topic.endswith('/color/image_raw/compressed') and kind == 'sensor_msgs/CompressedImage'))


def jpeg(msg):
    if hasattr(msg, 'format'):
        data = bytes(msg.data)
        if data.startswith(b'\xff\xd8\xff'):
            return 'jpeg', data
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'png', data
        raise ValueError('彩色压缩消息的实际编码未知')
    encoding = msg.encoding.lower()
    formats = {'rgb8': 'RGB', 'bgr8': 'BGR', 'rgba8': 'RGBA', 'bgra8': 'BGRA'}
    if encoding not in formats:
        raise ValueError('不支持的彩色图像编码 ' + encoding)
    channels = len(formats[encoding])
    if msg.width <= 0 or msg.height <= 0 or msg.step < channels * msg.width:
        raise ValueError('彩色图尺寸无效')
    try:
        from PIL import Image
    except ImportError:
        pass
    else:
        import io
        mode = 'RGBA' if channels == 4 else 'RGB'
        im = Image.frombytes(mode, (msg.width, msg.height), bytes(msg.data),
                             'raw', formats[encoding], msg.step, 1)
        output = io.BytesIO()
        im.convert('RGB').save(output, format='JPEG', quality=80)
        return 'jpeg', output.getvalue()
    import cv2
    import numpy as np
    code = {'bgr8': None, 'rgb8': cv2.COLOR_RGB2BGR, 'rgba8': cv2.COLOR_RGBA2BGR,
            'bgra8': cv2.COLOR_BGRA2BGR}
    pixels = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    pixels = pixels[:, :channels * msg.width].reshape(msg.height, msg.width, channels)
    if code[encoding] is not None:
        pixels = cv2.cvtColor(pixels, code[encoding])
    ok, buf = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError('JPEG 编码失败')
    return 'jpeg', buf.tobytes()


class Latest:
    """Single atomic, bounded latest-frame slot on the launcher's private tmpfs."""
    def __init__(self, path):
        self.path = path
        self.last = 0.
        self.last_error = 0.
        self.lock = threading.Lock()
        self.active = ''

    def receive(self, topic, msg):
        now = time.monotonic()
        with self.lock:
            if now - self.last < PERIOD:
                return
            try:
                fmt, data = jpeg(msg)
                if not 0 < len(data) <= MAX_BYTES:
                    raise ValueError('图像大小超出显示上限 8 MiB')
                stamp = msg.header.stamp
                ns = stamp.to_nsec() if hasattr(stamp, 'to_nsec') else stamp.sec*10**9 + stamp.nanosec
                self.write(topic, msg.header.frame_id, ns, fmt, data)
                self.last, self.active = now, topic
            except (ValueError, ImportError, RuntimeError) as exc:
                if now - self.last_error > 5:
                    print('D435i 彩色图像不可用:', exc, file=sys.stderr, flush=True)
                    self.last_error = now

    def write(self, source, frame, ns, fmt, data):
        meta = json.dumps([time.monotonic_ns(), source, frame, ns, fmt]).encode()
        temp = self.path.with_suffix('.new')
        with temp.open('wb') as out:
            out.write(meta + b'\n')
            out.write(data)
        temp.replace(self.path)


class Watch:
    """Prefer a compressed transport; try raw if an advertised transport is silent."""
    def __init__(self, latest, subscribe, cancel):
        self.latest, self.subscribe, self.cancel = latest, subscribe, cancel
        self.sub, self.name, self.started = None, '', 0.

    def update(self, names, now):
        choices = sorted(set(names), key=lambda n: (not n.endswith('/compressed'), n))
        old = self.name
        if old in choices:
            if now - max(self.started, self.latest.last) < 4 or len(choices) == 1:
                return
            name = choices[(choices.index(old) + 1) % len(choices)]
        else:
            name = choices[0] if choices else ''
        if name != old:
            if self.sub is not None:
                self.cancel(self.sub)
            self.sub = self.subscribe(name) if name else None
            self.name, self.started = name, now


def read_ros1(slot):
    import rospy
    from sensor_msgs.msg import CompressedImage, Image
    rospy.init_node('d435_color_input', anonymous=True)
    latest = Latest(slot)
    def subscribe(name):
        typ = CompressedImage if name.endswith('/compressed') else Image
        return rospy.Subscriber(name, typ, lambda m: latest.receive(name, m),
                                queue_size=1, buff_size=12*1024*1024)
    watch = Watch(latest, subscribe, lambda sub: sub.unregister())
    while not rospy.is_shutdown() and slot.parent.is_dir():
        watch.update([n for n, t in rospy.get_published_topics() if color_topic(n, t)], time.monotonic())
        time.sleep(.8)


def read_ros2(slot):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, Image
    rclpy.init()
    node = rclpy.create_node('go2_d435_color_input')
    latest = Latest(slot)
    def subscribe(name):
        typ = CompressedImage if name.endswith('/compressed') else Image
        return node.create_subscription(typ, name, lambda m: latest.receive(name, m), qos_profile_sensor_data)
    watch = Watch(latest, subscribe, node.destroy_subscription)
    next_scan = 0.
    while rclpy.ok() and slot.parent.is_dir() and Path('/run/go2_native/running').exists():
        now = time.monotonic()
        if now >= next_scan:
            watch.update([n for n, types in node.get_topic_names_and_types()
                          if any(color_topic(n, t) for t in types)], now)
            next_scan = now + 1.
        rclpy.spin_once(node, timeout_sec=.15)


def d435_rgb_device():
    # Read UVC capabilities on the attached D435i. Depth/IR and metadata
    # nodes live under the same USB product and must not supply the RGB panel.
    import errno
    import fcntl
    import struct
    candidates = []
    for node in sorted(Path('/sys/class/video4linux').glob('video[0-9]*')):
        usb = next((p for p in node.resolve().parents if (p/'idVendor').is_file()), None)
        if usb is None or (usb/'idVendor').read_text().strip().lower() != '8086':
            continue
        product = (usb/'product').read_text().lower() if (usb/'product').is_file() else ''
        if '435i' not in product or (node/'index').read_text().strip() != '0':
            continue
        path = Path('/dev')/node.name
        formats = set()
        with path.open('rb', buffering=0) as dev:
            for i in range(128):
                desc = bytearray(64)
                struct.pack_into('=II', desc, 0, i, 1)  # VIDEO_CAPTURE
                try:
                    fcntl.ioctl(dev.fileno(), 0xc0405602, desc)  # VIDIOC_ENUM_FMT
                except OSError as exc:
                    if exc.errno == errno.EINVAL:
                        break
                    raise
                formats.add(bytes(desc[44:48]))
        if formats & {b'YUYV', b'MJPG', b'RGB3', b'UYVY'} and not formats & {
                b'Z16 ', b'Y8  ', b'GREY', b'Y16 ', b'Y8I ', b'Y12I'}:
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError('识别到 %d 个可用 D435i UVC 彩色端点' % len(candidates))
    return candidates[0]


def read_uvc(slot):
    import cv2
    device = d435_rgb_device()
    stream = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
    try:
        stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        latest = Latest(slot)
        while slot.parent.is_dir():
            ok, pixels = stream.read()
            if not ok:
                raise RuntimeError('无法读取 D435i 彩色端点 ' + str(device))
            if time.monotonic() - latest.last < PERIOD:
                continue
            ok, buf = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                latest.write('D435i UVC '+str(device), 'camera_color_optical_frame', time.time_ns(), 'jpeg', buf.tobytes())
                latest.last = time.monotonic()
    finally:
        stream.release()


def read_usb(slot):
    try:
        import pyrealsense2 as rs
    except ImportError:
        return read_uvc(slot)
    import cv2
    import numpy as np
    devices = [dev for dev in rs.context().query_devices()
               if 'd435i' in dev.get_info(rs.camera_info.name).lower()]
    if len(devices) != 1:
        raise RuntimeError('USB D435i 数量为 %d' % len(devices))
    pipe, config = rs.pipeline(), rs.config()
    config.enable_device(devices[0].get_info(rs.camera_info.serial_number))
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    pipe.start(config)
    latest = Latest(slot)
    try:
        while slot.parent.is_dir():
            frame = pipe.wait_for_frames(3000).get_color_frame()
            if not frame or time.monotonic() - latest.last < PERIOD:
                continue
            stamp = time.time_ns()  # USB fallback is display-only, timestamped on receipt.
            ok, buf = cv2.imencode('.jpg', np.asanyarray(frame.get_data()), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                latest.write('USB D435i', 'camera_color_optical_frame', stamp, 'jpeg', buf.tobytes())
                latest.last = time.monotonic()
    finally:
        pipe.stop()


def read_slot(path, last_ns):
    try:
        with path.open('rb') as f:
            ns, topic, frame, stamp, fmt = json.loads(f.readline(4096))
            if ns <= last_ns or not 0 <= time.monotonic_ns() - ns <= 1_000_000_000:
                return None
            data = f.read(MAX_BYTES + 1)
        if not 0 < len(data) <= MAX_BYTES:
            return None
        return ns, topic, frame, stamp, fmt, data
    except FileNotFoundError:
        return None


def run(directory):
    import rospy
    from sensor_msgs.msg import CompressedImage
    rospy.init_node('go2_d435_display')
    children = []
    def spawn(cmd, env=None):
        p = subprocess.Popen(cmd, env=env)
        children.append(p)
        return p

    def stop():
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
    rospy.on_shutdown(stop)
    for name, uri in [('local', os.environ['ROS_MASTER_URI']), ('original', 'http://127.0.0.1:11311')]:
        spawn([sys.executable, '-u', __file__, 'ros1', str(directory/name)],
              dict(os.environ, ROS_MASTER_URI=uri))
    usb = None
    driver_started = None
    next_usb = time.monotonic() + 8
    have_driver = subprocess.run(['rospack', 'find', 'realsense2_camera'], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL).returncode == 0
    last, source, last_frame_time = 0, None, time.monotonic()
    pub = None
    next_check = 0.
    external_sub = None
    registered = ()
    discovered = external_rx = float('-inf')
    external_frame = ''
    def foreign(msg):
        nonlocal external_rx, external_frame
        name = (msg._connection_header or {}).get('callerid')
        if name and name != rospy.get_name() and len(msg.data) and (
                msg.data[:3] == b'\xff\xd8\xff' or msg.data[:8] == b'\x89PNG\r\n\x1a\n'):
            external_rx = time.monotonic()
            external_frame = msg.header.frame_id
    current = None
    try:
        while not rospy.is_shutdown():
            now = time.monotonic()
            if now >= next_check:
                publishers = dict(rospy.get_master().getSystemState()[2][0]).get(OUTPUT, [])
                others = tuple(sorted(n for n in publishers if n != rospy.get_name()))
                if others != registered:
                    registered, discovered = others, now
                    external_rx = float('-inf')
                if others and external_sub is None:
                    external_sub = rospy.Subscriber(OUTPUT, CompressedImage, foreign,
                                                    queue_size=1, buff_size=MAX_BYTES*2)
                if not others and external_sub is not None:
                    external_sub.unregister()
                    external_sub = None
                next_check = now + 1.
            collision = bool(registered) and (now-discovered < 3 or now-external_rx < 5)
            if collision:
                if now-external_rx < 5:
                    rospy.logwarn_throttle(10, '%s 已实收外部图像，发布者 %s、帧 %s；该面板由现有图像源占用',
                                           OUTPUT, registered, external_frame)
                if pub is not None:
                    pub.unregister()
                    pub = None
                    source = None
            elif registered:
                rospy.logwarn_throttle(10, '%s 有其他注册但近期无实际图像，继续供应 D435i', OUTPUT)
            sent = False
            for name in (current, 'local', 'dds', 'original', 'usb'):
                if not name:
                    continue
                item = read_slot(directory/name, last)
                if item is None:
                    # Keep one source during normal inter-frame intervals.
                    if name == current and now-last_frame_time < 1.5:
                        break
                    continue
                last, topic, frame, stamp, fmt, data = item
                current, last_frame_time, sent = name, now, True
                if not collision:
                    if pub is None:
                        pub = rospy.Publisher(OUTPUT, CompressedImage, queue_size=1)
                    msg = CompressedImage()
                    receipt = rospy.Time.now()
                    msg.header.stamp = (rospy.Time(*divmod(stamp, 10**9)) if stamp > 0 and
                                        abs(receipt.to_nsec()-stamp) < 2_000_000_000 else receipt)
                    msg.header.frame_id, msg.format, msg.data = frame, fmt, data
                    pub.publish(msg)
                    if source != (name, topic):
                        rospy.loginfo('D435i 彩色图像已发布 %s %s -> %s (%d bytes)', name, topic, OUTPUT, len(data))
                        source = name, topic
                break
            if not sent and not collision and now >= next_usb and now-last_frame_time >= 6:
                if (driver_started is not None and usb.poll() is None and
                        now - max(driver_started, last_frame_time) > 25):
                    rospy.logwarn('本次启动的 RealSense 节点尚无彩色图，改试直接读取 D435i USB')
                    usb.send_signal(signal.SIGINT)
                    have_driver = False
                    driver_started = None
                if usb is None or usb.poll() is not None:
                    if usb is not None and have_driver:
                        have_driver = False
                    cmd = (['roslaunch','-p','11321','realsense2_camera','rs_camera.launch','device_type:=d435i',
                            'camera:=go2_d435','enable_color:=true','enable_depth:=false','color_width:=640',
                            'color_height:=480','color_fps:=15'] if have_driver else
                           [sys.executable, '-u', __file__, 'usb', str(directory/'usb')])
                    rospy.loginfo('等待 D435i 彩色话题，尝试本机 %s', 'RealSense ROS 驱动' if have_driver else 'USB SDK')
                    children[:] = [p for p in children if p.poll() is None]
                    usb = spawn(cmd)
                    driver_started = now if have_driver else None
                rospy.logwarn_throttle(10, '等待 D435i 实际 RGB 图像；已检查 ROS1、Humble DDS 及 USB')
                next_usb = now + 12
            time.sleep(.04)
    finally:
        stop()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run(Path(os.environ['GO2_CAMERA_DIR']))
    else:
        {'ros1': read_ros1, 'ros2': read_ros2, 'usb': read_usb}[sys.argv[1]](Path(sys.argv[2]))
