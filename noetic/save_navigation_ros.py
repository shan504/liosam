"""Capture frame-labelled ROS 1 geometry from the current --go2 session."""
from collections import deque
import json
import math
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET


def fields(value):
    """Convert standard ROS geometry messages without rounding nanosecond stamps."""
    if hasattr(value, 'secs') and hasattr(value, 'nsecs'):
        return {'secs': value.secs, 'nsecs': value.nsecs}
    if isinstance(value, dict):
        return {key: fields(item) for key, item in value.items()}
    if hasattr(value, '__slots__'):
        return {key: fields(getattr(value, key)) for key in value.__slots__}
    if isinstance(value, (list, tuple)):
        return [fields(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def capture(destination):
    """Called after rospy.init_node. Returns manifest data; missing optional inputs are explicit."""
    import rospy
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry, Path as RosPath
    from sensor_msgs.msg import CameraInfo
    from tf2_msgs.msg import TFMessage
    from visualization_msgs.msg import MarkerArray

    out = Path(destination) / 'live'
    out.mkdir()
    def write(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                            separators=(',', ':')) + '\n', encoding='utf-8')

    mapping = '/lio_sam/mapping/odometry'
    native = '/go2_native/odometry'
    path = '/lio_sam/mapping/path'
    mount = '/go2_lio/body_from_lidar'
    loops = '/lio_sam/mapping/loop_closure_constraints'
    published = dict(rospy.get_published_topics())
    camera_topics = [topic for topic, typ in sorted(published.items()) if typ == 'sensor_msgs/CameraInfo']
    topics = {mapping: Odometry, native: Odometry, path: RosPath, mount: TransformStamped, loops: MarkerArray}
    topics.update({topic: CameraInfo for topic in camera_topics})
    latest, static_edges = {}, {}
    local_tf_sources = ['/tf_static', '/go2_native/mount_dds', '/go2_native/mount_ros1_11311']
    mapping_stamps = deque(maxlen=128)
    lock = threading.Lock()
    def sample(msg, topic):
        if topic == path and not msg.poses:
            return
        if topic in (mapping, native) and msg.header.stamp.to_nsec() == 0:
            return
        with lock:
            latest[topic] = msg
            if topic == mapping:
                mapping_stamps.append(msg.header.stamp)

    def statics(msg, topic):
        node = (getattr(msg, '_connection_header', {}) or {}).get('callerid', '')
        with lock:
            for tf in msg.transforms:
                static_edges[topic, node, tf.header.frame_id, tf.child_frame_id] = tf

    buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10), debug=False)
    listener = tf2_ros.TransformListener(buffer)
    subscribers = []
    started = rospy.Time.now().to_nsec()
    try:
        for topic, typ in topics.items():
            bufsize = 4 * 1024 * 1024 if topic in (path, loops) else 65536
            subscribers.append(rospy.Subscriber(topic, typ, sample, callback_args=topic, queue_size=1,
                                                buff_size=bufsize))
        for topic in local_tf_sources:
            subscribers.append(rospy.Subscriber(topic, TFMessage, statics, callback_args=topic, queue_size=32))
        # Collect concurrently. The newest mapping time can still be ahead of the
        # other TF publishers at unsubscribe time; use an interior mapping stamp.
        until = time.monotonic() + 3.5
        while time.monotonic() < until and not rospy.is_shutdown():
            time.sleep(.05)
    finally:
        for sub in subscribers:
            sub.unregister()
        listener.unregister()
    if rospy.is_shutdown():
        raise RuntimeError('ROS 在保存运行快照时已关闭')
    with lock:
        captured, static = dict(latest), dict(static_edges)
        stamps = sorted(mapping_stamps, key=lambda t: t.to_nsec())
    finished = rospy.Time.now().to_nsec()

    documents = {mapping: 'lio_odometry.json', native: 'go2_odometry.json',
                 path: 'keyframe_path.json', mount: 'body_from_lidar.json',
                 loops: 'loop_constraint_markers.json'}
    for topic, filename in documents.items():
        if topic in captured:
            write(filename, fields(captured[topic]))
    if camera_topics:
        write('camera_info.json', {name: fields(captured[name]) for name in camera_topics if name in captured})
    write('static_transforms.json', [
        {'topic': topic, 'publisher': node, 'transform': fields(value)}
        for (topic, node, _, _), value in sorted(static.items())])

    pairs = [('go2_ground', 'odom'), ('odom', 'go2_native/odom'), ('odom', 'lidar_link'),
             ('go2_ground', 'go2_visual/live_body')]
    transforms, missing = [], {}
    stamp = None
    if stamps:
        lag_ns = stamps[-1].to_nsec() - 350000000
        stamp = next((value for value in reversed(stamps) if value.to_nsec() <= lag_ns), stamps[0])
    for target, source in pairs:
        key = '%s <- %s' % (target, source)
        if stamp is None:
            missing[key] = '缺少 LIO 源时间戳'
            continue
        try:
            transform = buffer.lookup_transform(target, source, stamp)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            missing[key] = str(exc)
        else:
            transforms.append(fields(transform))
    write('reference_transforms.json', {
        'epoch_ns': stamp.to_nsec() if stamp is not None else None,
        'transforms': transforms, 'unavailable': missing})

    namespaces = ['/lio_sam', '/go2_ground_view', '/go2_native_lio_seed', '/go2_l1_style_dense',
                  '/go2_mapping_front_filter',
                  '/go2_d435/realsense2_camera', '/mid360s_attitude', '/use_sim_time',
                  '/rosdistro', '/run_id']
    absent = object()
    params = {}
    for key in namespaces:
        value = rospy.get_param(key, absent)
        if value is not absent:
            params[key] = fields(value)
    write('mapping_parameters.json', params)
    # Keep the exact URDF text, including references to external meshes. Only
    # identify explicitly robot-owned/global names on a shared master.
    urdf_param = None
    for key in ('/robot_description', '/go2/robot_description', '/go2_edu/robot_description'):
        urdf = rospy.get_param(key, '')
        if not isinstance(urdf, str) or not urdf.strip():
            continue
        try:
            robot = ET.fromstring(urdf)
        except ET.ParseError:
            continue
        if robot.tag != 'robot':
            continue
        (out / 'robot.urdf').write_text(urdf, encoding='utf-8')
        urdf_param = key
        break
    return {
        'robot_description_parameter': urdf_param,
        'loop_marker_count': len(captured[loops].markers) if loops in captured else None,
        'loop_marker_note': 'LIO-SAM visualization snapshot only; optimizer factors are not exported.',
        'snapshot_start_ros_ns': started, 'snapshot_end_ros_ns': finished,
        'topics_received': sorted(captured), 'topics_without_message': sorted(set(topics) - set(captured)),
        'camera_info_topics': camera_topics,
        'keyframe_count': len(captured[path].poses) if path in captured else 0,
        'static_edges': len(static), 'reference_epoch_ros_ns': stamp.to_nsec() if stamp is not None else None,
        'ground_tf_available': any(t['header']['frame_id'] == 'go2_ground' and t['child_frame_id'] == 'odom'
                                   for t in transforms),
        'reference_tf_unavailable': list(missing), 'saved_parameter_namespaces': sorted(params),
    }
