"""Transport static physical TF edges for an exact body-to-MID-360 lookup."""
import json
import math

PREFIX = 'GO2MOUNT1 '
TOPICS = {'dds': '/go2_native/mount_dds', 'ros1_11311': '/go2_native/mount_ros1_11311'}


def physical(frame):
    leaf = frame.lower().split('/')[-1]
    return bool(leaf and leaf not in ('map', 'odom', 'odom_mapping', 'world', 'go2_ground', 'go2_ground_focus')
                and not frame.lower().startswith(('go2_native/', 'go2_visual/')))


def valid_row(parent, child, pose):
    return (physical(parent) and physical(child) and child != parent and len(pose) == 7 and
            all(math.isfinite(v) for v in pose) and abs(sum(v*v for v in pose[3:]) - 1.) < .02)


def wire_statics(msg, source, edges):
    """Pass a per-source dictionary across callbacks. Emit a latched snapshot on change."""
    if source not in TOPICS:
        raise ValueError('未知静态 TF 来源 ' + str(source))
    changed = False
    for ts in msg.transforms:
        parent, child = ts.header.frame_id.lstrip('/'), ts.child_frame_id.lstrip('/')
        p, q = ts.transform.translation, ts.transform.rotation
        pose = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        if not valid_row(parent, child, pose):
            continue
        norm = math.sqrt(sum(v*v for v in pose[3:]))
        pose[3:] = [v/norm for v in pose[3:]]
        row = [parent, child, pose]
        if child not in edges:
            edges[child] = row
            changed = True
        elif edges[child] is not None:
            old_parent, _, old = edges[child]
            if (old_parent == parent and max(abs(a-b) for a,b in zip(old[:3], pose[:3])) <= 1e-6
                    and abs(sum(a*b for a,b in zip(old[3:], pose[3:]))) >= 1. - 1e-12):
                continue
            # Ambiguous or changing installation. Omit the edge, including
            # from future snapshots, rather than selecting one TF publisher.
            edges[child] = None
            changed = True
    if changed:
        rows = [r for r in edges.values() if r is not None]
        return PREFIX + json.dumps([source, rows], separators=(',', ':'))
    return None


def install_dds_reader(node):
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from tf2_msgs.msg import TFMessage
    edges = {}
    def receive(msg):
        wire = wire_statics(msg, 'dds', edges)
        if wire is not None:
            print(wire, flush=True)
    qos = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
    return node.create_subscription(TFMessage, '/tf_static', receive, qos)


def relay(line, publishers):
    """Private topics keep each source's static graph independent of ROS1 /tf."""
    import rospy
    from geometry_msgs.msg import TransformStamped
    from tf2_msgs.msg import TFMessage
    source, rows = json.loads(line[len(PREFIX):])
    target = TOPICS[source]
    if target not in publishers:
        publishers[target] = rospy.Publisher(target, TFMessage, queue_size=1, latch=True)
    msg = TFMessage()
    for parent, child, pose in rows:
        if not valid_row(parent, child, pose):
            continue
        ts = TransformStamped()
        ts.header.frame_id, ts.child_frame_id = parent, child
        p, q = ts.transform.translation, ts.transform.rotation
        p.x, p.y, p.z, q.x, q.y, q.z, q.w = pose
        msg.transforms.append(ts)
    publishers[target].publish(msg)
