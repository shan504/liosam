"""Read actual Go2 status messages on their native transport for the Noetic display."""
import json
import math
from threading import Lock
import time

PREFIX = 'GO2STATUS1 '
INPUT = '/go2_visual/status_input'
VALID = '/localization/valid'
STATUS = '/localization/status'
OPERATOR = '/operator/status'
SPORT = '/lf/sportmodestate'
PRODUCERS = {VALID: 'go2_gicp', STATUS: 'go2_gicp', OPERATOR: 'go2_operator'}


def status_callback(source, topic, send, trusted=None):
    last_time = float('-inf')
    last_data = None

    def received(msg):
        nonlocal last_time, last_data
        if topic == VALID and (getattr(msg, '_connection_header', None) or {}).get('latching') == '1':
            return
        now = time.monotonic()
        if topic == SPORT:
            if now - last_time < .5:
                return
            data = dict(mode=int(msg.mode), gait_type=int(msg.gait_type), error_code=int(msg.error_code))
        else:
            data = msg.data
            # Retain transitions and the freshness heartbeat expected by the old localizer.
            interval = .15 if topic == VALID else 1.
            if data == last_data and now - last_time < interval:
                return
        if trusted is not None and not trusted(msg):
            return
        last_time, last_data = now, data
        report = dict(source=source, topic=topic, data=data, received=now)
        send(PREFIX + json.dumps(report, separators=(',', ':')))

    return received


def subscribe_dds(node, send):
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from std_msgs.msg import Bool, String

    def subscribe(typ, topic, callback, qos):
        try:
            node.create_subscription(typ, topic, callback, qos)
        except Exception as exc:
            node.get_logger().warning('无法订阅状态 %s: %s' % (topic, exc))

    def only_original(topic):
        retry_at = 0.

        def trusted(_msg):
            nonlocal retry_at
            now = time.monotonic()
            if now < retry_at:
                return False
            try:
                endpoints = node.get_publishers_info_by_topic(topic)
            except Exception:
                endpoints = []
            # Exclude feedback through a generic ROS bridge. Humble callbacks
            # cannot identify which writer sent a sample if there are several.
            if len(endpoints) == 1 and endpoints[0].node_name == PRODUCERS[topic]:
                return True
            retry_at = now + 1.
            return False
        return trusted

    for typ, topic in ((Bool, VALID), (String, STATUS), (String, OPERATOR)):
        callback = status_callback('dds', topic, send, only_original(topic))
        subscribe(typ, topic, callback, qos_profile_sensor_data)
        # The old operator publishes its latest report transient-local/reliable.
        if topic == OPERATOR:
            subscribe(typ, topic, callback, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    try:
        from unitree_go.msg import SportModeState
    except ImportError as exc:
        node.get_logger().warning('未安装 Go2 机身状态消息: %s' % exc)
    else:
        subscribe(SportModeState, SPORT, status_callback('dds', SPORT, send), qos_profile_sensor_data)


def subscribe_original(rospy, send):
    from std_msgs.msg import Bool, String
    subs = []
    for typ, topic in ((Bool, VALID), (String, STATUS), (String, OPERATOR)):
        def trusted(msg, expected=PRODUCERS[topic]):
            header = getattr(msg, '_connection_header', None) or {}
            return header.get('callerid', '').rsplit('/', 1)[-1] == expected
        callback = status_callback('ros1_11311', topic, send, trusted)
        subs.append(rospy.Subscriber(topic, typ, callback, queue_size=4))
    return subs


def host_relay(rospy):
    from std_msgs.msg import String

    from rospy.topics import SubscribeListener

    class StatusCache(SubscribeListener):
        def __init__(self):
            self.latest = {}
            self.lock = Lock()

        def peer_subscribe(self, topic_name, topic_publish, peer_publish):
            # Replay each source/topic, retaining the original observation time.
            with self.lock:
                messages = [entry[1] for entry in self.latest.values()]
            for msg in messages:
                peer_publish(msg)

    cache = StatusCache()
    publisher = rospy.Publisher(INPUT, String, queue_size=16, subscriber_listener=cache)

    def relay(payload):
        report = json.loads(payload)
        key = report['source'], report['topic']
        if key[0] not in ('dds', 'ros1_11311') or key[1] not in (VALID, STATUS, OPERATOR, SPORT):
            return
        timestamp, data = report['received'], report['data']
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError('invalid status observation time')
        if key[1] == VALID and type(data) is not bool:
            raise ValueError('invalid Bool status')
        if key[1] in (STATUS, OPERATOR) and not isinstance(data, str):
            raise ValueError('invalid String status')
        if key[1] == SPORT and (not isinstance(data, dict) or
                not all(type(data.get(k)) is int for k in ('mode', 'gait_type', 'error_code'))):
            raise ValueError('invalid SportModeState')
        msg = String(data=payload)
        with cache.lock:
            previous = cache.latest.get(key)
            if previous and timestamp <= previous[0]:
                return
            cache.latest[key] = timestamp, msg
        publisher.publish(msg)
    return relay
