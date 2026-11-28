"""One acknowledged ROS request inside the running navigation container."""
import json
from pathlib import Path
import sys
import time
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent / 'original_go2/go2_3d'))
from route_store import validate_route


RESULT_PREFIX = 'GO2_ROUTE_RESULT '


def validate_request(request):
    if not isinstance(request, dict):
        raise ValueError('Expected route command object')
    identity = request.get('request_id')
    if not isinstance(identity, str) or str(UUID(identity)) != identity:
        raise ValueError('Invalid route request ID')
    command = request.get('command')
    if command not in ('snapshot', 'load', 'run', 'check', 'replace'):
        raise ValueError('Invalid route command')
    extra = {'route'} if command in ('load', 'run') else {'number'} if command == 'replace' else set()
    if set(request) != {'request_id', 'command'} | extra:
        raise ValueError('Unexpected route command fields')
    if command in ('load', 'run'):
        validate_route(request['route'])
    if command == 'replace' and (type(request['number']) is not int or request['number'] < 1):
        raise ValueError('Waypoint number must be a positive integer')
    return identity


def exchange(request, timeout=12.0):
    identity = validate_request(request)
    import rclpy
    from std_msgs.msg import String

    encoded = json.dumps(request, separators=(',', ':'), allow_nan=False)
    answers = []
    rclpy.init()
    node = None
    try:
        node = rclpy.create_node('go2_route_command_' + identity.replace('-', ''))

        def receive(message):
            try:
                answer = json.loads(message.data)
            except (ValueError, TypeError):
                return
            if isinstance(answer, dict) and answer.get('request_id') == identity:
                answers.append(answer)

        # Default depth QoS is reliable/volatile: commands are not latched or
        # replayed automatically after a navigation restart.
        subscription = node.create_subscription(String, '/navigation/route/result', receive, 10)
        publisher = node.create_publisher(String, '/navigation/route/command', 10)
        end = time.monotonic() + timeout
        next_send = 0.0
        while rclpy.ok() and time.monotonic() < end and not answers:
            now = time.monotonic()
            if publisher.get_subscription_count() > 0 and now >= next_send:
                # Same ID and payload on retry: the operator caches its result,
                # so losing a response cannot append/execute a route twice.
                publisher.publish(String(data=encoded))
                next_send = now + .75
            rclpy.spin_once(node, timeout_sec=.1)
        if not answers:
            raise TimeoutError('12 秒内未收到路线回执，结果未确认；请检查 /operator/status 和当前队列。')
        return answers[0]
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def main():
    try:
        raw = sys.stdin.read(131073)
        if len(raw.encode('utf-8')) > 131072:
            raise ValueError('Route request exceeds 128 KiB')
        answer = exchange(json.loads(raw))
        print(RESULT_PREFIX + json.dumps(answer, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
