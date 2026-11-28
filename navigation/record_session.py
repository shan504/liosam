#!/usr/bin/env python3
"""Record the current navigation input and diagnostic evidence inside a sidecar."""
import json
from pathlib import Path
import signal
import subprocess
import sys
from bag_topics import RECORD_TOPICS, REQUIRED_TOPICS


def main():
    import yaml
    directory = Path('/recording')
    process = subprocess.Popen(['ros2', 'bag', 'record', '--storage', 'sqlite3',
        '--output', str(directory/'bag'), '--max-bag-size', '2147483648',
        '--qos-profile-overrides-path', '/repo/navigation/bag_qos.yaml', *RECORD_TOPICS])
    def stop(signum, frame):
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    result = process.wait()
    metadata = directory/'bag/metadata.yaml'
    if not metadata.is_file():
        return result or 1
    info = yaml.safe_load(metadata.read_text())['rosbag2_bagfile_information']
    counts = {item['topic_metadata']['name']: item['message_count']
              for item in info.get('topics_with_message_count', [])}
    report = {'message_counts': counts, 'required_inputs_complete':
              all(counts.get(topic, 0) > 0 for topic in REQUIRED_TOPICS)}
    (directory/'recording_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print('消息数检查只确认输入存在；不代表定位质量或整个录包无丢帧。', flush=True)
    if not report['required_inputs_complete']:
        print('录包缺少有效点云或里程计；保留现场文件，但本包不能作为完整导航回放。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
