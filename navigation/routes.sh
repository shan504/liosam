#!/usr/bin/env bash
# Route files live on the host; ROS requests use the already running container.
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec /usr/bin/python3 "$here/routes.py" "$@"
