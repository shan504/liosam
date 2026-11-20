#!/usr/bin/env bash
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
upstream="$root/upstream/LIO-SAM"
patch="$root/patches/foxy-lio-sam.patch"
pin="$(git -C "$root" rev-parse HEAD:upstream/LIO-SAM)"
if [ ! -e "$upstream/.git" ]; then
  git -C "$root" submodule update --init --depth 1 -- upstream/LIO-SAM
fi
if [ "$(git -C "$upstream" rev-parse HEAD)" != "$pin" ]; then
  echo "Upstream checkout differs from the pinned commit $pin" >&2
  exit 1
fi
if git -C "$upstream" apply --reverse --check "$patch" 2>/dev/null; then
  echo "Foxy patch already present"
else
  git -C "$upstream" apply --check "$patch"
  git -C "$upstream" apply "$patch"
  echo "Applied Foxy patch to $pin"
fi
