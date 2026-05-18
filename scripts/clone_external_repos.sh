#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/external"

clone_if_missing() {
  local url="$1"
  local dst="$2"
  local ref="$3"
  if [ ! -d "$dst/.git" ]; then
    git clone "$url" "$dst"
    git -C "$dst" checkout "$ref"
  fi
}

clone_if_missing https://github.com/nickgkan/3d_diffuser_actor.git "$ROOT/external/3d_diffuser_actor" 4faf00b
clone_if_missing https://github.com/luca-medeiros/lang-segment-anything.git "$ROOT/external/langsam" 918043e
clone_if_missing https://github.com/advimman/lama.git "$ROOT/external/lama" master

echo "External repos are ready under external/."
