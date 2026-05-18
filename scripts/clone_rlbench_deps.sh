#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/external"

if [ ! -d "$ROOT/external/peract_rlbench" ]; then
  git clone https://github.com/MohitShridhar/RLBench.git "$ROOT/external/peract_rlbench"
  git -C "$ROOT/external/peract_rlbench" checkout -b peract --track origin/peract
fi

echo "Install inside graspcorrect-rlbench:"
echo "  conda activate graspcorrect-rlbench"
echo "  pip install -r external/peract_rlbench/requirements.txt"
echo "  pip install -e external/peract_rlbench"
echo
echo "PyRep and CoppeliaSim are system-level requirements."
echo "Follow: external/3d_diffuser_actor/README.md lines 62-82."
