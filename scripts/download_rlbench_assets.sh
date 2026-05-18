#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA:-conda}"
TASKS=("$@")
if [ "${#TASKS[@]}" -eq 0 ]; then
  TASKS=("insert_onto_square_peg")
fi

mkdir -p \
  "$ROOT/external/3d_diffuser_actor/train_logs" \
  "$ROOT/external/3d_diffuser_actor/instructions" \
  "$ROOT/external/3d_diffuser_actor/data/peract/raw/test"

echo "[1/3] Downloading 3D Diffuser Actor PerAct checkpoint"
if [ ! -f "$ROOT/external/3d_diffuser_actor/train_logs/diffuser_actor_peract.pth" ]; then
  curl -L --fail \
    -o "$ROOT/external/3d_diffuser_actor/train_logs/diffuser_actor_peract.pth" \
    https://huggingface.co/katefgroup/3d_diffuser_actor/resolve/main/diffuser_actor_peract.pth
fi

echo "[2/3] Downloading instruction embeddings"
if [ ! -f "$ROOT/external/3d_diffuser_actor/instructions/peract/instructions.pkl" ]; then
  curl -L --fail \
    -o "$ROOT/external/3d_diffuser_actor/instructions/instructions.zip" \
    https://huggingface.co/katefgroup/3d_diffuser_actor/resolve/main/instructions.zip
  unzip -q -o "$ROOT/external/3d_diffuser_actor/instructions/instructions.zip" \
    -d "$ROOT/external/3d_diffuser_actor/instructions"
  mkdir -p "$ROOT/external/3d_diffuser_actor/instructions/peract"
  cp "$ROOT/external/3d_diffuser_actor/instructions/instructions/peract/instructions.pkl" \
    "$ROOT/external/3d_diffuser_actor/instructions/peract/instructions.pkl"
fi

echo "[3/3] Downloading PerAct test demos for: ${TASKS[*]}"
for task in "${TASKS[@]}"; do
  zip_path="$ROOT/external/3d_diffuser_actor/data/peract/raw/test/$task.zip"
  task_dir="$ROOT/external/3d_diffuser_actor/data/peract/raw/test/$task"
  if [ ! -d "$task_dir" ]; then
    curl -L --fail \
      -o "$zip_path" \
      "https://huggingface.co/datasets/hqfang/rlbench-18-tasks/resolve/main/data/test/$task.zip"
    unzip -q -o "$zip_path" -d "$ROOT/external/3d_diffuser_actor/data/peract/raw/test"
  fi
  rm -f "$zip_path"
done

if ! command -v "$CONDA" >/dev/null 2>&1; then
  echo "Could not find '$CONDA' to run rearrange script. Set CONDA=/path/to/conda-or-micromamba." >&2
  exit 127
fi

"$CONDA" run -n graspcorrect-rlbench python \
  "$ROOT/external/3d_diffuser_actor/data_preprocessing/rearrange_rlbench_demos.py" \
  --root_dir "$ROOT/external/3d_diffuser_actor/data/peract/raw/test"

echo "Assets ready under external/3d_diffuser_actor/."
