#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA:-conda}"

if ! command -v "$CONDA" >/dev/null 2>&1; then
  echo "Could not find '$CONDA'. Install conda/mamba/micromamba or set CONDA=/path/to/micromamba." >&2
  exit 127
fi

echo "[1/4] Creating RLBench/3D Diffuser Actor env"
"$CONDA" env create -f "$ROOT/envs/rlbench_diffuser.yml" || "$CONDA" env update -f "$ROOT/envs/rlbench_diffuser.yml"

echo "[2/4] Installing GraspCorrect and 3D Diffuser Actor into graspcorrect-rlbench"
"$CONDA" run -n graspcorrect-rlbench pip install -e "$ROOT[train,vlm,vision,rlbench]"
"$CONDA" run -n graspcorrect-rlbench pip install -e "$ROOT/external/3d_diffuser_actor" --no-deps

echo "[3/4] Creating LangSAM sidecar env"
"$CONDA" env create -f "$ROOT/envs/langsam.yml" || "$CONDA" env update -f "$ROOT/envs/langsam.yml"
"$CONDA" run -n graspcorrect-langsam pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple \
  torch==2.3.1+cu121 torchvision==0.18.1+cu121
"$CONDA" run -n graspcorrect-langsam pip install -e "$ROOT/external/langsam"
"$CONDA" run -n graspcorrect-langsam pip install "transformers==4.46.3" "tokenizers<0.21,>=0.20"

echo "[4/4] Done"
LANGSAM_PY="$("$CONDA" run -n graspcorrect-langsam python - <<'PY'
import sys
print(sys.executable)
PY
)"
echo "Use this for RLBench commands:"
echo "  conda activate graspcorrect-rlbench"
echo "Use this env var so RLBench can call LangSAM:"
echo "  export GRASPCORRECT_LANGSAM_PYTHON=\"$LANGSAM_PY\""
echo
echo "RLBench still needs CoppeliaSim/PyRep and the PerAct RLBench fork used by 3D Diffuser Actor."
echo "Follow external/3d_diffuser_actor/README.md section 'Install RLBench locally'."
