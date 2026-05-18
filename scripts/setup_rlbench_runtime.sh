#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA:-conda}"
COPPELIASIM_VERSION="CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
COPPELIASIM_ROOT="$ROOT/external/coppeliasim/$COPPELIASIM_VERSION"

if ! command -v "$CONDA" >/dev/null 2>&1; then
  echo "Could not find '$CONDA'. Set CONDA=/path/to/conda-or-micromamba." >&2
  exit 127
fi

mkdir -p "$ROOT/external"

if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  echo "[1/6] Installing system packages for CoppeliaSim headless rendering"
  apt-get update
  apt-get install -y \
    xvfb xauth x11-utils \
    libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 \
    libavcodec-dev libavformat-dev libswscale-dev
else
  echo "[1/6] Skipping apt install. Install xvfb/xauth and CoppeliaSim X11 deps manually if needed."
fi

echo "[2/6] Cloning PerAct RLBench fork"
if [ ! -d "$ROOT/external/peract_rlbench" ]; then
  git clone https://github.com/MohitShridhar/RLBench.git "$ROOT/external/peract_rlbench"
  git -C "$ROOT/external/peract_rlbench" checkout -b peract --track origin/peract
  git -C "$ROOT/external/peract_rlbench" checkout ad991951
fi

echo "[3/6] Cloning PyRep"
if [ ! -d "$ROOT/external/PyRep" ]; then
  git clone https://github.com/stepjam/PyRep.git "$ROOT/external/PyRep"
  git -C "$ROOT/external/PyRep" checkout 8f420be
fi

echo "[4/6] Downloading CoppeliaSim 4.1 if missing"
mkdir -p "$ROOT/external/coppeliasim"
if [ ! -d "$COPPELIASIM_ROOT" ]; then
  curl -L --fail \
    -o "$ROOT/external/coppeliasim/$COPPELIASIM_VERSION.tar.xz" \
    "https://www.coppeliarobotics.com/files/V4_1_0/$COPPELIASIM_VERSION.tar.xz"
  tar -xf "$ROOT/external/coppeliasim/$COPPELIASIM_VERSION.tar.xz" -C "$ROOT/external/coppeliasim"
fi

export COPPELIASIM_ROOT
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"

echo "[5/6] Installing PerAct RLBench fork into graspcorrect-rlbench"
"$CONDA" run -n graspcorrect-rlbench pip install -r "$ROOT/external/peract_rlbench/requirements.txt"
"$CONDA" run -n graspcorrect-rlbench pip install -e "$ROOT/external/peract_rlbench"

echo "[6/6] Installing PyRep into graspcorrect-rlbench"
"$CONDA" run -n graspcorrect-rlbench pip install -r "$ROOT/external/PyRep/requirements.txt"
"$CONDA" run -n graspcorrect-rlbench pip install -e "$ROOT/external/PyRep"

cat <<EOF

Runtime env vars:
  export COPPELIASIM_ROOT="$COPPELIASIM_ROOT"
  export LD_LIBRARY_PATH="\$COPPELIASIM_ROOT:\${LD_LIBRARY_PATH:-}"
  export QT_QPA_PLATFORM_PLUGIN_PATH="\$COPPELIASIM_ROOT"

Use xvfb-run for RLBench commands on headless machines:
  xvfb-run -a -s '-screen 0 1280x1024x24' <command>
EOF
