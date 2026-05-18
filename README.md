# GraspCorrect RLBench Reproduction

이 repo는 GraspCorrect 논문을 **RLBench + 3D Diffuser Actor baseline**에 한정해 재현하는 코드입니다. 현재 구현된 실행 경로는 다음 세 가지입니다.

1. GPT-5.4 mini + LangSAM으로 object contour 위 grasp contact point 반복 선택
2. current/pre-grasp image와 object/gripper mask로 goal image 합성
3. GCBC diffusion policy로 baseline grasp action correction

CALVIN과 여러 baseline adapter는 현재 실행 경로에서 제외했습니다. 아래 명령은 repo root, 즉 `/workspace/GraspCorrect`에서 실행한다고 가정합니다.

## 전체 흐름

1. 외부 repo clone
2. conda/micromamba 환경 생성
3. CoppeliaSim, PyRep, RLBench runtime 설치
4. 3D Diffuser Actor checkpoint, instruction embedding, RLBench test demos 다운로드
5. GCBC dataset 생성
6. GCBC policy 학습
7. RLBench baseline 평가
8. RLBench + GraspCorrect 평가

## 환경 구조

환경은 두 개를 씁니다.

- `graspcorrect-rlbench`: Python 3.8, torch 1.13.1, RLBench, PyRep, 3D Diffuser Actor, GCBC 학습/평가
- `graspcorrect-langsam`: Python 3.10, torch 2.3.1, LangSAM sidecar

RLBench 명령은 항상 `graspcorrect-rlbench`에서 실행합니다. LangSAM은 RLBench 환경 안에 섞지 않고 `--langsam-python "$GRASPCORRECT_LANGSAM_PYTHON"`으로 subprocess 호출합니다. GraspCorrect evaluation 중 GPT-5.4 mini를 호출하므로 `.env`에 `OPENAI_API_KEY=...`가 있어야 합니다.

## 1. Conda 준비

이미 conda나 mamba가 있다면 이 정도만 잡으면 됩니다.

```bash
export CONDA="${CONDA:-conda}"
```

conda/mamba가 없다면 repo 내부 micromamba를 사용할 수 있습니다.

```bash
mkdir -p .local/bin .local/mamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C .local/bin --strip-components=1 bin/micromamba

export MAMBA_ROOT_PREFIX="$PWD/.local/mamba"
export CONDA="$PWD/.local/bin/micromamba"
```

이 repo에서 내가 사용한 방식은 로컬 micromamba입니다. 새 shell을 열 때마다 `CONDA`와 `MAMBA_ROOT_PREFIX`를 다시 export하면 됩니다.

## 2. 외부 코드와 환경 설치

```bash
bash scripts/clone_external_repos.sh
bash scripts/setup_envs.sh
bash scripts/setup_rlbench_runtime.sh
```

각 script가 하는 일은 다음과 같습니다.

- `clone_external_repos.sh`: 3D Diffuser Actor, LangSAM, LaMa 등 외부 repo clone
- `setup_envs.sh`: `graspcorrect-rlbench`, `graspcorrect-langsam` conda env 생성 및 Python package 설치
- `setup_rlbench_runtime.sh`: PerAct RLBench fork, PyRep, CoppeliaSim 4.1 설치

headless 서버에서는 RLBench/CoppeliaSim 명령을 `xvfb-run`으로 감싸야 합니다.

## 3. 매 shell마다 필요한 환경 변수

RLBench, GCBC 수집, 평가 명령을 실행하기 전에 아래를 잡아주세요.

```bash
export MAMBA_ROOT_PREFIX="$PWD/.local/mamba"
export CONDA="$PWD/.local/bin/micromamba"

export COPPELIASIM_ROOT="$PWD/external/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"

export GRASPCORRECT_LANGSAM_PYTHON="$($CONDA run -n graspcorrect-langsam python - <<'PY'
import sys
print(sys.executable)
PY
)"
```

conda를 쓰는 경우 첫 두 줄만 환경에 맞게 바꾸면 됩니다.

```bash
export CONDA=conda
```

## 4. Benchmark demo와 가중치 다운로드

`scripts/download_rlbench_assets.sh`는 세 가지를 받습니다.

- 3D Diffuser Actor PerAct checkpoint: `external/3d_diffuser_actor/train_logs/diffuser_actor_peract.pth`
- instruction embedding: `external/3d_diffuser_actor/instructions/peract/instructions.pkl`
- RLBench PerAct test demos: `external/3d_diffuser_actor/data/peract/raw/test/<task>/...`

한 task만 받을 때:

```bash
bash scripts/download_rlbench_assets.sh insert_onto_square_peg
```

논문 RLBench 18 tasks 전체를 받을 때:

```bash
TASKS=(
  close_jar
  insert_onto_square_peg
  light_bulb_in
  meat_off_grill
  open_drawer
  place_shape_in_shape_sorter
  place_wine_at_rack_location
  push_buttons
  put_groceries_in_cupboard
  put_item_in_drawer
  put_money_in_safe
  reach_and_drag
  slide_block_to_color_target
  stack_blocks
  stack_cups
  sweep_to_dustpan_of_size
  turn_tap
  place_cups
)

bash scripts/download_rlbench_assets.sh "${TASKS[@]}"
```

다운로드 후 script가 3D Diffuser Actor용 demo rearrange까지 수행합니다. 전체 test demos는 디스크를 많이 씁니다. 이 workspace에서는 `external/3d_diffuser_actor/data/peract/raw/test`가 약 22GB였습니다.

## 5. 설치 확인

Python import와 unit test:

```bash
$CONDA run -n graspcorrect-rlbench python -m compileall graspcorrect scripts -q
$CONDA run -n graspcorrect-rlbench python -m pytest -q
```

CUDA 확인:

```bash
$CONDA run -n graspcorrect-rlbench python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

LangSAM sidecar 확인:

```bash
$CONDA run -n graspcorrect-langsam python scripts/langsam_segment.py \
  --image paper/image.png \
  --prompt "red object" \
  --output runs/langsam_smoke/red_object_mask.npy \
  --metadata-output runs/langsam_smoke/red_object_meta.json
```

## 6. GCBC dataset 생성

GCBC 데이터 생성은 GPT-5.4 mini나 LangSAM을 쓰지 않습니다. 저장된 RLBench GT demo에서 gripper close transition을 찾고, grasp 직전까지 simulator에서 replay한 뒤 perturbed grasp action을 실행해서 `(current_rgb, goal_rgb, current_action, target_action)` pair를 저장합니다.

한 task smoke:

```bash
xvfb-run -a -s '-screen 0 1280x1024x24' \
  $CONDA run -n graspcorrect-rlbench python scripts/collect_rlbench_gcbc_dataset.py \
  --source stored \
  --tasks insert_onto_square_peg \
  --data-dir external/3d_diffuser_actor/data/peract/raw/test \
  --variations -1 \
  --samples-per-task 1 \
  --output-dir runs/gcbc_smoke \
  --headless 1 \
  --replay-max-tries 2
```

논문 설정에 맞춘 18 tasks x 200개, 총 3600개 생성:

```bash
TASKS=(
  close_jar
  insert_onto_square_peg
  light_bulb_in
  meat_off_grill
  open_drawer
  place_shape_in_shape_sorter
  place_wine_at_rack_location
  push_buttons
  put_groceries_in_cupboard
  put_item_in_drawer
  put_money_in_safe
  reach_and_drag
  slide_block_to_color_target
  stack_blocks
  stack_cups
  sweep_to_dustpan_of_size
  turn_tap
  place_cups
)

xvfb-run -a -s '-screen 0 1280x1024x24' \
  $CONDA run -n graspcorrect-rlbench python scripts/collect_rlbench_gcbc_dataset.py \
  --source stored \
  --tasks "${TASKS[@]}" \
  --data-dir external/3d_diffuser_actor/data/peract/raw/test \
  --variations -1 \
  --samples-per-task 200 \
  --output-dir data/gcbc_rlbench18_stored_200 \
  --headless 1 \
  --seed 7 \
  --replay-max-tries 2 \
  2>&1 | tee runs/logs/collect_gcbc_rlbench18_stored_200.log
```

결과 확인:

```bash
wc -l data/gcbc_rlbench18_stored_200/manifest.jsonl
find data/gcbc_rlbench18_stored_200/samples -type f -name '*.npz' | wc -l
python - <<'PY'
import json
s = json.load(open("data/gcbc_rlbench18_stored_200/collection_summary.json"))
for task, item in s.items():
    print(f"{task}: {item['written']}/{item['requested']}")
PY
```

이 workspace에서는 `3600` samples가 생성됐고, 각 task가 `200/200`으로 채워졌습니다.

## 7. GCBC policy 학습

Smoke 학습:

```bash
$CONDA run -n graspcorrect-rlbench python scripts/train_gcbc.py \
  --manifest runs/gcbc_smoke/manifest.jsonl \
  --output-dir runs/gcbc_smoke_policy \
  --device cuda \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --image-size 64 \
  --diffusion-steps 4
```

3600개 dataset으로 논문 설정에 가까운 학습:

```bash
$CONDA run -n graspcorrect-rlbench python scripts/train_gcbc.py \
  --manifest data/gcbc_rlbench18_stored_200/manifest.jsonl \
  --output-dir runs/gcbc_rlbench18_stored_200 \
  --device cuda \
  --epochs 50 \
  --batch-size 256 \
  --num-workers 4 \
  --image-size 224 \
  --diffusion-steps 100
```

결과 checkpoint:

```bash
ls -lh runs/gcbc_rlbench18_stored_200/policy.pt
ls -lh runs/gcbc_rlbench18_stored_200/latest.pt
```

이 workspace에서는 `runs/gcbc_rlbench18_stored_200/policy.pt`가 생성됐고, 마지막 training loss는 `0.419073`이었습니다.

Checkpoint load/inference smoke:

```bash
$CONDA run -n graspcorrect-rlbench python - <<'PY'
from graspcorrect.policies import GCBCDiffusionPolicy
from graspcorrect.data.gcbc_dataset import read_manifest, load_gcbc_sample
from graspcorrect.types import Action

policy = GCBCDiffusionPolicy.from_checkpoint(
    "runs/gcbc_rlbench18_stored_200/policy.pt",
    map_location="cpu",
)
entries = read_manifest("data/gcbc_rlbench18_stored_200/manifest.jsonl")
sample = load_gcbc_sample(entries[0])
out = policy.predict(
    sample["current_rgb"],
    sample["goal_rgb"],
    Action.from_vector(sample["current_action"]),
)
print("entries:", len(entries))
print("pred_shape:", out.as_vector().shape)
print("pred:", out.as_vector())
PY
```

## 8. RLBench 평가

현재 평가 baseline은 3D Diffuser Actor입니다.

중요한 점:

- 평가는 RLBench/CoppeliaSim에서 실제 action을 실행하므로 매우 느립니다.
- GPU는 model inference와 GCBC inference에 쓰이지만, simulation, IK, path planning은 CPU/CoppeliaSim 병목입니다.
- 결과 JSON은 task가 끝날 때마다 저장됩니다.
- 현재 evaluator에서 `--variations -1`은 전체 variation을 뜻합니다.
- 현재 evaluator에서 `--variations 0 1 2 3 4`는 사실상 variation `0..4`를 평가합니다. sparse variation set은 지원하지 않습니다.

### 8.1 빠른 smoke 평가

Baseline smoke:

```bash
xvfb-run -a -s '-screen 0 1280x1024x24' \
  $CONDA run -n graspcorrect-rlbench python scripts/eval_rlbench_diffuser.py \
  --tasks insert_onto_square_peg \
  --num-episodes 1 \
  --variations 0 \
  --headless 1 \
  --device cuda \
  --seed 11 \
  --output-file runs/rlbench_baseline_smoke.json
```

GraspCorrect smoke:

```bash
xvfb-run -a -s '-screen 0 1280x1024x24' \
  $CONDA run -n graspcorrect-rlbench python scripts/eval_rlbench_diffuser.py \
  --tasks insert_onto_square_peg \
  --num-episodes 1 \
  --variations 0 \
  --headless 1 \
  --device cuda \
  --seed 11 \
  --enable-graspcorrect \
  --gcbc-checkpoint runs/gcbc_rlbench18_stored_200/policy.pt \
  --langsam-python "$GRASPCORRECT_LANGSAM_PYTHON" \
  --output-file runs/rlbench_graspcorrect_smoke.json
```

Smoke는 “코드 경로가 끝까지 도는지” 확인하는 용도입니다. 성능 수치로 해석하면 안 됩니다. 짧은 episode에서는 gripper open-to-close correction trigger가 안 걸릴 수도 있습니다.

### 8.2 단일 task 평가

Baseline:

```bash
xvfb-run -a -s '-screen 0 1280x1024x24' \
  $CONDA run -n graspcorrect-rlbench python scripts/eval_rlbench_diffuser.py \
  --tasks close_jar \
  --num-episodes 100 \
  --variations -1 \
  --headless 1 \
  --device cuda \
  --seed 11 \
  --output-file runs/eval/baseline/close_jar.json \
  2>&1 | tee runs/logs/eval_baseline_close_jar.log
```

GraspCorrect:

```bash
xvfb-run -a -s '-screen 0 1280x1024x24' \
  $CONDA run -n graspcorrect-rlbench python scripts/eval_rlbench_diffuser.py \
  --tasks close_jar \
  --num-episodes 100 \
  --variations -1 \
  --headless 1 \
  --device cuda \
  --seed 11 \
  --enable-graspcorrect \
  --gcbc-checkpoint runs/gcbc_rlbench18_stored_200/policy.pt \
  --langsam-python "$GRASPCORRECT_LANGSAM_PYTHON" \
  --output-file runs/eval/graspcorrect/close_jar.json \
  2>&1 | tee runs/logs/eval_graspcorrect_close_jar.log
```

### 8.3 18 tasks 전체 평가, task-by-task 권장

한 번에 18 tasks를 넣어도 되지만, 오래 걸리고 중간 중단 시 관리가 어렵습니다. 아래처럼 task마다 JSON을 따로 저장하는 방식을 권장합니다.

Baseline:

```bash
mkdir -p runs/eval/baseline runs/logs

TASKS=(
  close_jar
  insert_onto_square_peg
  light_bulb_in
  meat_off_grill
  open_drawer
  place_shape_in_shape_sorter
  place_wine_at_rack_location
  push_buttons
  put_groceries_in_cupboard
  put_item_in_drawer
  put_money_in_safe
  reach_and_drag
  slide_block_to_color_target
  stack_blocks
  stack_cups
  sweep_to_dustpan_of_size
  turn_tap
  place_cups
)

for task in "${TASKS[@]}"; do
  xvfb-run -a -s '-screen 0 1280x1024x24' \
    $CONDA run -n graspcorrect-rlbench python scripts/eval_rlbench_diffuser.py \
    --tasks "$task" \
    --num-episodes 100 \
    --variations -1 \
    --headless 1 \
    --device cuda \
    --seed 11 \
    --output-file "runs/eval/baseline/${task}.json" \
    2>&1 | tee "runs/logs/eval_baseline_${task}.log"
done
```

GraspCorrect:

```bash
mkdir -p runs/eval/graspcorrect runs/logs

TASKS=(
  close_jar
  insert_onto_square_peg
  light_bulb_in
  meat_off_grill
  open_drawer
  place_shape_in_shape_sorter
  place_wine_at_rack_location
  push_buttons
  put_groceries_in_cupboard
  put_item_in_drawer
  put_money_in_safe
  reach_and_drag
  slide_block_to_color_target
  stack_blocks
  stack_cups
  sweep_to_dustpan_of_size
  turn_tap
  place_cups
)

for task in "${TASKS[@]}"; do
  xvfb-run -a -s '-screen 0 1280x1024x24' \
    $CONDA run -n graspcorrect-rlbench python scripts/eval_rlbench_diffuser.py \
    --tasks "$task" \
    --num-episodes 100 \
    --variations -1 \
    --headless 1 \
    --device cuda \
    --seed 11 \
    --enable-graspcorrect \
    --gcbc-checkpoint runs/gcbc_rlbench18_stored_200/policy.pt \
    --langsam-python "$GRASPCORRECT_LANGSAM_PYTHON" \
    --output-file "runs/eval/graspcorrect/${task}.json" \
    2>&1 | tee "runs/logs/eval_graspcorrect_${task}.log"
done
```

### 8.4 빠른 비교용 budgeted 평가

디버깅이나 속도 확인용으로 각 task의 variation 0, episode 0만 돌릴 수 있습니다.

```bash
mkdir -p runs/eval/baseline_budget runs/eval/graspcorrect_budget runs/logs

TASKS=(
  close_jar
  insert_onto_square_peg
  light_bulb_in
  meat_off_grill
  open_drawer
  place_shape_in_shape_sorter
  place_wine_at_rack_location
  push_buttons
  put_groceries_in_cupboard
  put_item_in_drawer
  put_money_in_safe
  reach_and_drag
  slide_block_to_color_target
  stack_blocks
  stack_cups
  sweep_to_dustpan_of_size
  turn_tap
  place_cups
)

for task in "${TASKS[@]}"; do
  xvfb-run -a -s '-screen 0 1280x1024x24' \
    $CONDA run -n graspcorrect-rlbench python scripts/eval_rlbench_diffuser.py \
    --tasks "$task" \
    --num-episodes 1 \
    --variations 0 \
    --headless 1 \
    --device cuda \
    --seed 11 \
    --output-file "runs/eval/baseline_budget/${task}.json" \
    2>&1 | tee "runs/logs/eval_baseline_budget_${task}.log"

  xvfb-run -a -s '-screen 0 1280x1024x24' \
    $CONDA run -n graspcorrect-rlbench python scripts/eval_rlbench_diffuser.py \
    --tasks "$task" \
    --num-episodes 1 \
    --variations 0 \
    --headless 1 \
    --device cuda \
    --seed 11 \
    --enable-graspcorrect \
    --gcbc-checkpoint runs/gcbc_rlbench18_stored_200/policy.pt \
    --langsam-python "$GRASPCORRECT_LANGSAM_PYTHON" \
    --output-file "runs/eval/graspcorrect_budget/${task}.json" \
    2>&1 | tee "runs/logs/eval_graspcorrect_budget_${task}.log"
done
```

이 결과는 논문 성능표로 쓰면 안 됩니다. 단지 pipeline이 task별로 돌아가는지 빠르게 보는 용도입니다.

## 9. 결과 요약하기

task별 JSON을 모아 baseline과 GraspCorrect 평균을 비교할 때:

```bash
python - <<'PY'
import json
from pathlib import Path

tasks = [
    "close_jar",
    "insert_onto_square_peg",
    "light_bulb_in",
    "meat_off_grill",
    "open_drawer",
    "place_shape_in_shape_sorter",
    "place_wine_at_rack_location",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "reach_and_drag",
    "slide_block_to_color_target",
    "stack_blocks",
    "stack_cups",
    "sweep_to_dustpan_of_size",
    "turn_tap",
    "place_cups",
]

def read_one(root, task):
    path = Path(root) / f"{task}.json"
    if not path.exists():
        return None
    data = json.load(open(path))
    item = data[task]
    return item["mean"], item["success_count"], item["valid_episodes"], item.get("corrections", 0)

print("| task | baseline | graspcorrect | corrections |")
print("|---|---:|---:|---:|")
for task in tasks:
    b = read_one("runs/eval/baseline", task)
    g = read_one("runs/eval/graspcorrect", task)
    b_txt = "-" if b is None else f"{100*b[0]:.1f}% ({b[1]}/{b[2]})"
    g_txt = "-" if g is None else f"{100*g[0]:.1f}% ({g[1]}/{g[2]})"
    c_txt = "-" if g is None else str(g[3])
    print(f"| {task} | {b_txt} | {g_txt} | {c_txt} |")
PY
```

budgeted 평가 결과를 요약하려면 root만 바꿉니다.

```bash
# 위 script에서
# runs/eval/baseline -> runs/eval/baseline_budget
# runs/eval/graspcorrect -> runs/eval/graspcorrect_budget
```

## 10. 속도와 GPU 관련 메모

- `--device cuda`로 3D Diffuser Actor와 GCBC는 GPU를 사용합니다.
- 그래도 RLBench 평가의 큰 병목은 CoppeliaSim simulation, IK, path planning입니다. 이 부분은 GPU를 적극 써도 크게 빨라지지 않습니다.
- 18 tasks x 100 episodes x baseline/GraspCorrect는 오래 걸립니다. task-by-task로 끊어서 실행하는 편이 안전합니다.
- 여러 GPU나 큰 GPU가 있더라도 CoppeliaSim 프로세스를 너무 많이 병렬 실행하면 불안정할 수 있습니다. 먼저 2개 병렬부터 확인하세요.
- GraspCorrect-on 평가는 GPT-5.4 mini와 LangSAM 호출이 추가되므로 baseline보다 훨씬 느릴 수 있습니다.

## 11. 현재 workspace에서 생성된 주요 산출물

이 workspace에는 이미 아래가 생성되어 있습니다.

- GCBC dataset: `data/gcbc_rlbench18_stored_200/manifest.jsonl`
- GCBC samples: `data/gcbc_rlbench18_stored_200/samples/*.npz`
- GCBC summary: `data/gcbc_rlbench18_stored_200/collection_summary.json`
- GCBC checkpoint: `runs/gcbc_rlbench18_stored_200/policy.pt`
- latest checkpoint: `runs/gcbc_rlbench18_stored_200/latest.pt`

다시 생성하고 싶지 않다면 같은 `--output-dir`를 덮어쓰지 마세요.

## 12. 현재 한계

- 현재 평가 wrapper는 3D Diffuser Actor baseline에 맞춰져 있습니다. Act3D는 같은 external repo 안에 구현과 script가 있지만, GraspCorrect wrapper에는 아직 연결하지 않았습니다.
- 논문은 LaMa inpainting을 사용하지만, 현재 코드는 LaMa checkpoint가 없으면 OpenCV Telea inpaint fallback을 씁니다.
- GraspCorrect correction은 gripper open-to-close 전환과 충분한 observation history가 있어야 발동합니다.
- GCBC dataset 생성/학습에는 GPT-5.4 mini와 LangSAM을 사용하지 않습니다. 이 둘은 GraspCorrect evaluation 중 visual goal generation에서만 사용됩니다.
