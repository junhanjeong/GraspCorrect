# GraspCorrect Reproduction

이 저장소는 `paper/GraspCorrect.md`와 `paper/GraspCorrect.pdf`를 기준으로 GraspCorrect 파이프라인을 재현하기 위한 Python 코드베이스입니다.

구현 범위는 논문의 3단계를 그대로 나눕니다.

1. **VLM-guided grasp detection**: grasp-guided prompt, object-aware contour sampling, iterative VQA.
2. **Visual goal generation**: object mask와 접촉점을 이용한 goal-state image composition.
3. **Action generation**: ResNet-34 + MLP 조건부 DDPM 형태의 GCBC 보정 정책.

RLBench와 CALVIN은 환경 의존성이 크기 때문에 이 저장소에는 공통 GraspCorrect 모듈, 학습/추론 CLI, 벤치마크 어댑터, 외부 베이스라인 설치 스크립트를 포함합니다. 논문처럼 모든 baseline을 내장 복사하지 않고, 공식 구현을 `external/` 아래에 설치한 뒤 adapter로 연결합니다. 우선 baseline은 RLBench와 CALVIN 모두를 지원하는 **3D Diffuser Actor**를 1순위로 지원하고, RVT-2/Act3D/PerAct는 동일한 adapter 인터페이스로 연결할 수 있게 두었습니다.

## 설치

가벼운 개발/테스트 환경:

```bash
cd /workspace/GraspCorrect
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[dev]"
```

GCBC 학습까지 필요한 환경:

```bash
python3 -m pip install -e ".[train,vlm,vision,dev]"
```

OpenAI VLM을 사용하려면:

```bash
export OPENAI_API_KEY=...
```

기본 모델은 논문 설정에 맞춰 `gpt-4o`입니다. 모델명은 CLI의 `--vlm-model` 또는 `configs/default.yaml`에서 바꿀 수 있습니다.

## 외부 벤치마크/베이스라인 설치

공식 저장소를 `external/`로 클론합니다.

```bash
python3 scripts/setup_external_repos.py --all
```

주요 외부 코드:

- RLBench: https://github.com/stepjam/RLBench
- CALVIN: https://github.com/mees/calvin
- 3D Diffuser Actor: https://github.com/nickgkan/3d_diffuser_actor
- RVT/RVT-2: https://github.com/NVlabs/RVT
- Act3D: https://github.com/zhouxian/act3d-chained-diffuser
- LangSAM: https://github.com/luca-medeiros/lang-segment-anything
- LaMa: https://github.com/advimman/lama

RLBench는 CoppeliaSim/PyRep, CALVIN은 자체 conda 환경과 dataset 다운로드가 필요합니다. 각 공식 README의 시스템 설치 절차를 먼저 따른 뒤 이 패키지를 editable로 설치하세요.

## 단일 이미지 smoke run

VLM 없이 mask 기반 heuristic grasp detection과 goal image 합성을 테스트할 수 있습니다.

```bash
python3 scripts/run_graspcorrect_image.py \
  --image paper/figure3.png \
  --task "insert the peg into the target hole" \
  --output outputs/figure3_goal.png
```

mask를 따로 제공하지 않으면 단순 foreground heuristic을 사용합니다. 실제 실험에서는 LangSAM 또는 벤치마크가 제공하는 target mask를 넘기는 쪽이 안정적입니다.

## GCBC 학습 데이터 포맷

학습 manifest는 JSONL입니다. 각 줄은 `.npz` 샘플 하나를 가리킵니다.

```json
{"path": "data/grasp_pairs/sample_000001.npz", "task": "insert peg"}
```

각 `.npz`는 다음 키를 포함합니다.

- `current_rgb`: correction-needed observation image, `H x W x 3`, uint8
- `goal_rgb`: generated visual goal image, `H x W x 3`, uint8
- `current_action`: 현재 policy grasp action, shape `(8,)`
- `target_action`: stable expert grasp action, shape `(8,)`

학습:

```bash
python3 scripts/train_gcbc.py \
  --manifest data/grasp_pairs/train.jsonl \
  --output runs/gcbc_rlbench \
  --config configs/default.yaml
```

## RLBench/CALVIN 실행 방식

환경과 baseline checkpoint 준비 후:

```bash
python3 scripts/eval_rlbench.py \
  --config configs/default.yaml \
  --baseline 3d_diffuser_actor \
  --baseline-root external/3d_diffuser_actor \
  --gcbc-checkpoint runs/gcbc_rlbench/policy.pt
```

```bash
python3 scripts/eval_calvin.py \
  --config configs/default.yaml \
  --baseline 3d_diffuser_actor \
  --baseline-root external/3d_diffuser_actor \
  --gcbc-checkpoint runs/gcbc_calvin/policy.pt
```

이 두 CLI는 벤치마크 패키지가 설치된 환경에서 동작합니다. 현재 저장소 단독 환경에서는 import guard가 걸려 있어 설치 안내 에러를 냅니다.

## 현재 재현상의 주의점

논문은 waypoint randomization을 통해 correction-needed grasp state 200개/task를 생성합니다. 공개 논문에는 benchmark별 waypoint 조작 코드가 포함되어 있지 않으므로, 이 저장소는 두 경로를 모두 제공합니다.

- `scripts/collect_grasp_pairs_from_manifest.py`: 저장된 demonstration/action sequence에서 gripper close transition을 찾아 synthetic perturbation pair를 생성합니다.
- `graspcorrect.data.rlbench_protocol`: RLBench 환경 안에서 waypoint perturbation collector를 구현할 때 필요한 hook interface입니다.

정확한 수치 재현을 위해서는 RLBench/CALVIN 공식 환경에서 task별 waypoint perturbation을 채워 넣어야 합니다. 그 외 GraspCorrect의 detection, goal generation, GCBC 학습/추론, benchmark wrapping 구조는 바로 확장 가능한 형태로 구현되어 있습니다.
