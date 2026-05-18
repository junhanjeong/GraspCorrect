# External Baselines

The paper evaluates PerAct, Act3D, 3D Diffuser Actor, RVT-2, GR-MG, MoDE, and SuSIE. This reproduction keeps those implementations external for license and environment reasons.

Clone known repositories:

```bash
python3 scripts/setup_external_repos.py --all
```

Recommended first integration:

```bash
python3 scripts/setup_external_repos.py --repo rlbench --repo calvin --repo 3d_diffuser_actor
```

`3D Diffuser Actor` is the most useful first target because the paper evaluates it on both RLBench and CALVIN.

To connect a baseline that exposes a class:

```python
from graspcorrect.baselines.base import PythonClassPolicyAdapter
from graspcorrect.benchmarks.base import GraspCorrectPolicyWrapper
from graspcorrect.pipeline import GraspCorrectPipeline
from graspcorrect.policies.gcbc import GCBCDiffusionPolicy

baseline = PythonClassPolicyAdapter("your_repo.inference:Policy", kwargs={...})
gcbc = GCBCDiffusionPolicy.from_checkpoint("runs/gcbc_rlbench/policy.pt")
policy = GraspCorrectPolicyWrapper(baseline=baseline, pipeline=GraspCorrectPipeline(policy=gcbc))
```

If the baseline only has a script entry point, implement a small script that reads JSON from stdin and writes:

```json
{"action": [x, y, z, qx, qy, qz, qw, gripper]}
```

Then wrap it with `SubprocessPolicyAdapter`.
