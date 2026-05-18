# Reproduction Notes

## Paper Mapping

`paper/GraspCorrect.md` describes three modules:

- Section 3.1 -> `graspcorrect.detection`
- Section 3.2 -> `graspcorrect.goal_generation`
- Section 3.3 -> `graspcorrect.policies.gcbc`

The default configuration follows Appendix A:

- pre-grasp window `W = 10`
- iterative VQA iterations `T = 4`
- `top_n = 3`
- GCBC position loss weight `lambda = 0.2`
- Adam learning rate `5e-4`
- batch size `256`

## Grasp Detection

`GraspDetector` first obtains a target mask, samples numbered candidates along the object contour, asks the VLM to select promising candidates, then resamples around selected contour indices with a 1D Gaussian. Left and right gripper contacts are selected sequentially, matching Appendix A.1.

When `OpenAIResponsesVLMClient` is not installed or configured, the detector can run with `HeuristicVLMClient` for smoke tests. Real reproduction should use the OpenAI client plus LangSAM masks.

## Goal Generation

`GoalComposer` creates the visual goal image by preserving the object mask from the pre-grasp frame, placing it on the current frame/background, and rendering parallel-jaw gripper contacts at the selected points. The class accepts an externally inpainted background so LaMa can be plugged in without changing the pipeline.

## Action Generation

`GCBCDiffusionPolicy` implements a conditional DDPM over the 7D pose part of the action. It conditions on:

- current image
- goal image
- current 8D action
- diffusion timestep
- noisy target pose

The output is a corrected 7D pose; the gripper state is copied from the baseline grasp action.

## Benchmark Integration

The wrapper class is `GraspCorrectPolicyWrapper`. It observes a baseline policy action stream and activates on the first open-to-close gripper transition. At that moment it uses the observation from `W` frames earlier as the pre-grasp frame and replaces the baseline grasp action with the GCBC output.

For exact paper numbers, the remaining environment-specific work is:

1. Install RLBench/CALVIN and the selected official baseline checkpoints.
2. Use the official baseline inference class through `PythonClassPolicyAdapter` or a JSON `SubprocessPolicyAdapter`.
3. Generate correction pairs with task-specific waypoint perturbations inside the simulator, or start with `scripts/collect_grasp_pairs_from_manifest.py` for a demonstration-based approximation.

For CALVIN, the official README asks custom agents to implement `reset()` and `step(obs, goal)`. Use `graspcorrect.benchmarks.calvin.CALVINGraspCorrectModel` as that `CustomModel` wrapper.
