from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from graspcorrect.pipeline import GraspCorrectPipeline
from graspcorrect.policies import GCBCDiffusionPolicy


ROOT = Path(__file__).resolve().parents[2]
DIFFUSER_ROOT = ROOT / "external" / "3d_diffuser_actor"
if str(DIFFUSER_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSER_ROOT))


@dataclass
class GraspCorrectRuntime:
    pipeline: GraspCorrectPipeline
    window: int = 10
    close_threshold: float = 0.5
    history: List[object] = field(default_factory=list)
    previous_gripper: Optional[float] = None
    already_corrected: bool = False
    object_prompt: Optional[str] = None

    def observe(self, obs: object) -> None:
        self.history.append(obs)
        if len(self.history) > self.window + 32:
            self.history = self.history[-(self.window + 32) :]

    def maybe_correct(self, current_obs: object, action: np.ndarray, task_desc: str) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        gripper = float(action[-1])
        previous = self.previous_gripper
        self.previous_gripper = gripper
        if self.already_corrected or previous is None:
            return action
        should = previous > self.close_threshold and gripper <= self.close_threshold
        if not should or len(self.history) <= self.window:
            return action
        pre_obs = self.history[-self.window - 1]
        result = self.pipeline.correct(
            current_obs=current_obs,
            pre_grasp_obs=pre_obs,
            baseline_action=action,
            task_desc=task_desc,
            object_prompt=self.object_prompt,
        )
        self.already_corrected = True
        corrected = result.corrected_action.as_vector()
        corrected[-1] = gripper
        return corrected.astype(np.float32)


class GraspCorrectRLBenchEnv:
    """3D Diffuser Actor RLBenchEnv with an extra correction camera enabled."""

    def __init__(self, correction_camera: str = "overhead", **kwargs) -> None:
        from utils.utils_with_rlbench import RLBenchEnv  # type: ignore

        class _Env(RLBenchEnv):
            def __init__(self, correction_camera_name: str, *args, **inner_kwargs):
                self._correction_camera_name = correction_camera_name
                super().__init__(*args, **inner_kwargs)

            def create_obs_config(self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **inner_kwargs):
                obs_config = super().create_obs_config(
                    image_size,
                    apply_rgb,
                    apply_depth,
                    apply_pc,
                    apply_cameras,
                    **inner_kwargs,
                )
                # The model still receives only apply_cameras. This merely makes
                # obs.overhead_rgb/front_rgb available to GraspCorrect.
                cam_cfg = getattr(obs_config, f"{self._correction_camera_name}_camera")
                cam_cfg.rgb = True
                cam_cfg.mask = False
                cam_cfg.image_size = image_size
                return obs_config

        self._env = _Env(correction_camera, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._env, name)


def run_diffuser_rlbench_eval(
    checkpoint: Path,
    tasks: Sequence[str],
    output_file: Path,
    data_dir: Path,
    instructions: Path,
    num_episodes: int = 100,
    variations: Sequence[int] = tuple(range(61)),
    device: str = "cuda",
    seed: int = 0,
    headless: bool = True,
    enable_graspcorrect: bool = False,
    gcbc_checkpoint: Optional[Path] = None,
    langsam_python: Optional[str] = None,
    correction_camera: str = "overhead",
    object_prompt: Optional[str] = None,
    use_langsam_subprocess: bool = True,
    increased_distance_px: float = 0.0,
) -> Dict[str, object]:
    from online_evaluation_rlbench.evaluate_policy import Arguments, load_models  # type: ignore
    from utils.common_utils import load_instructions, round_floats  # type: ignore
    from utils.utils_with_rlbench import Actioner, load_episodes  # type: ignore
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = Arguments().parse_args([])
    args.checkpoint = checkpoint
    args.device = device
    args.tasks = tuple(tasks)
    args.instructions = instructions
    args.data_dir = data_dir
    args.num_episodes = num_episodes
    args.variations = tuple(variations)
    args.headless = int(headless)
    args.gripper_loc_bounds_file = str(DIFFUSER_ROOT / "tasks" / "18_peract_tasks_location_bounds.json")
    args.cameras = ("left_shoulder", "right_shoulder", "wrist", "front")
    args.test_model = "3d_diffuser_actor"
    args.checkpoint = checkpoint
    args.quaternion_format = "wxyz"
    args.predict_trajectory = 1
    args.dense_interpolation = 1
    args.interpolation_length = 2
    args.num_history = 3
    args.embedding_dim = 120
    args.fps_subsampling_factor = 5
    args.lang_enhanced = 0
    args.relative_action = 0

    model = load_models(args)
    instruction = load_instructions(args.instructions)
    if instruction is None:
        raise RuntimeError(f"Could not load instructions from {args.instructions}")
    actioner = Actioner(
        policy=model,
        instructions=instruction,
        apply_cameras=args.cameras,
        action_dim=args.action_dim,
        predict_trajectory=bool(args.predict_trajectory),
    )
    env = GraspCorrectRLBenchEnv(
        correction_camera=correction_camera,
        data_path=args.data_dir,
        image_size=[int(x) for x in args.image_size.split(",")],
        apply_rgb=True,
        apply_pc=True,
        headless=bool(args.headless),
        apply_cameras=args.cameras,
        collision_checking=bool(args.collision_checking),
    )

    pipeline = None
    if enable_graspcorrect:
        if gcbc_checkpoint is None:
            raise ValueError("--gcbc-checkpoint is required with --enable-graspcorrect.")
        policy = GCBCDiffusionPolicy.from_checkpoint(gcbc_checkpoint, map_location=device)
        policy.to(torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu"))
        pipeline = GraspCorrectPipeline.build(
            policy=policy,
            use_langsam_subprocess=use_langsam_subprocess,
            langsam_python=langsam_python,
            correction_camera=correction_camera,
            increased_distance_px=increased_distance_px,
        )
        pipeline_object_prompt = object_prompt
    else:
        pipeline_object_prompt = None

    max_eps_dict = load_episodes()["max_episode_length"]
    results: Dict[str, object] = {}
    for task in tasks:
        task_result = evaluate_one_task(
            env=env,
            task_str=task,
            max_steps=max_eps_dict[task],
            num_variations=max(variations) + 1 if variations else -1,
            num_demos=num_episodes,
            actioner=actioner,
            runtime_pipeline=pipeline,
            object_prompt=pipeline_object_prompt,
            max_tries=2,
            dense_interpolation=True,
            interpolation_length=args.interpolation_length,
            num_history=args.num_history,
        )
        results[task] = task_result
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(round_floats(results), indent=2), encoding="utf-8")
    return results


def evaluate_one_task(
    env,
    task_str: str,
    max_steps: int,
    num_variations: int,
    num_demos: int,
    actioner,
    runtime_pipeline: Optional[GraspCorrectPipeline],
    object_prompt: Optional[str],
    max_tries: int,
    dense_interpolation: bool,
    interpolation_length: int,
    num_history: int,
) -> Dict[object, object]:
    from utils.utils_with_rlbench import task_file_to_task_class  # type: ignore

    env.env.launch()
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    task_variations = min(num_variations, task.variation_count()) if num_variations > 0 else task.variation_count()
    var_success_rates: Dict[object, object] = {}
    valid_counts: Dict[int, int] = {}
    try:
        for variation in range(task_variations):
            task.set_variation(variation)
            success_count, valid, valid_demos, correction_count = evaluate_variation(
                env=env,
                task_str=task_str,
                task=task,
                max_steps=max_steps,
                variation=variation,
                num_demos=num_demos // task_variations + 1,
                actioner=actioner,
                runtime_pipeline=runtime_pipeline,
                object_prompt=object_prompt,
                max_tries=max_tries,
                dense_interpolation=dense_interpolation,
                interpolation_length=interpolation_length,
                num_history=num_history,
            )
            if valid:
                var_success_rates[variation] = {
                    "success_rate": float(success_count) / max(1, valid_demos),
                    "success_count": int(success_count),
                    "valid_episodes": int(valid_demos),
                    "corrections": int(correction_count),
                }
                valid_counts[variation] = valid_demos
    finally:
        env.env.shutdown()
    denom = max(1, sum(valid_counts.values()))
    total_success = sum(int(v["success_count"]) for v in var_success_rates.values() if isinstance(v, dict))
    total_corrections = sum(int(v["corrections"]) for v in var_success_rates.values() if isinstance(v, dict))
    var_success_rates["mean"] = float(total_success / denom)
    var_success_rates["success_count"] = int(total_success)
    var_success_rates["valid_episodes"] = int(sum(valid_counts.values()))
    var_success_rates["corrections"] = int(total_corrections)
    return var_success_rates


def evaluate_variation(
    env,
    task_str: str,
    task,
    max_steps: int,
    variation: int,
    num_demos: int,
    actioner,
    runtime_pipeline: Optional[GraspCorrectPipeline],
    object_prompt: Optional[str],
    max_tries: int,
    dense_interpolation: bool,
    interpolation_length: int,
    num_history: int,
):
    from pyrep.errors import ConfigurationPathError, IKError  # type: ignore
    from rlbench.backend.exceptions import InvalidActionError  # type: ignore
    from utils.utils_with_rlbench import Mover  # type: ignore
    import torch
    import torch.nn.functional as F

    device = actioner.device
    success_count = 0
    valid_demos = 0
    correction_count = 0
    for demo_id in range(num_demos):
        try:
            demo = env.get_demo(task_str, variation, episode_index=demo_id)[0]
            valid_demos += 1
        except Exception:
            continue

        rgbs = torch.Tensor([]).to(device)
        pcds = torch.Tensor([]).to(device)
        grippers = torch.Tensor([]).to(device)
        descriptions, obs = task.reset_to_demo(demo)
        instruction = descriptions[0] if descriptions else task_str.replace("_", " ")
        actioner.load_episode(task_str, variation)
        move = Mover(task, max_tries=max_tries)
        runtime = GraspCorrectRuntime(runtime_pipeline) if runtime_pipeline is not None else None
        if runtime is not None:
            runtime.object_prompt = object_prompt
            runtime.observe(obs)

        reward = 0.0
        with torch.no_grad():
            for _step in range(max_steps):
                rgb, pcd, gripper = env.get_rgb_pcd_gripper_from_obs(obs)
                rgb = rgb.to(device)
                pcd = pcd.to(device)
                gripper = gripper.to(device)
                rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1)
                pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1)
                grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)

                rgbs_input = rgbs[:, -1:][:, :, :, :3]
                pcds_input = pcds[:, -1:]
                if num_history < 1:
                    gripper_input = grippers[:, -1]
                else:
                    gripper_input = grippers[:, -num_history:]
                    npad = num_history - gripper_input.shape[1]
                    gripper_input = F.pad(gripper_input, (0, 0, npad, 0), mode="replicate")

                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    interpolation_length=interpolation_length,
                )
                try:
                    if output.get("trajectory", None) is not None:
                        trajectory = output["trajectory"][-1].detach().cpu().numpy()
                        trajectory[:, -1] = trajectory[:, -1].round()
                        for action in tqdm(trajectory, leave=False):
                            if runtime is not None:
                                action = runtime.maybe_correct(obs, action, instruction)
                            obs, reward, terminate, _ = move(action, collision_checking=False)
                            if runtime is not None:
                                runtime.observe(obs)
                            if reward == 1:
                                break
                    else:
                        action = output["action"]
                        action[..., -1] = torch.round(action[..., -1])
                        action_np = action[-1].detach().cpu().numpy()
                        if runtime is not None:
                            action_np = runtime.maybe_correct(obs, action_np, instruction)
                        obs, reward, terminate, _ = move(action_np, collision_checking=False)
                        if runtime is not None:
                            runtime.observe(obs)
                    if reward == 1:
                        success_count += 1
                        break
                except (IKError, ConfigurationPathError, InvalidActionError):
                    reward = 0.0
                    break
        if runtime is not None and runtime.already_corrected:
            correction_count += 1
    if valid_demos == 0:
        return 0, False, 0, 0
    return int(success_count), True, valid_demos, correction_count
