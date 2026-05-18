from __future__ import annotations

import contextlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from graspcorrect.data import save_gcbc_sample
from graspcorrect.data.gcbc_dataset import write_manifest
from graspcorrect.utils.image import extract_observation_rgb


ROOT = Path(__file__).resolve().parents[2]
DIFFUSER_ROOT = ROOT / "external" / "3d_diffuser_actor"
if str(DIFFUSER_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSER_ROOT))


@dataclass
class WaypointPerturbConfig:
    position_sigma: float = 0.015
    rotation_sigma: float = 0.08
    perturb_all_waypoints: bool = False


@dataclass(frozen=True)
class StoredDemoRef:
    variation: int
    episode_offset: int
    episode_name: str


def collect_rlbench_gcbc_dataset_from_stored_demos(
    tasks: Sequence[str],
    output_dir: Path,
    samples_per_task: int,
    variations: Sequence[int],
    data_dir: Path,
    correction_camera: str = "overhead",
    headless: bool = True,
    seed: int = 7,
    perturb_config: Optional[WaypointPerturbConfig] = None,
    replay_max_tries: int = 2,
) -> Path:
    from rlbench.action_modes.action_mode import MoveArmThenGripper  # type: ignore
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning  # type: ignore
    from rlbench.action_modes.gripper_action_modes import Discrete  # type: ignore
    from rlbench.environment import Environment  # type: ignore
    from utils.utils_with_rlbench import Mover, keypoint_discovery, task_file_to_task_class  # type: ignore

    rng = np.random.default_rng(seed)
    random.seed(seed)
    perturb_config = perturb_config or WaypointPerturbConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "collection_summary.json"

    env = Environment(
        action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
        dataset_root=str(data_dir),
        obs_config=make_obs_config(correction_camera),
        headless=headless,
    )
    env.launch()
    written: List[Path] = []
    summary: Dict[str, Dict[str, object]] = {}
    try:
        for task_name in tasks:
            task_cls = task_file_to_task_class(task_name)
            task_env = env.get_task(task_cls)
            variation_list = resolve_variations(task_env, variations)
            demo_refs = list_stored_demo_refs(data_dir, task_name, variation_list)
            demo_refs = filter_stored_demo_refs_with_grasp(task_env, task_name, demo_refs)
            if not demo_refs:
                print(f"[skip-task] {task_name}: no stored demos with a gripper close transition under {data_dir}")
                summary[task_name] = {
                    "written": 0,
                    "requested": samples_per_task,
                    "attempts": 0,
                    "variations": variation_list,
                    "stored_demos": 0,
                }
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                continue

            task_written = 0
            attempts = 0
            max_attempts = samples_per_task * max(5, min(20, len(demo_refs)))
            while task_written < samples_per_task and attempts < max_attempts:
                ref = demo_refs[attempts % len(demo_refs)]
                attempts += 1
                task_env.set_variation(ref.variation)
                try:
                    demo = task_env.get_demos(
                        amount=1,
                        live_demos=False,
                        image_paths=False,
                        random_selection=False,
                        from_episode_number=ref.episode_offset,
                    )[0]
                    stable_idx = find_grasp_index(demo)
                    if stable_idx is None:
                        print(f"[skip] {task_name} {ref.episode_name}: no gripper close transition")
                        continue
                    stable_obs_at_grasp = demo[stable_idx]
                    stable_action = obs_action(stable_obs_at_grasp)
                    task_env.reset_to_demo(demo)
                    mover = Mover(task_env, max_tries=replay_max_tries)
                    for frame in replay_keypoints_before_grasp(demo, stable_idx, keypoint_discovery):
                        mover(obs_action(demo[frame]), collision_checking=False)
                    perturbed_action = perturb_action(stable_action, rng, perturb_config)
                    current_obs, _, _, _ = mover(perturbed_action, collision_checking=False)
                except Exception as exc:
                    print(f"[skip] {task_name} {ref.episode_name}: {exc}")
                    continue

                path = sample_dir / f"{task_name}_v{ref.variation}_{len(written):06d}.npz"
                save_gcbc_sample(
                    path=path,
                    current_rgb=extract_observation_rgb(current_obs, correction_camera),
                    goal_rgb=extract_observation_rgb(stable_obs_at_grasp, correction_camera),
                    current_action=perturbed_action,
                    target_action=stable_action,
                    task=np.asarray(task_name),
                    variation=np.asarray(ref.variation),
                    source_episode=np.asarray(ref.episode_name),
                    stable_grasp_index=np.asarray(stable_idx),
                )
                written.append(path)
                task_written += 1
                print(f"[write] {path}")

            summary[task_name] = {
                "written": task_written,
                "requested": samples_per_task,
                "attempts": attempts,
                "variations": variation_list,
                "stored_demos": len(demo_refs),
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        env.shutdown()
    write_manifest(written, manifest_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest_path


def collect_rlbench_gcbc_dataset(
    tasks: Sequence[str],
    output_dir: Path,
    samples_per_task: int,
    variations: Sequence[int],
    correction_camera: str = "overhead",
    headless: bool = True,
    seed: int = 7,
    perturb_config: Optional[WaypointPerturbConfig] = None,
    demo_max_attempts: int = 10,
) -> Path:
    from rlbench.action_modes.action_mode import MoveArmThenGripper  # type: ignore
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning  # type: ignore
    from rlbench.action_modes.gripper_action_modes import Discrete  # type: ignore
    from rlbench.environment import Environment  # type: ignore
    from utils.utils_with_rlbench import task_file_to_task_class  # type: ignore

    rng = np.random.default_rng(seed)
    random.seed(seed)
    perturb_config = perturb_config or WaypointPerturbConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "collection_summary.json"

    env = Environment(
        action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
        dataset_root="",
        obs_config=make_obs_config(correction_camera),
        headless=headless,
    )
    env.launch()
    written: List[Path] = []
    summary: Dict[str, Dict[str, object]] = {}
    try:
        for task_name in tasks:
            task_cls = task_file_to_task_class(task_name)
            task_env = env.get_task(task_cls)
            requested_variations = list(variations)
            total_variations = int(task_env.variation_count())
            variation_list = resolve_variations(task_env, variations)
            if not variation_list:
                print(f"[skip-task] {task_name}: no valid variations in {requested_variations}; task has {total_variations}")
                summary[task_name] = {
                    "written": 0,
                    "requested": samples_per_task,
                    "attempts": 0,
                    "variations": [],
                    "task_variation_count": total_variations,
                }
                continue
            task_written = 0
            attempts = 0
            max_attempts = samples_per_task * max(20, 20 * len(variation_list))
            while task_written < samples_per_task and attempts < max_attempts:
                variation = variation_list[attempts % len(variation_list)]
                attempts += 1
                task_env.set_variation(variation)
                state = np.random.get_state()
                stable_obs = []
                perturbed_obs = []
                try:
                    np.random.set_state(state)
                    stable_demo = task_env.get_demos(
                        amount=1,
                        live_demos=True,
                        callable_each_step=stable_obs.append,
                        max_attempts=demo_max_attempts,
                    )[0]
                    np.random.set_state(state)
                    with perturb_waypoints(task_env, rng, perturb_config):
                        perturbed_demo = task_env.get_demos(
                            amount=1,
                            live_demos=True,
                            callable_each_step=perturbed_obs.append,
                            max_attempts=demo_max_attempts,
                        )[0]
                except Exception as exc:
                    print(f"[skip] {task_name} variation={variation}: {exc}")
                    continue

                stable_idx = find_grasp_index(stable_obs or stable_demo)
                pert_idx = find_grasp_index(perturbed_obs or perturbed_demo)
                if stable_idx is None or pert_idx is None:
                    print(f"[skip] {task_name} variation={variation}: no gripper close transition")
                    continue

                current_obs = (perturbed_obs or perturbed_demo)[pert_idx]
                stable_obs_at_grasp = (stable_obs or stable_demo)[stable_idx]
                stable_action = obs_action(stable_obs_at_grasp)
                perturbed_action = obs_action(current_obs)
                path = sample_dir / f"{task_name}_v{variation}_{len(written):06d}.npz"
                save_gcbc_sample(
                    path=path,
                    current_rgb=extract_observation_rgb(current_obs, correction_camera),
                    goal_rgb=extract_observation_rgb(stable_obs_at_grasp, correction_camera),
                    current_action=perturbed_action,
                    target_action=stable_action,
                    task=np.asarray(task_name),
                    variation=np.asarray(variation),
                )
                written.append(path)
                task_written += 1
                print(f"[write] {path}")
            summary[task_name] = {
                "written": task_written,
                "requested": samples_per_task,
                "attempts": attempts,
                "variations": variation_list,
                "task_variation_count": total_variations,
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        env.shutdown()
    write_manifest(written, manifest_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest_path


@contextlib.contextmanager
def perturb_waypoints(task_env, rng: np.random.Generator, config: WaypointPerturbConfig):
    task = task_env._task
    original_get_waypoints = task._get_waypoints

    def patched_get_waypoints(*args, **kwargs):
        waypoints = original_get_waypoints(*args, **kwargs)
        for waypoint in waypoints:
            ext = waypoint.get_ext()
            should_perturb = config.perturb_all_waypoints or "close_gripper" in ext
            if not should_perturb:
                continue
            obj = waypoint.get_waypoint_object()
            pos = np.asarray(obj.get_position(), dtype=np.float32)
            euler = np.asarray(obj.get_orientation(), dtype=np.float32)
            pos = pos + rng.normal(0.0, config.position_sigma, size=3).astype(np.float32)
            euler = euler + rng.normal(0.0, config.rotation_sigma, size=3).astype(np.float32)
            obj.set_position(pos.tolist())
            obj.set_orientation(euler.tolist())
        return waypoints

    task._get_waypoints = patched_get_waypoints
    task._waypoints = None
    try:
        yield
    finally:
        task._get_waypoints = original_get_waypoints
        task._waypoints = None


def resolve_variations(task_env, variations: Sequence[int]) -> List[int]:
    requested_variations = list(variations)
    total_variations = int(task_env.variation_count())
    if requested_variations == [-1]:
        return list(range(total_variations))
    return [v for v in (requested_variations or [0]) if 0 <= v < total_variations]


def list_stored_demo_refs(data_dir: Path, task_name: str, variations: Sequence[int]) -> List[StoredDemoRef]:
    refs: List[StoredDemoRef] = []
    for variation in variations:
        episodes_dir = data_dir / task_name / f"variation{variation}" / "episodes"
        if not episodes_dir.exists():
            continue
        episodes = sorted(
            (p for p in episodes_dir.iterdir() if p.is_dir() and p.name.startswith("episode")),
            key=episode_sort_key,
        )
        refs.extend(StoredDemoRef(variation=variation, episode_offset=i, episode_name=p.name) for i, p in enumerate(episodes))
    return refs


def filter_stored_demo_refs_with_grasp(task_env, task_name: str, refs: Sequence[StoredDemoRef]) -> List[StoredDemoRef]:
    eligible: List[StoredDemoRef] = []
    for ref in refs:
        try:
            task_env.set_variation(ref.variation)
            demo = task_env.get_demos(
                amount=1,
                live_demos=False,
                image_paths=False,
                random_selection=False,
                from_episode_number=ref.episode_offset,
            )[0]
            if find_grasp_index(demo) is not None:
                eligible.append(ref)
        except Exception as exc:
            print(f"[skip-demo] {task_name} {ref.episode_name}: {exc}")
    return eligible


def episode_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def replay_keypoints_before_grasp(demo, grasp_index: int, keypoint_discovery_fn) -> List[int]:
    frames = [int(f) for f in keypoint_discovery_fn(demo) if 0 < int(f) < grasp_index]
    if not frames and grasp_index > 1:
        frames = [grasp_index - 1]
    return frames


def perturb_action(action: np.ndarray, rng: np.random.Generator, config: WaypointPerturbConfig) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    out[:3] += rng.normal(0.0, config.position_sigma, size=3).astype(np.float32)
    delta = euler_to_quat_xyzw(rng.normal(0.0, config.rotation_sigma, size=3))
    out[3:7] = normalize_quat_xyzw(quat_multiply_xyzw(delta, out[3:7]))
    out[7] = action[7]
    return out


def euler_to_quat_xyzw(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(euler, dtype=np.float32)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.asarray(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float32,
    )


def quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.asarray(a, dtype=np.float32)
    bx, by, bz, bw = np.asarray(b, dtype=np.float32)
    return np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    arr = np.asarray(quat, dtype=np.float32)
    return arr / max(float(np.linalg.norm(arr)), 1e-6)


def make_obs_config(camera: str):
    from pyrep.const import RenderMode  # type: ignore
    from rlbench.observation_config import CameraConfig, ObservationConfig  # type: ignore

    unused = CameraConfig()
    unused.set_all(False)
    used = CameraConfig(
        rgb=True,
        depth=False,
        point_cloud=False,
        mask=False,
        image_size=(256, 256),
        render_mode=RenderMode.OPENGL,
    )
    kwargs = {
        "front_camera": unused,
        "left_shoulder_camera": unused,
        "right_shoulder_camera": unused,
        "wrist_camera": unused,
        "overhead_camera": unused,
        f"{camera}_camera": used,
    }
    return ObservationConfig(
        joint_forces=False,
        joint_positions=False,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
        **kwargs,
    )


def find_grasp_index(observations: Iterable[object], threshold: float = 0.5) -> Optional[int]:
    obs_list = list(observations)
    if len(obs_list) < 2:
        return None
    prev = float(obs_list[0].gripper_open)
    for i, obs in enumerate(obs_list[1:], 1):
        value = float(obs.gripper_open)
        if prev > threshold and value <= threshold:
            return i
        prev = value
    return None


def obs_action(obs: object) -> np.ndarray:
    return np.concatenate([np.asarray(obs.gripper_pose, dtype=np.float32), np.asarray([float(obs.gripper_open)], dtype=np.float32)])
