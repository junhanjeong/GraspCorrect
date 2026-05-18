from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
from PIL import Image

try:  # pragma: no cover - optional train dependency
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object


REQUIRED_KEYS = ("current_rgb", "goal_rgb", "current_action", "target_action")


@dataclass
class ManifestEntry:
    path: Path
    task: str = ""


def read_manifest(path: str | Path) -> List[ManifestEntry]:
    manifest = Path(path)
    entries: List[ManifestEntry] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sample_path = Path(item["path"])
            if not sample_path.is_absolute():
                sample_path = manifest.parent / sample_path
            entries.append(ManifestEntry(path=sample_path, task=item.get("task", "")))
    return entries


def write_manifest(entries: Iterable[ManifestEntry], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps({"path": str(entry.path), "task": entry.task}) + "\n")


class GraspCorrectionDataset(Dataset):
    def __init__(self, manifest: str | Path, image_size: int = 224) -> None:
        if torch is None:
            raise ImportError("GraspCorrectionDataset requires torch. Install with `pip install -e .[train]`.")
        self.entries = read_manifest(manifest)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        entry = self.entries[idx]
        sample = load_npz_sample(entry.path)
        return {
            "current_image": _image_tensor(sample["current_rgb"], self.image_size),
            "goal_image": _image_tensor(sample["goal_rgb"], self.image_size),
            "current_action": torch.from_numpy(sample["current_action"].astype(np.float32)),
            "target_action": torch.from_numpy(sample["target_action"].astype(np.float32)),
        }


def load_npz_sample(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"{path} is missing required keys: {missing}")
        return {key: data[key] for key in REQUIRED_KEYS}


def save_npz_sample(
    path: str | Path,
    current_rgb: np.ndarray,
    goal_rgb: np.ndarray,
    current_action: np.ndarray,
    target_action: np.ndarray,
    **extra: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        current_rgb=np.asarray(current_rgb, dtype=np.uint8),
        goal_rgb=np.asarray(goal_rgb, dtype=np.uint8),
        current_action=np.asarray(current_action, dtype=np.float32),
        target_action=np.asarray(target_action, dtype=np.float32),
        **extra,
    )


def identify_grasp_index(actions: np.ndarray, gripper_index: int = 7, close_threshold: float = 0.5) -> Optional[int]:
    arr = np.asarray(actions)
    if arr.ndim != 2 or arr.shape[1] <= gripper_index:
        raise ValueError(f"actions must be TxD with D>{gripper_index}, got {arr.shape}.")
    gripper = arr[:, gripper_index]
    for i in range(1, len(gripper)):
        if gripper[i - 1] > close_threshold and gripper[i] <= close_threshold:
            return i
    return None


def perturb_action(
    action: np.ndarray,
    position_sigma: float = 0.015,
    rotation_sigma: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    out = np.asarray(action, dtype=np.float32).copy()
    out[:3] += rng.normal(0.0, position_sigma, size=3).astype(np.float32)
    quat = out[3:7] + rng.normal(0.0, rotation_sigma, size=4).astype(np.float32)
    norm = float(np.linalg.norm(quat))
    if norm > 1e-6:
        quat = quat / norm
    out[3:7] = quat
    return out


def _image_tensor(image: np.ndarray, image_size: int) -> "torch.Tensor":
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    pil = pil.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)
