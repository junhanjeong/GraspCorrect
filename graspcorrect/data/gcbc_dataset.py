from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
from PIL import Image, ImageEnhance

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object

PathLike = Union[str, Path]
REQUIRED_KEYS = ("current_rgb", "goal_rgb", "current_action", "target_action")


class GraspCorrectionDataset(Dataset):
    def __init__(self, manifest: PathLike, image_size: int = 224, augment: bool = False) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for GCBC training.")
        self.manifest = Path(manifest)
        self.entries = read_manifest(self.manifest)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, "torch.Tensor"]:
        sample = load_gcbc_sample(self.entries[index])
        current = Image.fromarray(sample["current_rgb"].astype(np.uint8)).convert("RGB")
        goal = Image.fromarray(sample["goal_rgb"].astype(np.uint8)).convert("RGB")
        if self.augment:
            current, goal = paired_image_augment(current, goal)
        return {
            "current_image": image_tensor(current, self.image_size),
            "goal_image": image_tensor(goal, self.image_size),
            "current_action": torch.from_numpy(sample["current_action"].astype(np.float32)),
            "target_action": torch.from_numpy(sample["target_action"].astype(np.float32)),
        }


def read_manifest(path: PathLike) -> List[Path]:
    path = Path(path)
    entries: List[Path] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            p = Path(item["path"])
            if not p.is_absolute():
                p = path.parent / p
            entries.append(p)
    return entries


def write_manifest(paths: Iterable[Path], output: PathLike) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for p in paths:
            path = Path(p)
            try:
                manifest_path = path.resolve().relative_to(output.parent.resolve())
            except ValueError:
                manifest_path = path.resolve()
            f.write(json.dumps({"path": str(manifest_path)}) + "\n")


def load_gcbc_sample(path: PathLike) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"{path} is missing keys: {missing}")
        return {k: data[k] for k in REQUIRED_KEYS}


def save_gcbc_sample(
    path: PathLike,
    current_rgb: np.ndarray,
    goal_rgb: np.ndarray,
    current_action: np.ndarray,
    target_action: np.ndarray,
    **metadata: np.ndarray,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        current_rgb=np.asarray(current_rgb, dtype=np.uint8),
        goal_rgb=np.asarray(goal_rgb, dtype=np.uint8),
        current_action=np.asarray(current_action, dtype=np.float32),
        target_action=np.asarray(target_action, dtype=np.float32),
        **metadata,
    )
    return path


def image_tensor(image: Image.Image, image_size: int) -> "torch.Tensor":
    image = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def paired_image_augment(current: Image.Image, goal: Image.Image):
    # Same random crop for both images; color jitter is independent but sampled
    # once so the pair remains visually coherent.
    rng = np.random.default_rng()
    w, h = current.size
    scale = float(rng.uniform(0.88, 1.0))
    crop_w = max(2, int(round(w * scale)))
    crop_h = max(2, int(round(h * scale)))
    x0 = int(rng.integers(0, max(1, w - crop_w + 1)))
    y0 = int(rng.integers(0, max(1, h - crop_h + 1)))
    box = (x0, y0, x0 + crop_w, y0 + crop_h)
    current = current.crop(box).resize((w, h), Image.BILINEAR)
    goal = goal.crop(box).resize((w, h), Image.BILINEAR)
    brightness = float(rng.uniform(0.85, 1.15))
    contrast = float(rng.uniform(0.85, 1.15))
    saturation = float(rng.uniform(0.85, 1.15))
    for enhancer, factor in (
        (ImageEnhance.Brightness, brightness),
        (ImageEnhance.Contrast, contrast),
        (ImageEnhance.Color, saturation),
    ):
        current = enhancer(current).enhance(factor)
        goal = enhancer(goal).enhance(factor)
    return current, goal
