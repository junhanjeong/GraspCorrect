from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class VLMConfig:
    model: str = "gpt-5.4-mini"
    detail: str = "high"
    temperature: float = 0.0


@dataclass
class DetectorConfig:
    window: int = 10
    iterations: int = 4
    candidates_per_iteration: int = 20
    top_n: int = 3
    gaussian_sigma_fraction: float = 0.06
    min_mask_area: int = 64
    seed: int = 7


@dataclass
class SegmenterConfig:
    mode: str = "langsam"
    langsam_python: Optional[str] = None
    sam_type: str = "sam2.1_hiera_small"
    box_threshold: float = 0.3
    text_threshold: float = 0.25


@dataclass
class GoalConfig:
    inpaint_radius: int = 5
    default_increased_distance_px: float = 0.0
    gripper_prompt: str = "robot gripper"
    correction_camera: str = "overhead"


def load_dotenv(path: str = ".env") -> Dict[str, str]:
    """Load simple KEY=VALUE pairs without adding a hard dependency."""

    env_path = Path(path)
    loaded: Dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded
