from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from graspcorrect.config import SegmenterConfig
from graspcorrect.utils.image import ensure_rgb, mask_to_bool


class Segmenter:
    def segment(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        raise NotImplementedError


@dataclass
class LangSAMSegmenter(Segmenter):
    """In-process LangSAM wrapper.

    Use this only in an environment where LangSAM can be imported. The current
    upstream LangSAM package requires Python >=3.10; RLBench/3D Diffuser Actor
    commonly run in Python 3.8, so SubprocessLangSAMSegmenter is often safer.
    """

    config: SegmenterConfig = field(default_factory=SegmenterConfig)

    def __post_init__(self) -> None:
        try:
            from lang_sam import LangSAM  # type: ignore
        except Exception as exc:
            raise ImportError("LangSAM is not importable in this environment.") from exc
        self.model = LangSAM(sam_type=self.config.sam_type)

    def segment(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        result = self.model.predict(
            [Image.fromarray(ensure_rgb(image))],
            [text_prompt],
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
        )[0]
        masks = np.asarray(result.get("masks", []))
        if masks.size == 0:
            raise RuntimeError(f"LangSAM returned no mask for prompt {text_prompt!r}.")
        scores = np.asarray(result.get("mask_scores", result.get("scores", np.ones(len(masks)))))
        idx = int(np.argmax(scores.reshape(-1))) if scores.size else 0
        return mask_to_bool(masks[idx])


@dataclass
class SubprocessLangSAMSegmenter(Segmenter):
    config: SegmenterConfig = field(default_factory=SegmenterConfig)
    script_path: Path = Path("scripts/langsam_segment.py")

    def segment(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        python = self.config.langsam_python or os.environ.get("GRASPCORRECT_LANGSAM_PYTHON") or sys.executable
        with tempfile.TemporaryDirectory(prefix="graspcorrect_langsam_") as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "image.png"
            mask_path = tmp_path / "mask.npy"
            meta_path = tmp_path / "meta.json"
            Image.fromarray(ensure_rgb(image)).save(image_path)
            cmd = [
                python,
                str(self.script_path),
                "--image",
                str(image_path),
                "--prompt",
                text_prompt,
                "--output",
                str(mask_path),
                "--metadata-output",
                str(meta_path),
                "--sam-type",
                self.config.sam_type,
                "--box-threshold",
                str(self.config.box_threshold),
                "--text-threshold",
                str(self.config.text_threshold),
            ]
            proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    "LangSAM subprocess failed.\n"
                    f"cmd={cmd}\nstdout={proc.stdout}\nstderr={proc.stderr}"
                )
            if not mask_path.exists():
                raise RuntimeError(f"LangSAM subprocess did not write {mask_path}.")
            if meta_path.exists():
                json.loads(meta_path.read_text(encoding="utf-8"))
            return mask_to_bool(np.load(mask_path))
