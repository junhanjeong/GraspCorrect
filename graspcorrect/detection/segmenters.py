from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from graspcorrect.utils.image import ensure_uint8_rgb


class Segmenter(Protocol):
    def segment(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        ...


@dataclass
class HeuristicSegmenter:
    """Simple foreground mask for smoke tests when LangSAM is unavailable."""

    saturation_threshold: int = 30
    value_delta: int = 18

    def segment(self, image: np.ndarray, text_prompt: str = "") -> np.ndarray:
        rgb = ensure_uint8_rgb(image).astype(np.float32)
        maxc = rgb.max(axis=-1)
        minc = rgb.min(axis=-1)
        saturation = maxc - minc
        gray = rgb.mean(axis=-1)
        border = np.concatenate([gray[:5, :].ravel(), gray[-5:, :].ravel(), gray[:, :5].ravel(), gray[:, -5:].ravel()])
        bg = float(np.median(border)) if border.size else float(np.median(gray))
        mask = (saturation > self.saturation_threshold) | (np.abs(gray - bg) > self.value_delta)
        mask = _keep_largest_component(mask)
        return mask.astype(np.uint8) * 255


@dataclass
class LangSAMSegmenter:
    """Thin wrapper around lang-segment-anything.

    The upstream LangSAM API has changed over time, so this wrapper tries the
    common call signatures and raises an actionable error if none match.
    """

    device: str = "cuda"

    def __post_init__(self) -> None:
        try:
            from lang_sam import LangSAM  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "LangSAM is not installed. Install it from "
                "https://github.com/luca-medeiros/lang-segment-anything or use HeuristicSegmenter."
            ) from exc
        self.model = LangSAM()

    def segment(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        pil = Image.fromarray(ensure_uint8_rgb(image))
        try:
            result = self.model.predict([pil], [text_prompt])
        except TypeError:  # pragma: no cover - optional dependency
            result = self.model.predict(pil, text_prompt)
        masks = _extract_masks(result)
        if not masks:
            raise RuntimeError(f"LangSAM returned no masks for prompt: {text_prompt!r}")
        areas = [int(np.asarray(mask).astype(bool).sum()) for mask in masks]
        return (np.asarray(masks[int(np.argmax(areas))]).astype(np.uint8) * 255)


def _extract_masks(result: object) -> list:
    if isinstance(result, dict):
        for key in ("masks", "mask"):
            if key in result:
                value = result[key]
                return list(value) if isinstance(value, (list, tuple)) else [value]
    if isinstance(result, (list, tuple)):
        if result and isinstance(result[0], dict):
            return _extract_masks(result[0])
        for item in result:
            arr = np.asarray(item)
            if arr.ndim >= 2:
                return list(item) if arr.ndim == 3 else [item]
    return []


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_pixels = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(pixels) > len(best_pixels):
                best_pixels = pixels
    out = np.zeros_like(mask, dtype=bool)
    if best_pixels:
        ys, xs = zip(*best_pixels)
        out[np.asarray(ys), np.asarray(xs)] = True
    return out
