from __future__ import annotations

import base64
import io
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from graspcorrect.types import GraspCandidate, Point2D


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def mask_to_bool(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    out = arr.astype(bool)
    if min_area and int(out.sum()) < min_area:
        raise ValueError(f"Mask area {int(out.sum())} is smaller than {min_area}.")
    return out


def image_to_data_url(image: np.ndarray) -> str:
    pil = Image.fromarray(ensure_rgb(image))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def draw_candidates(image: np.ndarray, candidates: Sequence[GraspCandidate]) -> np.ndarray:
    pil = Image.fromarray(ensure_rgb(image)).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for cand in candidates:
        x, y = cand.point
        r = 7
        xy = (x - r, y - r, x + r, y + r)
        draw.ellipse(xy, fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        draw.text((x + r + 2, y - r - 2), str(cand.label), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(pil)


def draw_cross(image: np.ndarray, point: Point2D, color: Tuple[int, int, int] = (230, 40, 40)) -> np.ndarray:
    pil = Image.fromarray(ensure_rgb(image)).convert("RGB")
    draw = ImageDraw.Draw(pil)
    x, y = point
    r = 9
    draw.line((x - r, y, x + r, y), fill=color, width=3)
    draw.line((x, y - r, x, y + r), fill=color, width=3)
    return np.asarray(pil)


def rgba_from_mask(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    rgba = Image.fromarray(ensure_rgb(image)).convert("RGBA")
    alpha = Image.fromarray(mask_to_bool(mask).astype(np.uint8) * 255, mode="L")
    rgba.putalpha(alpha)
    return rgba


def translate_layer(layer: Image.Image, dx: float, dy: float) -> Image.Image:
    return layer.convert("RGBA").transform(
        layer.size,
        Image.AFFINE,
        (1.0, 0.0, -dx, 0.0, 1.0, -dy),
        resample=Image.BICUBIC,
    )


def rotate_points(points: Iterable[Point2D], angle_degrees: float, center: Point2D) -> List[Point2D]:
    cx, cy = center
    theta = np.deg2rad(angle_degrees)
    rot = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    out = []
    for p in points:
        vec = np.asarray(p, dtype=np.float32) - np.asarray([cx, cy], dtype=np.float32)
        q = rot @ vec + np.asarray([cx, cy], dtype=np.float32)
        out.append((float(q[0]), float(q[1])))
    return out


def scale_layer_about_point(layer: Image.Image, center: Point2D, scale: float) -> Image.Image:
    if abs(scale - 1.0) < 1e-6:
        return layer.convert("RGBA")
    cx, cy = center
    return layer.convert("RGBA").transform(
        layer.size,
        Image.AFFINE,
        (1.0 / scale, 0.0, cx - cx / scale, 0.0, 1.0 / scale, cy - cy / scale),
        resample=Image.BICUBIC,
    )


def inpaint_background(image: np.ndarray, remove_mask: np.ndarray, radius: int = 5) -> np.ndarray:
    rgb = ensure_rgb(image)
    mask = mask_to_bool(remove_mask)
    try:
        import cv2  # type: ignore

        return cv2.inpaint(rgb, (mask.astype(np.uint8) * 255), radius, cv2.INPAINT_TELEA)
    except Exception:
        pil = Image.fromarray(rgb)
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=max(radius, 1)))
        out = np.asarray(pil).copy()
        out[mask] = np.asarray(blurred)[mask]
        return out


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    binary = mask_to_bool(mask)
    try:
        import cv2  # type: ignore

        count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        comps = []
        for label in range(1, count):
            ys, xs = np.nonzero(labels == label)
            comps.append(np.stack([ys, xs], axis=1))
        return comps
    except Exception:
        seen = np.zeros(binary.shape, dtype=bool)
        comps: List[np.ndarray] = []
        h, w = binary.shape
        for y, x in zip(*np.nonzero(binary)):
            if seen[y, x]:
                continue
            stack = [(int(y), int(x))]
            seen[y, x] = True
            pts = []
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if binary[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append(np.asarray(pts, dtype=np.int32))
        return comps


def extract_observation_rgb(raw_obs: object, camera: str) -> np.ndarray:
    if isinstance(raw_obs, np.ndarray):
        return ensure_rgb(raw_obs)
    attr = f"{camera}_rgb"
    if hasattr(raw_obs, attr):
        return ensure_rgb(getattr(raw_obs, attr))
    if hasattr(raw_obs, "rgb"):
        return ensure_rgb(getattr(raw_obs, "rgb"))
    raise AttributeError(f"Observation has no RGB image for camera {camera!r}.")
