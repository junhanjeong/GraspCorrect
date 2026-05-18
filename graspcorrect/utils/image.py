from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from graspcorrect.types import GraspCandidate, GraspPair, Point2D


Color = Tuple[int, int, int]


def ensure_uint8_rgb(image: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 image, got {arr.shape}.")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def to_pil(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(ensure_uint8_rgb(image), mode="RGB")


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_rgb(image: np.ndarray | Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil(image).save(path)


def image_to_data_url(image: np.ndarray | Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    to_pil(image).save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def mask_to_bool(mask: np.ndarray, min_area: int = 1) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    out = arr > 0
    if int(out.sum()) < min_area:
        raise ValueError(f"Mask area {int(out.sum())} is smaller than min_area={min_area}.")
    return out


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask_to_bool(mask))
    if xs.size == 0:
        raise ValueError("Cannot compute a bounding box for an empty mask.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def draw_numbered_circles(
    image: np.ndarray | Image.Image,
    candidates: Sequence[GraspCandidate],
    radius: int = 9,
    fill: Color = (255, 220, 0),
    outline: Color = (0, 0, 0),
    text: Color = (0, 0, 0),
    selected: Optional[Iterable[int]] = None,
) -> np.ndarray:
    pil = to_pil(image)
    draw = ImageDraw.Draw(pil)
    selected_set = set(selected or [])
    for candidate in candidates:
        x, y = candidate.point
        color = (230, 45, 45) if candidate.label in selected_set else fill
        box = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(box, fill=color, outline=outline, width=2)
        label = str(candidate.label)
        draw.text((x + radius + 2, y - radius - 2), label, fill=text)
    return np.asarray(pil)


def draw_cross(
    image: np.ndarray | Image.Image,
    point: Point2D,
    color: Color = (230, 45, 45),
    radius: int = 10,
    width: int = 3,
) -> np.ndarray:
    pil = to_pil(image)
    draw = ImageDraw.Draw(pil)
    x, y = point
    draw.line((x - radius, y, x + radius, y), fill=color, width=width)
    draw.line((x, y - radius, x, y + radius), fill=color, width=width)
    return np.asarray(pil)


def draw_grasp_pair(
    image: np.ndarray | Image.Image,
    pair: GraspPair,
    color: Color = (35, 150, 255),
    radius: int = 7,
) -> np.ndarray:
    pil = to_pil(image)
    draw = ImageDraw.Draw(pil)
    for point in (pair.left.point, pair.right.point):
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    draw.line((*pair.left.point, *pair.right.point), fill=color, width=2)
    return np.asarray(pil)


def foreground_rgba(image: np.ndarray | Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = to_pil(image).convert("RGBA")
    alpha = Image.fromarray((mask_to_bool(mask).astype(np.uint8) * 255), mode="L")
    rgb.putalpha(alpha)
    return rgb


def median_fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = ensure_uint8_rgb(image).copy()
    mask_bool = mask_to_bool(mask)
    background = rgb[~mask_bool]
    if background.size == 0:
        fill = np.asarray([255, 255, 255], dtype=np.uint8)
    else:
        fill = np.median(background.reshape(-1, 3), axis=0).astype(np.uint8)
    rgb[mask_bool] = fill
    return rgb


def blur_mask_region(image: np.ndarray, mask: np.ndarray, radius: int = 7) -> np.ndarray:
    rgb = ensure_uint8_rgb(image)
    pil = to_pil(rgb)
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    mask_img = Image.fromarray(mask_to_bool(mask).astype(np.uint8) * 255, mode="L")
    pil.paste(blurred, mask=mask_img)
    return np.asarray(pil)


def paste_rgba(base: np.ndarray | Image.Image, rgba: Image.Image, xy: Tuple[int, int]) -> np.ndarray:
    pil = to_pil(base).convert("RGBA")
    pil.alpha_composite(rgba.convert("RGBA"), dest=xy)
    return np.asarray(pil.convert("RGB"))
