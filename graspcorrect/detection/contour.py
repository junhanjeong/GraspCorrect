from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from graspcorrect.types import GraspCandidate
from graspcorrect.utils.image import mask_to_bool


@dataclass
class OrderedContour:
    """A 1D parameterization of the largest object-mask contour."""

    points_xy: np.ndarray

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "OrderedContour":
        binary = mask_to_bool(mask)
        points = _opencv_contour(binary)
        if points is None:
            points = _fallback_contour(binary)
        if len(points) < 4:
            raise ValueError("Object contour is too small for grasp sampling.")
        return cls(points_xy=np.asarray(points, dtype=np.float32))

    def sample_uniform(self, count: int) -> List[GraspCandidate]:
        n = len(self.points_xy)
        indices = np.linspace(0, n, count, endpoint=False, dtype=np.int64)
        return self._candidates_from_indices(indices)

    def sample_gaussian(
        self,
        selected: Sequence[GraspCandidate],
        count: int,
        sigma_fraction: float,
        rng: np.random.Generator,
    ) -> List[GraspCandidate]:
        n = len(self.points_xy)
        centers = np.asarray([self.nearest_index(c.point) for c in selected], dtype=np.float32)
        if centers.size == 0:
            return self.sample_uniform(count)
        sigma = max(1.0, float(n) * float(sigma_fraction))
        sampled = []
        attempts = 0
        while len(sampled) < count and attempts < count * 20:
            attempts += 1
            center = float(rng.choice(centers))
            idx = int(round(rng.normal(center, sigma))) % n
            if idx not in sampled:
                sampled.append(idx)
        if len(sampled) < count:
            extra = rng.choice(np.arange(n), size=count - len(sampled), replace=False)
            sampled.extend([int(x) for x in extra])
        return self._candidates_from_indices(np.asarray(sampled[:count], dtype=np.int64))

    def nearest_index(self, point: Iterable[float]) -> int:
        p = np.asarray(point, dtype=np.float32).reshape(1, 2)
        dist = np.sum((self.points_xy - p) ** 2, axis=1)
        return int(np.argmin(dist))

    def _candidates_from_indices(self, indices: np.ndarray) -> List[GraspCandidate]:
        out: List[GraspCandidate] = []
        for label, idx in enumerate(indices, 1):
            x, y = self.points_xy[int(idx) % len(self.points_xy)]
            out.append(GraspCandidate(label=label, point=(float(x), float(y)), contour_index=int(idx)))
        return out


def relabel(candidates: Sequence[GraspCandidate]) -> List[GraspCandidate]:
    return [
        GraspCandidate(label=i + 1, point=c.point, contour_index=c.contour_index)
        for i, c in enumerate(candidates)
    ]


def select_by_labels(candidates: Sequence[GraspCandidate], labels: Sequence[int]) -> List[GraspCandidate]:
    by_label = {c.label: c for c in candidates}
    selected = []
    for label in labels:
        cand = by_label.get(int(label))
        if cand is not None:
            selected.append(cand)
    return selected


def _opencv_contour(binary: np.ndarray):
    try:
        import cv2  # type: ignore

        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        pts = contour.reshape(-1, 2)
        return pts
    except Exception:
        return None


def _fallback_contour(binary: np.ndarray) -> np.ndarray:
    padded = np.pad(binary, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    eroded = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = binary & ~eroded
    ys, xs = np.nonzero(boundary)
    if len(xs) == 0:
        ys, xs = np.nonzero(binary)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    return pts[np.argsort(angles)]
