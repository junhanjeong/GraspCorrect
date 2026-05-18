from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

from graspcorrect.types import GraspCandidate, Point2D
from graspcorrect.utils.image import mask_to_bool


def boundary_points(mask: np.ndarray) -> np.ndarray:
    """Return boundary points as Nx2 array in x,y order."""

    binary = mask_to_bool(mask)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    neighbors = [
        padded[:-2, 1:-1],
        padded[2:, 1:-1],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[:-2, :-2],
        padded[:-2, 2:],
        padded[2:, :-2],
        padded[2:, 2:],
    ]
    interior = center.copy()
    for neighbor in neighbors:
        interior &= neighbor
    boundary = center & ~interior
    ys, xs = np.nonzero(boundary)
    if xs.size == 0:
        ys, xs = np.nonzero(binary)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def order_points_by_angle(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    return points[order]


def ordered_contour(mask: np.ndarray) -> np.ndarray:
    return order_points_by_angle(boundary_points(mask))


def sample_initial_candidates(
    mask: np.ndarray,
    num_candidates: int,
    rng: Optional[np.random.Generator] = None,
) -> List[GraspCandidate]:
    contour = ordered_contour(mask)
    if contour.shape[0] == 0:
        raise ValueError("Cannot sample grasp candidates from an empty contour.")
    if rng is None:
        rng = np.random.default_rng()
    if contour.shape[0] >= num_candidates:
        indices = np.linspace(0, contour.shape[0] - 1, num_candidates, dtype=int)
    else:
        indices = rng.choice(contour.shape[0], size=num_candidates, replace=True)
    return [
        GraspCandidate(label=i + 1, point=tuple(map(float, contour[idx])), contour_index=int(idx))
        for i, idx in enumerate(indices)
    ]


def gaussian_resample_candidates(
    mask: np.ndarray,
    selected: Sequence[GraspCandidate],
    num_candidates: int,
    sigma_fraction: float = 0.08,
    rng: Optional[np.random.Generator] = None,
) -> List[GraspCandidate]:
    contour = ordered_contour(mask)
    if contour.shape[0] == 0:
        raise ValueError("Cannot sample grasp candidates from an empty contour.")
    if not selected:
        return sample_initial_candidates(mask, num_candidates, rng)
    if rng is None:
        rng = np.random.default_rng()
    sigma = max(1.0, float(contour.shape[0]) * sigma_fraction)
    centers = np.asarray([c.contour_index for c in selected], dtype=np.float32)
    sampled = []
    attempts = 0
    while len(sampled) < num_candidates and attempts < num_candidates * 20:
        attempts += 1
        center = float(rng.choice(centers))
        idx = int(round(rng.normal(center, sigma))) % contour.shape[0]
        if idx not in sampled:
            sampled.append(idx)
    while len(sampled) < num_candidates:
        sampled.append(int(rng.integers(0, contour.shape[0])))
    return [
        GraspCandidate(label=i + 1, point=tuple(map(float, contour[idx])), contour_index=int(idx))
        for i, idx in enumerate(sampled[:num_candidates])
    ]


def candidates_by_label(
    candidates: Sequence[GraspCandidate],
    labels: Iterable[int],
) -> List[GraspCandidate]:
    lookup = {candidate.label: candidate for candidate in candidates}
    return [lookup[label] for label in labels if label in lookup]


def contour_centroid(mask: np.ndarray) -> Point2D:
    binary = mask_to_bool(mask)
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        raise ValueError("Cannot compute centroid of an empty mask.")
    return float(xs.mean()), float(ys.mean())


def antipodal_score(left: Point2D, right: Point2D, centroid: Point2D) -> float:
    left_v = np.asarray(left, dtype=np.float32) - np.asarray(centroid, dtype=np.float32)
    right_v = np.asarray(right, dtype=np.float32) - np.asarray(centroid, dtype=np.float32)
    distance = float(np.linalg.norm(left_v - right_v))
    if np.linalg.norm(left_v) < 1e-6 or np.linalg.norm(right_v) < 1e-6:
        opposite = 0.0
    else:
        opposite = float(-np.dot(left_v, right_v) / (np.linalg.norm(left_v) * np.linalg.norm(right_v)))
    horizontal = abs(float(left[1] - right[1]))
    return distance + 30.0 * opposite - 0.2 * horizontal
