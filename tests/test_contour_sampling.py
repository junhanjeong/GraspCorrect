import numpy as np

from graspcorrect.detection.contour import gaussian_resample_candidates, sample_initial_candidates


def test_sample_initial_candidates_on_square_mask():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 20:44] = 255
    candidates = sample_initial_candidates(mask, 12, np.random.default_rng(0))
    assert len(candidates) == 12
    assert all(0 <= x < 64 and 0 <= y < 64 for c in candidates for x, y in [c.point])


def test_gaussian_resample_preserves_candidate_count():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 20:44] = 255
    candidates = sample_initial_candidates(mask, 12, np.random.default_rng(0))
    refined = gaussian_resample_candidates(mask, candidates[:3], 12, rng=np.random.default_rng(1))
    assert len(refined) == 12
    assert {c.label for c in refined} == set(range(1, 13))
