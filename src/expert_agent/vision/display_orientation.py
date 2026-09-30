from __future__ import annotations

import numpy as np


def match_rgb_to_native(
    rgb: np.ndarray,
    science: np.ndarray,
    min_correlation: float = 0.3,
) -> tuple[np.ndarray, str, float] | None:
    """Orient an RGB preview onto the native FITS pixel grid."""
    from scipy import ndimage

    colour = np.asarray(rgb)
    native = np.asarray(science, dtype=float)
    if colour.ndim != 3 or colour.shape[:2] != native.shape:
        return None
    gray = np.asarray(colour, dtype=float).mean(axis=2)

    def normalized(values: np.ndarray) -> np.ndarray:
        finite = values[np.isfinite(values)]
        if not finite.size:
            return np.zeros_like(values, dtype=float)
        low, high = np.percentile(finite, [1.0, 99.0])
        clipped = np.clip(values, low, high)
        std = float(np.nanstd(clipped))
        return (clipped - float(np.nanmean(clipped))) / max(std, 1e-9)

    target = normalized(ndimage.gaussian_filter(native, 1.5)).ravel()
    candidates = {
        "identity": colour,
        "flip_y": colour[::-1, :],
        "flip_x": colour[:, ::-1],
        "rotate_180": colour[::-1, ::-1],
    }
    scored: list[tuple[float, str]] = []
    for name, candidate in candidates.items():
        candidate_gray = np.asarray(candidate, dtype=float).mean(axis=2)
        aligned = normalized(ndimage.gaussian_filter(candidate_gray, 1.5)).ravel()
        if float(np.std(target)) <= 1e-12 or float(np.std(aligned)) <= 1e-12:
            score = float("nan")
        else:
            score = float(np.corrcoef(target, aligned)[0, 1])
        scored.append((score, name))
    finite_scores = [item for item in scored if np.isfinite(item[0])]
    if not finite_scores:
        return None
    score, mapping = max(finite_scores)
    if score < float(min_correlation):
        return None
    return np.asarray(candidates[mapping]), mapping, float(score)
