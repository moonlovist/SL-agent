"""Local shape measurements of observed pixels; no route-filled templates."""
from __future__ import annotations

import numpy as np


def measure_curved_geometry(mask: np.ndarray, center_yx: tuple[int, int]) -> dict:
    ys, xs = np.where(mask)
    if not len(xs):
        return dict(angular_occupied_deg=0.0, radial_fractional_width=None,
                    local_tangential_fraction=None, local_shape_count=0)
    dx, dy = xs - center_yx[1], ys - center_yx[0]
    radius = np.hypot(dx, dy)
    angles = np.degrees(np.arctan2(dy, dx)) % 360
    occupied = len(np.unique(np.floor(angles / 5).astype(int))) * 5.0
    median = float(np.median(radius))
    width = float((np.percentile(radius, 90) - np.percentile(radius, 10)) / median) if median > 0 else None
    aligned = []
    # Non-overlapping 30-degree sectors; only sectors with enough pixels and a
    # defined elongated axis have a meaningful local tangent measurement.
    for sector in range(12):
        use = (angles >= sector * 30) & (angles < (sector + 1) * 30)
        if use.sum() < 5:
            continue
        xy = np.column_stack((dx[use], dy[use])).astype(float)
        values, vectors = np.linalg.eigh(np.cov(xy, rowvar=False, bias=True))
        if values[-1] <= 0 or np.sqrt(values[-1] / max(values[0], 1e-9)) < 1.5:
            continue
        mean = xy.mean(axis=0)
        if np.linalg.norm(mean) == 0:
            continue
        tangent = np.array([-mean[1], mean[0]]) / np.linalg.norm(mean)
        cosine = np.clip(abs(np.dot(vectors[:, -1], tangent)), 0, 1)
        aligned.append(bool(np.degrees(np.arccos(cosine)) <= 35))
    return dict(angular_occupied_deg=occupied, radial_fractional_width=width,
                local_tangential_fraction=float(np.mean(aligned)) if aligned else None,
                local_shape_count=len(aligned))
