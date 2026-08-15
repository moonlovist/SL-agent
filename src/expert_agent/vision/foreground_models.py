from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .preprocess import PreprocessedImage, robust_rms


@dataclass
class ForegroundModelComparison:
    method: str
    success: bool
    model: np.ndarray | None = None
    residual: np.ndarray | None = None
    residual_robust_rms: float | None = None
    coefficients: list[float] = field(default_factory=list)
    gaussian_sigmas_px: list[float] = field(default_factory=list)
    psf_applied: bool = False
    flags: list[str] = field(default_factory=list)


def _normalized_psf(psf: np.ndarray | None) -> np.ndarray | None:
    if psf is None:
        return None
    values = np.asarray(psf, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    # Negative sidelobes may be meaningful in a reconstructed PSF, so retain
    # them; only require a finite, non-zero normalization.
    total = float(values.sum())
    if abs(total) <= 1e-12:
        return None
    return values / total


def fit_psf_mge_comparison(prep: PreprocessedImage) -> ForegroundModelComparison:
    """Fit a conservative, fixed-geometry PSF-convolved MGE for audit only.

    This model is deliberately not used by detection. It provides an
    independent subtraction view so model-sensitive residuals can be exposed.
    """

    try:
        from scipy.optimize import lsq_linear
        from scipy.signal import fftconvolve
    except ImportError:
        return ForegroundModelComparison(
            method="psf_convolved_fixed_geometry_mge",
            success=False,
            flags=["scipy_unavailable"],
        )

    psf = _normalized_psf(prep.image.psf)
    if psf is None:
        return ForegroundModelComparison(
            method="psf_convolved_fixed_geometry_mge",
            success=False,
            flags=["valid_psf_unavailable"],
        )

    data = np.asarray(prep.image.data, dtype=float)
    height, width = data.shape
    cy, cx = prep.nucleus_yx
    yy, xx = np.indices(data.shape)
    dx = xx - float(cx)
    dy = yy - float(cy)
    pa = float(prep.foreground_position_angle_deg or 0.0) * np.pi / 180.0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    axis_ratio = float(prep.foreground_axis_ratio or 1.0)
    q = float(np.clip(1.0 / max(axis_ratio, 1.0), 0.25, 1.0))
    elliptical_radius = np.sqrt(x_rot**2 + (y_rot / q) ** 2)

    sigmas = np.asarray([0.7, 1.1, 1.7, 2.6, 4.0, 6.2, 9.5, 14.5, 22.0, 34.0])
    components = []
    for sigma in sigmas:
        intrinsic = np.exp(-0.5 * np.square(elliptical_radius / sigma))
        components.append(fftconvolve(intrinsic, psf, mode="same"))

    fit_radius = min(50.0, 0.45 * min(height, width))
    valid = (
        np.isfinite(data)
        & ~prep.fit_mask
        & (np.hypot(dx, dy) <= fit_radius)
    )
    if int(valid.sum()) < len(components) * 8:
        return ForegroundModelComparison(
            method="psf_convolved_fixed_geometry_mge",
            success=False,
            flags=["too_few_unmasked_fit_pixels"],
        )

    design_full = np.column_stack(
        [component.ravel() for component in components]
        + [np.ones(data.size, dtype=float)]
    )
    selected = valid.ravel()
    coefficients = None
    try:
        for _ in range(3):
            design = design_full[selected]
            target = data.ravel()[selected]
            lower = np.concatenate([np.zeros(len(components)), [-np.inf]])
            upper = np.full(len(components) + 1, np.inf)
            result = lsq_linear(design, target, bounds=(lower, upper), max_iter=300)
            coefficients = result.x
            residual_fit = target - design @ coefficients
            scale = robust_rms(residual_fit)
            if scale <= 0:
                break
            keep_local = np.abs(residual_fit - np.median(residual_fit)) <= 3.5 * scale
            selected_indices = np.flatnonzero(selected)
            updated = np.zeros_like(selected)
            updated[selected_indices[keep_local]] = True
            if int(updated.sum()) == int(selected.sum()):
                break
            selected = updated
        if coefficients is None or not np.all(np.isfinite(coefficients)):
            raise RuntimeError("non_finite_coefficients")
        model = (design_full @ coefficients).reshape(data.shape)
        residual = data - model
        return ForegroundModelComparison(
            method="psf_convolved_fixed_geometry_mge",
            success=True,
            model=model,
            residual=residual,
            residual_robust_rms=robust_rms(residual[valid]),
            coefficients=[float(value) for value in coefficients],
            gaussian_sigmas_px=[float(value) for value in sigmas],
            psf_applied=True,
            flags=["experimental_audit_only", "not_used_for_detection"],
        )
    except Exception as exc:
        return ForegroundModelComparison(
            method="psf_convolved_fixed_geometry_mge",
            success=False,
            psf_applied=True,
            flags=[f"fit_failed:{type(exc).__name__}"],
        )
