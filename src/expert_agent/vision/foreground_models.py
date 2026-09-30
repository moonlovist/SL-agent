from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..core.config import VisionConfig
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


def _selected_center(prep: PreprocessedImage) -> tuple[float, float]:
    selected = prep.center_selection.get("selected_centroid_xy")
    if (
        isinstance(selected, (list, tuple))
        and len(selected) == 2
        and np.all(np.isfinite(selected))
    ):
        return tuple(map(float, selected))
    return float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0])


def _background_bounds(
    prep: PreprocessedImage,
    cx: float,
    cy: float,
    min_radius: float,
) -> tuple[float, float]:
    """Constrain the additive sky term from distant, source-masked pixels."""
    yy, xx = np.indices(prep.image.data.shape)
    radius = np.hypot(xx - cx, yy - cy)
    valid = (
        (radius >= min_radius)
        & np.isfinite(prep.image.data)
        & np.isfinite(prep.image.rms)
        & (prep.image.rms > 0)
        & ~prep.source_mask
        & ~prep.fit_mask
    )
    if int(valid.sum()) >= 100:
        sky = float(np.median(prep.image.data[valid]))
        noise = float(np.median(prep.image.rms[valid]))
        scatter = float(robust_rms(prep.image.data[valid]))
    else:
        sky = float(prep.background_median)
        noise = float(prep.background_rms)
        scatter = noise
    # The sky is not another galaxy component.  A half-pixel-noise allowance
    # absorbs modest background uncertainty while preventing a broad profile
    # plus a free constant from shifting the whole residual by several sigma.
    tolerance = max(0.5 * max(noise, 1e-8), 0.1 * max(scatter, 1e-8))
    return sky - tolerance, sky + tolerance


def _selected_foreground_pixels(prep: PreprocessedImage) -> np.ndarray:
    """Centrosymmetric pixels assigned to the single pre-fit foreground.

    A deblender can join a one-sided lensed arc to the selected galaxy.  The
    complete segment must therefore not be treated as guaranteed foreground
    light.  For geometry and robust-fit retention we keep the central core and
    the part of the segment that has a counterpart after a 180-degree rotation
    about the selected centroid.  This matches the leading-order symmetry of
    ordinary galaxy light without asserting that every galaxy is perfectly
    symmetric or using the test as lens evidence.
    """
    label = prep.center_selection.get("selected_sep_label")
    if not isinstance(label, int):
        return np.zeros(prep.image.data.shape, dtype=bool)
    selected = np.asarray(prep.segmentation) == label
    if prep.source_protection_weight is not None:
        selected &= np.asarray(prep.source_protection_weight) > 0
    centroid = prep.center_selection.get("selected_centroid_xy")
    if not (
        isinstance(centroid, (list, tuple))
        and len(centroid) == 2
        and np.all(np.isfinite(centroid))
    ):
        return selected
    cx, cy = map(float, centroid)
    yy, xx = np.indices(selected.shape)
    reflected_y = np.rint(2.0 * cy - yy).astype(int)
    reflected_x = np.rint(2.0 * cx - xx).astype(int)
    inside = (
        (reflected_y >= 0)
        & (reflected_y < selected.shape[0])
        & (reflected_x >= 0)
        & (reflected_x < selected.shape[1])
    )
    reflected_selected = np.zeros_like(selected)
    reflected_selected[inside] = selected[reflected_y[inside], reflected_x[inside]]
    symmetric = selected & reflected_selected
    # Preserve the fitted nucleus even when pixelisation makes its reflected
    # mask imperfect; outside the core, only paired light constrains geometry.
    core = np.hypot(xx - cx, yy - cy) <= 3.0
    return selected & (symmetric | core)


def fit_psf_mge_comparison(prep: PreprocessedImage) -> ForegroundModelComparison:
    """Fit a conservative, fixed-geometry PSF-convolved MGE candidate.

    The caller decides whether it is the detection model or the sensitivity
    comparison.  It is fitted to the same exposure, so it is never described
    as independent evidence.
    """

    try:
        from scipy.optimize import lsq_linear
        from scipy.signal import fftconvolve
    except ImportError:
        return ForegroundModelComparison(
            method="psf_convolved_robust_fixed_geometry_mge",
            success=False,
            flags=["scipy_unavailable"],
        )

    psf = _normalized_psf(prep.image.psf)
    if psf is None:
        return ForegroundModelComparison(
            method="psf_convolved_robust_fixed_geometry_mge",
            success=False,
            flags=["valid_psf_unavailable"],
        )

    data = np.asarray(prep.image.data, dtype=float)
    height, width = data.shape
    # The integer peak is useful for display and array indexing, but a
    # PSF-scale light model must not be forced onto that pixel centre.  SEP's
    # flux-weighted centroid is retained by the centre-selection audit.
    selected_centroid = prep.center_selection.get("selected_centroid_xy")
    if (
        isinstance(selected_centroid, (list, tuple))
        and len(selected_centroid) == 2
        and np.all(np.isfinite(selected_centroid))
    ):
        cx, cy = map(float, selected_centroid)
    else:
        cy, cx = map(float, prep.nucleus_yx)
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
        & np.isfinite(prep.image.rms) & (prep.image.rms > 0)
        & ~prep.fit_mask
        & (np.hypot(dx, dy) <= fit_radius)
    )
    protection = prep.source_protection_weight
    if protection is not None:
        valid &= np.asarray(protection) > 0
    if int(valid.sum()) < len(components) * 8:
        return ForegroundModelComparison(
            method="psf_convolved_robust_fixed_geometry_mge",
            success=False,
            flags=["too_few_unmasked_fit_pixels"],
        )

    design_full = np.column_stack(
        [component.ravel() for component in components]
        + [np.ones(data.size, dtype=float)]
    )
    selected = valid.ravel()
    foreground_pixels = _selected_foreground_pixels(prep).ravel()
    coefficients = None
    try:
        for _ in range(3):
            noise = prep.image.rms.ravel()[selected]
            if protection is not None:
                noise = noise / np.sqrt(
                    np.clip(np.asarray(protection).ravel()[selected], 1e-3, 1.0)
                )
            design = design_full[selected] / noise[:, None]
            target = data.ravel()[selected] / noise
            sky_low, sky_high = _background_bounds(
                prep, cx, cy, min_radius=37.5
            )
            lower = np.concatenate([np.zeros(len(components)), [sky_low]])
            upper = np.concatenate([np.full(len(components), np.inf), [sky_high]])
            result = lsq_linear(design, target, bounds=(lower, upper), max_iter=300)
            if not result.success:
                raise RuntimeError("optimizer_did_not_converge")
            coefficients = result.x
            residual_fit = target - design @ coefficients
            scale = robust_rms(residual_fit)
            if scale <= 0:
                break
            keep_local = np.abs(residual_fit - np.median(residual_fit)) <= 3.5 * scale
            selected_indices = np.flatnonzero(selected)
            # Retain only the centrosymmetric part of the selected foreground;
            # a one-sided arc blended into the same SEP segment stays clip-able.
            keep_local |= foreground_pixels[selected_indices]
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
            method="psf_convolved_robust_fixed_geometry_mge",
            success=True,
            model=model,
            residual=residual,
            residual_robust_rms=robust_rms(residual[valid]),
            coefficients=[float(value) for value in coefficients],
            gaussian_sigmas_px=[float(value) for value in sigmas],
            psf_applied=True,
            flags=["experimental_audit_only", "not_used_for_detection", "input_rms_weighted", "same_data_not_independent"],
        )
    except Exception as exc:
        return ForegroundModelComparison(
            method="psf_convolved_robust_fixed_geometry_mge",
            success=False,
            psf_applied=True,
            flags=[f"fit_failed:{type(exc).__name__}"],
        )


def promote_psf_mge_primary(
    prep: PreprocessedImage,
    mge_fit: ForegroundModelComparison,
    *,
    ring_outer_px: float,
    near_core_radius_px: float,
) -> tuple[PreprocessedImage, ForegroundModelComparison]:
    """Use a successful PSF-aware MGE for detection and retain the old model.

    The returned comparison is the pre-existing isophotal/radial subtraction.
    Keeping it as an audit channel makes model dependence visible after the
    PSF-aware model becomes primary; it is not an independent exposure.
    """

    if not mge_fit.success or mge_fit.model is None or mge_fit.residual is None:
        return prep, mge_fit

    old_comparison = ForegroundModelComparison(
        method=prep.model_method,
        success=bool(np.all(np.isfinite(prep.model)) and np.all(np.isfinite(prep.residual))),
        model=np.asarray(prep.model, dtype=float).copy(),
        residual=np.asarray(prep.residual, dtype=float).copy(),
        residual_robust_rms=float(prep.residual_robust_rms),
        psf_applied=False,
        flags=[
            "audit_only_alternative_foreground_model",
            "not_used_for_detection",
            "same_data_not_independent",
            *( [] if prep.model_success else ["alternative_model_reported_fit_failure"] ),
        ],
    )

    residual = np.asarray(mge_fit.residual, dtype=float)
    rms = np.asarray(prep.image.rms, dtype=float)
    residual_snr = np.divide(
        residual,
        rms,
        out=np.zeros_like(residual),
        where=np.isfinite(rms) & (rms > 0),
    )
    yy, xx = np.indices(residual.shape)
    cy, cx = prep.nucleus_yx
    radius = np.hypot(xx - cx, yy - cy)
    background_region = (radius > ring_outer_px * 1.5) & ~prep.source_mask
    if not background_region.any():
        background_region = ~prep.source_mask
    core_region = radius <= near_core_radius_px
    comparison_region = radius <= ring_outer_px
    denominator = float(np.sum(np.abs(residual[comparison_region])))
    core_fraction = (
        float(np.sum(np.abs(residual[core_region])) / denominator)
        if denominator > 0
        else 0.0
    )
    symmetric_half_y = min(cy, residual.shape[0] - 1 - cy)
    symmetric_half_x = min(cx, residual.shape[1] - 1 - cx)
    tidal_asymmetry = prep.tidal_asymmetry
    if symmetric_half_y >= 1 and symmetric_half_x >= 1:
        symmetric = residual[
            cy - symmetric_half_y : cy + symmetric_half_y + 1,
            cx - symmetric_half_x : cx + symmetric_half_x + 1,
        ]
        rotated = np.rot90(symmetric, 2)
        asymmetry_denominator = float(
            np.sum(np.abs(symmetric)) + np.sum(np.abs(rotated))
        )
        tidal_asymmetry = (
            float(np.sum(np.abs(symmetric - rotated)) / asymmetry_denominator)
            if asymmetry_denominator > 0
            else 0.0
        )

    retained_flags = [
        flag
        for flag in prep.flags
        if not flag.startswith("ellipse_model_failed:")
    ]
    retained_flags.append("psf_mge_primary_with_isophote_sensitivity_audit")
    promoted = replace(
        prep,
        model=np.asarray(mge_fit.model, dtype=float),
        residual=residual,
        residual_snr=residual_snr,
        residual_median=float(np.median(residual[background_region])),
        residual_robust_rms=robust_rms(residual[background_region]),
        model_method=mge_fit.method,
        model_success=True,
        tidal_asymmetry=tidal_asymmetry,
        core_residual_fraction=core_fraction,
        flags=retained_flags,
    )
    return promoted, old_comparison


def build_source_protection(
    prep: PreprocessedImage, config: VisionConfig
) -> tuple[np.ndarray, dict]:
    """Fit one continuous foreground/source boundary from radial minima.

    A minimum is considered only when a radial profile contains both the
    declining foreground light on its inner side and a resolved outer rise.
    Adjacent angular samples must agree in radius; isolated minima are rejected
    instead of producing isolated wedge weights.  A supported radius defines a
    closed ellipse using the foreground's inner light geometry.  Pixels inside
    the ellipse constrain the foreground model, the ambiguous source annulus is
    excluded, and a distant sky annulus remains available for the background
    term.  The model itself is still evaluated over the full image.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return np.ones(prep.image.data.shape, dtype=float), {
            "method": "continuous_elliptical_radial_minimum_boundary",
            "boundary_detected": False,
            "flags": ["scipy_unavailable"],
        }

    data = np.asarray(prep.image.data, dtype=float)
    rms = np.asarray(prep.image.rms, dtype=float)
    selected = prep.center_selection.get("selected_centroid_xy")
    if isinstance(selected, (list, tuple)) and len(selected) == 2:
        cx, cy = map(float, selected)
    else:
        cy, cx = map(float, prep.nucleus_yx)
    yy, xx = np.indices(data.shape)
    dx, dy = xx - cx, yy - cy
    circular_radius = np.hypot(dx, dy)
    pa = np.radians(float(prep.foreground_position_angle_deg or 0.0))
    q = float(np.clip(1.0 / max(float(prep.foreground_axis_ratio or 1.0), 1.0), 0.25, 1.0))
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    xrot = dx * cos_pa + dy * sin_pa
    yrot = -dx * sin_pa + dy * cos_pa
    elliptical_radius = np.sqrt(xrot * xrot + (yrot / q) ** 2)

    bins = max(int(config.source_boundary_angle_bins), 24)
    rmax = float(max(config.ring_outer_px, config.near_core_radius_px + 3.0))
    radial_step = 0.5
    sample_radii = np.arange(0.0, rmax + radial_step, radial_step)
    sample_angles = np.arange(bins, dtype=float) * 2.0 * np.pi / bins
    # A unit radius in this coordinate traces the measured foreground ellipse.
    xr = sample_radii[None, :] * np.cos(sample_angles)[:, None]
    yr = q * sample_radii[None, :] * np.sin(sample_angles)[:, None]
    sample_x = cx + xr * cos_pa - yr * sin_pa
    sample_y = cy + xr * sin_pa + yr * cos_pa
    smoothed = ndimage.gaussian_filter(data, 1.0)
    polar = ndimage.map_coordinates(smoothed, [sample_y, sample_x], order=1, mode="nearest")
    polar = ndimage.gaussian_filter(polar, sigma=(1.0, 1.0), mode=("wrap", "nearest"))
    polar_rms = ndimage.map_coordinates(rms, [sample_y, sample_x], order=1, mode="nearest")

    start_radius = max(3.0, 0.65 * float(config.near_core_radius_px))
    start_index = int(np.searchsorted(sample_radii, start_radius))
    end_index = int(np.searchsorted(sample_radii, rmax - 1.0))
    candidates: list[dict | None] = []
    for angle_index in range(bins):
        profile = polar[angle_index]
        noise_profile = np.maximum(polar_rms[angle_index], 1e-8)
        angle_candidates = []
        for valley_index in range(start_index, end_index):
            if not (
                profile[valley_index] <= profile[valley_index - 1]
                and profile[valley_index] <= profile[valley_index + 1]
            ):
                continue
            inner_start = max(1, valley_index - int(round(6.0 / radial_step)))
            outer_stop = min(
                profile.size,
                valley_index + int(round(10.0 / radial_step)) + 1,
            )
            inner_peak = float(np.max(profile[inner_start:valley_index]))
            outer_slice = profile[valley_index + 1 : outer_stop]
            if not outer_slice.size:
                continue
            outer_relative = int(np.argmax(outer_slice))
            outer_peak_index = valley_index + 1 + outer_relative
            outer_peak = float(profile[outer_peak_index])
            valley = float(profile[valley_index])
            local_noise = float(np.median(noise_profile[max(0, valley_index - 1) : valley_index + 2]))
            inner_sigma = (inner_peak - valley) / local_noise
            outer_sigma = (outer_peak - valley) / local_noise
            significance = min(inner_sigma, outer_sigma)
            # Both sides must be brighter.  This distinguishes a source/light
            # separation valley from the ordinary monotonic wing of a galaxy.
            if significance >= float(config.source_boundary_min_contrast_sigma):
                angle_candidates.append(
                    {
                        "radius_px": float(sample_radii[valley_index]),
                        "outer_peak_radius_px": float(sample_radii[outer_peak_index]),
                        "significance_sigma": float(significance),
                    }
                )
        candidates.append(
            max(angle_candidates, key=lambda item: item["significance_sigma"])
            if angle_candidates
            else None
        )

    valid = np.asarray([item is not None for item in candidates], dtype=bool)
    # Find circular runs of adjacent supported profiles.  A radius jump larger
    # than the tolerance also starts a new run, preventing unrelated minima
    # from being joined into a boundary.
    runs: list[list[int]] = []
    if valid.any():
        starts = [
            index
            for index in range(bins)
            if valid[index]
            and (
                not valid[(index - 1) % bins]
                or abs(
                    candidates[index]["radius_px"]
                    - candidates[(index - 1) % bins]["radius_px"]
                )
                > float(config.source_boundary_radius_tolerance_px)
            )
        ]
        if not starts and valid.all():
            starts = [0]
        visited: set[int] = set()
        for start in starts:
            run = []
            index = start
            while index not in visited and valid[index]:
                if run and abs(
                    candidates[index]["radius_px"] - candidates[run[-1]]["radius_px"]
                ) > float(config.source_boundary_radius_tolerance_px):
                    break
                visited.add(index)
                run.append(index)
                index = (index + 1) % bins
            if run:
                runs.append(run)

    min_support = max(2, int(config.source_boundary_min_angular_support_bins))
    supported_runs = [run for run in runs if len(run) >= min_support]
    weight = np.ones(data.shape, dtype=float)
    boundary_radius = None
    selected_run: list[int] = []
    if supported_runs:
        selected_run = max(
            supported_runs,
            key=lambda run: (
                len(run),
                np.median([candidates[index]["significance_sigma"] for index in run]),
            ),
        )
        boundary_radius = float(
            np.median([candidates[index]["radius_px"] for index in selected_run])
        )
        sky_start = float(config.ring_outer_px) * float(config.source_boundary_sky_start_scale)
        ambiguous_annulus = (elliptical_radius > boundary_radius) & (circular_radius < sky_start)
        weight[ambiguous_annulus] = 0.0

    fallback_radius = None
    fallback_used = False
    if not supported_runs and prep.human_foreground_seed is not None:
        seed = np.asarray(prep.human_foreground_seed, dtype=bool)
        seed_radii = elliptical_radius[seed]
        if seed_radii.size:
            fallback_radius = float(
                np.clip(
                    np.percentile(seed_radii, 90.0),
                    max(4.0, float(config.near_core_radius_px)),
                    float(config.ring_outer_px),
                )
            )
            sky_start = float(config.ring_outer_px) * float(
                config.source_boundary_sky_start_scale
            )
            ambiguous_annulus = (
                (elliptical_radius > fallback_radius)
                & (circular_radius < sky_start)
            )
            weight[ambiguous_annulus] = 0.0
            fallback_used = True

    audit = {
        "method": "continuous_elliptical_radial_minimum_boundary",
        "angle_bins": bins,
        "radial_step_px": radial_step,
        "candidate_minimum_count": int(valid.sum()),
        "supported_run_lengths": [len(run) for run in supported_runs],
        "selected_angle_indices": selected_run,
        "selected_angular_support_deg": float(len(selected_run) * 360.0 / bins),
        "boundary_detected": boundary_radius is not None,
        "boundary_radius_px": boundary_radius,
        "human_core_fallback_used": fallback_used,
        "human_core_fallback_radius_px": fallback_radius,
        "boundary_axis_ratio_q": q,
        "boundary_position_angle_deg": float(np.degrees(pa) % 180.0),
        "minimum_contrast_sigma": (
            None
            if not selected_run
            else float(np.median([candidates[index]["significance_sigma"] for index in selected_run]))
        ),
        "fit_policy": (
            "inside_continuous_boundary_plus_distant_sky; ambiguous_annulus_excluded"
            if boundary_radius is not None
            else "human_core_seed_plus_distant_sky; ambiguous_annulus_excluded"
            if fallback_used
            else "no_boundary_fallback_available"
        ),
        "isolated_sector_weights_removed": True,
        "flags": [] if boundary_radius is not None else ["no_supported_foreground_source_minimum"],
    }
    return weight, audit


def _geometry_from_moments(
    prep: PreprocessedImage, cx: float, cy: float, rmin: float, rmax: float
) -> tuple[float, float]:
    data = np.asarray(prep.image.data, float)
    yy, xx = np.indices(data.shape)
    dx, dy = xx - cx, yy - cy
    radius = np.hypot(dx, dy)
    valid = (radius >= rmin) & (radius <= rmax) & ~prep.fit_mask
    foreground_pixels = _selected_foreground_pixels(prep)
    selected_valid = valid & foreground_pixels
    # Geometry follows the selected foreground's centrosymmetric light.  Using
    # every positive annulus pixel circularises edge-on discs through sky noise;
    # using the whole selected segment can instead absorb a blended one-sided
    # arc.  Fall back only when paired foreground pixels are too sparse.
    if int(selected_valid.sum()) >= 12:
        valid = selected_valid
    values = np.clip(data - prep.background_median, 0.0, None)
    if prep.source_protection_weight is not None:
        values = values * np.asarray(prep.source_protection_weight)
    values = np.where(valid, values, 0.0)
    total = float(values.sum())
    if total <= 0:
        q = 1.0 / max(float(prep.foreground_axis_ratio or 1.0), 1.0)
        return float(np.clip(q, 0.25, 1.0)), np.radians(float(prep.foreground_position_angle_deg or 0.0))
    covariance = np.array([
        [np.sum(values * dx * dx), np.sum(values * dx * dy)],
        [np.sum(values * dx * dy), np.sum(values * dy * dy)],
    ]) / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major, minor = eigenvalues[order[0]], eigenvalues[order[1]]
    vector = eigenvectors[:, order[0]]
    q = np.sqrt(max(minor, 1e-8) / max(major, 1e-8))
    pa = np.arctan2(vector[1], vector[0])
    return float(np.clip(q, 0.25, 1.0)), float(pa)


def _elliptical_radius(shape, cx, cy, q, pa):
    yy, xx = np.indices(shape)
    dx, dy = xx - cx, yy - cy
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    xrot = dx * cos_pa + dy * sin_pa
    yrot = -dx * sin_pa + dy * cos_pa
    return np.sqrt(xrot * xrot + (yrot / max(q, 0.2)) ** 2)


def _linear_fit(
    prep: PreprocessedImage,
    components: list[np.ndarray],
    valid: np.ndarray,
    sky_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    from scipy.optimize import lsq_linear

    selected = valid.ravel().copy()
    if prep.source_protection_weight is not None:
        selected &= np.asarray(prep.source_protection_weight).ravel() > 0
    foreground_pixels = _selected_foreground_pixels(prep).ravel()
    design_full = np.column_stack(
        [component.ravel() for component in components]
        + [np.ones(prep.image.data.size)]
    )
    coefficients = None
    for _ in range(3):
        noise = prep.image.rms.ravel()[selected]
        if prep.source_protection_weight is not None:
            noise = noise / np.sqrt(np.clip(prep.source_protection_weight.ravel()[selected], 1e-3, 1.0))
        design = design_full[selected] / noise[:, None]
        target = prep.image.data.ravel()[selected] / noise
        lower = np.r_[np.zeros(len(components)), sky_bounds[0]]
        upper = np.r_[np.full(len(components), np.inf), sky_bounds[1]]
        result = lsq_linear(design, target, bounds=(lower, upper), max_iter=200)
        if not result.success or not np.all(np.isfinite(result.x)):
            return None
        coefficients = result.x
        residual = target - design @ coefficients
        scale = robust_rms(residual)
        if scale <= 0:
            break
        keep = np.abs(residual - np.median(residual)) <= 3.5 * scale
        indices = np.flatnonzero(selected)
        keep |= foreground_pixels[indices]
        updated = np.zeros_like(selected)
        updated[indices[keep]] = True
        if updated.sum() == selected.sum():
            break
        selected = updated
    if coefficients is None:
        return None
    return coefficients, (design_full @ coefficients).reshape(prep.image.data.shape)


def fit_double_geometry_mge(
    prep: PreprocessedImage, config: VisionConfig
) -> ForegroundModelComparison:
    try:
        from scipy.signal import fftconvolve
    except ImportError:
        return ForegroundModelComparison("psf_convolved_double_geometry_mge", False, flags=["scipy_unavailable"])
    psf = _normalized_psf(prep.image.psf)
    if psf is None:
        return ForegroundModelComparison("psf_convolved_double_geometry_mge", False, flags=["valid_psf_unavailable"])
    selected = prep.center_selection.get("selected_centroid_xy")
    cx, cy = map(float, selected) if isinstance(selected, (list, tuple)) and len(selected) == 2 else (float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0]))
    q1, pa1 = _geometry_from_moments(prep, cx, cy, 0.0, 7.0)
    q2, pa2 = _geometry_from_moments(prep, cx, cy, 6.0, min(30.0, config.ring_outer_px * 1.2))
    sigmas1 = [0.7, 1.1, 1.8, 2.8, 4.3, 6.5]
    sigmas2 = [4.0, 6.5, 10.0, 15.5, 24.0, 36.0]
    r1 = _elliptical_radius(prep.image.data.shape, cx, cy, q1, pa1)
    r2 = _elliptical_radius(prep.image.data.shape, cx, cy, q2, pa2)
    components = [fftconvolve(np.exp(-0.5 * (r1 / sigma) ** 2), psf, mode="same") for sigma in sigmas1]
    components += [fftconvolve(np.exp(-0.5 * (r2 / sigma) ** 2), psf, mode="same") for sigma in sigmas2]
    yy, xx = np.indices(prep.image.data.shape)
    radius = np.hypot(xx - cx, yy - cy)
    angle_bin = np.floor(((np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi) * 20).astype(int)
    valid = np.isfinite(prep.image.data) & (prep.image.rms > 0) & ~prep.fit_mask & (radius <= 50) & (angle_bin % 5 != 0)
    if valid.sum() < 100:
        return ForegroundModelComparison("psf_convolved_double_geometry_mge", False, flags=["too_few_unmasked_fit_pixels"])
    fitted = _linear_fit(
        prep,
        components,
        valid,
        _background_bounds(
            prep, cx, cy, min_radius=1.5 * float(config.ring_outer_px)
        ),
    )
    if fitted is None:
        return ForegroundModelComparison("psf_convolved_double_geometry_mge", False, psf_applied=True, flags=["fit_failed"])
    coefficients, model = fitted
    residual = prep.image.data - model
    return ForegroundModelComparison(
        method="psf_convolved_robust_double_geometry_mge",
        success=True,
        model=model,
        residual=residual,
        residual_robust_rms=robust_rms(residual[valid]),
        coefficients=[float(value) for value in coefficients],
        gaussian_sigmas_px=[*sigmas1, *sigmas2],
        psf_applied=True,
        flags=["input_rms_weighted", "nonnegative_amplitudes", "robust_clipping", "heldout_angular_sectors", "same_data_not_independent"],
    )


def fit_double_sersic(
    prep: PreprocessedImage, config: VisionConfig
) -> ForegroundModelComparison:
    """Fit a bounded, coarse double-Sersic bank with a shared centre."""
    try:
        from scipy.optimize import lsq_linear
        from scipy.signal import fftconvolve
    except ImportError:
        return ForegroundModelComparison("psf_convolved_bounded_grid_double_sersic", False, flags=["scipy_unavailable"])
    psf = _normalized_psf(prep.image.psf)
    if psf is None:
        return ForegroundModelComparison("psf_convolved_bounded_grid_double_sersic", False, flags=["valid_psf_unavailable"])
    selected = prep.center_selection.get("selected_centroid_xy")
    cx, cy = map(float, selected) if isinstance(selected, (list, tuple)) and len(selected) == 2 else (float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0]))
    q1, pa1 = _geometry_from_moments(prep, cx, cy, 0.0, 7.0)
    q2, pa2 = _geometry_from_moments(prep, cx, cy, 6.0, min(30.0, config.ring_outer_px * 1.2))
    r1 = _elliptical_radius(prep.image.data.shape, cx, cy, q1, pa1)
    r2 = _elliptical_radius(prep.image.data.shape, cx, cy, q2, pa2)
    core_grid = [(re, n) for re in (2.0, 4.0, 7.0) for n in (2.0, 4.0, 6.0)]
    outer_grid = [(re, n) for re in (7.0, 12.0, 20.0, 30.0) for n in (0.7, 1.0, 2.0, 4.0)]
    def component(radius, re, n):
        bn = 2.0 * n - 1.0 / 3.0
        intrinsic = np.exp(-bn * (np.power(np.maximum(radius / re, 1e-6), 1.0 / n) - 1.0))
        return fftconvolve(intrinsic, psf, mode="same")
    core_components = [component(r1, re, n) for re, n in core_grid]
    outer_components = [component(r2, re, n) for re, n in outer_grid]
    yy, xx = np.indices(prep.image.data.shape)
    radius = np.hypot(xx - cx, yy - cy)
    angle_bin = np.floor(((np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi) * 20).astype(int)
    train = np.isfinite(prep.image.data) & (prep.image.rms > 0) & ~prep.fit_mask & (radius <= 50) & (angle_bin % 5 != 0)
    validate = np.isfinite(prep.image.data) & (prep.image.rms > 0) & ~prep.fit_mask & (radius <= 50) & (angle_bin % 5 == 0)
    if prep.source_protection_weight is not None:
        included = np.asarray(prep.source_protection_weight) > 0
        train &= included
        validate &= included
    if train.sum() < 100 or validate.sum() < 20:
        return ForegroundModelComparison("psf_convolved_bounded_grid_double_sersic", False, flags=["too_few_unmasked_fit_pixels"])
    best = None
    data_flat = prep.image.data.ravel()
    noise_flat = prep.image.rms.ravel()
    protection = prep.source_protection_weight.ravel() if prep.source_protection_weight is not None else np.ones(data_flat.size)
    for i, core in enumerate(core_components):
        for j, outer in enumerate(outer_components):
            design_full = np.column_stack([core.ravel(), outer.ravel(), np.ones(data_flat.size)])
            noise = noise_flat[train.ravel()] / np.sqrt(np.clip(protection[train.ravel()], 1e-3, 1.0))
            sky_low, sky_high = _background_bounds(
                prep, cx, cy, min_radius=1.5 * float(config.ring_outer_px)
            )
            result = lsq_linear(
                design_full[train.ravel()] / noise[:, None],
                data_flat[train.ravel()] / noise,
                bounds=([0, 0, sky_low], [np.inf, np.inf, sky_high]),
                max_iter=80,
            )
            if not result.success:
                continue
            model = (design_full @ result.x).reshape(prep.image.data.shape)
            normalized = (prep.image.data - model)[validate] / prep.image.rms[validate]
            score = robust_rms(normalized * np.sqrt(np.clip(prep.source_protection_weight[validate] if prep.source_protection_weight is not None else 1.0, 1e-3, 1.0)))
            key = (score, i, j)
            if best is None or key < best[0]:
                best = (key, result.x, model, i, j)
    if best is None:
        return ForegroundModelComparison("psf_convolved_bounded_grid_double_sersic", False, psf_applied=True, flags=["fit_failed"])
    _, coefficients, model, i, j = best
    residual = prep.image.data - model
    re1, n1 = core_grid[i]
    re2, n2 = outer_grid[j]
    return ForegroundModelComparison(
        method="psf_convolved_bounded_grid_double_sersic",
        success=True,
        model=model,
        residual=residual,
        residual_robust_rms=robust_rms(residual[train]),
        coefficients=[float(coefficients[0]), float(coefficients[1]), float(coefficients[2]), re1, n1, re2, n2],
        psf_applied=True,
        flags=["shared_subpixel_centroid", "bounded_sersic_indices", "heldout_angular_sectors", "source_protection_weighted", "same_data_not_independent"],
    )


def _model_score(
    prep: PreprocessedImage,
    fit: ForegroundModelComparison,
    config: VisionConfig,
) -> float:
    if not fit.success or fit.residual is None or not np.all(np.isfinite(fit.residual)):
        return float("inf")
    selected = prep.center_selection.get("selected_centroid_xy")
    cx, cy = map(float, selected) if isinstance(selected, (list, tuple)) and len(selected) == 2 else (float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0]))
    yy, xx = np.indices(prep.image.data.shape)
    radius = np.hypot(xx - cx, yy - cy)
    angle_bin = np.floor(((np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi) * 20).astype(int)
    validation = (radius <= 50) & ~prep.fit_mask & (angle_bin % 5 == 0) & np.isfinite(fit.residual) & (prep.image.rms > 0)
    if prep.source_protection_weight is not None:
        validation &= np.asarray(prep.source_protection_weight) > 0
    if validation.sum() < 20:
        return float("inf")
    weight = prep.source_protection_weight[validation] if prep.source_protection_weight is not None else 1.0
    normalized = fit.residual[validation] / prep.image.rms[validation] * np.sqrt(np.clip(weight, 1e-3, 1.0))
    scale = robust_rms(normalized)
    quality = _foreground_model_quality(prep, fit, config)
    background_bias = abs(float(quality.get("background_median_sigma") or 0.0))
    radial_bias = abs(float(quality.get("annular_median_sigma") or 0.0))
    dipole = float(quality.get("core_dipole_scale_sigma") or 0.0)
    complexity = max(len(fit.coefficients) - 1, 1)
    return float(
        scale
        + 0.70 * background_bias
        + 0.25 * radial_bias
        + 0.08 * dipole
        + 0.002 * complexity
    )


def _foreground_model_quality(
    prep: PreprocessedImage,
    fit: ForegroundModelComparison,
    config: VisionConfig,
) -> dict[str, object]:
    """Measure residual failures without declaring any residual to be a lens."""
    if not fit.success or fit.residual is None:
        return {
            "quality_pass": False,
            "core_residual_unreliable": True,
            "failure_reasons": ["fit_not_available"],
        }
    residual = np.asarray(fit.residual, dtype=float)
    rms = np.asarray(prep.image.rms, dtype=float)
    cx, cy = _selected_center(prep)
    yy, xx = np.indices(residual.shape)
    radius = np.hypot(xx - cx, yy - cy)
    finite = np.isfinite(residual) & np.isfinite(rms) & (rms > 0)
    normalized = np.divide(
        residual, rms, out=np.zeros_like(residual), where=finite
    )
    background = (
        finite
        & (radius >= 1.5 * float(config.ring_outer_px))
        & ~prep.source_mask
        & ~prep.fit_mask
    )
    if int(background.sum()) < 100:
        background = finite & (radius >= float(config.ring_outer_px))
    annulus = (
        finite
        & (radius >= float(config.near_core_radius_px))
        & (radius <= float(config.ring_outer_px))
        & ~prep.fit_mask
    )
    if prep.human_fit_exclude is not None:
        annulus &= ~np.asarray(prep.human_fit_exclude, dtype=bool)
    background_median = (
        float(np.median(normalized[background])) if background.any() else None
    )
    background_scale = (
        float(robust_rms(normalized[background])) if background.any() else None
    )
    annular_median = (
        float(np.median(normalized[annulus])) if annulus.any() else None
    )

    core = finite & (radius <= float(config.near_core_radius_px))
    reflected_y = 2.0 * cy - yy[core]
    reflected_x = 2.0 * cx - xx[core]
    if int(core.sum()) >= 6:
        from scipy import ndimage

        reflected_residual = ndimage.map_coordinates(
            residual, [reflected_y, reflected_x], order=1, mode="nearest"
        )
        reflected_rms = ndimage.map_coordinates(
            rms, [reflected_y, reflected_x], order=1, mode="nearest"
        )
        denominator = np.sqrt(
            np.square(rms[core]) + np.square(reflected_rms)
        )
        dipole_values = np.divide(
            residual[core] - reflected_residual,
            denominator,
            out=np.zeros_like(reflected_residual),
            where=denominator > 0,
        )
        core_dipole = float(robust_rms(dipole_values))
        core_positive_fraction = float(np.mean(normalized[core] >= 3.0))
        core_negative_fraction = float(np.mean(normalized[core] <= -3.0))
    else:
        core_dipole = None
        core_positive_fraction = 0.0
        core_negative_fraction = 0.0

    reasons: list[str] = []
    if background_median is None or abs(background_median) > 1.0:
        reasons.append("background_residual_not_zero_centered")
    if background_scale is None or not (0.25 <= background_scale <= 3.0):
        reasons.append("background_residual_scale_inconsistent")
    if annular_median is None or abs(annular_median) > 2.0:
        reasons.append("broad_annular_residual_bias")
    core_unreliable = bool(
        core_dipole is None
        or core_dipole > 2.5
        or (
            core_positive_fraction >= 0.05
            and core_negative_fraction >= 0.05
        )
    )
    return {
        "quality_pass": not reasons,
        "background_median_sigma": background_median,
        "background_robust_scale_sigma": background_scale,
        "annular_median_sigma": annular_median,
        "core_dipole_scale_sigma": core_dipole,
        "core_positive_3sigma_fraction": core_positive_fraction,
        "core_negative_3sigma_fraction": core_negative_fraction,
        "core_residual_unreliable": core_unreliable,
        "failure_reasons": reasons,
    }


def fit_foreground_model_bank(
    prep: PreprocessedImage, config: VisionConfig
) -> tuple[PreprocessedImage, ForegroundModelComparison]:
    """Fit the approved foreground bank and expose one deterministic primary."""
    weight, boundary_audit = build_source_protection(prep, config)
    weighted = replace(prep, source_protection_weight=weight)
    fixed = fit_psf_mge_comparison(weighted)
    double_mge = fit_double_geometry_mge(weighted, config)
    double_sersic = fit_double_sersic(weighted, config)
    isophote = ForegroundModelComparison(
        method=weighted.model_method,
        success=bool(np.all(np.isfinite(weighted.model)) and np.all(np.isfinite(weighted.residual))),
        model=np.asarray(weighted.model, float).copy(),
        residual=np.asarray(weighted.residual, float).copy(),
        residual_robust_rms=float(weighted.residual_robust_rms),
        flags=["same_data_not_independent", "isophote_model_bank_candidate"],
    )
    # The preprocessing isophote was fitted before the minimum boundary was
    # known.  It remains a valid candidate only when no boundary is detected;
    # otherwise allowing it to win would silently bypass the requested fit
    # exclusion.
    boundary_detected = bool(boundary_audit.get("boundary_detected"))
    candidates = [double_mge, double_sersic, fixed]
    # A preprocessing isophote was fitted before a human override existed and
    # cannot be allowed to bypass the human centre or do-not-fit mask.
    human_override = "human_foreground_override" in weighted.flags
    if not boundary_detected and not human_override:
        candidates.insert(2, isophote)
    scored = [
        (
            float(_model_score(weighted, candidate, config)),
            index,
            candidate,
            _foreground_model_quality(weighted, candidate, config),
        )
        for index, candidate in enumerate(candidates)
    ]
    scored.sort(key=lambda item: (item[0], item[1]))
    acceptable = [item for item in scored if item[3].get("quality_pass")]
    best_score, _, best, best_quality = (
        acceptable[0] if acceptable else scored[0]
    )
    audit = {
        "selection_method": "heldout_residual_with_background_annulus_and_core_quality_gates",
        "source_boundary": boundary_audit,
        "candidates": [
            {
                "method": candidate.method,
                "success": bool(candidate.success),
                "selection_score": None if not np.isfinite(score) else float(score),
                "quality": quality,
            }
            for score, _, candidate, quality in scored
        ],
        "selected_method": best.method,
        "selected_score": None if not np.isfinite(best_score) else float(best_score),
        "selected_quality": best_quality,
        "quality_pass": bool(acceptable),
        "single_primary_output": True,
    }
    weighted = replace(weighted, foreground_fit_candidates=audit)
    if best is isophote or not np.isfinite(best_score):
        runner_up = next((candidate for score, _, candidate, _ in scored if candidate is not isophote and np.isfinite(score)), fixed)
        return weighted, runner_up
    promoted, comparison = promote_psf_mge_primary(
        weighted,
        best,
        ring_outer_px=config.ring_outer_px,
        near_core_radius_px=config.near_core_radius_px,
    )
    promoted = replace(promoted, foreground_fit_candidates=audit, source_protection_weight=weight)
    if bool(best_quality.get("core_residual_unreliable")):
        promoted = replace(
            promoted,
            flags=[*promoted.flags, "foreground_core_residual_unreliable"],
        )
    if not acceptable:
        promoted = replace(
            promoted,
            model_success=False,
            flags=[*promoted.flags, "foreground_model_quality_failed"],
        )
    return promoted, comparison
