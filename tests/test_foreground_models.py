from pathlib import Path

import numpy as np

from expert_agent.core.config import VisionConfig
from expert_agent.vision.foreground_models import (
    ForegroundModelComparison,
    _foreground_model_quality,
    _geometry_from_moments,
    _selected_foreground_pixels,
    build_source_protection,
    fit_psf_mge_comparison,
    promote_psf_mge_primary,
)
from expert_agent.vision.preprocess import ImageData, PreprocessedImage


def test_psf_mge_comparison_recovers_smooth_central_source() -> None:
    size = 61
    yy, xx = np.indices((size, size))
    radius2 = (xx - 30.0) ** 2 + (yy - 30.0) ** 2
    data = 3.0 * np.exp(-0.5 * radius2 / 4.0**2) + 0.2
    psf = np.zeros((9, 9), dtype=float)
    psf[4, 4] = 1.0
    image = ImageData(
        path=Path("synthetic.fits"), object_id="synthetic",
        input_format="fits", band="VIS", data=data,
        rms=np.full_like(data, 0.01), supplied_rms=True,
        finite_fraction=1.0, wcs=None, pixel_scale_arcsec=None, psf=psf,
    )
    zeros = np.zeros_like(data)
    false_mask = np.zeros_like(data, dtype=bool)
    prep = PreprocessedImage(
        image=image, model=zeros, residual=data, residual_snr=zeros,
        source_mask=false_mask, fit_mask=false_mask,
        segmentation=np.zeros_like(data, dtype=np.int32), nucleus_yx=(30, 30),
        background_median=0.2, background_rms=0.01, residual_median=0.0,
        residual_robust_rms=0.01, model_method="synthetic", model_success=True,
        foreground_detected=True, foreground_peak_value=float(data.max()),
        foreground_integrated_flux=float(data.sum()), foreground_ellipticity=0.0,
        foreground_position_angle_deg=0.0, foreground_axis_ratio=1.0,
        foreground_reliability=1.0, multiple_core_count=1,
        tidal_asymmetry=0.0, core_residual_fraction=0.0, flags=[],
    )

    result = fit_psf_mge_comparison(prep)

    assert result.success
    assert result.psf_applied
    assert result.model is not None
    assert result.residual is not None
    assert result.residual_robust_rms is not None
    assert result.residual_robust_rms < 1e-4
    assert "not_used_for_detection" in result.flags

    promoted, comparison = promote_psf_mge_primary(
        prep, result, ring_outer_px=25.0, near_core_radius_px=6.5
    )
    assert promoted.model_method == "psf_convolved_robust_fixed_geometry_mge"
    assert promoted.model_success
    assert np.allclose(promoted.model, result.model)
    assert comparison.method == "synthetic"
    assert comparison.psf_applied is False
    assert "not_used_for_detection" in comparison.flags


def _boundary_prep(data: np.ndarray, rms: float = 0.02) -> PreprocessedImage:
    size = data.shape[0]
    center = size // 2
    image = ImageData(
        path=Path("boundary.fits"), object_id="boundary",
        input_format="fits", band="VIS", data=data,
        rms=np.full_like(data, rms), supplied_rms=True,
        finite_fraction=1.0, wcs=None, pixel_scale_arcsec=None, psf=None,
    )
    zeros = np.zeros_like(data)
    false_mask = np.zeros_like(data, dtype=bool)
    return PreprocessedImage(
        image=image, model=zeros, residual=data, residual_snr=zeros,
        source_mask=false_mask, fit_mask=false_mask,
        segmentation=np.zeros_like(data, dtype=np.int32),
        nucleus_yx=(center, center), background_median=0.0,
        background_rms=rms, residual_median=0.0,
        residual_robust_rms=rms, model_method="synthetic",
        model_success=True, foreground_detected=True,
        foreground_peak_value=float(data.max()),
        foreground_integrated_flux=float(data.sum()),
        foreground_ellipticity=0.0, foreground_position_angle_deg=0.0,
        foreground_axis_ratio=1.0, foreground_reliability=1.0,
        multiple_core_count=1, tidal_asymmetry=0.0,
        core_residual_fraction=0.0, flags=[],
        center_selection={"selected_centroid_xy": [center, center]},
    )


def test_minimum_boundary_is_closed_not_an_isolated_sector() -> None:
    size = 81
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - 40.0, yy - 40.0)
    angle = np.degrees(np.arctan2(yy - 40.0, xx - 40.0))
    foreground = 6.0 * np.exp(-0.5 * (radius / 3.0) ** 2)
    partial_arc = 2.0 * np.exp(-0.5 * ((radius - 13.0) / 1.2) ** 2) * (np.abs(angle) <= 35.0)
    prep = _boundary_prep(foreground + partial_arc)

    weight, audit = build_source_protection(prep, VisionConfig())

    assert audit["boundary_detected"]
    assert audit["selected_angular_support_deg"] >= 15.0
    assert 5.0 <= audit["boundary_radius_px"] <= 11.5
    assert set(np.unique(weight)) <= {0.0, 1.0}
    test_radius = audit["boundary_radius_px"] + 2.0
    ring = np.abs(radius - test_radius) <= 0.4
    assert ring.any() and np.all(weight[ring] == 0.0)


def test_monotonic_foreground_does_not_invent_a_minimum_boundary() -> None:
    size = 81
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - 40.0, yy - 40.0)
    prep = _boundary_prep(5.0 * np.exp(-0.5 * (radius / 5.0) ** 2))

    weight, audit = build_source_protection(prep, VisionConfig())

    assert not audit["boundary_detected"]
    assert audit["isolated_sector_weights_removed"]
    assert np.all(weight == 1.0)


def test_human_core_is_conservative_fallback_when_minimum_is_absent() -> None:
    size = 81
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - 40.0, yy - 40.0)
    prep = _boundary_prep(5.0 * np.exp(-0.5 * (radius / 5.0) ** 2))
    prep.human_foreground_seed = radius <= 7.0

    weight, audit = build_source_protection(prep, VisionConfig())

    assert not audit["boundary_detected"]
    assert audit["human_core_fallback_used"]
    assert audit["human_core_fallback_radius_px"] >= 6.5
    assert np.all(weight[(radius >= 12.0) & (radius <= 25.0)] == 0.0)
    assert np.all(weight[radius >= 38.0] == 1.0)


def test_foreground_quality_rejects_large_background_offset() -> None:
    data = np.zeros((81, 81), dtype=float)
    prep = _boundary_prep(data, rms=0.02)
    bad_residual = np.full_like(data, -0.08)
    fit = ForegroundModelComparison(
        method="synthetic_bad_sky",
        success=True,
        model=np.full_like(data, 0.08),
        residual=bad_residual,
        residual_robust_rms=0.02,
    )

    quality = _foreground_model_quality(prep, fit, VisionConfig())

    assert not quality["quality_pass"]
    assert "background_residual_not_zero_centered" in quality["failure_reasons"]


def test_foreground_geometry_uses_selected_disc_not_annular_sky() -> None:
    size = 81
    yy, xx = np.indices((size, size))
    elliptical_radius = np.sqrt((xx - 40.0) ** 2 + ((yy - 40.0) / 0.30) ** 2)
    data = 4.0 * np.exp(-elliptical_radius / 5.0)
    prep = _boundary_prep(data)
    prep.segmentation[data > 0.08] = 7
    prep.center_selection["selected_sep_label"] = 7

    q, _ = _geometry_from_moments(prep, 40.0, 40.0, 3.0, 25.0)

    assert q < 0.45


def test_foreground_geometry_mask_rejects_one_sided_blended_arc() -> None:
    size = 81
    yy, xx = np.indices((size, size))
    disc = ((xx - 40.0) / 13.0) ** 2 + ((yy - 40.0) / 3.0) ** 2 <= 1.0
    one_sided_arc = (
        (np.hypot(xx - 40.0, yy - 40.0) >= 10.0)
        & (np.hypot(xx - 40.0, yy - 40.0) <= 12.0)
        & (yy < 36.0)
        & (xx > 40.0)
    )
    prep = _boundary_prep(disc.astype(float) + one_sided_arc.astype(float))
    prep.segmentation[disc | one_sided_arc] = 7
    prep.center_selection["selected_sep_label"] = 7

    foreground = _selected_foreground_pixels(prep)

    assert foreground[40, 30]
    assert foreground[40, 50]
    assert not np.any(foreground & one_sided_arc)
