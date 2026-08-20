from pathlib import Path

import numpy as np

from expert_agent.core.config import VisionConfig
from expert_agent.vision.detectors import posterior_ring_candidates
from expert_agent.vision.preprocess import ImageData, PreprocessedImage


def _prep_from_residual_snr(residual_snr: np.ndarray) -> PreprocessedImage:
    data = np.asarray(residual_snr, dtype=float)
    zeros = np.zeros_like(data)
    false_mask = np.zeros_like(data, dtype=bool)
    size = data.shape[0]
    image = ImageData(
        path=Path("synthetic.fits"), object_id="synthetic", input_format="fits",
        band="VIS", data=data, rms=np.ones_like(data), supplied_rms=True,
        finite_fraction=1.0, wcs=None, pixel_scale_arcsec=None, psf=None,
    )
    return PreprocessedImage(
        image=image, model=zeros, residual=data, residual_snr=data,
        source_mask=false_mask, fit_mask=false_mask,
        segmentation=np.zeros_like(data, dtype=np.int32),
        nucleus_yx=(size // 2, size // 2), background_median=0.0,
        background_rms=1.0, residual_median=0.0, residual_robust_rms=1.0,
        model_method="synthetic", model_success=True, foreground_detected=True,
        foreground_peak_value=1.0, foreground_integrated_flux=1.0,
        foreground_ellipticity=0.0, foreground_position_angle_deg=0.0,
        foreground_axis_ratio=1.0, foreground_reliability=1.0,
        multiple_core_count=0, tidal_asymmetry=0.0,
        core_residual_fraction=0.0, flags=[],
    )


def test_posterior_ring_search_recovers_low_per_pixel_snr_ring() -> None:
    size = 61
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - 30.0, yy - 30.0)
    # Each pixel is only modestly significant, but it coherently follows one
    # annulus -- the precise failure mode the posterior model targets.
    residual_snr = 1.1 * np.exp(-0.5 * ((radius - 12.0) / 1.0) ** 2)
    prep = _prep_from_residual_snr(residual_snr)
    config = VisionConfig(
        ring_inner_px=4.0, ring_outer_px=20.0,
        ring_model_center_offset_px=0.0, ring_model_center_steps=1,
        ring_model_axis_ratios=(1.0,), ring_model_position_angles_deg=(0.0,),
        ring_model_max_candidates=3,
    )

    candidates = posterior_ring_candidates(prep, config, [])

    assert candidates
    best = candidates[0]
    assert best.passes_initial_screen
    assert abs(best.semi_major_radius_px - 12.0) <= 1.0
    assert best.positive_coverage_fraction >= 0.80
    assert best.on_minus_off_ring_snr > 0.0
