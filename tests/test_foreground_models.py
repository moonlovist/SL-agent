from pathlib import Path

import numpy as np

from expert_agent.vision.foreground_models import fit_psf_mge_comparison
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
