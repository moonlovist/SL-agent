"""Measure every pairwise colour for an accepted arc chain and foreground galaxy.

This is a post-processing companion to the current Lens Expert Agent multiband
measurement.  It deliberately reuses the agent's common-PSF, calibrated-flux,
protected elliptical-profile machinery instead of introducing a second colour
pipeline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve

from expert_agent.schemas.observation import ObservationDocument
from expert_agent.vision.foreground_models import _normalized_psf
from expert_agent.vision.supplementary import (
    _elliptical_foreground_residual,
    _gaussian_matching_kernel,
    _magnitude_record,
    _robust_scatter,
    _rotate_mask,
    aperture_measurement,
    psf_width,
)


BAND_ORDER = ("VIS", "NIR_Y", "NIR_J", "NIR_H")


def _magnitude(flux: float) -> float | None:
    return float(25.0 - 2.5 * np.log10(flux)) if flux > 0 else None


def _colour(first: dict, second: dict) -> tuple[float | None, float | None]:
    if first.get("magnitude") is None or second.get("magnitude") is None:
        return None, None
    value = float(first["magnitude"] - second["magnitude"])
    errors = (first.get("magnitude_error"), second.get("magnitude_error"))
    error = float(np.hypot(*errors)) if all(v is not None for v in errors) else None
    return value, error


def calculate(observation_path: Path, artifacts_path: Path, fits_path: Path) -> dict:
    document = ObservationDocument.model_validate_json(
        observation_path.read_text(encoding="utf-8")
    )
    if not document.arc_chains:
        raise ValueError("The observation contains no accepted arc chain")
    chain = max(document.arc_chains, key=lambda item: item.pixel_count)
    member_ids = list(chain.source_candidate_ids)
    if not member_ids:
        raise ValueError(f"Arc chain {chain.id} has no source candidate ids")

    with np.load(artifacts_path) as saved:
        arrays = {key: np.asarray(saved[key]) for key in saved.files}
    member_masks = [
        np.asarray(arrays[f"model_candidate_mask_{candidate_id}"], bool)
        for candidate_id in member_ids
    ]
    chain_detection_mask = np.logical_or.reduce(member_masks)

    with fits.open(fits_path, memmap=False) as hdus:
        names = {hdu.name for hdu in hdus}
        calibrated = {}
        for band in BAND_ORDER:
            required = [f"{band}_{kind}" for kind in ("FLUX", "RMS", "PSF")]
            if not all(name in names for name in required):
                continue
            flux, rms, psf = [np.asarray(hdus[name].data, float) for name in required]
            zero_point = float(hdus[required[0]].header["MAGZERO"])
            normalized_psf = _normalized_psf(psf)
            if normalized_psf is None:
                raise ValueError(f"Invalid PSF in {band}")
            scale = 10 ** (-0.4 * (zero_point - 25.0))
            calibrated[band] = (flux * scale, rms * scale, normalized_psf)

    if len(calibrated) < 2:
        raise ValueError("Fewer than two calibrated bands are available")
    widths = {band: psf_width(values[2]) for band, values in calibrated.items()}
    if any(width is None for width in widths.values()):
        raise ValueError("At least one PSF width could not be measured")
    target_fwhm = max(widths.values())
    kernels = {
        band: _gaussian_matching_kernel(width, target_fwhm)
        for band, width in widths.items()
    }

    yy, xx = np.indices(document.image_shape)
    radius = np.hypot(xx - document.nucleus_xy[0], yy - document.nucleus_xy[1])
    sky = (radius >= 45) & (radius <= 65)
    source_mask = np.asarray(
        arrays.get("source_mask", np.zeros(document.image_shape)), bool
    )
    sky &= ~binary_dilation(source_mask, iterations=3)

    aperture_growth = max(1, int(np.ceil(target_fwhm / 2.0)))
    chain_aperture = binary_dilation(chain_detection_mask, iterations=aperture_growth)

    review = document.candidate_handoff_review
    all_detection_masks = []
    if review is not None:
        for entry in review.entries:
            key = f"model_candidate_mask_{entry.candidate_id}"
            if key in arrays:
                all_detection_masks.append(np.asarray(arrays[key], bool))
    all_apertures = [
        binary_dilation(mask, iterations=aperture_growth)
        for mask in all_detection_masks
    ]
    candidate_union = (
        np.logical_or.reduce(all_apertures)
        if all_apertures
        else np.zeros(document.image_shape, bool)
    )
    profile_exclude = binary_dilation(candidate_union, iterations=3)
    profile_exclude |= binary_dilation(source_mask, iterations=2)

    arc_bands: dict[str, dict] = {}
    model_bands: dict[str, dict] = {}
    for band, (data, rms, _) in calibrated.items():
        kernel = kernels[band]
        matched = fftconvolve(data, kernel, mode="same")
        residual = _elliptical_foreground_residual(
            matched, document, profile_exclude
        )
        if residual is None:
            raise ValueError(f"Foreground colour model failed in {band}")
        foreground_model = matched - residual

        _, diagonal_error = aperture_measurement(
            data, rms, kernel, chain_aperture, sky
        )
        control_fluxes = []
        for angle in np.arange(24.0, 360.0, 24.0):
            control = _rotate_mask(
                chain_aperture, document.nucleus_xy, float(angle)
            )
            if control.sum() < max(3, int(0.8 * chain_aperture.sum())):
                continue
            if np.mean(source_mask[control]) > 0.15:
                continue
            control_fluxes.append(float(np.sum(residual[control])))
        local_scatter = _robust_scatter(control_fluxes)
        arc_error = max(diagonal_error, local_scatter or 0.0)
        arc_record = _magnitude_record(
            float(np.sum(residual[chain_aperture])), arc_error
        )
        arc_record.update(
            {
                "diagonal_noise_with_common_psf": float(diagonal_error),
                "local_control_scatter": local_scatter,
                "local_control_count": len(control_fluxes),
            }
        )
        arc_bands[band] = arc_record

        model_flux = float(np.sum(foreground_model[chain_aperture]))
        model_bands[band] = {
            "flux_zp25": model_flux,
            "magnitude": _magnitude(model_flux),
            "magnitude_error": None,
            "status": "measured_model_no_uncertainty",
        }

    multiband = document.supplementary_diagnostics["multiband"]
    galaxy_core_bands: dict[str, dict] = {}
    for band, record in multiband["apertures"]["foreground"].items():
        snr = record.get("snr_nominal")
        galaxy_core_bands[band] = {
            **record,
            "magnitude_error": (
                float(2.5 / np.log(10) / snr)
                if record.get("magnitude") is not None and snr
                else None
            ),
        }

    pairwise = []
    for first, second in combinations(calibrated, 2):
        arc_value, arc_error = _colour(arc_bands[first], arc_bands[second])
        model_value, _ = _colour(model_bands[first], model_bands[second])
        core_value, core_error = _colour(
            galaxy_core_bands[first], galaxy_core_bands[second]
        )
        pairwise.append(
            {
                "band_pair": f"{first}-{second}",
                "arc_color_mag": arc_value,
                "arc_color_error_mag": arc_error,
                "galaxy_model_at_arc_color_mag": model_value,
                "arc_minus_galaxy_model_color_mag": (
                    float(arc_value - model_value)
                    if arc_value is not None and model_value is not None
                    else None
                ),
                "galaxy_core_color_mag": core_value,
                "galaxy_core_color_error_mag": core_error,
                "arc_minus_galaxy_core_color_mag": (
                    float(arc_value - core_value)
                    if arc_value is not None and core_value is not None
                    else None
                ),
                "arc_minus_galaxy_core_error_mag": (
                    float(np.hypot(arc_error, core_error))
                    if arc_error is not None and core_error is not None
                    else None
                ),
            }
        )

    return {
        "object_id": document.object_id,
        "schema_version": "arc_galaxy_pairwise_colours_v1",
        "arc_definition": {
            "chain_id": chain.id,
            "member_candidate_ids": member_ids,
            "detection_pixel_count": int(chain_detection_mask.sum()),
            "common_psf_aperture_pixel_count": int(chain_aperture.sum()),
            "detection_mask_sha256": hashlib.sha256(
                chain_detection_mask.astype(np.uint8).tobytes()
            ).hexdigest(),
        },
        "photometric_policy": {
            "zero_point": 25.0,
            "psf_matching": "same Gaussian second-moment matching as current agent",
            "arc": "accepted chain union after protected elliptical-profile subtraction",
            "galaxy_model_at_arc": "elliptical radial foreground model integrated in the identical arc aperture",
            "galaxy_core": "current-agent raw foreground aperture (radius <= 4 px), sky subtracted",
            "delta_sign": "arc colour minus galaxy colour",
        },
        "psf_fwhm_px": widths,
        "target_psf_fwhm_px": target_fwhm,
        "arc_bands": arc_bands,
        "galaxy_model_at_arc_bands": model_bands,
        "galaxy_core_bands": galaxy_core_bands,
        "pairwise_colours": pairwise,
        "limitations": [
            "Galaxy-model colour has no propagated foreground-model uncertainty.",
            "Arc errors include rotated local-control scatter and diagonal noise, but not native covariance, PSF uncertainty, or foreground-model uncertainty.",
            "Pairwise colours share bands and are therefore correlated; the six rows are not six independent tests.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("observation", type=Path)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("fits", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    result = calculate(args.observation, args.artifacts, args.fits)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = result["pairwise_colours"]
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
