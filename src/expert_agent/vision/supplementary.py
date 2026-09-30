"""Deterministic, label-blind auxiliary measurements (not physical diagnoses).

Keep morphology at native VIS resolution. Colour apertures use a common PSF;
arc chains contain only the union of observed pixels, never interpolated gaps.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve

from .candidate_handoff import _measure
from .foreground_models import _normalized_psf


def psf_width(psf):
    if psf is None or not np.isfinite(psf).all() or np.sum(psf) <= 0:
        return None
    p = np.maximum(psf, 0).astype(float)
    p /= p.sum()
    y, x = np.indices(p.shape)
    variance = np.sum(p * ((x - np.sum(x*p))**2 + (y - np.sum(y*p))**2)) / 2
    return float(2.35482 * np.sqrt(variance)) if variance > 0 else None


def measure_arc_chains(document, arrays, fwhm):
    """Local proximity + tangent compatibility; grouping is not multiple imaging."""
    review = document.candidate_handoff_review
    if review is None:
        return [], []
    entries = {e.candidate_id: e for e in review.entries if e.measurement is not None}
    eligible = {
        cid: e for cid, e in entries.items()
        if e.measurement.axis_ratio is not None and e.measurement.axis_ratio >= 1.5
        and e.measurement.tangential_misalignment_deg is not None
        and e.measurement.tangential_misalignment_deg <= 35
        and e.measurement.integrated_flux > 0 and e.mask_key in arrays
    }
    adjacency = {cid: set() for cid in eligible}
    distances = {cid: distance_transform_edt(~np.asarray(arrays[e.mask_key], bool)) for cid, e in eligible.items()}
    gap = max(2.0, min(4.0, fwhm or 2.0))
    for a, b in combinations(sorted(eligible), 2):
        ma, mb = (eligible[c].measurement for c in (a, b))
        if abs(ma.median_radius_px - mb.median_radius_px) > gap:
            continue
        if distances[a][np.asarray(arrays[eligible[b].mask_key], bool)].min() <= gap:
            adjacency[a].add(b)
            adjacency[b].add(a)
    chains, records, visited = [], [], set()
    candidates = {c.id: c for c in document.model_candidates}
    for root in sorted(adjacency):
        if root in visited:
            continue
        stack, group = [root], set()
        while stack:
            cid = stack.pop()
            if cid in group:
                continue
            group.add(cid)
            stack.extend(sorted(adjacency[cid] - group))
        visited.update(group)
        if len(group) < 2:
            continue
        ids = sorted(group)
        mask = np.logical_or.reduce([np.asarray(arrays[eligible[c].mask_key], bool) for c in ids])
        chain_id = "CHAIN_" + "_".join(ids)
        proxy = candidates[ids[0]].model_copy(update={"id": chain_id, "pixel_count": int(mask.sum())})
        measurement, *_ = _measure(document, proxy,
            {**arrays, f"model_candidate_mask_{chain_id}": mask}, 2.0, 4.0)
        measurement.id = chain_id
        measurement.source_candidate_ids = ids
        # A union must not inherit the first fragment's persistence fraction.
        levels = document.provenance.parameters.get("snr_thresholds", [2., 2.58, 3.34, 4.31, 5.57, 7.19, 9.29, 12.])
        survived = 0
        for level in levels:
            labels, _ = label(mask & (arrays["residual"] / arrays["rms"] >= level))
            counts = np.bincount(labels.ravel())[1:]
            survived += int(bool(counts.size and counts.max() >= 3))
        measurement.threshold_survival_fraction = survived/len(levels) if levels else None
        measurement.sep_survival_fraction = None  # union SEP grid was not rerun
        measurement.candidate_audit_ids = sorted({a for c in ids for a in eligible[c].measurement.candidate_audit_ids})
        measurement.flags.extend(["observed_pixel_union_no_gap_filling", "fragment_group_not_physical_image_count"])
        # A group must itself have measured curved geometry, not just proximity.
        accepted = bool(measurement.local_shape_count >= 2 and
                        (measurement.local_tangential_fraction or 0) >= 0.67)
        records.append({"id": chain_id, "members": ids, "accepted": accepted,
                        "gap_limit_px": gap, "pixel_count": int(mask.sum()),
                        "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()})
        if accepted:
            chains.append(measurement)
    return chains, records


def aperture_measurement(data, rms, kernel, aperture, sky):
    """Propagate the aperture's adjoint convolution weights on native pixels.

    Includes covariance introduced by OUR convolution, not pre-existing input
    covariance, PSF uncertainty, foreground contamination, or selection trials.
    """
    weights = aperture.astype(float) - sky.astype(float) * aperture.sum() / sky.sum()
    effective = fftconvolve(weights, kernel[::-1, ::-1], mode="same")
    flux = float(np.sum(effective * data))
    error = float(np.sqrt(np.sum((effective * rms)**2)))
    return flux, error


def _rotate_mask(mask, center_xy, angle_deg):
    """Nearest-pixel rigid rotation for control apertures; never a signal mask."""
    ys, xs = np.where(mask)
    cx, cy = center_xy
    angle = np.radians(angle_deg)
    dx, dy = xs-cx, ys-cy
    xr = np.rint(cx + dx*np.cos(angle) - dy*np.sin(angle)).astype(int)
    yr = np.rint(cy + dx*np.sin(angle) + dy*np.cos(angle)).astype(int)
    result = np.zeros(mask.shape, bool)
    inside = (xr >= 0) & (xr < mask.shape[1]) & (yr >= 0) & (yr < mask.shape[0])
    result[yr[inside], xr[inside]] = True
    return result


def _elliptical_foreground_residual(data, document, exclude):
    """Protected, fixed-geometry radial profile for colour measurement only."""
    yy, xx = np.indices(data.shape)
    cx, cy = document.nucleus_xy
    dx, dy = xx-cx, yy-cy
    galaxy = document.foreground_galaxy
    pa = np.radians(galaxy.position_angle_deg or 0.0) if galaxy else 0.0
    q = 1.0 / max(galaxy.axis_ratio or 1.0, 1.0) if galaxy else 1.0
    xrot = dx*np.cos(pa) + dy*np.sin(pa)
    yrot = -dx*np.sin(pa) + dy*np.cos(pa)
    radius = np.sqrt(xrot**2 + (yrot/max(q, 0.25))**2)
    centers, values = [], []
    for lo in np.arange(0.0, min(40.0, radius.max()), 0.75):
        use = (radius >= lo) & (radius < lo+0.75) & ~exclude & np.isfinite(data)
        if use.sum() >= 12:
            centers.append(lo+0.375)
            values.append(float(np.median(data[use])))
    if len(centers) < 8:
        return None
    profile = np.interp(radius, centers, values, left=values[0], right=values[-1])
    return data-profile


def _robust_scatter(values):
    values = np.asarray(values, float)
    if values.size < 4:
        return None
    scale = 1.4826*np.median(np.abs(values-np.median(values)))
    return float(scale) if np.isfinite(scale) and scale > 0 else None


def _gaussian_matching_kernel(source_fwhm, target_fwhm):
    """Documented width-matching approximation; no deconvolution."""
    sigma = np.sqrt(max(target_fwhm**2-source_fwhm**2, 0.0))/2.35482
    if sigma < 0.05:
        return np.ones((1, 1), float)
    half = max(2, int(np.ceil(4*sigma)))
    yy, xx = np.mgrid[-half:half+1, -half:half+1]
    kernel = np.exp(-0.5*(xx**2+yy**2)/sigma**2)
    return kernel/kernel.sum()


def _magnitude_record(flux, error, min_snr=3.0):
    snr = flux/error if error > 0 else None
    measured = bool(snr is not None and np.isfinite(snr) and snr >= min_snr and flux > 0)
    return {"flux_zp25": float(flux), "error_including_local_controls": float(error),
            "snr": float(snr) if snr is not None else None,
            "magnitude": float(25-2.5*np.log10(flux)) if measured else None,
            "magnitude_error": float(2.5/np.log(10)/snr) if measured else None,
            "status": "measured" if measured else "low_snr_or_nonpositive"}


def _flux_scale_sed_diagnostic(first, second, min_common_bands=2):
    """Fit F1 ~= alpha*F2 in flux space without imposing a grade threshold.

    Unlike magnitude colours, finite low-S/N and negative flux measurements
    remain usable here.  The result is diagnostic only because native
    covariance and foreground-model uncertainty are not fully propagated.
    """
    bands = []
    for band in sorted(set(first).intersection(second)):
        a, b = first[band], second[band]
        values = (
            a.get("flux_zp25"), a.get("error_including_local_controls"),
            b.get("flux_zp25"), b.get("error_including_local_controls"),
        )
        if all(value is not None and np.isfinite(value) for value in values) and values[1] > 0 and values[3] > 0:
            bands.append(band)
    if len(bands) < min_common_bands:
        return {"status": "unknown", "reason": "insufficient_finite_flux_bands", "bands": bands}

    flux_a = np.asarray([first[band]["flux_zp25"] for band in bands], float)
    error_a = np.asarray([first[band]["error_including_local_controls"] for band in bands], float)
    flux_b = np.asarray([second[band]["flux_zp25"] for band in bands], float)
    error_b = np.asarray([second[band]["error_including_local_controls"] for band in bands], float)

    def objective(log_alpha):
        alpha = float(np.exp(log_alpha))
        variance = error_a**2 + alpha**2*error_b**2
        return float(np.sum((flux_a-alpha*flux_b)**2/variance))

    fit = minimize_scalar(objective, bounds=(-12.0, 12.0), method="bounded")
    if not fit.success or not np.isfinite(fit.fun):
        return {"status": "unknown", "reason": "flux_scale_fit_failed", "bands": bands}
    alpha = float(np.exp(fit.x))
    variance = error_a**2 + alpha**2*error_b**2
    residual = flux_a-alpha*flux_b
    chi2 = float(np.sum(residual**2/variance))
    dof = len(bands)-1
    return {
        "status": "measured_not_calibrated_for_grading",
        "bands": bands,
        "band_count": len(bands),
        "best_flux_scale_alpha": alpha,
        "same_sed_chi2": chi2,
        "same_sed_dof": dof,
        "same_sed_reduced_chi2": chi2/max(dof, 1),
        "per_band_diagnostics": {
            band: {
                "first_flux_zp25": float(flux_a[index]),
                "second_flux_zp25": float(flux_b[index]),
                "residual_flux_zp25": float(residual[index]),
                "residual_sigma": float(residual[index]/np.sqrt(variance[index])),
            }
            for index, band in enumerate(bands)
        },
        "interpretation": "Continuous same-SED flux-shape diagnostic; no Q1 grade threshold is applied.",
    }


def _gray_colour_test(
    first,
    second,
    systematic_floor_mag=0.10,
    min_common_bands=3,
    min_band_snr=None,
    consistent_max_departure_mag=0.20,
    consistent_max_reduced_chi2=2.0,
    inconsistent_min_departure_mag=0.35,
    inconsistent_min_sigma=3.0,
):
    """Test whether two SEDs differ only by an achromatic flux scale."""
    flux_scale_diagnostic = _flux_scale_sed_diagnostic(first, second)
    bands = sorted(set(first).intersection(second))
    bands = [
        b
        for b in bands
        if first[b].get("magnitude") is not None
        and second[b].get("magnitude") is not None
        and (
            min_band_snr is None
            or (
                first[b].get("snr") is not None
                and second[b].get("snr") is not None
                and first[b]["snr"] >= min_band_snr
                and second[b]["snr"] >= min_band_snr
            )
        )
    ]
    if len(bands) < min_common_bands:
        return {"status": "unknown", "reason": "fewer_than_three_reliable_common_bands", "bands": bands,
                "flux_scale_diagnostic": flux_scale_diagnostic}
    difference = np.array([first[b]["magnitude"]-second[b]["magnitude"] for b in bands])
    variance = np.array([first[b]["magnitude_error"]**2 + second[b]["magnitude_error"]**2
                         + systematic_floor_mag**2 for b in bands])
    offset = float(np.sum(difference/variance)/np.sum(1/variance))
    departures = difference-offset
    significance = np.abs(departures)/np.sqrt(variance)
    maximum = float(np.max(np.abs(departures)))
    max_sigma = float(np.max(significance))
    chi2 = float(np.sum(departures**2/variance))
    dof = len(bands)-1
    reduced_chi2 = chi2/max(dof, 1)
    if maximum >= inconsistent_min_departure_mag and max_sigma >= inconsistent_min_sigma:
        status = "strongly_inconsistent"
    elif maximum <= consistent_max_departure_mag and reduced_chi2 <= consistent_max_reduced_chi2:
        status = "consistent_with_achromatic_scaling"
    else:
        status = "inconclusive"
    per_band = {
        band: {
            "magnitude_difference": float(difference[index]),
            "difference_error_including_systematic": float(np.sqrt(variance[index])),
            "departure_from_gray_offset_mag": float(departures[index]),
            "departure_sigma": float(significance[index]),
        }
        for index, band in enumerate(bands)
    }
    return {"status": status, "bands": bands, "reliable_common_band_count": len(bands),
            "gray_offset_mag": offset,
            "maximum_color_departure_mag": maximum, "maximum_departure_sigma": max_sigma,
            "chi2": chi2, "degrees_of_freedom": dof, "reduced_chi2": reduced_chi2,
            "systematic_floor_mag": systematic_floor_mag,
            "per_band_diagnostics": per_band,
            "flux_scale_diagnostic": flux_scale_diagnostic,
            "interpretation": "Consistency supports common-source imaging but is not unique to lensing; inconsistency can also arise from contamination, dust, variability or microlensing."}


def _tiered_colour_test(first, second, config):
    """Return decisive high-quality evidence or a non-decisive weak tier.

    The suggestive tier is available through either three bands at the lower
    S/N boundary or two bands at the high S/N boundary.  Its status is kept
    distinct so it can never inherit the high-tier A-cap semantics.
    """
    common = {
        "systematic_floor_mag": config.colour_systematic_floor_mag,
        "consistent_max_departure_mag": config.colour_consistent_max_departure_mag,
        "consistent_max_reduced_chi2": config.colour_consistent_max_reduced_chi2,
        "inconsistent_min_departure_mag": config.colour_inconsistent_min_departure_mag,
        "inconsistent_min_sigma": config.colour_inconsistent_min_sigma,
    }
    high = _gray_colour_test(
        first,
        second,
        min_common_bands=config.colour_min_common_bands,
        min_band_snr=config.colour_min_band_snr,
        **common,
    )
    if high["status"] != "unknown":
        high["evidence_tier"] = "high_confidence"
        return high

    lower_snr_three_band = _gray_colour_test(
        first,
        second,
        min_common_bands=config.colour_min_common_bands,
        min_band_snr=config.colour_suggestive_min_band_snr,
        **common,
    )
    high_snr_two_band = _gray_colour_test(
        first,
        second,
        min_common_bands=config.colour_suggestive_min_common_bands,
        min_band_snr=config.colour_min_band_snr,
        **common,
    )
    usable = [
        item
        for item in (lower_snr_three_band, high_snr_two_band)
        if item["status"] != "unknown"
    ]
    if not usable:
        high["evidence_tier"] = "flux_diagnostic_only"
        high["suggestive_attempts"] = {
            "lower_snr_three_band": lower_snr_three_band,
            "high_snr_two_band": high_snr_two_band,
        }
        return high
    selected = max(
        usable,
        key=lambda item: (
            item.get("reliable_common_band_count", 0),
            -item.get("reduced_chi2", float("inf")),
        ),
    )
    selected = dict(selected)
    selected["high_confidence_test"] = high
    selected["evidence_tier"] = "suggestive_non_decisive"
    selected["status"] = {
        "consistent_with_achromatic_scaling": "suggestive_consistent_with_achromatic_scaling",
        "strongly_inconsistent": "suggestive_colour_conflict",
        "inconclusive": "suggestive_inconclusive",
    }[selected["status"]]
    selected["interpretation"] = (
        "Suggestive two-band or S/N 2.5-3 colour relation. It is auditable "
        "corroboration only and cannot promote, veto, or cap a Q1 grade."
    )
    return selected


def _colour_relationship_audit(decontaminated, raw, masks=None, config=None):
    from ..core.config import VisionConfig

    config = config or VisionConfig()
    candidates = sorted(decontaminated)
    pairs = []
    for first, second in combinations(candidates, 2):
        overlap = None
        if masks and first in masks and second in masks:
            overlap = float(np.sum(masks[first] & masks[second]) /
                            max(1, min(masks[first].sum(), masks[second].sum())))
        test = _tiered_colour_test(
            decontaminated[first], decontaminated[second], config
        )
        if overlap is not None and overlap > config.colour_candidate_overlap_max:
            test = {"status": "unknown", "reason": "common_psf_apertures_overlap",
                    "bands": test.get("bands", []), "overlap_fraction": overlap}
        else:
            test["overlap_fraction"] = overlap
        pairs.append({"first": first, "second": second, **test})
    foreground = []
    if "foreground" in raw:
        for candidate in candidates:
            test = _tiered_colour_test(
                decontaminated[candidate], raw["foreground"], config
            )
            overlap = None
            if masks and candidate in masks and "foreground" in masks:
                overlap = float(np.sum(masks[candidate] & masks["foreground"])/
                                max(1, masks[candidate].sum()))
            if overlap is not None and overlap > config.colour_foreground_core_overlap_max:
                test = {"status": "unknown", "reason": "candidate_aperture_overlaps_foreground_core",
                        "bands": test.get("bands", []), "foreground_core_overlap_fraction": overlap}
            else:
                status = test["status"]
                test["status"] = ({
                    "strongly_inconsistent": "distinct_from_foreground",
                    "consistent_with_achromatic_scaling": "indistinguishable_from_foreground",
                    "suggestive_colour_conflict": "suggestive_distinct_from_foreground",
                    "suggestive_consistent_with_achromatic_scaling": "suggestive_indistinguishable_from_foreground",
                }.get(status, status))
                test["foreground_core_overlap_fraction"] = overlap
            foreground.append({"candidate": candidate, **test})
    thresholds = {
        "minimum_band_snr": config.colour_min_band_snr,
        "minimum_reliable_common_bands": config.colour_min_common_bands,
        "suggestive_minimum_band_snr": config.colour_suggestive_min_band_snr,
        "suggestive_minimum_common_bands": config.colour_suggestive_min_common_bands,
        "minimum_local_controls": config.colour_min_local_controls,
        "systematic_floor_mag": config.colour_systematic_floor_mag,
        "consistent_max_departure_mag": config.colour_consistent_max_departure_mag,
        "consistent_max_reduced_chi2": config.colour_consistent_max_reduced_chi2,
        "inconsistent_min_departure_mag": config.colour_inconsistent_min_departure_mag,
        "inconsistent_min_sigma": config.colour_inconsistent_min_sigma,
        "candidate_aperture_overlap_max": config.colour_candidate_overlap_max,
        "foreground_core_overlap_max": config.colour_foreground_core_overlap_max,
    }
    return {"policy": "achromatic_sed_shape_v3_tiered_configuration_gated", "component_pairs": pairs,
            "candidate_vs_foreground": foreground,
            "thresholds": thresholds,
            "grade_use": "Only high-confidence pairs selected later by a supported lens-configuration hypothesis may affect a grade. Suggestive relations are audit-only and cannot promote, veto, or cap. Reliable inconsistency can weaken that configuration; consistency supports but never creates it. Foreground similarity is non-discriminating and never acts alone as a grade cap.",
            "grade_semantics": {
                "A": "Compatible colour evidence may corroborate clear morphology; unexplained reliable conflict in a required same-source pair can cap A at B.",
                "B": "Colour support cannot promote incomplete morphology; conflict or unavailable colour can leave a probable system at B.",
                "C": "Colour conflict may support ambiguity when morphology already has a viable non-lens interpretation; no standalone C boundary is calibrated yet.",
                "unknown": "Missing or failed colour measurement is recorded as unknown evidence and cannot by itself make the whole object unknown.",
            },
            "limitations": ["Thresholds and the systematic floor are provisional measurement categories, not Q1-calibrated A/B/C cuts.",
                            "Dust, time delay plus variability, microlensing and remaining subtraction error can produce real colour differences."]}


def measure_colours(document, arrays, path, config=None):
    from astropy.io import fits
    from astropy.wcs import WCS
    from ..core.config import VisionConfig

    config = config or VisionConfig()
    result = {"policy": "common_psf_protected_elliptical_profile_v3_tiered", "status": "unavailable",
              "bands": {}, "apertures": {}, "colours": {}, "decontaminated_apertures": {},
              "relationship_audit": {},
              "limitations": ["Raw-image aperture colours are retained separately and are not grade evidence.",
                  "Bands are approximately matched to the broadest measured PSF second-moment width without deconvolution.",
                  "Protected fixed-geometry elliptical-profile subtraction is used for candidate colours; local rotated controls inflate errors.",
                  "S/N 2.5-3 or two-band relations are retained as suggestive non-decisive evidence; only the original S/N>=3 three-band tier can affect grading.",
                  "Native input covariance, PSF uncertainty and colour gradients are not fully calibrated.",
                  "No blue-source requirement or exact-colour-equality rule is used."]}
    if document.input_format != "fits":
        result["reason"] = "calibrated_fits_required"
        return result
    with fits.open(path, memmap=False) as hdus:
        names = {h.name for h in hdus}
        bands = {}
        h, w = document.image_shape
        y, x = np.mgrid[0:h:complex(0, 5), 0:w:complex(0, 5)]
        x, y = np.minimum(x, w-1), np.minimum(y, h-1)
        ref = WCS(hdus[f"{document.band}_FLUX"].header).celestial if f"{document.band}_FLUX" in names else None
        for band in dict.fromkeys([document.band, "NIR_Y", "NIR_J", "NIR_H"]):
            required = [f"{band}_{ext}" for ext in ("FLUX", "RMS", "PSF")]
            if not all(n in names for n in required):
                result["bands"][band] = {"status": "unavailable", "reason": "missing_flux_rms_or_psf"}
                continue
            flux, rms, psf = [np.asarray(hdus[n].data, float) for n in required]
            zp = hdus[required[0]].header.get("MAGZERO")
            reason = None
            if flux.shape != document.image_shape or rms.shape != flux.shape:
                reason = "different_pixel_grid_requires_validated_reprojection"
            elif not np.isfinite(flux).all() or not np.isfinite(rms).all() or not (rms > 0).all():
                reason = "invalid_pixels"
            elif zp is None or not np.isfinite(float(zp)):
                reason = "missing_magnitude_zeropoint"
            else:
                bwcs = WCS(hdus[required[0]].header).celestial
                if ref is None or not ref.has_celestial or not bwcs.has_celestial:
                    reason = "missing_celestial_wcs"
                else:
                    bx, by = bwcs.world_to_pixel_values(*ref.pixel_to_world_values(x, y))
                    if not np.isfinite([bx, by]).all() or np.max(np.hypot(bx-x, by-y)) > 0.05:
                        reason = "wcs_mismatch_requires_validated_reprojection"
            p = _normalized_psf(psf)
            if p is None or any(n % 2 == 0 for n in psf.shape) or not np.isfinite(psf).all():
                reason = "invalid_or_even_sized_psf"
            if reason:
                result["bands"][band] = {"status": "unavailable", "reason": reason}
                continue
            scale = 10 ** (-0.4 * (float(zp) - 25.0))
            bands[band] = (flux * scale, rms * scale, p)
            result["bands"][band] = {"status": "measured", "magzero": float(zp), "common_zero_point": 25.0}
        if document.band not in bands or len(bands) < 2:
            result["reason"] = "fewer_than_two_calibrated_aligned_bands_including_primary"
            return result
        widths = {band: psf_width(values[2]) for band, values in bands.items()}
        if any(value is None for value in widths.values()):
            result["reason"] = "psf_width_measurement_failed"
            return result
        target_fwhm = max(widths.values())
        kernels = {band: _gaussian_matching_kernel(width, target_fwhm)
                   for band, width in widths.items()}
        result["psf_matching"] = {"method": "gaussian_second_moment_width_no_deconvolution",
                                  "input_fwhm_px": widths, "target_fwhm_px": target_fwhm,
                                  "limitation": "Matches one PSF width moment, not the complete non-Gaussian PSF shape."}
        yy, xx = np.indices(document.image_shape)
        radius = np.hypot(xx-document.nucleus_xy[0], yy-document.nucleus_xy[1])
        sky = (radius >= 45) & (radius <= 65)
        # Mask detected neighbours in the sky aperture. This same sky is used in all bands.
        if "source_mask" in arrays:
            sky &= ~binary_dilation(np.asarray(arrays["source_mask"], bool), iterations=3)
        if sky.sum() < 100:
            result["reason"] = "insufficient_common_sky_pixels"
            return result
        masks = {"foreground": radius <= 4}
        if document.candidate_handoff_review:
            masks.update({e.candidate_id: np.asarray(arrays[e.mask_key], bool)
                          for e in document.candidate_handoff_review.entries if e.mask_key in arrays})
        original_candidate_masks = {key: value for key, value in masks.items() if key != "foreground"}
        aperture_growth = max(1, int(np.ceil(target_fwhm/2.0)))
        candidate_masks = {key: binary_dilation(value, iterations=aperture_growth)
                           for key, value in original_candidate_masks.items()}
        masks.update(candidate_masks)
        result["candidate_aperture_growth_px"] = aperture_growth
        candidate_union = (np.logical_or.reduce(list(candidate_masks.values()))
                           if candidate_masks else np.zeros(document.image_shape, bool))
        protected = binary_dilation(candidate_union, iterations=3)
        contaminant = np.asarray(arrays.get("source_mask", np.zeros(document.image_shape)), bool)
        profile_exclude = protected | binary_dilation(contaminant, iterations=2)
        for band, (data, rms, _) in bands.items():
            kernel = kernels[band]
            matched = fftconvolve(data, kernel, mode="same")
            foreground_residual = _elliptical_foreground_residual(matched, document, profile_exclude)
            for key, mask in masks.items():
                if not mask.any():
                    continue
                flux, err = aperture_measurement(data, rms, kernel, mask, sky)
                raw_snr = flux/err if err > 0 else None
                raw_measured = bool(
                    flux > 0
                    and raw_snr is not None
                    and raw_snr >= config.colour_suggestive_min_band_snr
                )
                record = {"flux_zp25": flux, "noise_diagonal_native": err,
                          "snr_nominal": raw_snr,
                          "magnitude": float(25 - 2.5*np.log10(flux)) if raw_measured else None}
                result["apertures"].setdefault(key, {})[band] = record
            if foreground_residual is not None:
                for key, mask in candidate_masks.items():
                    raw_error = result["apertures"][key][band]["noise_diagonal_native"]
                    control_fluxes = []
                    for angle in np.arange(24.0, 360.0, 24.0):
                        control = _rotate_mask(mask, document.nucleus_xy, float(angle))
                        if control.sum() < max(3, int(0.8*mask.sum())):
                            continue
                        # Other candidate flux is deliberately allowed in controls:
                        # it inflates the empirical scatter instead of making the
                        # uncertainty unavailable around a populated ring.
                        blocked = contaminant
                        if np.mean(blocked[control]) > 0.15:
                            continue
                        control_fluxes.append(float(np.sum(foreground_residual[control])))
                    local_scatter = _robust_scatter(control_fluxes)
                    error = max(raw_error, local_scatter or 0.0)
                    record = _magnitude_record(
                        float(np.sum(foreground_residual[mask])), error,
                        min_snr=config.colour_suggestive_min_band_snr,
                    )
                    record.update({"diagonal_noise_with_common_psf": raw_error,
                                   "local_control_scatter": local_scatter,
                                   "local_control_count": len(control_fluxes),
                                   "local_controls_may_include_candidate_flux": True,
                                   "color_aperture_pixel_count": int(mask.sum()),
                                   "source_detection_mask_sha256": hashlib.sha256(
                                       original_candidate_masks[key].astype(np.uint8).tobytes()).hexdigest(),
                                   "foreground_method": "protected_fixed_geometry_elliptical_radial_profile"})
                    if len(control_fluxes) < config.colour_min_local_controls:
                        record["status"] = "insufficient_local_controls"
                        record["magnitude"] = record["magnitude_error"] = None
                    result["decontaminated_apertures"].setdefault(key, {})[band] = record
        for records in result["decontaminated_apertures"].values():
            finite_fluxes = [float(record["flux_zp25"]) for record in records.values()
                             if record.get("flux_zp25") is not None and np.isfinite(record["flux_zp25"])]
            signed_sum = float(np.sum(finite_fluxes)) if finite_fluxes else 0.0
            absolute_sum = float(np.sum(np.abs(finite_fluxes))) if finite_fluxes else 0.0
            for record in records.values():
                flux = record.get("flux_zp25")
                record["normalized_flux_by_signed_sum"] = (
                    float(flux/signed_sum) if flux is not None and abs(signed_sum) > 0 else None
                )
                record["normalized_flux_by_absolute_sum"] = (
                    float(flux/absolute_sum) if flux is not None and absolute_sum > 0 else None
                )
        for key, records in result["apertures"].items():
            for a, b in combinations(bands, 2):
                ra, rb = records[a], records[b]
                available = ra["magnitude"] is not None and rb["magnitude"] is not None
                result["colours"].setdefault(key, {})[f"{a}-{b}"] = {
                    "status": "measured_contaminated_aperture" if available else "low_snr_unknown",
                    "value_mag": ra["magnitude"]-rb["magnitude"] if available else None,
                    "error_mag_nominal": float(2.5/np.log(10)*np.hypot(1/ra["snr_nominal"], 1/rb["snr_nominal"])) if available else None}
        raw_for_relationship = {}
        for key, records in result["apertures"].items():
            raw_for_relationship[key] = {}
            for band, record in records.items():
                snr = record["snr_nominal"]
                raw_for_relationship[key][band] = {
                    "magnitude": record["magnitude"],
                    "magnitude_error": float(2.5/np.log(10)/snr) if record["magnitude"] is not None else None,
                    "snr": snr,
                }
        result["relationship_audit"] = _colour_relationship_audit(
            result["decontaminated_apertures"], raw_for_relationship, masks, config)
        result["status"] = "measured"
        result["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result


def attach_supplementary(document, arrays, image, config=None):
    width = psf_width(image.psf)
    document.arc_chains, chains = measure_arc_chains(document, arrays, width)
    document.supplementary_diagnostics = {
        "policy": "scientific_revision_v0220", "psf_fwhm_px": width,
        "psf_width_definition": "positive_psf_second_moment_gaussian_equivalent_not_measured_halfmaximum",
        "arc_chain_audit": chains,
        "signed_route_apertures": measure_signed_routes(arrays),
        "multiband": measure_colours(document, arrays, image.path, config),
        "origin_diagnosis": {"psf_artifact": "unknown", "intrinsic_morphology": "unknown",
                             "reason": "residual_connectivity_asymmetry_core_fraction_are_not_origin_tests"},
    }
    return document


def measure_signed_routes(arrays):
    """Whole searched apertures, including negative/subthreshold pixels.

    These are not search-corrected detections. Keep the data available rather
    than equating failure of three threshold layers with a nonexistent arc.
    """
    records = []
    for key in sorted(arrays):
        if not key.startswith("geometric_path_mask_"):
            continue
        mask = np.asarray(arrays[key], bool).copy()
        if "source_mask" in arrays:
            mask &= ~np.asarray(arrays["source_mask"], bool)
        residual, rms = arrays["residual"], arrays["rms"]
        mask &= np.isfinite(residual) & np.isfinite(rms) & (rms > 0)
        if not mask.any():
            continue
        flux = float(np.sum(residual[mask]))
        noise = float(np.sqrt(np.sum(rms[mask]**2)))
        records.append({"mask_key": key, "pixel_count": int(mask.sum()),
            "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
            "signed_flux": flux, "nominal_snr_diagonal": flux/noise,
            "negative_pixel_fraction": float(np.mean(residual[mask] < 0)),
            "above_2sigma_fraction": float(np.mean(residual[mask]/rms[mask] >= 2)),
            "status": "diagnostic_not_calibrated_detection",
            "limitations": ["Search trials, native pixel covariance and foreground uncertainty are not calibrated.",
                            "A full ellipse can dilute a partial arc; low mean is not a rejection."]})
    return records
