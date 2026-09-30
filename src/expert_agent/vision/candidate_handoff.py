"""Complete measurement handoff for the integrated grading workflow.

A comparison image made from the same exposure is a model-sensitivity check,
not an independent observation. No threshold occupancy proves physical origin.
"""
from __future__ import annotations

from collections.abc import Mapping
from collections import ChainMap
import hashlib
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

from ..schemas.observation import (
    CandidateHandoffEntry, CandidateHandoffReview, ObservationDocument,
    StructureObservation,
)
from .curved_geometry import measure_curved_geometry


def _psf_width(psf) -> float | None:
    """Gaussian-equivalent second-moment width, matching supplementary.py."""
    if psf is None:
        return None
    values = np.asarray(psf, dtype=float)
    if not np.isfinite(values).all() or values.sum() <= 0:
        return None
    values = np.maximum(values, 0)
    values /= values.sum()
    yy, xx = np.indices(values.shape)
    x0, y0 = np.sum(xx * values), np.sum(yy * values)
    variance = np.sum(values * ((xx - x0) ** 2 + (yy - y0) ** 2)) / 2
    return float(2.35482 * np.sqrt(variance)) if variance > 0 else None


def _input_radial_visibility(data, rms, mask, center_xy, exclude):
    """Measure input-image contrast against same-radius control pixels.

    This is deliberately not a second foreground fit.  Each candidate pixel
    is compared with robust controls in a one-pixel foreground-centric radial
    bin, excluding all detected candidate masks and a two-pixel guard band.
    The result approximates whether the feature is visible in the input VIS
    image rather than existing only after subtraction.
    """
    data = np.asarray(data, dtype=float)
    rms = np.asarray(rms, dtype=float)
    yy, xx = np.indices(mask.shape)
    cx, cy = center_xy
    radius_bin = np.floor(np.hypot(xx - cx, yy - cy)).astype(int)
    guarded = maximum_filter(np.asarray(exclude, dtype=np.uint8), size=5) > 0
    valid = np.isfinite(data) & np.isfinite(rms) & (rms > 0) & ~guarded
    candidate_y, candidate_x = np.where(mask)
    contrast = np.full(candidate_y.size, np.nan)
    baseline_error = np.full(candidate_y.size, np.nan)
    for radial_bin in np.unique(radius_bin[mask]):
        controls = valid & (radius_bin == radial_bin)
        values = data[controls]
        if values.size < 12:
            continue
        baseline = float(np.median(values))
        mad = float(1.4826 * np.median(np.abs(values - baseline)))
        use = radius_bin[candidate_y, candidate_x] == radial_bin
        contrast[use] = data[candidate_y[use], candidate_x[use]] - baseline
        baseline_error[use] = mad / np.sqrt(values.size)
    measured = np.isfinite(contrast)
    coverage = float(np.mean(measured)) if measured.size else 0.0
    if not measured.any():
        return None, None, coverage
    local_rms = rms[candidate_y[measured], candidate_x[measured]]
    noise = float(np.sqrt(np.sum(local_rms ** 2 + baseline_error[measured] ** 2)))
    snr = float(np.sum(contrast[measured]) / noise) if noise > 0 else None
    positive = float(np.mean(contrast[measured] / local_rms >= 1.0))
    return snr, positive, coverage


def _measure(document, candidate, arrays, threshold, inner_radius):
    key = f"model_candidate_mask_{candidate.id}"
    required = (key, "residual", "rms")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"missing pixel arrays: {', '.join(missing)}")
    raw_mask = np.asarray(arrays[key])
    if raw_mask.shape != document.image_shape or not np.isin(raw_mask, [0, 1]).all():
        raise ValueError("candidate mask must be binary and match the image shape")
    mask = raw_mask.astype(bool)
    if not mask.any():
        raise ValueError("empty candidate mask")
    if int(mask.sum()) != candidate.pixel_count:
        raise ValueError("saved mask pixel count disagrees with candidate provenance")
    residual, rms = (np.asarray(arrays[name], dtype=float) for name in ("residual", "rms"))
    for name, array in (("residual", residual), ("rms", rms)):
        if array.shape != mask.shape or not np.isfinite(array[mask]).all():
            raise ValueError(f"{name} invalid at candidate pixels")
    if not (rms[mask] > 0).all():
        raise ValueError("input RMS must be positive at candidate pixels")
    xs_center, ys_center = document.nucleus_xy
    if "nucleus_yx" in arrays and not np.allclose(
        np.asarray(arrays["nucleus_yx"]), [ys_center, xs_center], atol=1e-6, rtol=0,
    ):
        raise ValueError("saved pixel center disagrees with observation")
    ys, xs = np.where(mask)
    centroid = (float(xs.mean()), float(ys.mean()))
    dx, dy = xs - xs_center, ys - ys_center
    radii = np.hypot(dx, dy)
    core_radius = float(document.provenance.parameters.get("near_core_radius_px", 6.5))
    if not np.isfinite(core_radius) or core_radius <= 0:
        raise ValueError("core radius must be finite and positive")
    core_overlap = float(np.mean(radii <= core_radius))
    radial_width_px = float(np.percentile(radii, 90) - np.percentile(radii, 10))
    psf_fwhm = _psf_width(arrays["psf"]) if "psf" in arrays else None
    radial_width_psf = (
        radial_width_px / psf_fwhm
        if psf_fwhm is not None and psf_fwhm > 0
        else None
    )
    ordered_angles = np.sort(np.degrees(np.arctan2(dy, dx)) % 360)
    extent = float(360 - np.max(np.diff(np.r_[ordered_angles, ordered_angles[0] + 360])))
    axis_ratio = misalignment = None
    if len(xs) >= 2:
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(np.column_stack((xs, ys)), rowvar=False, bias=True))
        if eigenvalues[-1] > 0:
            axis_ratio = float(np.sqrt(eigenvalues[-1] / max(eigenvalues[0], 1e-9)))
            radial = np.array([centroid[0] - xs_center, centroid[1] - ys_center])
            if np.linalg.norm(radial) > 0:
                tangent = np.array([-radial[1], radial[0]]) / np.linalg.norm(radial)
                misalignment = float(np.degrees(np.arccos(np.clip(abs(eigenvectors[:, -1] @ tangent), 0, 1))))
    noise = float(np.sqrt(np.sum(rms[mask] ** 2)))
    flux = float(residual[mask].sum())
    if not np.isfinite([noise, flux]).all() or noise <= 0:
        raise ValueError("non-finite or zero aperture-noise measurement")
    input_contrast_snr = input_positive_fraction = input_control_coverage = None
    if "data" in arrays:
        data = np.asarray(arrays["data"], dtype=float)
        if data.shape == mask.shape:
            exclude = np.asarray(
                arrays.get("all_candidate_mask", mask), dtype=bool
            )
            input_contrast_snr, input_positive_fraction, input_control_coverage = (
                _input_radial_visibility(
                    data, rms, mask, (xs_center, ys_center), exclude
                )
            )
    survival = nearby = negative = comparison_flux = None
    detection = "unavailable"
    notes = [
        "Geometry and detection reuse the same residual pixels; they are correlated, search-conditioned measurements.",
        "Aperture SNR uses diagonal input RMS only; pixel covariance, foreground-fit uncertainty and search trials are not calibrated.",
        "Comparison threshold occupancy is not a retained-flux fraction or a probability of being real.",
        "This review reports pixel geometry; sky coordinates are not recomputed.",
    ]
    comparison = document.foreground_model_comparison
    if comparison is not None and comparison.comparison_success and "comparison_psf_mge_residual" in arrays:
        comp = np.asarray(arrays["comparison_psf_mge_residual"], dtype=float)
        if comp.shape == mask.shape and np.isfinite(comp[mask]).all():
            valid = np.isfinite(comp) & np.isfinite(rms) & (rms > 0)
            comp_snr = np.full(mask.shape, np.nan)
            np.divide(comp, rms, out=comp_snr, where=valid)
            survival = float(np.mean(comp_snr[mask] >= threshold))
            negative = float(np.mean(comp_snr[mask] <= -threshold))
            nearby = float(np.mean(maximum_filter((comp_snr >= threshold).astype(float), size=3, mode="constant")[mask]))
            comparison_flux = float(comp[mask].sum())
            detection = "detected" if survival > 0 else "below_threshold"
            if survival == 0:
                notes.append("No candidate pixels exceed the comparison threshold; this does not establish erasure, artifact or intrinsic origin.")
        else:
            notes.append("Comparison pixels are invalid; model sensitivity is unavailable, not negative evidence.")
    if detection == "unavailable":
        notes.append("No successful, valid comparison residual is available.")
    count, total = candidate.snr_threshold_survival_count, candidate.snr_threshold_total_count
    persistence = count / total if count is not None and total else candidate.snr_threshold_survival_fraction
    inner = float(np.mean(radii <= inner_radius + 1.0))
    audits = [a.id for a in document.candidate_structure_audits if a.candidate_id == candidate.id]
    measurement = StructureObservation(
        id=f"REVIEW_{candidate.id}", centroid_xy=centroid,
        radius_px=float(np.hypot(centroid[0] - xs_center, centroid[1] - ys_center)),
        position_angle_deg=float(np.degrees(np.arctan2(-(centroid[0] - xs_center), -(centroid[1] - ys_center))) % 360),
        pixel_count=len(xs), integrated_flux=flux, integrated_snr=flux / noise,
        median_radius_px=float(np.median(radii)), angular_extent_deg=extent,
        axis_ratio=axis_ratio, tangential_misalignment_deg=misalignment,
        threshold_survival_fraction=persistence, sep_survival_fraction=candidate.sep_survival_fraction,
        inner_connection_fraction=inner, connects_to_inner_region=inner > 0,
        near_core=core_overlap > 0, subtraction_core_overlap_fraction=core_overlap,
        comparison_integrated_snr=comparison_flux / noise if comparison_flux is not None else None,
        occupied_angle_bins_5deg=sorted(set((ordered_angles // 5).astype(int).tolist())),
        foreground_comparison_survival_fraction=survival,
        comparison_nearby_survival_fraction=nearby, comparison_negative_fraction=negative,
        input_radial_contrast_snr=input_contrast_snr,
        input_radial_positive_fraction=input_positive_fraction,
        input_radial_control_coverage=input_control_coverage,
        radial_width_px=radial_width_px,
        radial_width_psf=radial_width_psf,
        **measure_curved_geometry(mask, (ys_center, xs_center)),
        source_candidate_ids=[candidate.id], candidate_audit_ids=audits,
        flags=["complete_review_measurement", "eligible_for_integrated_grading", "physical_origin_unresolved"],
    )
    return measurement, noise, comparison_flux, detection, notes


def attach_candidate_review(
    document: ObservationDocument, arrays: Mapping | None = None, *,
    source: str = "saved_pixel_artifacts", output_dir: Path | None = None,
) -> ObservationDocument:
    """Return a copy. Missing/invalid masks remain explicit unmeasurable entries."""
    if arrays is None:
        path = document.artifacts.get("vision_arrays")
        if path and Path(path).is_file():
            with np.load(path, allow_pickle=False) as saved:
                return attach_candidate_review(
                    document, saved, source=source, output_dir=output_dir,
                )
        arrays = {}
    parameters = document.provenance.parameters
    threshold = float(parameters.get("arc_snr_low") or parameters.get("residual_structure_snr_low", 2.0))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("comparison threshold must be finite and positive")
    inner = float(parameters.get("ring_inner_px", 4.0))
    candidates = document.model_candidates
    if len({c.id for c in candidates}) != len(candidates):
        raise ValueError("duplicate model candidate IDs")
    formal = {cid: s.id for s in document.structures for cid in s.source_candidate_ids}
    audits = {a.candidate_id: a for a in document.candidate_structure_audits}
    entries = []
    candidate_masks = [
        np.asarray(arrays[f"model_candidate_mask_{candidate.id}"], dtype=bool)
        for candidate in candidates
        if f"model_candidate_mask_{candidate.id}" in arrays
        and np.asarray(arrays[f"model_candidate_mask_{candidate.id}"]).shape
        == document.image_shape
    ]
    all_candidate_mask = (
        np.logical_or.reduce(candidate_masks)
        if candidate_masks
        else np.zeros(document.image_shape, dtype=bool)
    )
    measurement_arrays = ChainMap(
        {"all_candidate_mask": all_candidate_mask}, arrays
    )
    for candidate in candidates:
        audit = audits.get(candidate.id)
        values = dict(
            candidate_id=candidate.id, formal_structure_id=formal.get(candidate.id),
            legacy_verdict=audit.verdict if audit else None,
            legacy_handoff_class=audit.handoff_class if audit else None,
            mask_key=f"model_candidate_mask_{candidate.id}", comparison_threshold_sigma=threshold,
        )
        try:
            measurement, noise, flux, detection, notes = _measure(
                document, candidate, measurement_arrays, threshold, inner
            )
            mask_hash = hashlib.sha256(np.asarray(arrays[values["mask_key"]], dtype=np.uint8).tobytes()).hexdigest()
            entries.append(CandidateHandoffEntry(
                **values, status="formal_reference" if candidate.id in formal else "measured_for_review",
                mask_sha256=mask_hash,
                measurement=measurement, aperture_noise_diagonal=noise, comparison_aperture_flux=flux,
                comparison_detection=detection, uncertainty_notes=notes,
            ))
        except ValueError as exc:
            entries.append(CandidateHandoffEntry(
                **values, status="unmeasurable", comparison_detection="unavailable",
                uncertainty_notes=[str(exc), "Measurement unavailable; this is not evidence against the candidate."],
            ))
    result = document.model_copy(deep=True)
    result.schema_version = "1.18"
    result.candidate_handoff_review = CandidateHandoffReview(
        source=source, entries=entries,
        grading_policy="integrated_candidate_evidence_v1",
    )
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if candidates and all(name in arrays for name in ("data", "residual")):
            path = output_dir / "candidate_handoff_review.png"
            _draw_review(result, arrays, path)
            result.artifacts["candidate_handoff_review"] = str(path.resolve())
    return result


def _draw_review(document, arrays, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    from matplotlib.lines import Line2D

    entries = {e.candidate_id: e for e in document.candidate_handoff_review.entries}
    xs_center, ys_center = document.nucleus_xy
    half = max(30, *(c.radius_px + 12 for c in document.model_candidates))
    x0, x1 = max(0, int(xs_center - half)), min(document.image_shape[1], int(xs_center + half + 1))
    y0, y1 = max(0, int(ys_center - half)), min(document.image_shape[0], int(ys_center + half + 1))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    try:
        for ax, key, title in zip(axes, ["data", "residual", "comparison_psf_mge_residual"], ["Input", "Primary residual", "Comparison residual"]):
            ax.set_title(title)
            if key in arrays:
                cut = np.asarray(arrays[key])[y0:y1, x0:x1]
                finite = cut[np.isfinite(cut)]
                if finite.size:
                    low, high = np.percentile(finite, [1, 99.5])
                    high = max(float(high), float(low) + 1e-9)
                    ax.imshow(cut, origin="lower", cmap="gray", norm=AsinhNorm(linear_width=(high-low)/20, vmin=low, vmax=high), extent=[x0-.5,x1-.5,y0-.5,y1-.5])
            else:
                ax.text(.5, .5, "Unavailable", transform=ax.transAxes, ha="center")
            for candidate in document.model_candidates:
                entry = entries[candidate.id]
                if entry.measurement is None:
                    continue
                color = "#ed55e9" if candidate.origin == "path_recovery" else "#54e577" if candidate.geometry_consistent_path_ids else "#ffad42"
                mask = np.asarray(arrays[entry.mask_key], dtype=float)
                # Contours outline actual detected pixels; no geometric gap filling.
                ax.contour(mask, levels=[.5], colors=[color], linewidths=.8)
                x, y = entry.measurement.centroid_xy
                ax.annotate(candidate.id, (x, y), xytext=(3, 3), textcoords="offset points", color=color, fontsize=8)
            ax.plot(xs_center, ys_center, "+", color="cyan", ms=8)
            ax.set(xlim=(x0-.5, x1-.5), ylim=(y0-.5, y1-.5), xlabel="x [pixel]")
        axes[0].set_ylabel("y [pixel]")
        fig.legend(handles=[Line2D([0],[0],color=color,label=label) for color,label in [("#ffad42","Off-route SEP"),("#54e577","SEP + geometry"),("#ed55e9","Route recovery")]], loc="outside lower center", ncol=3)
        fig.suptitle(f"{document.object_id}\nComplete candidate evidence: outlines show measured masks, not lens confidence", fontsize=11)
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
