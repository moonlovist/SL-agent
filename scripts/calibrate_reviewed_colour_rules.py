"""Reanalyse independent colour measurements after human structure review.

This calibration script is deliberately separate from the production agent.
It never changes a Q1 grade.  It uses the reviewed semantic manifest to:

* union components that are fragments of one continuous arc/ring;
* restrict same-source colour tests to explicitly reviewed image pairs;
* use accepted manual pixel masks when the saved Ci masks are inadequate;
* exclude objects that remain unmeasurable from threshold calibration;
* retain continuous measurements, quality diagnostics, and current-policy
  classifications so that later rule changes are traceable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.signal import fftconvolve
from scipy.stats import mannwhitneyu

from independent_colour_measurement import (
    BANDS,
    COLOURS,
    aperture_error,
    colours,
    gaussian_kernel,
    local_background,
    mask_hash,
    psf_fwhm,
    radial_model,
    record_photometry,
    sed_fit,
)


PRIMARY_COLOURS = ("I_Y", "Y_J", "J_H")
CURRENT_THRESHOLDS = {
    "minimum_band_snr": 3.0,
    "minimum_reliable_common_bands": 3,
    "systematic_floor_mag": 0.10,
    "consistent_max_departure_mag": 0.20,
    "consistent_max_reduced_chi2": 2.0,
    "inconsistent_min_departure_mag": 0.35,
    "inconsistent_min_sigma": 3.0,
}


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def portable(value: Any) -> Any:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: portable(row.get(key)) for key in fields})


def reliable_bands(photometry: dict[str, dict[str, Any]]) -> list[str]:
    return [band for band in BANDS if photometry[band].get("magnitude") is not None]


def gray_colour_test(
    first: dict[str, dict[str, Any]],
    second: dict[str, dict[str, Any]],
    thresholds: dict[str, float | int] = CURRENT_THRESHOLDS,
) -> dict[str, Any]:
    """Reproduce the current achromatic-scale test without changing agent code."""
    bands = [
        band
        for band in BANDS
        if first[band].get("magnitude") is not None
        and second[band].get("magnitude") is not None
    ]
    flux_diagnostic = sed_fit(first, second)
    if len(bands) < int(thresholds["minimum_reliable_common_bands"]):
        return {
            "status": "unknown",
            "reason": "insufficient_reliable_common_bands",
            "bands": bands,
            "reliable_common_band_count": len(bands),
            "flux_scale_same_sed_reduced_chi2": flux_diagnostic.get("same_sed_reduced_chi2"),
        }
    difference = np.asarray(
        [first[band]["magnitude"] - second[band]["magnitude"] for band in bands],
        dtype=float,
    )
    variance = np.asarray(
        [
            first[band]["magnitude_error"] ** 2
            + second[band]["magnitude_error"] ** 2
            + float(thresholds["systematic_floor_mag"]) ** 2
            for band in bands
        ],
        dtype=float,
    )
    offset = float(np.sum(difference / variance) / np.sum(1.0 / variance))
    departures = difference - offset
    significance = np.abs(departures) / np.sqrt(variance)
    maximum = float(np.max(np.abs(departures)))
    max_sigma = float(np.max(significance))
    chi2 = float(np.sum(departures**2 / variance))
    dof = len(bands) - 1
    reduced = chi2 / max(dof, 1)
    if (
        maximum >= float(thresholds["inconsistent_min_departure_mag"])
        and max_sigma >= float(thresholds["inconsistent_min_sigma"])
    ):
        status = "strongly_inconsistent"
    elif (
        maximum <= float(thresholds["consistent_max_departure_mag"])
        and reduced <= float(thresholds["consistent_max_reduced_chi2"])
    ):
        status = "consistent_with_achromatic_scaling"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "bands": bands,
        "reliable_common_band_count": len(bands),
        "gray_offset_mag": offset,
        "maximum_color_departure_mag": maximum,
        "maximum_departure_sigma": max_sigma,
        "chi2": chi2,
        "degrees_of_freedom": dof,
        "reduced_chi2": reduced,
        "flux_scale_same_sed_reduced_chi2": flux_diagnostic.get("same_sed_reduced_chi2"),
    }


def max_colour_shift(
    first: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]]
) -> float | None:
    c1, c2 = colours(first), colours(second)
    values = []
    for name in PRIMARY_COLOURS:
        a, b = c1.get(f"colour_{name}"), c2.get(f"colour_{name}")
        if a is not None and b is not None:
            values.append(abs(float(a) - float(b)))
    return max(values) if values else None


def load_inputs(
    object_id: str,
    objects_by_id: dict[str, dict[str, str]],
    measurement_root: Path,
) -> dict[str, Any]:
    object_row = objects_by_id[object_id]
    fits_path = Path(object_row["fits_path"])
    with fits.open(fits_path, memmap=False) as hdus:
        flux = {band: np.asarray(hdus[f"{band}_FLUX"].data, float) for band in BANDS}
        rms = {band: np.asarray(hdus[f"{band}_RMS"].data, float) for band in BANDS}
        psf = {band: np.asarray(hdus[f"{band}_PSF"].data, float) for band in BANDS}
        headers = {band: hdus[f"{band}_FLUX"].header.copy() for band in BANDS}
    scaled_flux = {
        band: flux[band] * 10 ** (-0.4 * (float(headers[band]["MAGZERO"]) - 25.0))
        for band in BANDS
    }
    scaled_rms = {
        band: rms[band] * 10 ** (-0.4 * (float(headers[band]["MAGZERO"]) - 25.0))
        for band in BANDS
    }
    fwhm = {band: psf_fwhm(psf[band]) for band in BANDS}
    if any(value is None for value in fwhm.values()):
        raise ValueError(f"{object_id}: invalid PSF")
    target = max(float(value) for value in fwhm.values() if value is not None)
    kernels = {band: gaussian_kernel(float(fwhm[band]), target) for band in BANDS}
    matched = {
        band: fftconvolve(scaled_flux[band], kernels[band], mode="same")
        for band in BANDS
    }
    wider_target = target * 1.10
    wider_kernels = {
        band: gaussian_kernel(float(fwhm[band]), wider_target) for band in BANDS
    }
    wider_matched = {
        band: fftconvolve(scaled_flux[band], wider_kernels[band], mode="same")
        for band in BANDS
    }
    mask_dir = measurement_root / "masks" / object_id
    masks: dict[str, np.ndarray] = {}
    for path in sorted(mask_dir.glob("C*_mask.fits")):
        masks[path.stem.removesuffix("_mask")] = fits.getdata(path).astype(bool)
    diagnostics = json.loads((mask_dir / "diagnostics.json").read_text(encoding="utf-8"))
    center = tuple(float(value) for value in diagnostics["center_xy"])
    return {
        "fits_path": fits_path,
        "matched": matched,
        "wider_matched": wider_matched,
        "rms": scaled_rms,
        "kernels": kernels,
        "wider_kernels": wider_kernels,
        "target": target,
        "wider_target": wider_target,
        "masks": masks,
        "center": center,
    }


def measure(
    images: dict[str, np.ndarray],
    rms: dict[str, np.ndarray],
    kernels: dict[str, np.ndarray],
    mask: np.ndarray,
    exclusions: np.ndarray,
) -> dict[str, dict[str, Any]]:
    return {
        band: record_photometry(images[band], rms[band], kernels[band], mask, exclusions)[0]
        for band in BANDS
    }


def flatten_photometry(prefix: str, photometry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for band in BANDS:
        for key in ("flux_local_bg_subtracted", "flux_error", "snr", "magnitude", "magnitude_error", "upper_limit"):
            row[f"{prefix}_{band}_{key}"] = photometry[band].get(key)
    for key, value in colours(photometry).items():
        row[f"{prefix}_{key}"] = value
    row[f"{prefix}_reliable_band_count"] = len(reliable_bands(photometry))
    return row


def analyse_structure(
    review: dict[str, Any],
    loaded: dict[str, Any],
    manual_mask_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    masks = loaded["masks"]
    components = review.get("arc_components", [])
    missing = [component for component in components if component not in masks]
    if missing:
        raise ValueError(f"{review['object_id']}: missing masks {missing}")
    manual_arc_path = review.get("manual_arc_mask")
    if manual_arc_path:
        if manual_mask_root is None:
            raise ValueError(f"{review['object_id']}: --manual-mask-root is required")
        arc_mask = fits.getdata(manual_mask_root / manual_arc_path).astype(bool)
        mask_source = "accepted_manual_mask"
    else:
        if not components:
            raise ValueError(f"{review['object_id']}: no arc mask or arc components")
        arc_mask = np.logical_or.reduce([masks[component] for component in components])
        mask_source = "reviewed_existing_components"
    reference_mask = masks["C0"]
    reference_id = "C0"
    manual_foreground_path = review.get("manual_foreground_mask")
    if manual_foreground_path:
        if manual_mask_root is None:
            raise ValueError(f"{review['object_id']}: --manual-mask-root is required")
        reference_mask = fits.getdata(manual_mask_root / manual_foreground_path).astype(bool)
        reference_id = "MANUAL_FOREGROUND"
    expected_shape = next(iter(masks.values())).shape
    if arc_mask.shape != expected_shape or reference_mask.shape != expected_shape:
        raise ValueError(
            f"{review['object_id']}: manual mask shape does not match measurement images"
        )
    all_mask_parts = list(masks.values()) + [arc_mask, reference_mask]
    all_masks = np.logical_or.reduce(all_mask_parts)
    candidate_masks = [mask for name, mask in masks.items() if name != "C0"]
    candidate_masks.append(arc_mask)
    candidate_union = (
        np.logical_or.reduce(candidate_masks)
        if candidate_masks
        else np.zeros_like(arc_mask, dtype=bool)
    )
    exclusions = binary_dilation(all_masks, iterations=2) & ~arc_mask
    reference_exclusions = binary_dilation(all_masks, iterations=2) & ~reference_mask
    nominal = measure(loaded["matched"], loaded["rms"], loaded["kernels"], arc_mask, exclusions)
    foreground_reference = measure(
        loaded["matched"],
        loaded["rms"],
        loaded["kernels"],
        reference_mask,
        reference_exclusions,
    )
    radial_images = {
        band: loaded["matched"][band]
        - radial_model(
            loaded["matched"][band],
            loaded["center"],
            binary_dilation(candidate_union, iterations=2),
        )
        for band in BANDS
    }
    radial = measure(radial_images, loaded["rms"], loaded["kernels"], arc_mask, exclusions)
    variants: dict[str, dict[str, dict[str, Any]]] = {
        "nominal": nominal,
        "radial_subtracted": radial,
    }
    for name, mask in (
        ("erode_1px", binary_erosion(arc_mask, iterations=1)),
        ("dilate_1px", binary_dilation(arc_mask, iterations=1)),
        ("dilate_2px", binary_dilation(arc_mask, iterations=2)),
    ):
        if int(mask.sum()) >= 3:
            variant_exclusions = binary_dilation(all_masks, iterations=2) & ~mask
            variants[name] = measure(
                loaded["matched"], loaded["rms"], loaded["kernels"], mask, variant_exclusions
            )
    outer = measure(loaded["matched"], loaded["rms"], loaded["kernels"], arc_mask, exclusions)
    for band in BANDS:
        background, scatter, count = local_background(
            loaded["matched"][band], arc_mask, exclusions, "outer"
        )
        if background is None:
            continue
        raw = outer[band]["flux_raw"]
        flux = float(raw - background * arc_mask.sum())
        bg_error = arc_mask.sum() * float(scatter) / np.sqrt(count) if scatter is not None and count else 0.0
        diagonal_error = aperture_error(loaded["rms"][band], loaded["kernels"][band], arc_mask)
        error = float(np.hypot(diagonal_error, bg_error))
        snr = flux / error if error > 0 else None
        reliable = bool(snr is not None and snr >= 3.0 and flux > 0)
        outer[band].update(
            {
                "flux_local_bg_subtracted": flux,
                "flux_error": error,
                "snr": snr,
                "magnitude": float(25.0 - 2.5 * np.log10(flux)) if reliable else None,
                "magnitude_error": float(2.5 / np.log(10) / snr) if reliable else None,
                "upper_limit": float(3.0 * error) if not reliable else None,
            }
        )
    variants["outer_background"] = outer
    variants["psf_target_plus10pct"] = measure(
        loaded["wider_matched"],
        loaded["rms"],
        loaded["wider_kernels"],
        arc_mask,
        exclusions,
    )
    aperture_changes = [
        max_colour_shift(nominal, variants[name])
        for name in ("erode_1px", "dilate_1px", "dilate_2px")
        if name in variants
    ]
    aperture_changes = [value for value in aperture_changes if value is not None]
    nominal_foreground = gray_colour_test(nominal, foreground_reference)
    radial_foreground = gray_colour_test(radial, foreground_reference)
    row: dict[str, Any] = {
        "object_id": review["object_id"],
        "cohort": review["cohort"],
        "structure_type": review["structure_type"],
        "calibration_status": review["calibration_status"],
        "mask_source": mask_source,
        "manual_arc_mask": manual_arc_path,
        "foreground_reference_id": reference_id,
        "manual_foreground_mask": manual_foreground_path,
        "arc_components": ";".join(components),
        "arc_component_count": len(components),
        "arc_union_pixel_count": int(arc_mask.sum()),
        "arc_union_mask_sha256": mask_hash(arc_mask),
        "review_note": review.get("note"),
        "foreground_test_nominal_status": nominal_foreground["status"],
        "foreground_test_radial_status": radial_foreground["status"],
        "foreground_test_radial_common_bands": radial_foreground.get("reliable_common_band_count"),
        "foreground_test_radial_max_departure_mag": radial_foreground.get("maximum_color_departure_mag"),
        "foreground_test_radial_max_departure_sigma": radial_foreground.get("maximum_departure_sigma"),
        "foreground_test_radial_reduced_chi2": radial_foreground.get("reduced_chi2"),
        "foreground_test_radial_flux_reduced_chi2": radial_foreground.get("flux_scale_same_sed_reduced_chi2"),
        "aperture_sensitivity_max_primary_colour_mag": max(aperture_changes) if aperture_changes else None,
        "foreground_subtraction_shift_max_primary_colour_mag": max_colour_shift(nominal, radial),
        "background_sensitivity_max_primary_colour_mag": max_colour_shift(nominal, outer),
        "psf_sensitivity_max_primary_colour_mag": max_colour_shift(
            nominal, variants["psf_target_plus10pct"]
        ),
    }
    row.update(flatten_photometry("arc_nominal", nominal))
    row.update(flatten_photometry("arc_radial", radial))
    row.update(flatten_photometry("foreground_reference", foreground_reference))
    return row, {
        review["object_id"]: variants | {"foreground_reference": foreground_reference}
    }


def pair_record(
    review: dict[str, Any],
    loaded: dict[str, Any],
    first: str,
    second: str,
) -> dict[str, Any]:
    masks = loaded["masks"]
    all_masks = np.logical_or.reduce(list(masks.values()))
    candidate_masks = [mask for name, mask in masks.items() if name != "C0"]
    candidate_union = np.logical_or.reduce(candidate_masks)
    radial_images = {
        band: loaded["matched"][band]
        - radial_model(
            loaded["matched"][band],
            loaded["center"],
            binary_dilation(candidate_union, iterations=2),
        )
        for band in BANDS
    }
    photometry = {}
    for component in (first, second):
        mask = masks[component]
        exclusions = binary_dilation(all_masks, iterations=2) & ~mask
        photometry[component] = measure(
            radial_images, loaded["rms"], loaded["kernels"], mask, exclusions
        )
    test = gray_colour_test(photometry[first], photometry[second])
    return {
        "object_id": review["object_id"],
        "cohort": review["cohort"],
        "calibration_status": review["calibration_status"],
        "component_1": first,
        "component_2": second,
        "reviewed_pair_certainty": (
            "tentative" if review["calibration_status"] == "exploratory_uncertain_pair" else "accepted"
        ),
        "current_policy_status": test["status"],
        "reliable_common_band_count": test.get("reliable_common_band_count"),
        "maximum_color_departure_mag": test.get("maximum_color_departure_mag"),
        "maximum_departure_sigma": test.get("maximum_departure_sigma"),
        "reduced_chi2": test.get("reduced_chi2"),
        "flux_scale_same_sed_reduced_chi2": test.get("flux_scale_same_sed_reduced_chi2"),
        "review_note": review.get("note"),
    }


def group_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = {
        cohort: [finite(row.get(metric)) for row in rows if row["cohort"] == cohort]
        for cohort in ("A", "B")
    }
    values = {key: [value for value in group if value is not None] for key, group in values.items()}
    result: dict[str, Any] = {
        "metric": metric,
        "A_n": len(values["A"]),
        "B_n": len(values["B"]),
        "A_median": float(np.median(values["A"])) if values["A"] else None,
        "B_median": float(np.median(values["B"])) if values["B"] else None,
    }
    if values["A"] and values["B"]:
        u, p = mannwhitneyu(values["A"], values["B"], alternative="two-sided")
        result["mann_whitney_u"] = float(u)
        result["two_sided_p"] = float(p)
        result["rank_auc_A_greater"] = float(u / (len(values["A"]) * len(values["B"])))
    else:
        result.update({"mann_whitney_u": None, "two_sided_p": None, "rank_auc_A_greater": None})
    return result


def make_plots(rows: list[dict[str, Any]], output: Path) -> None:
    metrics = [
        ("foreground_test_radial_max_departure_mag", "Arc vs foreground-reference maximum colour departure (mag)"),
        ("foreground_test_radial_reduced_chi2", "Arc vs foreground-reference reduced chi-square"),
        ("foreground_subtraction_shift_max_primary_colour_mag", "Foreground-subtraction colour shift (mag)"),
        ("aperture_sensitivity_max_primary_colour_mag", "Aperture sensitivity (mag)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    rng = np.random.default_rng(20260919)
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        for index, cohort in enumerate(("A", "B")):
            subset = [row for row in rows if row["cohort"] == cohort]
            values = [finite(row.get(metric)) for row in subset]
            values = [value for value in values if value is not None]
            jitter = rng.normal(0.0, 0.035, len(values)) if values else []
            axis.scatter(np.full(len(values), index) + jitter, values, s=42, alpha=0.85, label=cohort)
            if values:
                axis.hlines(np.median(values), index - 0.18, index + 0.18, color="black", linewidth=2)
        axis.set_xticks([0, 1], ["A", "B"])
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    fig.suptitle("Reviewed-mask colour diagnostics; accepted manual masks included")
    fig.tight_layout()
    fig.savefig(output / "ab_colour_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    categories = ("A", "B")
    statuses = ("consistent_with_achromatic_scaling", "inconclusive", "strongly_inconsistent", "unknown")
    bottoms = np.zeros(2, dtype=int)
    colours_by_status = ("#4c9f70", "#d9a441", "#b64b4b", "#8b8b8b")
    for status, colour in zip(statuses, colours_by_status):
        counts = [
            sum(1 for row in rows if row["cohort"] == cohort and row["foreground_test_radial_status"] == status)
            for cohort in categories
        ]
        axis.bar(categories, counts, bottom=bottoms, label=status, color=colour)
        bottoms += np.asarray(counts)
    axis.set_ylabel("Object count")
    axis.set_title("Current-policy arc vs foreground-reference colour classification")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "ab_foreground_status_counts.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manual-mask-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with (args.measurements / "objects.csv").open(encoding="utf-8-sig", newline="") as handle:
        objects = list(csv.DictReader(handle))
    objects_by_id = {row["object_id"]: row for row in objects}
    review_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    remask_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for review in manifest["objects"]:
        review_rows.append(
            {
                "object_id": review["object_id"],
                "cohort": review["cohort"],
                "structure_type": review["structure_type"],
                "arc_components": ";".join(review.get("arc_components", [])),
                "same_source_pairs": json.dumps(review.get("same_source_pairs", []), ensure_ascii=False),
                "calibration_status": review["calibration_status"],
                "review_note": review.get("note"),
            }
        )
        if review["calibration_status"] == "needs_manual_remask":
            remask_rows.append(review_rows[-1])
            continue
        if review["calibration_status"].startswith("discarded_"):
            excluded_rows.append(review_rows[-1])
            continue
        loaded = load_inputs(review["object_id"], objects_by_id, args.measurements)
        structure, _ = analyse_structure(review, loaded, args.manual_mask_root)
        structure_rows.append(structure)
        for first, second in review.get("same_source_pairs", []):
            pair_rows.append(pair_record(review, loaded, first, second))
    calibration_rows = [
        row
        for row in structure_rows
        if row["calibration_status"] in {"usable_existing_mask", "usable_manual_mask"}
    ]
    metrics = [
        "foreground_test_radial_max_departure_mag",
        "foreground_test_radial_max_departure_sigma",
        "foreground_test_radial_reduced_chi2",
        "foreground_test_radial_flux_reduced_chi2",
        "foreground_subtraction_shift_max_primary_colour_mag",
        "background_sensitivity_max_primary_colour_mag",
        "aperture_sensitivity_max_primary_colour_mag",
        "psf_sensitivity_max_primary_colour_mag",
        "arc_radial_reliable_band_count",
    ]
    group_statistics = [group_summary(calibration_rows, metric) for metric in metrics]
    status_counts = {
        cohort: {
            status: sum(
                1
                for row in calibration_rows
                if row["cohort"] == cohort and row["foreground_test_radial_status"] == status
            )
            for status in ("consistent_with_achromatic_scaling", "inconclusive", "strongly_inconsistent", "unknown")
        }
        for cohort in ("A", "B")
    }
    summary = {
        "schema": "reviewed_colour_calibration_summary_v2",
        "input_manifest": str(args.manifest.resolve()),
        "input_measurements": str(args.measurements.resolve()),
        "input_manual_mask_root": (
            str(args.manual_mask_root.resolve()) if args.manual_mask_root else None
        ),
        "current_thresholds_evaluated_without_modification": CURRENT_THRESHOLDS,
        "reviewed_object_count": len(review_rows),
        "measured_structure_count": len(structure_rows),
        "existing_mask_structure_count": sum(
            row["mask_source"] == "reviewed_existing_components" for row in structure_rows
        ),
        "manual_mask_structure_count": sum(
            row["mask_source"] == "accepted_manual_mask" for row in structure_rows
        ),
        "threshold_calibration_object_count": len(calibration_rows),
        "manual_remask_count": len(remask_rows),
        "discarded_object_count": len(excluded_rows),
        "same_source_pair_count": len(pair_rows),
        "accepted_same_source_pair_count": sum(
            row["reviewed_pair_certainty"] == "accepted" for row in pair_rows
        ),
        "foreground_status_counts": status_counts,
        "group_statistics": group_statistics,
        "interpretation_constraints": [
            "The 20-object set contains only A and B labels and cannot calibrate C or unknown boundaries.",
            "A and B are both lens-candidate classes; arc-versus-foreground colour contrast is not expected to separate them by itself.",
            "Only one reviewed same-source pair is accepted, so same-source thresholds cannot be fitted from this set.",
            "Exploratory, unconfirmed-remask, and discarded objects do not set thresholds.",
        ],
    }
    write_csv(args.output / "review_manifest.csv", review_rows)
    write_csv(args.output / "reviewed_structure_colours.csv", structure_rows)
    write_csv(args.output / "reviewed_same_source_pairs.csv", pair_rows)
    write_csv(args.output / "manual_remask_queue.csv", remask_rows)
    write_csv(args.output / "excluded_objects.csv", excluded_rows)
    write_csv(args.output / "ab_metric_statistics.csv", group_statistics)
    (args.output / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=portable), encoding="utf-8"
    )
    make_plots(calibration_rows, args.output)
    print(json.dumps({key: summary[key] for key in (
        "reviewed_object_count",
        "measured_structure_count",
        "existing_mask_structure_count",
        "manual_mask_structure_count",
        "threshold_calibration_object_count",
        "manual_remask_count",
        "discarded_object_count",
        "same_source_pair_count",
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
