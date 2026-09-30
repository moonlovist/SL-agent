"""Analyse manually labelled colour structures without changing LensAgent grades.

The input is the standalone napari annotation output.  The script performs
common-PSF, protected foreground-profile photometry on the human masks and
keeps three evidence families separate:

* colour consistency between explicitly grouped independent-image candidates;
* colour contrast between a candidate and the manually marked foreground;
* geometry and annotation observability diagnostics.

It writes auditable CSV/JSON/Markdown outputs.  It does not import annotations
into the production Agent and does not alter any grading threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.signal import fftconvolve
from scipy.stats import mannwhitneyu

from independent_colour_measurement import BANDS, aperture_error, gaussian_kernel, mask_hash, psf_fwhm
from expert_agent.vision.supplementary import _gray_colour_test, _rotate_mask, _robust_scatter


ROLE_CANDIDATE = {"independent_image_candidate", "arc_or_ring_single_structure"}
ROLE_LEGACY_MAP = {
    "compact_image_candidate": "independent_image_candidate",
    "arc_fragment_same_structure": "arc_or_ring_single_structure",
}


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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: portable(row.get(key)) for key in fields})


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def annotation_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("annotation.json") if path.parent != root)


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.where(mask)
    if not len(xx):
        raise ValueError("empty mask")
    return float(np.mean(xx)), float(np.mean(yy))


def mask_shape(mask: np.ndarray) -> dict[str, float | None]:
    yy, xx = np.where(mask)
    if len(xx) < 2:
        return {"axis_ratio": None, "major_axis_angle_deg": None, "elongation": None}
    coords = np.column_stack((xx - np.mean(xx), yy - np.mean(yy)))
    covariance = np.cov(coords, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    major, minor = float(max(values[order[0]], 0.0)), float(max(values[order[1]], 0.0))
    vector = vectors[:, order[0]]
    angle = float(np.mod(np.degrees(np.arctan2(vector[1], vector[0])), 180.0))
    ratio = float(np.sqrt(minor / major)) if major > 0 else None
    return {
        "axis_ratio": ratio,
        "major_axis_angle_deg": angle,
        "elongation": float(1.0 - ratio) if ratio is not None else None,
    }


def foreground_geometry(mask: np.ndarray) -> tuple[tuple[float, float], float, float]:
    center = mask_centroid(mask)
    shape = mask_shape(mask)
    q = shape["axis_ratio"] if shape["axis_ratio"] is not None else 1.0
    q = float(np.clip(q, 0.25, 1.0))
    pa = float(shape["major_axis_angle_deg"] or 0.0)
    return center, q, pa


def elliptical_residual(
    data: np.ndarray,
    center: tuple[float, float],
    axis_ratio: float,
    position_angle_deg: float,
    exclude: np.ndarray,
) -> np.ndarray | None:
    yy, xx = np.indices(data.shape)
    dx, dy = xx - center[0], yy - center[1]
    angle = np.radians(position_angle_deg)
    xrot = dx * np.cos(angle) + dy * np.sin(angle)
    yrot = -dx * np.sin(angle) + dy * np.cos(angle)
    radius = np.sqrt(xrot**2 + (yrot / max(axis_ratio, 0.25)) ** 2)
    centers: list[float] = []
    values: list[float] = []
    for low in np.arange(0.0, min(40.0, float(radius.max())), 0.75):
        use = (radius >= low) & (radius < low + 0.75) & ~exclude & np.isfinite(data)
        if int(use.sum()) >= 12:
            centers.append(float(low + 0.375))
            values.append(float(np.median(data[use])))
    if len(centers) < 8:
        return None
    model = np.interp(radius, centers, values, left=values[0], right=values[-1])
    return data - model


def magnitude_record(flux: float, error: float, local_count: int) -> dict[str, Any]:
    snr = flux / error if error > 0 else None
    measured = bool(snr is not None and np.isfinite(snr) and snr >= 3.0 and flux > 0 and local_count >= 4)
    return {
        "flux_zp25": float(flux),
        "error_including_local_controls": float(error),
        "snr": float(snr) if snr is not None else None,
        "magnitude": float(25.0 - 2.5 * np.log10(flux)) if measured else None,
        "magnitude_error": float(2.5 / np.log(10) / snr) if measured else None,
        "local_control_count": int(local_count),
        "status": "measured" if measured else "low_snr_or_insufficient_controls",
    }


def load_multiband(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=False) as hdus:
        flux = {band: np.asarray(hdus[f"{band}_FLUX"].data, float) for band in BANDS}
        rms = {band: np.asarray(hdus[f"{band}_RMS"].data, float) for band in BANDS}
        psf = {band: np.asarray(hdus[f"{band}_PSF"].data, float) for band in BANDS}
        zeropoint = {band: float(hdus[f"{band}_FLUX"].header["MAGZERO"]) for band in BANDS}
    scaled_flux = {band: flux[band] * 10 ** (-0.4 * (zeropoint[band] - 25.0)) for band in BANDS}
    scaled_rms = {band: rms[band] * 10 ** (-0.4 * (zeropoint[band] - 25.0)) for band in BANDS}
    widths = {band: psf_fwhm(psf[band]) for band in BANDS}
    if any(value is None for value in widths.values()):
        raise ValueError(f"invalid PSF in {path}")
    target = max(float(value) for value in widths.values() if value is not None)
    kernels = {band: gaussian_kernel(float(widths[band]), target) for band in BANDS}
    matched = {band: fftconvolve(scaled_flux[band], kernels[band], mode="same") for band in BANDS}
    return {
        "flux": scaled_flux,
        "rms": scaled_rms,
        "kernels": kernels,
        "matched": matched,
        "target_fwhm": target,
    }


def foreground_photometry(
    loaded: dict[str, Any], foreground: np.ndarray, center: tuple[float, float]
) -> dict[str, dict[str, Any]]:
    yy, xx = np.indices(foreground.shape)
    radius = np.hypot(xx - center[0], yy - center[1])
    sky = (radius >= 45.0) & (radius <= 65.0)
    records: dict[str, dict[str, Any]] = {}
    for band in BANDS:
        data = loaded["matched"][band]
        noise = loaded["rms"][band]
        sky_values = data[sky]
        sky_level = float(np.median(sky_values))
        flux = float(np.sum(data[foreground] - sky_level))
        error = aperture_error(noise, loaded["kernels"][band], foreground)
        records[band] = magnitude_record(flux, error, 99)
    return records


def candidate_photometry(
    loaded: dict[str, Any],
    mask: np.ndarray,
    center: tuple[float, float],
    residuals: dict[str, np.ndarray | None],
    blocked: np.ndarray,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for band in BANDS:
        residual = residuals[band]
        if residual is None:
            records[band] = {
                "flux_zp25": None,
                "error_including_local_controls": None,
                "snr": None,
                "magnitude": None,
                "magnitude_error": None,
                "local_control_count": 0,
                "status": "foreground_model_failed",
            }
            continue
        diagonal = aperture_error(loaded["rms"][band], loaded["kernels"][band], mask)
        controls = []
        for angle in np.arange(24.0, 360.0, 24.0):
            control = _rotate_mask(mask, center, float(angle))
            if int(control.sum()) < max(3, int(0.8 * mask.sum())):
                continue
            if float(np.mean(blocked[control])) > 0.15:
                continue
            controls.append(float(np.sum(residual[control])))
        local_scatter = _robust_scatter(controls)
        error = max(diagonal, local_scatter or 0.0)
        record = magnitude_record(float(np.sum(residual[mask])), error, len(controls))
        record["diagonal_noise_with_common_psf"] = diagonal
        record["local_control_scatter"] = local_scatter
        records[band] = record
    return records


def flatten_photometry(prefix: str, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    reliable = 0
    for band in BANDS:
        record = records[band]
        for key in ("flux_zp25", "error_including_local_controls", "snr", "magnitude", "magnitude_error", "local_control_count", "status"):
            row[f"{prefix}_{band}_{key}"] = record.get(key)
        reliable += record.get("magnitude") is not None
    row[f"{prefix}_reliable_band_count"] = int(reliable)
    return row


def circular_separation(first: float, second: float) -> float:
    delta = abs(first - second) % 360.0
    return float(min(delta, 360.0 - delta))


def angular_occupancy(mask: np.ndarray, center: tuple[float, float], bins: int = 36) -> float:
    yy, xx = np.where(mask)
    if not len(xx):
        return 0.0
    angles = np.mod(np.arctan2(yy - center[1], xx - center[0]), 2 * np.pi)
    occupied = np.unique(np.floor(angles / (2 * np.pi) * bins).astype(int) % bins)
    return float(len(occupied) * 360.0 / bins)


def structure_geometry(mask: np.ndarray, center: tuple[float, float]) -> dict[str, Any]:
    cx, cy = mask_centroid(mask)
    dx, dy = cx - center[0], cy - center[1]
    radius = float(np.hypot(dx, dy))
    angle = float(np.mod(np.degrees(np.arctan2(dy, dx)), 360.0))
    shape = mask_shape(mask)
    major = shape["major_axis_angle_deg"]
    tangentiality = None
    if major is not None and radius > 0:
        tangent = (angle + 90.0) % 180.0
        difference = abs(((major - tangent + 90.0) % 180.0) - 90.0)
        tangentiality = float(np.cos(np.radians(difference)))
    yy, xx = np.where(mask)
    radii = np.hypot(xx - center[0], yy - center[1])
    return {
        "centroid_x": cx,
        "centroid_y": cy,
        "centroid_radius_px": radius,
        "centroid_angle_deg": angle,
        "radial_min_px": float(np.min(radii)),
        "radial_max_px": float(np.max(radii)),
        "angular_occupancy_deg": angular_occupancy(mask, center),
        "tangentiality": tangentiality,
        **shape,
    }


def group_geometry(rows: list[dict[str, Any]], union_mask: np.ndarray, center: tuple[float, float]) -> dict[str, Any]:
    radii = [float(row["centroid_radius_px"]) for row in rows]
    angles = [float(row["centroid_angle_deg"]) for row in rows]
    separations = [circular_separation(a, b) for a, b in combinations(angles, 2)]
    return {
        "image_count": len(rows),
        "radius_mean_px": float(np.mean(radii)) if radii else None,
        "radius_cv": float(np.std(radii) / np.mean(radii)) if radii and np.mean(radii) > 0 else None,
        "radius_max_min_ratio": float(max(radii) / min(radii)) if radii and min(radii) > 0 else None,
        "maximum_pair_angle_separation_deg": max(separations) if separations else None,
        "opposition_closeness_deg": min(abs(180.0 - value) for value in separations) if separations else None,
        "angular_occupancy_deg": angular_occupancy(union_mask, center),
        "median_tangentiality": float(np.median([row["tangentiality"] for row in rows if row["tangentiality"] is not None])) if any(row["tangentiality"] is not None for row in rows) else None,
    }


def metric_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = {
        cohort: [finite(row.get(metric)) for row in rows if row["cohort"] == cohort]
        for cohort in ("A", "B")
    }
    values = {key: [value for value in group if value is not None] for key, group in values.items()}
    result: dict[str, Any] = {"metric": metric}
    for cohort in ("A", "B"):
        group = values[cohort]
        result[f"{cohort}_n"] = len(group)
        result[f"{cohort}_median"] = float(np.median(group)) if group else None
        result[f"{cohort}_q25"] = float(np.quantile(group, 0.25)) if group else None
        result[f"{cohort}_q75"] = float(np.quantile(group, 0.75)) if group else None
    if values["A"] and values["B"]:
        u, p = mannwhitneyu(values["A"], values["B"], alternative="two-sided")
        result["mann_whitney_u"] = float(u)
        result["two_sided_p"] = float(p)
        result["rank_auc_A_greater"] = float(u / (len(values["A"]) * len(values["B"])))
    else:
        result.update({"mann_whitney_u": None, "two_sided_p": None, "rank_auc_A_greater": None})
    return result


def analyse_one(annotation_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    cohort = str(annotation["cohort"])
    object_id = str(annotation["object_id"])
    input_path = Path(annotation["input_path"])
    with fits.open(annotation_path.parent / annotation["masks_fits"], memmap=False) as hdus:
        foreground = np.asarray(hdus["FOREGROUND"].data, bool)
        labels = np.asarray(hdus["STRUCTURES"].data, np.int16)
    if not foreground.any():
        raise ValueError(f"{object_id}: no foreground mask")
    center, foreground_q, foreground_pa = foreground_geometry(foreground)
    loaded = load_multiband(input_path)
    growth = max(1, int(np.ceil(float(loaded["target_fwhm"]) / 2.0)))
    records_by_label = {int(item["label"]): item for item in annotation["structures"]}
    empty_labels = [label for label in records_by_label if not np.any(labels == label)]
    records_by_label = {
        label: item for label, item in records_by_label.items() if label not in empty_labels
    }
    original_masks = {label: labels == label for label in records_by_label}
    apertures = {label: binary_dilation(mask, iterations=growth) for label, mask in original_masks.items()}
    candidate_union = np.logical_or.reduce(list(apertures.values())) if apertures else np.zeros_like(labels, bool)
    profile_exclude = binary_dilation(candidate_union, iterations=3)
    residuals = {
        band: elliptical_residual(loaded["matched"][band], center, foreground_q, foreground_pa, profile_exclude)
        for band in BANDS
    }
    contaminant_masks = [
        original_masks[label]
        for label, item in records_by_label.items()
        if ROLE_LEGACY_MAP.get(str(item["role"]), str(item["role"]))
        == "unrelated_or_contaminant"
    ]
    # Match the production colour protocol: rotated same-radius controls may
    # contain other candidate light (which increases empirical scatter), and
    # are rejected only for explicit contaminant overlap.  Blocking the
    # foreground core here would remove nearly every control for small-radius
    # arcs even after protected foreground subtraction.
    blocked = (
        binary_dilation(np.logical_or.reduce(contaminant_masks), iterations=2)
        if contaminant_masks
        else np.zeros_like(foreground, dtype=bool)
    )
    foreground_phot = foreground_photometry(loaded, foreground, center)
    structure_rows: list[dict[str, Any]] = []
    photometry: dict[int, dict[str, dict[str, Any]]] = {}
    geometry: dict[int, dict[str, Any]] = {}
    for label, item in records_by_label.items():
        role = ROLE_LEGACY_MAP.get(str(item["role"]), str(item["role"]))
        geometry[label] = structure_geometry(original_masks[label], center)
        photometry[label] = candidate_photometry(loaded, apertures[label], center, residuals, blocked)
        row = {
            "cohort": cohort,
            "object_id": object_id,
            "sample_status": annotation["sample_status"],
            "structure_id": item["structure_id"],
            "label": label,
            "role": role,
            "source_group": str(item.get("source_group", "")).strip(),
            "note": item.get("note", ""),
            "native_pixel_count": int(original_masks[label].sum()),
            "photometry_aperture_pixel_count": int(apertures[label].sum()),
            "native_mask_sha256": mask_hash(original_masks[label]),
            "target_psf_fwhm_px": loaded["target_fwhm"],
            "aperture_growth_px": growth,
            **geometry[label],
        }
        row.update(flatten_photometry("candidate", photometry[label]))
        structure_rows.append(row)
    pair_rows: list[dict[str, Any]] = []
    groups: dict[str, list[int]] = defaultdict(list)
    for label, item in records_by_label.items():
        role = ROLE_LEGACY_MAP.get(str(item["role"]), str(item["role"]))
        group = str(item.get("source_group", "")).strip()
        if group and role == "independent_image_candidate":
            groups[group].append(label)
    for group, members in groups.items():
        if len(members) < 2:
            continue
        for first, second in combinations(members, 2):
            test = _gray_colour_test(photometry[first], photometry[second])
            flux_test = test.get("flux_scale_diagnostic", {})
            pair_rows.append({
                "cohort": cohort,
                "object_id": object_id,
                "sample_status": annotation["sample_status"],
                "source_group": group,
                "structure_1": f"S{first}",
                "structure_2": f"S{second}",
                "status": test["status"],
                "reliable_common_band_count": test.get("reliable_common_band_count"),
                "maximum_color_departure_mag": test.get("maximum_color_departure_mag"),
                "maximum_departure_sigma": test.get("maximum_departure_sigma"),
                "reduced_chi2": test.get("reduced_chi2"),
                "flux_shape_status": flux_test.get("status"),
                "flux_shape_band_count": flux_test.get("band_count"),
                "flux_shape_reduced_chi2": flux_test.get("same_sed_reduced_chi2"),
                "angular_separation_deg": circular_separation(geometry[first]["centroid_angle_deg"], geometry[second]["centroid_angle_deg"]),
                "radius_ratio": max(geometry[first]["centroid_radius_px"], geometry[second]["centroid_radius_px"]) / max(min(geometry[first]["centroid_radius_px"], geometry[second]["centroid_radius_px"]), 1e-6),
            })
    foreground_rows: list[dict[str, Any]] = []
    for label, item in records_by_label.items():
        role = ROLE_LEGACY_MAP.get(str(item["role"]), str(item["role"]))
        if role not in ROLE_CANDIDATE:
            continue
        test = _gray_colour_test(photometry[label], foreground_phot)
        flux_test = test.get("flux_scale_diagnostic", {})
        mapped = {
            "strongly_inconsistent": "distinct_from_foreground",
            "consistent_with_achromatic_scaling": "indistinguishable_from_foreground",
        }.get(test["status"], test["status"])
        foreground_rows.append({
            "cohort": cohort,
            "object_id": object_id,
            "sample_status": annotation["sample_status"],
            "structure_id": f"S{label}",
            "role": role,
            "status": mapped,
            "reliable_common_band_count": test.get("reliable_common_band_count"),
            "maximum_color_departure_mag": test.get("maximum_color_departure_mag"),
            "maximum_departure_sigma": test.get("maximum_departure_sigma"),
            "reduced_chi2": test.get("reduced_chi2"),
            "flux_shape_status": flux_test.get("status"),
            "flux_shape_band_count": flux_test.get("band_count"),
            "flux_shape_reduced_chi2": flux_test.get("same_sed_reduced_chi2"),
        })
    candidate_labels = [
        label for label, item in records_by_label.items()
        if ROLE_LEGACY_MAP.get(str(item["role"]), str(item["role"])) in ROLE_CANDIDATE
    ]
    candidate_geometry = [geometry[label] for label in candidate_labels]
    candidate_mask = np.logical_or.reduce([original_masks[label] for label in candidate_labels]) if candidate_labels else np.zeros_like(labels, bool)
    system_geometry = group_geometry(candidate_geometry, candidate_mask, center) if candidate_labels else {}
    pair_for_object = pair_rows
    foreground_for_object = foreground_rows
    object_row = {
        "cohort": cohort,
        "object_id": object_id,
        "sample_status": annotation["sample_status"],
        "sample_note": annotation.get("sample_note", ""),
        "structure_count": len(records_by_label),
        "empty_saved_structure_count": len(empty_labels),
        "candidate_structure_count": len(candidate_labels),
        "independent_image_count": sum(ROLE_LEGACY_MAP.get(str(item["role"]), str(item["role"])) == "independent_image_candidate" for item in records_by_label.values()),
        "arc_ring_structure_count": sum(ROLE_LEGACY_MAP.get(str(item["role"]), str(item["role"])) == "arc_or_ring_single_structure" for item in records_by_label.values()),
        "unresolved_count": sum(str(item["role"]) == "unresolved" for item in records_by_label.values()),
        "unresolved_fraction": sum(str(item["role"]) == "unresolved" for item in records_by_label.values()) / max(len(records_by_label), 1),
        "foreground_pixel_count": int(foreground.sum()),
        "foreground_axis_ratio": foreground_q,
        "same_source_group_count": sum(len(members) >= 2 for members in groups.values()),
        "same_source_pair_count": len(pair_for_object),
        "same_source_pair_measured_count": sum(row["status"] != "unknown" for row in pair_for_object),
        "same_source_consistent_count": sum(row["status"] == "consistent_with_achromatic_scaling" for row in pair_for_object),
        "same_source_inconsistent_count": sum(row["status"] == "strongly_inconsistent" for row in pair_for_object),
        "same_source_max_departure_mag": max((row["maximum_color_departure_mag"] for row in pair_for_object if row["maximum_color_departure_mag"] is not None), default=None),
        "same_source_max_departure_sigma": max((row["maximum_departure_sigma"] for row in pair_for_object if row["maximum_departure_sigma"] is not None), default=None),
        "same_source_flux_shape_median_reduced_chi2": float(np.median([row["flux_shape_reduced_chi2"] for row in pair_for_object if row["flux_shape_reduced_chi2"] is not None])) if any(row["flux_shape_reduced_chi2"] is not None for row in pair_for_object) else None,
        "foreground_relation_measured_count": sum(row["status"] not in {"unknown", "inconclusive"} for row in foreground_for_object),
        "foreground_distinct_count": sum(row["status"] == "distinct_from_foreground" for row in foreground_for_object),
        "foreground_indistinguishable_count": sum(row["status"] == "indistinguishable_from_foreground" for row in foreground_for_object),
        "foreground_distinct_fraction": sum(row["status"] == "distinct_from_foreground" for row in foreground_for_object) / max(sum(row["status"] not in {"unknown", "inconclusive"} for row in foreground_for_object), 1),
        **{f"geometry_{key}": value for key, value in system_geometry.items()},
    }
    return structure_rows, pair_rows, foreground_rows, object_row


def report_markdown(summary: dict[str, Any], stats: list[dict[str, Any]]) -> str:
    stat = {row["metric"]: row for row in stats}
    def fmt(metric: str, cohort: str) -> str:
        value = stat.get(metric, {}).get(f"{cohort}_median")
        return "不可测" if value is None else f"{value:.3f}"
    return f"""# 第二批人工结构标注：颜色与几何校准结论

## 数据范围

- 输入标注共 {summary['input_annotation_count']} 个样本；其中 {summary['object_count']} 个完成颜色/几何测量，{summary['measurement_error_count']} 个因缺少必要掩膜而保留为不可测。
- 可测样本：A={summary['cohort_counts']['A']}，B={summary['cohort_counts']['B']}。
- 人工结构：{summary['structure_count']} 个；A 中无法确定 {summary['unresolved_counts']['A']} 个，B 中无法确定 {summary['unresolved_counts']['B']} 个。
- 明确标成同源且至少含两个独立像的组：{summary['same_source_group_count']} 组；生成 {summary['same_source_pair_count']} 个像对。
- 人工标注不会被导入正式 Agent；本批证据经审查后只用于修订规则语义与构型准入方式。

## 直接结论

1. “无法确定结构比例”A 中位数为 {fmt('unresolved_fraction', 'A')}，B 为 {fmt('unresolved_fraction', 'B')}。它可作为观测完整性/构型完整性的置信度，不应成为非透镜否决项。
2. 同源像颜色比较只使用人工明确分组的独立像；未分组结构、同一弧的碎片和无法确定结构均不参与阈值拟合。
3. 候选与前景的颜色差异只说明两者更可能不是同一种光成分。颜色相似不否决透镜，颜色差异也不能单独把 B 升为 A。
4. 椭圆势、源位置、外剪切和不同像类型本来就可产生不同像半径，因此不得使用“半径必须一致”的硬门槛。半径相似仍是支持证据；仅用宽松的最大/最小半径物理包络排除明显无关的近远结构组合。
5. 环/长弧优先使用掩膜的角向覆盖度和切向性，不应按被分割出的短段数量套用“至少四像”。

## 建议的颜色规则

- 支持证据：在至少三个可靠公共波段中，同源像经一个无色倍率因子缩放后颜色形状一致。只能增强已有几何构型的可信度。
- 冲突证据：只有明确属于同一构型、测量质量合格的像对同时满足“大颜色偏离”和“高显著性”时，才允许把 A 上限压到 B。
- 未知证据：公共可靠波段不足、前景模型失败、局部对照不足或结构角色不确定时，颜色状态为 unknown，不降级。
- 前景关系：`distinct_from_foreground` 是背景来源的辅助支持；`indistinguishable_from_foreground` 记为非判别，不设等级上限。
- 当前阈值仍应视为测量分类阈值，而非从这 32 个样本拟合出的 Q1 分级边界。样本量和 B 的角色不确定性不足以安全地重新拟合 0.20/0.35 mag 数值。

## 建议的几何规则

- 多像：使用共同中心、角向分布、切向关系和颜色一致性的联合证据；相似半径加分，但 fold、cusp 和紧致多像不再要求严格共同半径。
- 长弧/环：以单一结构的角向覆盖度、曲率中心和切向性作为主要证据；断裂碎片先合并，再判构型。
- B 中大量小结构不确定，说明应降低“碎片数/紧致结构数”的权重，并提高“能否组成统一构型”的权重。
- 对不可测或不确定的小结构采取证据缺失，而不是负证据。

## 已采纳到 Agent 的修改

- 保持正式颜色阈值不变：可靠公共波段至少 3 个、系统误差底限 0.10 mag、一致类别最大偏离 0.20 mag、强冲突同时要求至少 0.35 mag 与 3σ。
- 新增中等可信颜色层：至少 3 个公共波段达到 S/N≥2.5，或至少 2 个公共波段达到 S/N≥3。该层只记录提示性一致、冲突或中间状态，不能升级、否决或限制等级。
- 新增 `low_snr_flux_shape_diagnostic_only`：低信噪比四波段通量形状会进入审计记录，但不支持、不否决、也不改变等级。
- 新增 `foreground_distinct_support`：候选与前景颜色可靠不同时明确记录为来源分离支持，但不能单独建立透镜构型或升级。
- fold、cusp 和紧致多像改用最大/最小半径比不超过 3 的宽松构型包络；严格共同半径仍只作为完整统一系统的强证据，避免把无关第三结构并入颜色比较。

## 下一批 B 环样本的作用

B 环样本最适合补充两项目前不足的证据：一是环结构相对前景的颜色是否可稳定分离，二是完整/半完整环在前景模型变化下的颜色稳定性。它不能单独标定 A/B 边界，因为仍缺少配对的 A 环和 C/unknown 对照；建议沿用同一人工掩膜协议后与本批结果合并分析。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = annotation_paths(args.annotations)
    structures: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    foreground: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            s_rows, p_rows, f_rows, o_row = analyse_one(path)
            structures.extend(s_rows)
            pairs.extend(p_rows)
            foreground.extend(f_rows)
            objects.append(o_row)
        except Exception as exc:  # retain per-object failure for audit
            errors.append({"annotation": str(path), "error": f"{type(exc).__name__}: {exc}"})
    metrics = [
        "unresolved_fraction",
        "candidate_structure_count",
        "same_source_pair_count",
        "same_source_max_departure_mag",
        "same_source_max_departure_sigma",
        "same_source_flux_shape_median_reduced_chi2",
        "foreground_distinct_fraction",
        "geometry_radius_cv",
        "geometry_radius_max_min_ratio",
        "geometry_maximum_pair_angle_separation_deg",
        "geometry_opposition_closeness_deg",
        "geometry_angular_occupancy_deg",
        "geometry_median_tangentiality",
    ]
    stats = [metric_summary(objects, metric) for metric in metrics]
    summary = {
        "schema": "manual_colour_geometry_calibration_v1",
        "annotations": str(args.annotations.resolve()),
        "input_annotation_count": len(paths),
        "object_count": len(objects),
        "cohort_counts": {cohort: sum(row["cohort"] == cohort for row in objects) for cohort in ("A", "B")},
        "structure_count": len(structures),
        "unresolved_counts": {cohort: sum(row["cohort"] == cohort and row["role"] == "unresolved" for row in structures) for cohort in ("A", "B")},
        "same_source_group_count": sum(row["same_source_group_count"] for row in objects),
        "same_source_pair_count": len(pairs),
        "same_source_status_counts": {status: sum(row["status"] == status for row in pairs) for status in ("consistent_with_achromatic_scaling", "inconclusive", "strongly_inconsistent", "unknown")},
        "candidate_foreground_status_counts": {status: sum(row["status"] == status for row in foreground) for status in ("distinct_from_foreground", "indistinguishable_from_foreground", "inconclusive", "unknown")},
        "measurement_error_count": len(errors),
        "measurement_errors": errors,
        "metric_statistics": stats,
        "interpretation_constraints": [
            "A/B labels are not independent physical truth labels and cannot calibrate C/unknown boundaries.",
            "Human unresolved labels measure observability and role ambiguity, not evidence against lensing.",
            "Candidate-versus-foreground colour contrast is source-separation support, not an A/B classifier.",
            "Same-source thresholds require explicit image grouping and must not be fitted from arc fragments.",
            "The small sample is used to audit rule direction and failure modes, not to optimize cut points.",
        ],
    }
    write_csv(args.output / "structures.csv", structures)
    write_csv(args.output / "same_source_pairs.csv", pairs)
    write_csv(args.output / "candidate_vs_foreground.csv", foreground)
    write_csv(args.output / "objects.csv", objects)
    write_csv(args.output / "metric_statistics.csv", stats)
    write_csv(args.output / "measurement_errors.csv", errors)
    (args.output / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=portable), encoding="utf-8"
    )
    (args.output / "COLOUR_GEOMETRY_RULE_REVISION_CN.md").write_text(
        report_markdown(summary, stats), encoding="utf-8"
    )
    print(json.dumps({
        "object_count": len(objects),
        "structure_count": len(structures),
        "same_source_pair_count": len(pairs),
        "candidate_foreground_relation_count": len(foreground),
        "measurement_error_count": len(errors),
    }, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
