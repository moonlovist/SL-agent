"""Export traceable colour measurements for A/B/C/unknown rule calibration.

The labels supplied to this script are joined only after a LensAgent run.  They
are never passed back into the vision or grading pipeline.  Three long-form CSV
tables preserve object-, pair-, and aperture-level information so provisional
colour thresholds can be assessed without rerunning photometry.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decision_conditions(reasoning: dict[str, Any]) -> dict[str, Any]:
    for item in reversed(reasoning.get("reasoning_trace", [])):
        if item.get("action") == "decision_conditions":
            return item.get("content") or {}
    return {}


def _published_grade(directory: Path, reasoning: dict[str, Any]) -> str | None:
    for name in ("summary.json", "run_manifest.json"):
        path = directory / name
        if path.exists():
            value = _load(path).get("published_grade")
            if value is not None:
                return value
    return (reasoning.get("expert_assessment") or {}).get("vote_category")


def _status_columns(prefix: str, items: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(item.get("status", "missing")) for item in items)
    names = (
        "consistent_with_achromatic_scaling", "strongly_inconsistent",
        "inconclusive", "unknown", "distinct_from_foreground",
        "indistinguishable_from_foreground", "missing",
    )
    return {f"{prefix}_{name}": counts[name] for name in names}


def _locate_observations(root: Path) -> list[Path]:
    preferred = sorted(root.rglob("reasoning_observation.json"))
    paths = preferred or sorted(root.rglob("observation.json"))
    return [path for path in paths if "code_snapshot" not in path.parts]


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=PATH, for example A=outputs_A")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if label not in {"A", "B", "C", "unknown"}:
        raise argparse.ArgumentTypeError("LABEL must be A, B, C, or unknown")
    path = Path(raw_path.strip())
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Dataset path does not exist: {path}")
    return label, path


def export(datasets: list[tuple[str, Path]], output: Path) -> dict[str, Any]:
    object_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    foreground_rows: list[dict[str, Any]] = []
    aperture_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for known_label, root in datasets:
        for observation_path in _locate_observations(root):
            observation = _load(observation_path)
            object_id = str(observation.get("object_id"))
            identity = known_label, object_id
            if identity in seen:
                raise ValueError(f"Duplicate object in label {known_label}: {object_id}")
            seen.add(identity)
            directory = observation_path.parent
            reasoning_path = directory / "reasoning.json"
            reasoning = _load(reasoning_path) if reasoning_path.exists() else {}
            decision = _decision_conditions(reasoning)
            multiband = (
                observation.get("supplementary_diagnostics", {}).get("multiband", {})
            )
            relation = multiband.get("relationship_audit", {})
            pairs = list(relation.get("component_pairs", []))
            foreground = list(relation.get("candidate_vs_foreground", []))
            grading_pairs = list(decision.get("colour_configuration_pairs", []))
            grading_lookup = {
                tuple(sorted((str(item.get("first")), str(item.get("second"))))): item
                for item in grading_pairs
            }

            measured_bands = [
                band for band, item in multiband.get("bands", {}).items()
                if item.get("status") == "measured"
            ]
            unavailable_bands = {
                band: item.get("reason")
                for band, item in multiband.get("bands", {}).items()
                if item.get("status") != "measured"
            }
            controls = [
                _finite(record.get("local_control_count"))
                for records in multiband.get("decontaminated_apertures", {}).values()
                for record in records.values()
            ]
            controls = [value for value in controls if value is not None]
            departures = [
                value for item in grading_pairs
                if (value := _finite(item.get("maximum_color_departure_mag"))) is not None
            ]
            significances = [
                value for item in grading_pairs
                if (value := _finite(item.get("maximum_departure_sigma"))) is not None
            ]
            object_row = {
                "object_id": object_id,
                "known_label": known_label,
                "published_grade": _published_grade(directory, reasoning),
                "decision_audit_passed": (reasoning.get("decision_audit") or {}).get("audit_passed"),
                "multiband_status": multiband.get("status"),
                "multiband_reason": multiband.get("reason"),
                "measured_bands": ";".join(measured_bands),
                "measured_band_count": len(measured_bands),
                "unavailable_bands_json": _json_cell(unavailable_bands),
                "colour_policy": relation.get("policy"),
                "colour_thresholds_json": _json_cell(relation.get("thresholds", {})),
                "candidate_aperture_count": len(multiband.get("decontaminated_apertures", {})),
                "all_candidate_pair_count": len(pairs),
                "configuration_linked_pair_count": len(grading_pairs),
                "colour_evidence_state": decision.get("colour_evidence_state"),
                "colour_caps_A": decision.get("colour_caps_A"),
                "colour_support": decision.get("colour_support"),
                "foreground_only_ambiguity": any(
                    item.get("id") == "PC-colour-foreground-ambiguity"
                    for item in (reasoning.get("evidence_dossier") or {}).get("processing_cautions", [])
                ),
                "maximum_configuration_departure_mag": max(departures) if departures else None,
                "maximum_configuration_departure_sigma": max(significances) if significances else None,
                "median_local_control_count": median(controls) if controls else None,
                "psf_target_fwhm_px": _finite((multiband.get("psf_matching") or {}).get("target_fwhm_px")),
                "input_sha256": multiband.get("input_sha256"),
                "source_observation": str(observation_path),
                **_status_columns("all_pairs", pairs),
                **_status_columns("grading_pairs", grading_pairs),
                **_status_columns("foreground", foreground),
            }
            object_rows.append(object_row)

            for item in pairs:
                key = tuple(sorted((str(item.get("first")), str(item.get("second")))))
                grading = grading_lookup.get(key)
                flux_fit = item.get("flux_scale_diagnostic") or {}
                pair_rows.append({
                    "object_id": object_id, "known_label": known_label,
                    "published_grade": object_row["published_grade"],
                    "first": item.get("first"), "second": item.get("second"),
                    "configuration_linked": grading is not None,
                    "configuration_ids": ";".join((grading or {}).get("configuration_ids", [])),
                    "status": item.get("status"), "reason": item.get("reason"),
                    "bands": ";".join(item.get("bands", [])),
                    "reliable_common_band_count": item.get("reliable_common_band_count"),
                    "gray_offset_mag": item.get("gray_offset_mag"),
                    "maximum_color_departure_mag": item.get("maximum_color_departure_mag"),
                    "maximum_departure_sigma": item.get("maximum_departure_sigma"),
                    "chi2": item.get("chi2"), "degrees_of_freedom": item.get("degrees_of_freedom"),
                    "reduced_chi2": item.get("reduced_chi2"),
                    "systematic_floor_mag": item.get("systematic_floor_mag"),
                    "overlap_fraction": item.get("overlap_fraction"),
                    "per_band_diagnostics_json": _json_cell(item.get("per_band_diagnostics", {})),
                    "flux_fit_status": flux_fit.get("status"),
                    "flux_fit_bands": ";".join(flux_fit.get("bands", [])),
                    "flux_fit_band_count": flux_fit.get("band_count"),
                    "best_flux_scale_alpha": flux_fit.get("best_flux_scale_alpha"),
                    "same_sed_flux_chi2": flux_fit.get("same_sed_chi2"),
                    "same_sed_flux_dof": flux_fit.get("same_sed_dof"),
                    "same_sed_flux_reduced_chi2": flux_fit.get("same_sed_reduced_chi2"),
                    "flux_fit_per_band_diagnostics_json": _json_cell(flux_fit.get("per_band_diagnostics", {})),
                })

            for item in foreground:
                foreground_rows.append({
                    "object_id": object_id, "known_label": known_label,
                    "published_grade": object_row["published_grade"],
                    "candidate": item.get("candidate"), "status": item.get("status"),
                    "reason": item.get("reason"), "bands": ";".join(item.get("bands", [])),
                    "reliable_common_band_count": item.get("reliable_common_band_count"),
                    "gray_offset_mag": item.get("gray_offset_mag"),
                    "maximum_color_departure_mag": item.get("maximum_color_departure_mag"),
                    "maximum_departure_sigma": item.get("maximum_departure_sigma"),
                    "reduced_chi2": item.get("reduced_chi2"),
                    "foreground_core_overlap_fraction": item.get("foreground_core_overlap_fraction"),
                    "per_band_diagnostics_json": _json_cell(item.get("per_band_diagnostics", {})),
                })

            raw = multiband.get("apertures", {})
            for candidate, records in multiband.get("decontaminated_apertures", {}).items():
                for band, item in records.items():
                    raw_item = raw.get(candidate, {}).get(band, {})
                    aperture_rows.append({
                        "object_id": object_id, "known_label": known_label,
                        "published_grade": object_row["published_grade"],
                        "candidate": candidate, "band": band, "status": item.get("status"),
                        "flux_zp25": item.get("flux_zp25"),
                        "error_including_local_controls": item.get("error_including_local_controls"),
                        "snr": item.get("snr"), "magnitude": item.get("magnitude"),
                        "magnitude_error": item.get("magnitude_error"),
                        "normalized_flux_by_signed_sum": item.get("normalized_flux_by_signed_sum"),
                        "normalized_flux_by_absolute_sum": item.get("normalized_flux_by_absolute_sum"),
                        "diagonal_noise_with_common_psf": item.get("diagonal_noise_with_common_psf"),
                        "local_control_scatter": item.get("local_control_scatter"),
                        "local_control_count": item.get("local_control_count"),
                        "color_aperture_pixel_count": item.get("color_aperture_pixel_count"),
                        "raw_flux_zp25": raw_item.get("flux_zp25"),
                        "raw_snr_nominal": raw_item.get("snr_nominal"),
                        "raw_magnitude": raw_item.get("magnitude"),
                        "source_detection_mask_sha256": item.get("source_detection_mask_sha256"),
                    })

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "colour_objects.csv", object_rows)
    _write_csv(output / "colour_component_pairs.csv", pair_rows)
    _write_csv(output / "colour_candidate_vs_foreground.csv", foreground_rows)
    _write_csv(output / "colour_apertures_by_band.csv", aperture_rows)
    manifest = {
        "schema": "colour_rule_calibration_dataset_v1",
        "labels": dict(Counter(row["known_label"] for row in object_rows)),
        "object_count": len(object_rows),
        "component_pair_count": len(pair_rows),
        "foreground_relation_count": len(foreground_rows),
        "aperture_band_count": len(aperture_rows),
        "datasets": [{"label": label, "root": str(root)} for label, root in datasets],
        "important_boundary": "Labels are joined only by this post-run exporter and never enter LensAgent measurement or grading.",
    }
    (output / "colour_export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", action="append", type=_parse_dataset, required=True,
        help="Repeatable LABEL=PATH; use A/B/C, X or nonlens controls, and unknown separately.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.dataset, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
