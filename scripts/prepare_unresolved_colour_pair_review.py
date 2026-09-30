"""Prepare review targets for unresolved manually masked colour structures."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path


BANDS = ("VIS", "NIR_Y", "NIR_J", "NIR_H")


def _number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def compare_rows(first: dict[str, str], second: dict[str, str]) -> dict | None:
    measurements = []
    for band in BANDS:
        first_mag = _number(first, f"candidate_{band}_magnitude")
        second_mag = _number(second, f"candidate_{band}_magnitude")
        first_error = _number(first, f"candidate_{band}_magnitude_error")
        second_error = _number(second, f"candidate_{band}_magnitude_error")
        if None in (first_mag, second_mag, first_error, second_error):
            continue
        variance = first_error**2 + second_error**2 + 0.10**2
        measurements.append((band, first_mag - second_mag, variance))
    if len(measurements) < 3:
        return None
    denominator = sum(1.0 / variance for _, _, variance in measurements)
    offset = sum(difference / variance for _, difference, variance in measurements) / denominator
    departures = [abs(difference - offset) for _, difference, _ in measurements]
    significances = [
        departure / math.sqrt(variance)
        for departure, (_, _, variance) in zip(departures, measurements)
    ]
    chi2 = sum(
        (difference - offset) ** 2 / variance
        for _, difference, variance in measurements
    )
    reduced_chi2 = chi2 / max(len(measurements) - 1, 1)
    maximum = max(departures)
    maximum_sigma = max(significances)
    if maximum >= 0.35 and maximum_sigma >= 3.0:
        status = "strongly_inconsistent"
    elif maximum <= 0.20 and reduced_chi2 <= 2.0:
        status = "consistent_with_achromatic_scaling"
    else:
        status = "inconclusive"
    return {
        "common_bands": len(measurements),
        "bands": [band for band, _, _ in measurements],
        "maximum_departure_mag": round(maximum, 3),
        "maximum_departure_sigma": round(maximum_sigma, 2),
        "reduced_chi2": round(reduced_chi2, 2),
        "current_status": status,
    }


def build_manifest(structures_csv: Path) -> dict:
    with structures_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_object = defaultdict(list)
    for row in rows:
        by_object[row["object_id"]].append(row)
    objects = []
    for object_id, object_rows in sorted(by_object.items()):
        pairs = []
        unresolved = [row for row in object_rows if row.get("role") == "unresolved"]
        comparison = [row for row in object_rows if row.get("role") != "unresolved"]
        for first in unresolved:
            for second in comparison:
                diagnostic = compare_rows(first, second)
                if diagnostic is None:
                    continue
                pairs.append({
                    "first_label": int(first["label"]),
                    "second_label": int(second["label"]),
                    "first_role": first["role"],
                    "second_role": second["role"],
                    **diagnostic,
                })
        if pairs:
            objects.append({
                "cohort": object_rows[0]["cohort"],
                "object_id": object_id,
                "pairs": pairs,
            })
    return {
        "schema": "unresolved_colour_pair_review_manifest_v1",
        "source_structures_csv": str(structures_csv.resolve()),
        "selection": (
            "Unresolved versus non-unresolved structure in the same object with at least "
            "three common S/N>=3 magnitude bands. Proposed comparisons are not labels."
        ),
        "object_count": len(objects),
        "pair_count": sum(len(item["pairs"]) for item in objects),
        "objects": objects,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structures_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    payload = build_manifest(args.structures_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output_json.resolve()),
        "object_count": payload["object_count"],
        "pair_count": payload["pair_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
