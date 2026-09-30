"""Deterministic measurement union BEFORE the sole interpretation pass.

Never reads labels or casts a vote. Original Observation is never modified.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
import numpy as np

from ..schemas.observation import SceneGeometryObservation

POLICY = "deterministic_evidence_union_v2"


def prepare_evidence(source, config):
    document = source.model_copy(deep=True)
    trace = {"policy": POLICY, "parameters": asdict(config), "items": [],
             "limitations": ["Heuristic morphology is not a calibrated human visual judgement.",
                             "Nominal SNR excludes search trials, covariance and foreground-fit uncertainty."],
             "source_observation_sha256": hashlib.sha256(
                 json.dumps(source.model_dump(mode="json"), sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    review = source.candidate_handoff_review
    if review is None:
        trace["limitations"].append("Legacy Observation has no complete pixel handoff; missing measurements are not reconstructed.")
        return document, trace
    if len({e.candidate_id for e in review.entries}) != len(review.entries):
        raise ValueError("duplicate candidate handoff IDs")
    cids = {e.candidate_id for e in review.entries}
    # Keep only historical structures not represented by the complete handoff.
    structures = [s for s in document.structures if not cids.intersection(s.source_candidate_ids)]
    seen_masks = {}
    for entry in sorted(review.entries, key=lambda e: e.candidate_id):
        item = entry.measurement
        reasons = []
        if item is None:
            reasons.append("measurement_unavailable")
        else:
            if not np.isfinite(item.integrated_snr) or not np.isfinite(item.integrated_flux):
                reasons.append("nonfinite_signal")
            if item.pixel_count < config.candidate_grade_min_pixels:
                reasons.append("pixel_count_below_measurement_policy")
            if item.integrated_snr < config.candidate_grade_min_snr or item.integrated_flux <= 0:
                reasons.append("nominal_signal_below_measurement_policy")
        if entry.mask_sha256 and entry.mask_sha256 in seen_masks:
            reasons.append("duplicate_pixel_mask:" + seen_masks[entry.mask_sha256])
        elif entry.mask_sha256:
            seen_masks[entry.mask_sha256] = entry.candidate_id
        # No route, old isolation verdict, centroid, or model occupancy veto.
        target_id = entry.formal_structure_id or f"REVIEW_{entry.candidate_id}"
        trace["items"].append({
            "candidate_id": entry.candidate_id, "included": not reasons,
            "structure_id": target_id if not reasons else None,
            "source_ref": f"candidate_handoff_review.entries.{entry.candidate_id}",
            "mask_key": entry.mask_key, "mask_sha256": entry.mask_sha256,
            "reasons": reasons or ["measurable_positive_signal_not_lens_proof"],
            "measurements": item.model_dump(mode="json") if item is not None else None,
        })
        if not reasons:
            structures.append(item.model_copy(update={"id": target_id}))
    # Replace fragments only with a measured union, never with a filled route.
    for chain in document.arc_chains:
        members = set(chain.source_candidate_ids)
        if chain.integrated_flux <= 0 or chain.integrated_snr < config.candidate_grade_min_snr:
            continue
        if not members.issubset(cids):
            raise ValueError("arc chain references missing candidates")
        structures = [s for s in structures if not members.intersection(s.source_candidate_ids)]
        structures.append(chain)
        trace.setdefault("arc_chain_replacements", []).append({"id": chain.id, "members": sorted(members), "no_gap_filling": True})
    if len({s.id for s in structures}) != len(structures):
        raise ValueError("duplicate canonical structure IDs")
    document.structures = sorted(structures, key=lambda s: s.id)
    # Old scene statistics refer to old masks; do not reuse or sum arc extents.
    if structures and all(s.occupied_angle_bins_5deg is not None for s in structures):
        radii = np.array([s.median_radius_px for s in structures])
        angles = [s.tangential_misalignment_deg for s in structures if s.tangential_misalignment_deg is not None]
        bins = set().union(*(s.occupied_angle_bins_5deg for s in structures))
        document.scene_geometry = SceneGeometryObservation(
            structure_count=len(structures), common_radius_cv=float(radii.std()/radii.mean()) if radii.mean() > 0 else None,
            total_angular_coverage_deg=5.0 * len(bins),
            mean_tangential_misalignment_deg=float(np.mean(angles)) if angles else None,
            tangential_alignment_fraction=float(np.mean(np.array(angles) <= config.tangential_misalignment_max_deg)) if angles else None,
        )
    else:
        document.scene_geometry = None
        trace["limitations"].append("Union scene geometry unavailable without occupied pixel-angle bins for every structure.")
    trace["canonical_structure_ids"] = [s.id for s in document.structures]
    return document, trace
