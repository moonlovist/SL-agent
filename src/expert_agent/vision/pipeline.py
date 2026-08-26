from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..core.config import VisionConfig
from ..schemas.observation import (
    AlternativeIndicators,
    AstrometryObservation,
    CandidateSystemObservation,
    DataQuality,
    ForegroundGalaxyObservation,
    ForegroundModelComparisonObservation,
    GeometricPathObservation,
    Measurement,
    ModelCandidateObservation,
    ObservationDocument,
    Provenance,
    RingCandidateObservation,
    SceneGeometryObservation,
    StructureObservation,
)
from .astrometry import (
    axial_misalignment_deg,
    orientation_vectors,
    pixel_to_sky,
    sky_separation_and_pa,
)
from .detectors import (
    ResidualStructure,
    StabilityDetection,
    circular_span,
    consensus_residual_structures,
    geometric_path_search,
    path_pixel_consistency,
    partition_sep_segments_by_routes,
    position_angle,
    sep_consensus,
    threshold_stability,
)
from .diagnostics import write_visual_audit_bundle
from .foreground_models import fit_psf_mge_comparison
from .preprocess import load_image, preprocess
from .summary_cn import write_observation_summary_cn
from .structure_audit import audit_structure_candidates


class VisionPipeline:
    def __init__(self, config: VisionConfig | None = None):
        self.config = config or VisionConfig()

    def run(
        self, input_path: str | Path, output_dir: str | Path | None = None
    ) -> ObservationDocument:
        output: Path | None = None
        if output_dir is not None:
            output = Path(output_dir).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            cache_dir = output / ".matplotlib"
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Photutils can import Matplotlib before diagnostics are rendered,
            # so configure a writable cache before any preprocessing import path
            # reaches it.
            os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        image = load_image(input_path, self.config.band)
        runtime_config = self.config
        if image.input_format == "png":
            # Rendered PNGs lack calibrated flux/RMS and can contain thousands
            # of thresholded display-noise islands. Use a conservative reduced
            # grid so PNG remains a supported observation mode without
            # pretending to have FITS-level precision.
            runtime_config = replace(
                self.config,
                source_detect_sigma=max(2.5, self.config.source_detect_sigma),
                snr_thresholds=(3.0, 4.0, 6.0),
                snr_min_survival=2,
                sep_nthresh=(16,),
                sep_deblend_cont=(0.01,),
                sep_detect_sigma=max(3.0, self.config.sep_detect_sigma),
                sep_min_survival=1,
                residual_structure_snr_low=max(
                    3.0, self.config.structure_snr_low
                ),
            )
        prep = preprocess(image, runtime_config)
        threshold_detections, level_map = threshold_stability(prep, runtime_config)
        consensus_detections = sep_consensus(prep, runtime_config)
        segments, residual_segmentation, footprint, sep_seed_mask = consensus_residual_structures(
            prep, runtime_config, consensus_detections
        )
        geometric_paths_raw = geometric_path_search(prep, runtime_config, segments)
        comparison = fit_psf_mge_comparison(prep)
        comparison_snr: np.ndarray | None = None
        residual_correlation: float | None = None
        if comparison.success and comparison.residual is not None:
            comparison_snr = np.divide(
                comparison.residual,
                prep.image.rms,
                out=np.zeros_like(comparison.residual),
                where=prep.image.rms > 0,
            )
            yy, xx = np.indices(prep.image.data.shape)
            radius = np.hypot(xx - prep.nucleus_yx[1], yy - prep.nucleus_yx[0])
            annulus = (
                (radius >= runtime_config.ring_inner_px)
                & (radius <= runtime_config.ring_outer_px)
                & np.isfinite(prep.residual)
                & np.isfinite(comparison.residual)
            )
            if int(annulus.sum()) >= 3:
                value = float(
                    np.corrcoef(prep.residual[annulus], comparison.residual[annulus])[0, 1]
                )
                residual_correlation = value if np.isfinite(value) else None

        measurements: list[Measurement] = []
        # Raw SEP output remains trace data. Only candidates that pass the
        # explicit candidate-structure audit become measured structures.
        structures: list[StructureObservation] = []
        ring_candidates: list[RingCandidateObservation] = []  # legacy field; no longer emitted
        geometric_paths: list[GeometricPathObservation] = []
        model_candidates: list[ModelCandidateObservation] = []
        candidate_systems: list[CandidateSystemObservation] = []
        model_candidate_masks: dict[str, np.ndarray] = {}
        downgraded_sep_trace_mask = np.zeros(prep.image.data.shape, dtype=bool)

        def comparison_fraction(mask: np.ndarray) -> float | None:
            if comparison_snr is None or not np.any(mask):
                return None
            return float(
                np.mean(
                    comparison_snr[mask] >= runtime_config.structure_snr_low
                )
            )

        local_partitions, _supported_seed_indices_by_path = (
            partition_sep_segments_by_routes(
                geometric_paths_raw,
                segments,
                min_supported_area=runtime_config.snr_min_area,
            )
        )
        candidate_ids_by_seed: dict[int, list[str]] = {
            index: [] for index in range(len(segments))
        }
        supported_candidate_ids_by_path: dict[int, list[str]] = {
            index: [] for index in range(len(geometric_paths_raw))
        }

        for partition in local_partitions:
            seed_index = partition.segment_index
            segment = segments[seed_index]
            raw_detection = (
                consensus_detections[segment.label - 1]
                if 0 < segment.label <= len(consensus_detections)
                else None
            )
            sep_fraction = (
                raw_detection.survival / raw_detection.total_runs
                if raw_detection is not None and raw_detection.total_runs
                else None
            )
            mask = np.zeros(prep.image.data.shape, dtype=bool)
            mask[partition.pixels_y, partition.pixels_x] = True
            if partition.status == "sep_only":
                downgraded_sep_trace_mask |= mask
                # Removing route-supported pixels can leave isolated SEP
                # crumbs. Preserve them in the trace mask, but do not promote
                # a sub-minimum fragment into a new structural candidate.
                if int(mask.sum()) < runtime_config.snr_min_area:
                    continue
            candidate_id = f"C{len(model_candidates) + 1}"
            candidate_ids_by_seed[seed_index].append(candidate_id)
            centroid_x = float(partition.pixels_x.mean())
            centroid_y = float(partition.pixels_y.mean())
            pixel_radii = np.hypot(
                partition.pixels_x - prep.nucleus_yx[1],
                partition.pixels_y - prep.nucleus_yx[0],
            )
            radius_px = float(np.median(pixel_radii))
            pixel_angles = (
                np.degrees(
                    np.arctan2(
                        -(partition.pixels_x - prep.nucleus_yx[1]),
                        -(partition.pixels_y - prep.nucleus_yx[0]),
                    )
                )
                + 360.0
            ) % 360.0
            integrated_snr = float(
                prep.residual_snr[mask].sum() / np.sqrt(max(int(mask.sum()), 1))
            )
            path_indices = [
                path_index
                for path_index, path_candidate in enumerate(geometric_paths_raw)
                if np.any(path_candidate.route_mask & mask)
            ]
            consistent_path_indices = partition.path_indices
            path_ids = [f"G{index + 1}" for index in path_indices]
            consistent_path_ids = [
                f"G{index + 1}" for index in consistent_path_indices
            ]
            status = partition.status
            for path_index in consistent_path_indices:
                supported_candidate_ids_by_path[path_index].append(candidate_id)
            near_core = radius_px <= runtime_config.near_core_radius_px
            model_survival = comparison_fraction(mask)
            measurement_ids: list[str] = []
            for kind, value, reliability in [
                ("integrated_snr", integrated_snr, min(0.6, abs(integrated_snr) / 20.0)),
                ("radius", radius_px, 0.6),
                ("sep_survival_fraction", sep_fraction, sep_fraction or 0.0),
                ("foreground_comparison_survival_fraction", model_survival, 0.55 if model_survival is not None else 0.0),
            ]:
                measurement_id = f"M_{candidate_id}_{kind}"
                measurement_ids.append(measurement_id)
                measurements.append(
                    Measurement(
                        id=measurement_id,
                        kind=f"model_candidate_{kind}",
                        value=value,
                        unit="px" if kind == "radius" else None,
                        reliability=float(np.clip(reliability, 0.0, 1.0)),
                        method="sep_to_geometry_model_chain",
                        description=(
                            f"{kind.replace('_', ' ')} for {candidate_id}; retained for model-chain audit "
                            "and not exposed as a scored structure"
                        ),
                        structure_id=candidate_id,
                        correlation_group="model_chain_candidates",
                        flags=["not_scoring_input", "model_chain_unvalidated"],
                    )
                )
            flags = ["not_scoring_input", "model_chain_unvalidated"]
            if near_core:
                flags.append("near_core_subtraction_sensitive")
            if model_survival is not None and model_survival < 1.0:
                flags.append("foreground_comparison_changes_pixels")
            if model_survival == 0.0:
                flags.append("not_surviving_comparison_model")
            path_metrics = {
                f"G{path_index + 1}": path_pixel_consistency(
                    geometric_paths_raw[path_index],
                    partition.pixels_y,
                    partition.pixels_x,
                )
                for path_index in path_indices
            }
            model_candidates.append(
                ModelCandidateObservation(
                    id=candidate_id,
                    origin="sep_seed",
                    status=status,
                    structural_support_level=(
                        "sep_plus_geometry" if consistent_path_ids else "raw_sep_only"
                    ),
                    support_sources=(
                        ["sep_parameter_consensus", "geometric_route_consistency"]
                        if consistent_path_ids
                        else ["sep_parameter_consensus"]
                    ),
                    parent_sep_component_id=f"SEP{segment.label}",
                    pixel_partition=(
                        "route_supported"
                        if consistent_path_ids
                        else "off_route_remainder"
                    ),
                    review_priority=(
                        "priority_keep"
                        if consistent_path_ids
                        else "downgraded_trace"
                    ),
                    raw_detection_ids=[f"R{segment.label}"],
                    path_ids=path_ids,
                    geometry_consistent_path_ids=consistent_path_ids,
                    path_overlap_fractions={
                        path_id: values[0]
                        for path_id, values in path_metrics.items()
                        if values[0] > 0.0
                    },
                    path_median_deviation_px={
                        path_id: values[1]
                        for path_id, values in path_metrics.items()
                        if values[0] > 0.0
                    },
                    centroid_xy=(centroid_x, centroid_y),
                    radius_px=radius_px,
                    position_angle_deg=position_angle(
                        centroid_x, centroid_y, prep.nucleus_yx
                    ),
                    pixel_count=int(mask.sum()),
                    integrated_snr=integrated_snr,
                    angular_extent_deg=circular_span(pixel_angles),
                    sep_survival_fraction=sep_fraction,
                    foreground_comparison_survival_fraction=model_survival,
                    near_core=near_core,
                    measurement_ids=measurement_ids,
                    flags=flags,
                )
            )
            model_candidate_masks[candidate_id] = mask

        recovered_ids_by_path: dict[int, list[str]] = {}
        for path_index, path_candidate in enumerate(geometric_paths_raw, start=1):
            path_id = f"G{path_index}"
            for recovered in path_candidate.recovered_segments:
                candidate_id = f"C{len(model_candidates) + 1}"
                recovered_ids_by_path.setdefault(path_index - 1, []).append(candidate_id)
                mask = np.zeros(prep.image.data.shape, dtype=bool)
                mask[recovered.pixels_y, recovered.pixels_x] = True
                dx = recovered.x - prep.nucleus_yx[1]
                dy = recovered.y - prep.nucleus_yx[0]
                radius_px = float(np.hypot(dx, dy))
                angle_deg = position_angle(recovered.x, recovered.y, prep.nucleus_yx)
                near_core = radius_px <= runtime_config.near_core_radius_px
                model_survival = comparison_fraction(mask)
                measurement_ids: list[str] = []
                for kind, value, reliability in [
                    ("integrated_snr", recovered.integrated_snr, min(0.55, abs(recovered.integrated_snr) / 20.0)),
                    ("radius", radius_px, 0.55),
                    ("foreground_comparison_survival_fraction", model_survival, 0.55 if model_survival is not None else 0.0),
                ]:
                    measurement_id = f"M_{candidate_id}_{kind}"
                    measurement_ids.append(measurement_id)
                    measurements.append(
                        Measurement(
                            id=measurement_id,
                            kind=f"model_candidate_{kind}",
                            value=value,
                            unit="px" if kind == "radius" else None,
                            reliability=float(np.clip(reliability, 0.0, 1.0)),
                            method="geometry_masked_sep",
                            description=(
                                f"{kind.replace('_', ' ')} for {candidate_id}; detected by SEP inside {path_id} "
                                "and not yet promoted to a scored structure"
                            ),
                            structure_id=candidate_id,
                            correlation_group="model_chain_candidates",
                            flags=["not_scoring_input", "model_chain_unvalidated"],
                        )
                    )
                flags = ["not_scoring_input", "model_chain_unvalidated", "geometry_masked_sep"]
                if near_core:
                    flags.append("near_core_subtraction_sensitive")
                if model_survival is not None and model_survival < 1.0:
                    flags.append("foreground_comparison_changes_pixels")
                if model_survival == 0.0:
                    flags.append("not_surviving_comparison_model")
                model_candidates.append(
                    ModelCandidateObservation(
                        id=candidate_id,
                        origin="path_recovery",
                        status="geometry_recovered",
                        structural_support_level="route_recovered",
                        support_sources=["geometric_route", "route_masked_sep"],
                        pixel_partition="route_recovery",
                        review_priority="priority_keep",
                        path_ids=[path_id],
                        geometry_consistent_path_ids=[path_id],
                        centroid_xy=(recovered.x, recovered.y),
                        radius_px=radius_px,
                        position_angle_deg=angle_deg,
                        pixel_count=recovered.pixel_count,
                        integrated_snr=recovered.integrated_snr,
                        angular_extent_deg=recovered.angular_extent_deg,
                        foreground_comparison_survival_fraction=model_survival,
                        near_core=near_core,
                        measurement_ids=measurement_ids,
                        flags=flags,
                    )
                )
                model_candidate_masks[candidate_id] = mask

        candidate_system_membership: dict[str, list[str]] = {
            candidate.id: [] for candidate in model_candidates
        }
        candidate_by_id = {candidate.id: candidate for candidate in model_candidates}
        seen_system_members: set[tuple[str, ...]] = set()
        for path_index, path_candidate in enumerate(geometric_paths_raw, start=1):
            path_id = f"G{path_index}"
            sep_ids = supported_candidate_ids_by_path.get(path_index - 1, [])
            recovered_ids = recovered_ids_by_path.get(path_index - 1, [])
            member_ids = list(dict.fromkeys([*sep_ids, *recovered_ids]))
            if not member_ids:
                continue
            membership_key = tuple(sorted(member_ids))
            if membership_key in seen_system_members:
                continue
            seen_system_members.add(membership_key)
            system_id = f"CS{len(candidate_systems) + 1}"
            union_mask = np.zeros(prep.image.data.shape, dtype=bool)
            audit_flags: list[str] = []
            for member_id in member_ids:
                union_mask |= model_candidate_masks[member_id]
                candidate_system_membership[member_id].append(system_id)
                audit_flags.extend(candidate_by_id[member_id].flags)
            union_y, union_x = np.where(union_mask)
            union_angles = (
                np.degrees(
                    np.arctan2(
                        -(union_x - prep.nucleus_yx[1]),
                        -(union_y - prep.nucleus_yx[0]),
                    )
                )
                + 360.0
            ) % 360.0
            member_radii = [candidate_by_id[item].radius_px for item in member_ids]
            candidate_systems.append(
                CandidateSystemObservation(
                    id=system_id,
                    path_id=path_id,
                    member_candidate_ids=member_ids,
                    sep_candidate_ids=sep_ids,
                    recovered_candidate_ids=recovered_ids,
                    component_count=len(member_ids),
                    union_pixel_count=int(union_mask.sum()),
                    radius_range_px=(min(member_radii), max(member_radii)),
                    angular_extent_deg=circular_span(union_angles),
                    support_sources=list(
                        dict.fromkeys(
                            source
                            for member_id in member_ids
                            for source in candidate_by_id[member_id].support_sources
                        )
                    ),
                    audit_flags=sorted(
                        {
                            flag for flag in audit_flags
                            if flag in {
                                "near_core_subtraction_sensitive",
                                "foreground_comparison_changes_pixels",
                                "not_surviving_comparison_model",
                            }
                        }
                    ),
                    flags=["not_scoring_input", "candidate_system_unvalidated"],
                )
            )
        model_candidates = [
            candidate.model_copy(
                update={
                    "candidate_system_ids": candidate_system_membership[candidate.id]
                }
            )
            for candidate in model_candidates
        ]

        candidate_structure_audits = audit_structure_candidates(
            model_candidates,
            data_usable=(image.finite_fraction > 0.99 and prep.background_rms > 0),
            foreground_model_success=prep.model_success,
        )
        candidate_by_id = {candidate.id: candidate for candidate in model_candidates}
        promoted_structure_id_by_candidate: dict[str, str] = {}
        structure_masks: dict[str, np.ndarray] = {}
        for audit in candidate_structure_audits:
            if audit.verdict != "promoted":
                continue
            candidate = candidate_by_id[audit.candidate_id]
            mask = model_candidate_masks[candidate.id]
            ys, xs = np.where(mask)
            if not ys.size:
                continue
            structure_id = f"S{len(structures) + 1}"
            promoted_structure_id_by_candidate[candidate.id] = structure_id
            structure_masks[structure_id] = mask
            centroid_x = float(xs.mean())
            centroid_y = float(ys.mean())
            centered = np.column_stack((xs - centroid_x, ys - centroid_y))
            if len(centered) >= 2:
                covariance = np.cov(centered, rowvar=False, bias=True)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                minor_value = max(float(eigenvalues[0]), 1e-9)
                major_value = max(float(eigenvalues[-1]), minor_value)
                axis_ratio = float(np.sqrt(major_value / minor_value))
                major_vector = eigenvectors[:, -1]
                major_axis_deg = float(
                    np.degrees(np.arctan2(major_vector[1], major_vector[0])) % 180.0
                )
            else:
                axis_ratio = 1.0
                major_axis_deg = 0.0
            radial_axis_deg = float(
                np.degrees(
                    np.arctan2(
                        centroid_y - prep.nucleus_yx[0],
                        centroid_x - prep.nucleus_yx[1],
                    )
                )
                % 180.0
            )
            tangential_misalignment = axial_misalignment_deg(
                major_axis_deg, (radial_axis_deg + 90.0) % 180.0
            )
            shape_reliability = float(
                np.clip((axis_ratio - 1.0) / 1.5, 0.0, 0.7)
            )
            pixel_radii = np.hypot(
                xs - prep.nucleus_yx[1], ys - prep.nucleus_yx[0]
            )
            pixel_angles = (
                np.degrees(
                    np.arctan2(
                        -(xs - prep.nucleus_yx[1]),
                        -(ys - prep.nucleus_yx[0]),
                    )
                )
                + 360.0
            ) % 360.0
            inner_region = pixel_radii <= (runtime_config.ring_inner_px + 1.0)
            inner_fraction = float(np.mean(inner_region))
            threshold_fraction = float(
                np.mean(level_map[mask] / max(len(runtime_config.snr_thresholds), 1))
            )
            centroid_ra, centroid_dec = pixel_to_sky(
                prep.image, centroid_x, centroid_y
            )
            radius_arcsec, sky_pa = sky_separation_and_pa(
                prep.image,
                (float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0])),
                (centroid_x, centroid_y),
            )
            integrated_flux = float(prep.residual[mask].sum())
            measurement_ids: list[str] = []
            structure_measurements = [
                ("integrated_snr", candidate.integrated_snr, None, min(0.7, abs(candidate.integrated_snr) / 20.0), "audited_candidate_signal"),
                ("radius", candidate.radius_px, "px", 0.65, "audited_candidate_geometry"),
                ("angular_extent", candidate.angular_extent_deg, "deg", 0.6, "audited_candidate_geometry"),
                ("axis_ratio", axis_ratio, None, shape_reliability, "audited_candidate_morphology"),
                ("tangential_misalignment", tangential_misalignment, "deg", shape_reliability, "audited_candidate_morphology"),
                ("threshold_survival_fraction", threshold_fraction, None, threshold_fraction, "shared_residual_detection"),
                ("sep_survival_fraction", candidate.sep_survival_fraction, None, candidate.sep_survival_fraction or 0.0, "shared_residual_detection"),
                ("inner_connection_fraction", inner_fraction, None, 0.55, "subtraction_quality"),
                ("foreground_comparison_survival_fraction", candidate.foreground_comparison_survival_fraction, None, 0.55 if candidate.foreground_comparison_survival_fraction is not None else 0.0, "foreground_model_comparison"),
            ]
            for kind, value, unit, reliability, correlation_group in structure_measurements:
                measurement_id = f"M_{structure_id}_{kind}"
                measurement_ids.append(measurement_id)
                measurements.append(
                    Measurement(
                        id=measurement_id,
                        kind=kind,
                        value=value,
                        unit=unit,
                        reliability=float(np.clip(reliability, 0.0, 1.0)),
                        method="candidate_structure_audit_promotion",
                        description=(
                            f"{kind.replace('_', ' ')} for {structure_id}, promoted "
                            f"from visual candidate {candidate.id}; this is a measured "
                            "residual structure, not a lens label"
                        ),
                        structure_id=structure_id,
                        correlation_group=correlation_group,
                        flags=["audited_structure_measurement", f"source_candidate:{candidate.id}"],
                    )
                )
            structures.append(
                StructureObservation(
                    id=structure_id,
                    centroid_xy=(centroid_x, centroid_y),
                    radius_px=candidate.radius_px,
                    position_angle_deg=candidate.position_angle_deg,
                    centroid_ra_deg=centroid_ra,
                    centroid_dec_deg=centroid_dec,
                    radius_arcsec=radius_arcsec,
                    sky_position_angle_deg=sky_pa,
                    pixel_count=candidate.pixel_count,
                    integrated_flux=integrated_flux,
                    integrated_snr=candidate.integrated_snr,
                    median_radius_px=float(np.median(pixel_radii)),
                    angular_extent_deg=candidate.angular_extent_deg,
                    axis_ratio=axis_ratio,
                    tangential_misalignment_deg=tangential_misalignment,
                    threshold_survival_fraction=threshold_fraction,
                    sep_survival_fraction=candidate.sep_survival_fraction,
                    inner_connection_fraction=inner_fraction,
                    connects_to_inner_region=inner_fraction > 0.0,
                    near_core=candidate.near_core,
                    source_candidate_ids=[candidate.id],
                    candidate_audit_ids=[audit.id],
                    measurement_ids=measurement_ids,
                    flags=[
                        "candidate_structure_audit_promoted",
                        "not_a_lens_label",
                        f"audit_discovery_context:{audit.discovery_context}",
                        "geometry_route_relation_recorded"
                        if candidate.geometry_consistent_path_ids
                        else "geometry_route_relation_absent",
                        *audit.caution_flags,
                    ],
                )
            )
        candidate_structure_audits = [
            audit.model_copy(
                update={
                    "structure_id": promoted_structure_id_by_candidate.get(
                        audit.candidate_id
                    )
                }
            )
            for audit in candidate_structure_audits
        ]

        sep_only_mask = downgraded_sep_trace_mask.copy()
        sep_geometry_mask = np.zeros(prep.image.data.shape, dtype=bool)
        geometry_recovered_mask = np.zeros(prep.image.data.shape, dtype=bool)
        model_candidate_mask = np.zeros(prep.image.data.shape, dtype=bool)
        for candidate in model_candidates:
            mask = model_candidate_masks[candidate.id]
            model_candidate_mask |= mask
            if candidate.status == "sep_only":
                sep_only_mask |= mask
            elif candidate.status == "sep_geometry_consistent":
                sep_geometry_mask |= mask
            elif candidate.status == "geometry_recovered":
                geometry_recovered_mask |= mask
        audited_structure_mask = np.zeros_like(model_candidate_mask, dtype=bool)
        for structure_mask in structure_masks.values():
            audited_structure_mask |= structure_mask
        audit_review_mask = np.zeros_like(model_candidate_mask, dtype=bool)
        audit_trace_mask = np.zeros_like(model_candidate_mask, dtype=bool)
        for audit in candidate_structure_audits:
            candidate_mask = model_candidate_masks[audit.candidate_id]
            if audit.verdict == "hold_for_review":
                audit_review_mask |= candidate_mask
            elif audit.verdict == "trace_only":
                audit_trace_mask |= candidate_mask
        comparison_survival = comparison_fraction(audited_structure_mask)
        measurements.extend(
            [
                Measurement(
                    id="M_model_comparison_success",
                    kind="foreground_model_comparison_success",
                    value=comparison.success,
                    reliability=1.0,
                    method=comparison.method,
                    description="Whether the audit-only PSF-convolved MGE comparison fit completed",
                    correlation_group="foreground_model_comparison",
                    flags=comparison.flags,
                ),
                Measurement(
                    id="M_model_comparison_structure_survival",
                    kind="measured_structure_pixel_survival_fraction",
                    value=comparison_survival,
                    reliability=0.65 if comparison_survival is not None else 0.0,
                    method=comparison.method,
                    description="Fraction of audited structure pixels still above the same S/N threshold under the comparison subtraction",
                    correlation_group="foreground_model_comparison",
                    flags=["audit_only", "not_used_for_detection"],
                ),
                Measurement(
                    id="M_model_comparison_residual_correlation",
                    kind="foreground_subtraction_residual_correlation",
                    value=residual_correlation,
                    reliability=0.65 if residual_correlation is not None else 0.0,
                    method=comparison.method,
                    description="Correlation of primary and comparison residuals inside the configured annulus",
                    correlation_group="foreground_model_comparison",
                    flags=["audit_only", "not_used_for_detection"],
                ),
            ]
        )
        for index, candidate in enumerate(geometric_paths_raw, start=1):
            candidate_id = f"G{index}"
            entries = [
                ("semi_major_radius", candidate.semi_major_radius_px, "px", "geometric_path_proposal"),
                ("axis_ratio", candidate.axis_ratio, None, "geometric_path_proposal"),
                ("position_angle", candidate.position_angle_deg, "deg", "geometric_path_proposal"),
                ("half_width", candidate.half_width_px, "px", "seed_width_psf_and_fit_uncertainty"),
                ("fit_scatter", candidate.fit_scatter_px, "px", "seed_path_fit"),
                ("above_local_background_fraction", candidate.above_local_background_fraction, None, "adjacent_path_comparison"),
                ("longest_above_local_background", candidate.longest_above_local_background_deg, "deg", "adjacent_path_comparison"),
                ("median_on_path_snr", candidate.median_on_path_snr, None, "path_snr_sampling"),
                ("on_minus_off_path_snr", candidate.on_minus_off_path_snr, None, "adjacent_path_comparison"),
                ("recovered_segment_count", len(candidate.recovered_segments), None, "guided_residual_recovery"),
            ]
            measurement_ids: list[str] = []
            for kind, value, unit, method in entries:
                measurement_id = f"M_{candidate_id}_{kind}"
                measurement_ids.append(measurement_id)
                measurements.append(
                    Measurement(
                        id=measurement_id,
                        kind=f"geometric_path_{kind}",
                        value=value,
                        unit=unit,
                        reliability=0.45,
                        method=method,
                        description=(
                            f"{kind.replace('_', ' ')} measured for {candidate_id}; "
                            "the path guides a pixel search and is not an observed ring or lens label"
                        ),
                        structure_id=candidate_id,
                        correlation_group="geometry_guided_residual_search",
                        flags=["not_a_lens_decision", "search_route_not_structure"],
                    )
                )
            seed_ids = [
                candidate_id
                for seed_index in candidate.seed_indices
                for candidate_id in candidate_ids_by_seed.get(seed_index, [])
            ]
            linked_ids = supported_candidate_ids_by_path.get(index - 1, [])
            recovered_ids = recovered_ids_by_path.get(index - 1, [])
            flags = ["not_a_lens_decision", "search_route_not_structure", "single_iteration"]
            geometric_paths.append(
                GeometricPathObservation(
                    id=candidate_id,
                    mode=candidate.mode,
                    seed_structure_ids=seed_ids,
                    linked_structure_ids=linked_ids,
                    center_xy=(candidate.center_x, candidate.center_y),
                    semi_major_radius_px=candidate.semi_major_radius_px,
                    axis_ratio=candidate.axis_ratio,
                    position_angle_deg=candidate.position_angle_deg,
                    half_width_px=candidate.half_width_px,
                    fit_scatter_px=candidate.fit_scatter_px,
                    above_local_background_fraction=candidate.above_local_background_fraction,
                    longest_above_local_background_deg=candidate.longest_above_local_background_deg,
                    median_on_path_snr=candidate.median_on_path_snr,
                    on_minus_off_path_snr=candidate.on_minus_off_path_snr,
                    recovered_segments=[],
                    seed_candidate_ids=seed_ids,
                    linked_candidate_ids=linked_ids,
                    recovered_candidate_ids=recovered_ids,
                    measurement_ids=measurement_ids,
                    flags=flags,
                )
            )

        all_angles: list[np.ndarray] = []
        for structure in structures:
            structure_mask = structure_masks[structure.id]
            structure_y, structure_x = np.where(structure_mask)
            all_angles.append(
                (
                    np.degrees(
                        np.arctan2(
                            -(structure_x - prep.nucleus_yx[1]),
                            -(structure_y - prep.nucleus_yx[0]),
                        )
                    )
                    + 360.0
                )
                % 360.0
            )
        radii = [item.median_radius_px for item in structures]
        common_radius_cv = (
            float(np.std(radii) / max(np.mean(radii), 1e-9)) if len(radii) >= 2 else None
        )
        total_coverage = (
            circular_span(np.concatenate(all_angles)) if all_angles else 0.0
        )
        valid_misalignments = [
            item.tangential_misalignment_deg
            for item in structures
            if item.tangential_misalignment_deg is not None
        ]
        mean_misalignment = (
            float(np.mean(valid_misalignments)) if valid_misalignments else None
        )
        tangential_fraction = (
            float(np.mean([value <= 30.0 for value in valid_misalignments]))
            if valid_misalignments
            else None
        )
        scene_entries = [
            (
                "M_scene_structure_count",
                "structure_count",
                len(structures),
                None,
                1.0,
                "segmentation_summary",
                "Number of visual candidates promoted by the candidate-structure audit",
            ),
            (
                "M_scene_common_radius_cv",
                "common_radius_cv",
                common_radius_cv,
                None,
                0.8 if len(radii) >= 2 else 0.0,
                "geometry_summary",
                "Coefficient of variation of structure radii; lower means a more common radius",
            ),
            (
                "M_scene_total_angular_coverage",
                "total_angular_coverage",
                total_coverage,
                "deg",
                min(1.0, sum(item.pixel_count for item in structures) / 40.0),
                "residual_structure_union",
                "Minimum circular span containing all audited structure pixels",
            ),
            (
                "M_scene_model_success",
                "galaxy_model_success",
                prep.model_success,
                None,
                1.0,
                prep.model_method,
                "Whether the preferred foreground-galaxy model converged",
            ),
            (
                "M_scene_mean_tangential_misalignment",
                "mean_tangential_misalignment",
                mean_misalignment,
                "deg",
                0.8 if valid_misalignments else 0.0,
                "geometry_summary",
                "Mean local tangential misalignment across measured structures",
            ),
            (
                "M_scene_tangential_alignment_fraction",
                "tangential_alignment_fraction",
                tangential_fraction,
                None,
                0.8 if valid_misalignments else 0.0,
                "geometry_summary",
                "Fraction of measured structures within 30 degrees of the local tangent",
            ),
        ]
        for mid, kind, value, unit, reliability, method, description in scene_entries:
            measurements.append(
                Measurement(
                    id=mid,
                    kind=kind,
                    value=value,
                    unit=unit,
                    reliability=reliability,
                    method=method,
                    description=description,
                    correlation_group="scene_geometry" if "model" not in kind else "subtraction_quality",
                )
            )

        foreground_entries = [
            (
                "M_foreground_detected",
                "foreground_detected",
                prep.foreground_detected,
                None,
                prep.foreground_reliability,
                "sep_central_source",
                "Whether a central foreground light concentration was measured",
            ),
            (
                "M_foreground_peak_value",
                "foreground_peak_value",
                prep.foreground_peak_value,
                "image_value",
                prep.foreground_reliability,
                "central_pixel",
                "Image value at the measured central light peak",
            ),
            (
                "M_foreground_integrated_flux",
                "foreground_integrated_flux",
                prep.foreground_integrated_flux,
                "image_value",
                prep.foreground_reliability if prep.foreground_integrated_flux is not None else 0.0,
                "sep_central_source",
                "SEP-integrated flux of the measured central light concentration",
            ),
            (
                "M_foreground_ellipticity",
                "foreground_ellipticity",
                prep.foreground_ellipticity,
                None,
                prep.foreground_reliability if prep.foreground_ellipticity is not None else 0.0,
                "sep_second_moments",
                "Ellipticity of the central light distribution",
            ),
            (
                "M_foreground_position_angle",
                "foreground_position_angle",
                prep.foreground_position_angle_deg,
                "deg",
                prep.foreground_reliability if prep.foreground_position_angle_deg is not None else 0.0,
                "sep_second_moments",
                "Position angle of the central light distribution",
            ),
            (
                "M_foreground_axis_ratio",
                "foreground_axis_ratio",
                prep.foreground_axis_ratio,
                None,
                prep.foreground_reliability if prep.foreground_axis_ratio is not None else 0.0,
                "sep_second_moments",
                "Major-to-minor axis ratio of the central light distribution",
            ),
        ]
        for mid, kind, value, unit, reliability, method, description in foreground_entries:
            measurements.append(
                Measurement(
                    id=mid,
                    kind=kind,
                    value=value,
                    unit=unit,
                    reliability=float(np.clip(reliability, 0.0, 1.0)),
                    method=method,
                    description=description,
                    correlation_group="foreground_light",
                    flags=[] if prep.foreground_detected else ["central_source_not_sep_detected"],
                )
            )

        alternative_entries = [
            (
                "M_alt_multiple_core_count",
                "multiple_core_count",
                prep.multiple_core_count,
                None,
                0.55 if prep.multiple_core_count is not None else 0.0,
                "sep_central_source_count",
                "Number of secondary central sources with at least 10% of the primary SEP flux",
                ["proxy_measurement"],
            ),
            (
                "M_alt_tidal_asymmetry",
                "tidal_asymmetry",
                prep.tidal_asymmetry,
                None,
                0.45 if prep.tidal_asymmetry is not None else 0.0,
                "residual_180_degree_asymmetry",
                "Rotation-asymmetry proxy measured on the foreground-subtracted residual",
                ["proxy_measurement", "subtraction_correlated"],
            ),
            (
                "M_alt_core_residual_fraction",
                "core_residual_fraction",
                prep.core_residual_fraction,
                None,
                0.55 if prep.core_residual_fraction is not None else 0.0,
                "absolute_residual_flux_ratio",
                "Fraction of absolute residual flux inside the subtraction-sensitive core",
                ["proxy_measurement", "subtraction_correlated"],
            ),
            (
                "M_alt_bad_pixel_fraction",
                "bad_pixel_fraction",
                1.0 - image.finite_fraction,
                None,
                1.0,
                "finite_input_pixel_count",
                "Fraction of input pixels requiring non-finite-value replacement",
                [],
            ),
            (
                "M_alt_psf_information_available",
                "psf_information_available",
                image.psf is not None,
                None,
                1.0,
                "input_metadata_check",
                "Whether a PSF image is present in the selected FITS band",
                [] if image.psf is not None else ["not_available"],
            ),
        ]
        for mid, kind, value, unit, reliability, method, description, item_flags in alternative_entries:
            measurements.append(
                Measurement(
                    id=mid,
                    kind=kind,
                    value=value,
                    unit=unit,
                    reliability=reliability,
                    method=method,
                    description=description,
                    correlation_group=(
                        "subtraction_quality"
                        if kind in {"tidal_asymmetry", "core_residual_fraction"}
                        else "interaction_morphology"
                        if kind == "multiple_core_count"
                        else "input_quality"
                    ),
                    flags=item_flags,
                )
            )

        artifacts: dict[str, str] = {}
        if output is not None:
            artifact_path = output / "vision_artifacts.npz"
            arrays = {
                "data": prep.image.data,
                "rms": prep.image.rms,
                "model": prep.model,
                "residual": prep.residual,
                "residual_snr": prep.residual_snr,
                "source_mask": prep.source_mask.astype(np.uint8),
                "fit_mask": prep.fit_mask.astype(np.uint8),
                "initial_segmentation": prep.segmentation,
                "residual_segmentation": residual_segmentation,
                "sep_footprint": footprint.astype(np.uint8),
                "sep_raw_consensus_segmentation": residual_segmentation,
                "sep_raw_consensus_footprint": footprint.astype(np.uint8),
                "sep_geometry_seed_mask": sep_seed_mask.astype(np.uint8),
                "model_candidate_mask": model_candidate_mask.astype(np.uint8),
                "sep_only_candidate_mask": sep_only_mask.astype(np.uint8),
                "sep_geometry_consistent_mask": sep_geometry_mask.astype(np.uint8),
                "geometry_recovered_candidate_mask": geometry_recovered_mask.astype(np.uint8),
                "residual_structure_mask": audited_structure_mask.astype(np.uint8),
                "signal_audit_verified_mask": audited_structure_mask.astype(np.uint8),
                "signal_audit_review_mask": audit_review_mask.astype(np.uint8),
                "signal_audit_trace_mask": audit_trace_mask.astype(np.uint8),
                # Compatibility alias for v0.5 and earlier analysis notebooks.
                "arc_mask": audited_structure_mask.astype(np.uint8),
                "threshold_level_map": level_map,
                "nucleus_yx": np.asarray(prep.nucleus_yx),
            }
            for candidate_id, candidate_mask in model_candidate_masks.items():
                arrays[f"model_candidate_mask_{candidate_id}"] = candidate_mask.astype(np.uint8)
            for index, candidate in enumerate(geometric_paths_raw, start=1):
                arrays[f"geometric_path_mask_G{index}"] = candidate.route_mask.astype(np.uint8)
                arrays[f"geometric_path_recovered_mask_G{index}"] = candidate.recovered_mask.astype(np.uint8)
            if prep.image.psf is not None:
                arrays["psf"] = prep.image.psf
            if comparison.success and comparison.model is not None and comparison.residual is not None:
                arrays["comparison_psf_mge_model"] = comparison.model
                arrays["comparison_psf_mge_residual"] = comparison.residual
            np.savez_compressed(artifact_path, **arrays)
            artifacts["vision_arrays"] = str(artifact_path)
            diagnostic_path = self._write_diagnostic(
                prep, audited_structure_mask, [], runtime_config, output
            )
            if diagnostic_path:
                artifacts["diagnostic_image"] = str(diagnostic_path)
            artifacts.update(
                write_visual_audit_bundle(
                    prep=prep,
                    config=runtime_config,
                    threshold_detections=threshold_detections,
                    consensus_detections=consensus_detections,
                    level_map=level_map,
                    footprint=footprint,
                    residual_structure_mask=audited_structure_mask,
                    sep_only_candidate_mask=sep_only_mask,
                    sep_geometry_mask=sep_geometry_mask,
                    geometry_recovered_mask=geometry_recovered_mask,
                    model_candidate_mask=model_candidate_mask,
                    audit_review_mask=audit_review_mask,
                    audit_trace_mask=audit_trace_mask,
                    segments=[],
                    geometric_paths=geometric_paths_raw,
                    foreground_comparison=comparison,
                    output=output,
                    candidate_points=[
                        (
                            candidate.id,
                            candidate.centroid_xy[0],
                            candidate.centroid_xy[1],
                            candidate.status,
                        )
                        for candidate in model_candidates
                    ],
                )
            )

        finite_fraction = image.finite_fraction
        flags = list(prep.flags)
        if not structures:
            flags.append("no_audited_residual_structure")
        if model_candidates:
            flags.append("model_chain_candidates_audited")
        limitations = [
            "All residual detectors share the same galaxy-subtracted image and are correlated.",
            "A single-band run cannot test source-color consistency.",
            "Disk connectivity, multiple-core count, and residual asymmetry are screening proxies rather than calibrated morphology classifiers.",
            "Merged SEP components are locally partitioned by route relation, but off-route status is provenance rather than a signal-audit veto.",
            "Candidate-structure audit validates detector-level signal usability without using geometric route membership as a verdict condition; promotion is not an arc, ring, multiple-image, or lens label.",
            "Geometric paths only guide masked SEP searches; unfilled path regions are not treated as rings or missing evidence against lensing.",
        ]
        if image.wcs is None or image.pixel_scale_arcsec is None:
            limitations.append(
                "No validated celestial WCS/pixel scale is available; angular measurements remain in pixel geometry."
            )
        if image.psf is None:
            limitations.append(
                "No PSF image is present in the selected band; PSF-specific artifact rejection is unavailable."
            )
        else:
            limitations.append(
                "A PSF image is present and used by the audit-only comparison model, but the formal detection subtraction does not yet use PSF convolution."
            )
        if image.input_format == "png":
            limitations.append("PNG noise was estimated from image pixels; no supplied RMS map was available.")
        if output is not None:
            artifacts["observation_summary_cn"] = str(
                output / "observation_summary_cn.md"
            )
        center_xy = (float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0]))
        center_ra, center_dec = pixel_to_sky(image, *center_xy)
        north_vector, east_vector = orientation_vectors(image, center_xy)
        wcs_valid = (
            image.wcs is not None
            and center_ra is not None
            and center_dec is not None
            and image.pixel_scale_arcsec is not None
        )
        document = ObservationDocument(
            object_id=image.object_id,
            input_path=str(image.path),
            input_format=image.input_format,
            band=image.band,
            image_shape=image.data.shape,
            nucleus_xy=(float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0])),
            data_quality=DataQuality(
                usable=finite_fraction > 0.99 and prep.background_rms > 0,
                supplied_rms=image.supplied_rms,
                finite_fraction=finite_fraction,
                background_median=prep.background_median,
                background_rms=prep.background_rms,
                model_method=prep.model_method,
                model_success=prep.model_success,
                residual_median=prep.residual_median,
                residual_robust_rms=prep.residual_robust_rms,
                flags=flags,
            ),
            foreground_galaxy=ForegroundGalaxyObservation(
                detected=prep.foreground_detected,
                center_xy=(float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0])),
                peak_value=prep.foreground_peak_value,
                integrated_flux=prep.foreground_integrated_flux,
                ellipticity=prep.foreground_ellipticity,
                position_angle_deg=prep.foreground_position_angle_deg,
                axis_ratio=prep.foreground_axis_ratio,
                reliability=prep.foreground_reliability,
                method="sep_central_source_and_second_moments",
                measurement_ids=[item[0] for item in foreground_entries],
                flags=[] if prep.foreground_detected else ["central_source_not_sep_detected"],
            ),
            scene_geometry=SceneGeometryObservation(
                structure_count=len(structures),
                common_radius_cv=common_radius_cv,
                total_angular_coverage_deg=total_coverage,
                mean_tangential_misalignment_deg=mean_misalignment,
                tangential_alignment_fraction=tangential_fraction,
                measurement_ids=[item[0] for item in scene_entries if "model" not in item[1]],
            ),
            alternative_indicators=AlternativeIndicators(
                inner_connected_structure_count=sum(
                    bool(item.connects_to_inner_region) for item in structures
                ),
                multiple_core_count=prep.multiple_core_count,
                tidal_asymmetry=prep.tidal_asymmetry,
                core_residual_fraction=prep.core_residual_fraction,
                bad_pixel_fraction=1.0 - image.finite_fraction,
                psf_information_available=image.psf is not None,
                measurement_ids=[item[0] for item in alternative_entries],
                limitations=[
                    "Connectivity and asymmetry are residual-image proxies.",
                    "No PSF-specific measurement is available.",
                ],
            ),
            astrometry=AstrometryObservation(
                wcs_available=image.wcs is not None,
                wcs_valid=wcs_valid,
                frame="ICRS" if image.wcs is not None else None,
                projection=(
                    "/".join(str(item) for item in image.wcs.wcs.ctype)
                    if image.wcs is not None
                    else None
                ),
                pixel_scale_arcsec=image.pixel_scale_arcsec,
                center_ra_deg=center_ra,
                center_dec_deg=center_dec,
                north_vector_xy=north_vector,
                east_vector_xy=east_vector,
                display_origin="lower" if image.input_format == "fits" else "upper",
                orientation_description=(
                    "Native FITS pixel grid displayed with WCS north/east arrows; no reprojection applied."
                    if image.wcs is not None
                    else "Raster pixel orientation; no celestial WCS available."
                ),
                flags=[] if wcs_valid else ["wcs_unavailable_or_invalid"],
            ),
            foreground_model_comparison=ForegroundModelComparisonObservation(
                primary_method=prep.model_method,
                comparison_method=comparison.method,
                comparison_success=comparison.success,
                psf_applied=comparison.psf_applied,
                used_for_detection=False,
                primary_residual_robust_rms=prep.residual_robust_rms,
                comparison_residual_robust_rms=comparison.residual_robust_rms,
                measured_structure_pixel_survival_fraction=comparison_survival,
                residual_correlation=residual_correlation,
                flags=comparison.flags,
            ),
            ring_candidates=ring_candidates,
            geometric_paths=geometric_paths,
            model_candidates=model_candidates,
            candidate_systems=candidate_systems,
            candidate_structure_audits=candidate_structure_audits,
            structures=structures,
            measurements=measurements,
            artifacts=artifacts,
            limitations=limitations,
            provenance=Provenance(parameters=runtime_config.to_dict()),
        )
        if output is not None:
            write_observation_summary_cn(document, output)
        return document

    @staticmethod
    def _write_diagnostic(
        prep,
        residual_structure_mask: np.ndarray,
        segments: list[ResidualStructure],
        config: VisionConfig,
        output: Path,
    ) -> Path | None:
        cache_dir = output / ".matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        y, x = prep.nucleus_yx
        half = min(60, y, x, prep.image.data.shape[0] - y, prep.image.data.shape[1] - x)
        view = np.s_[y - half : y + half, x - half : x + half]
        figure, axes = plt.subplots(1, 4, figsize=(16, 4))
        panels = [prep.image.data, prep.model, prep.residual_snr, residual_structure_mask]
        titles = [
            f"{prep.image.band}_FLUX input (central crop)",
            "foreground light model",
            "foreground-subtracted residual S/N",
            "signal-audited residual structures (not lens labels)",
        ]
        cmaps = ["gray", "gray", "coolwarm", "gray"]
        origin = "lower" if prep.image.input_format == "fits" else "upper"
        for panel_index, (axis, panel, title, cmap) in enumerate(
            zip(axes, panels, titles, cmaps)
        ):
            shown = panel[view]
            kwargs = {}
            finite = shown[np.isfinite(shown)]
            if finite.size and panel_index in {0, 1}:
                low, high = np.percentile(finite, [1.0, 99.5])
                if high > low:
                    kwargs.update(vmin=float(low), vmax=float(high))
            elif finite.size and panel_index == 2:
                limit = float(np.percentile(np.abs(finite), 99.0))
                if limit > 0:
                    kwargs.update(vmin=-limit, vmax=limit)
            axis.imshow(shown, origin=origin, cmap=cmap, **kwargs)
            axis.plot(half, half, "c+", ms=10, mew=1.5)
            for radius, style in (
                (config.ring_inner_px, "--"),
                (config.ring_outer_px, "-"),
            ):
                axis.add_patch(
                    plt.Circle(
                        (half, half), radius, fill=False, color="yellow",
                        linestyle=style, linewidth=0.8, alpha=0.9,
                    )
                )
            for structure_index, segment in enumerate(segments, start=1):
                local_x = segment.x - (prep.nucleus_yx[1] - half)
                local_y = segment.y - (prep.nucleus_yx[0] - half)
                axis.plot(local_x, local_y, "mx", ms=6, mew=1.2)
                axis.annotate(
                    f"S{structure_index}", (local_x + 2, local_y + 2),
                    color="magenta", fontsize=7, weight="bold",
                )
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
        north, east = orientation_vectors(
            prep.image,
            (float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0])),
        )
        if north is not None and east is not None:
            base_x, base_y, length = 12.0, 12.0, 10.0
            for vector, label, color in (
                (north, "N", "lime"), (east, "E", "cyan")
            ):
                axes[0].annotate(
                    "",
                    xy=(base_x + vector[0] * length, base_y + vector[1] * length),
                    xytext=(base_x, base_y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                )
                axes[0].text(
                    base_x + vector[0] * (length + 3),
                    base_y + vector[1] * (length + 3),
                    label, color=color, fontsize=8, weight="bold",
                )
        figure.tight_layout()
        path = output / "vision_diagnostic.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        return path
