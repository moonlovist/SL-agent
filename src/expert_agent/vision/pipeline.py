from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..core.config import VisionConfig
from ..schemas.observation import (
    AlternativeIndicators,
    AstrometryObservation,
    DataQuality,
    ForegroundGalaxyObservation,
    ForegroundModelComparisonObservation,
    GeometricPathObservation,
    Measurement,
    ObservationDocument,
    Provenance,
    RecoveredPathSegmentObservation,
    RingCandidateObservation,
    SceneGeometryObservation,
    StructureObservation,
)
from .astrometry import orientation_vectors, pixel_to_sky
from .detectors import (
    ResidualStructure,
    StabilityDetection,
    circular_span,
    extract_residual_structures,
    geometric_path_search,
    sep_consensus,
    threshold_stability,
)
from .diagnostics import write_visual_audit_bundle
from .foreground_models import fit_psf_mge_comparison
from .preprocess import load_image, preprocess
from .summary_cn import write_observation_summary_cn


def _nearest(
    segment: ResidualStructure, detections: list[StabilityDetection], max_distance: float = 4.0
) -> StabilityDetection | None:
    if not detections:
        return None
    ranked = sorted(
        detections,
        key=lambda item: np.hypot(item.x - segment.x, item.y - segment.y),
    )
    distance = float(np.hypot(ranked[0].x - segment.x, ranked[0].y - segment.y))
    return ranked[0] if distance <= max_distance else None


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
        segments, residual_segmentation, footprint, residual_structure_mask = extract_residual_structures(
            prep, runtime_config
        )
        geometric_paths_raw = geometric_path_search(prep, runtime_config, segments)
        comparison = fit_psf_mge_comparison(prep)
        comparison_survival: float | None = None
        residual_correlation: float | None = None
        if comparison.success and comparison.residual is not None:
            comparison_snr = np.divide(
                comparison.residual,
                prep.image.rms,
                out=np.zeros_like(comparison.residual),
                where=prep.image.rms > 0,
            )
            measured_pixels = residual_structure_mask.astype(bool)
            if measured_pixels.any():
                comparison_survival = float(
                    np.mean(
                        comparison_snr[measured_pixels]
                        >= runtime_config.structure_snr_low
                    )
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
        structures: list[StructureObservation] = []
        ring_candidates: list[RingCandidateObservation] = []  # legacy field; no longer emitted
        geometric_paths: list[GeometricPathObservation] = []
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
                    description="Fraction of primary measured-structure pixels still above the same S/N threshold under the comparison subtraction",
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
        all_angles: list[np.ndarray] = []
        for index, segment in enumerate(segments, start=1):
            structure_id = f"S{index}"
            threshold_match = _nearest(segment, threshold_detections)
            consensus_match = _nearest(segment, consensus_detections)
            threshold_fraction = (
                threshold_match.survival / threshold_match.total_runs
                if threshold_match
                else None
            )
            consensus_fraction = (
                consensus_match.survival / consensus_match.total_runs
                if consensus_match
                else None
            )
            axis_ratio = (
                consensus_match.axis_ratio
                if consensus_match and consensus_match.axis_ratio is not None
                else segment.axis_ratio
            )
            misalignment = (
                consensus_match.tangential_misalignment_deg
                if consensus_match
                and consensus_match.tangential_misalignment_deg is not None
                else segment.tangential_misalignment_deg
            )
            base_reliability = min(1.0, max(0.0, abs(segment.snr) / 20.0))
            entries = [
                ("integrated_snr", float(segment.snr), None, "residual_structure_pixels", base_reliability),
                ("radius", float(segment.median_radius_px), "px", "residual_structure_pixels", base_reliability),
                ("radius_arcsec", segment.radius_arcsec, "arcsec", "wcs_transform", 1.0 if segment.radius_arcsec is not None else 0.0),
                ("centroid_ra", segment.centroid_ra_deg, "deg", "wcs_transform", 1.0 if segment.centroid_ra_deg is not None else 0.0),
                ("centroid_dec", segment.centroid_dec_deg, "deg", "wcs_transform", 1.0 if segment.centroid_dec_deg is not None else 0.0),
                ("sky_position_angle", segment.sky_position_angle_deg, "deg", "wcs_transform", 1.0 if segment.sky_position_angle_deg is not None else 0.0),
                ("angular_extent", float(segment.angular_extent_deg), "deg", "residual_structure_pixels", base_reliability),
                ("axis_ratio", float(axis_ratio), None, "sep_consensus", consensus_fraction or 0.4),
                ("tangential_misalignment", float(misalignment), "deg", "sep_consensus", consensus_fraction or 0.4),
                (
                    "threshold_survival_fraction",
                    threshold_fraction,
                    None,
                    "snr_threshold_grid",
                    threshold_fraction or 0.0,
                ),
                (
                    "sep_survival_fraction",
                    consensus_fraction,
                    None,
                    "sep_parameter_grid",
                    consensus_fraction or 0.0,
                ),
                (
                    "inner_connection_fraction",
                    float(segment.inner_connection_fraction),
                    None,
                    "residual_component_connectivity",
                    min(0.7, base_reliability),
                ),
            ]
            measurement_ids = []
            for kind, value, unit, method, reliability in entries:
                measurement_id = f"M_{structure_id}_{kind}"
                measurement_ids.append(measurement_id)
                measurements.append(
                    Measurement(
                        id=measurement_id,
                        kind=kind,
                        value=value,
                        unit=unit,
                        reliability=float(np.clip(reliability, 0.0, 1.0)),
                        method=method,
                        description=f"{kind.replace('_', ' ')} measured for {structure_id}",
                        structure_id=structure_id,
                        correlation_group=(
                            "shared_residual_detection"
                            if "survival" in kind
                            else "astrometry"
                            if method == "wcs_transform"
                            else "galaxy_connectivity"
                            if kind == "inner_connection_fraction"
                            else "residual_morphology"
                        ),
                    )
                )
            near_core = segment.median_radius_px <= runtime_config.near_core_radius_px
            flags = ["near_core_subtraction_sensitive"] if near_core else []
            structures.append(
                StructureObservation(
                    id=structure_id,
                    centroid_xy=(segment.x, segment.y),
                    radius_px=segment.median_radius_px,
                    position_angle_deg=segment.position_angle_deg,
                    centroid_ra_deg=segment.centroid_ra_deg,
                    centroid_dec_deg=segment.centroid_dec_deg,
                    radius_arcsec=segment.radius_arcsec,
                    sky_position_angle_deg=segment.sky_position_angle_deg,
                    pixel_count=segment.pixel_count,
                    integrated_flux=segment.flux,
                    integrated_snr=segment.snr,
                    median_radius_px=segment.median_radius_px,
                    angular_extent_deg=segment.angular_extent_deg,
                    axis_ratio=axis_ratio,
                    tangential_misalignment_deg=misalignment,
                    threshold_survival_fraction=threshold_fraction,
                    sep_survival_fraction=consensus_fraction,
                    inner_connection_fraction=segment.inner_connection_fraction,
                    connects_to_inner_region=segment.connects_to_inner_region,
                    near_core=near_core,
                    measurement_ids=measurement_ids,
                    flags=flags,
                )
            )
            yy = segment.pixels_y
            xx = segment.pixels_x
            angles = (
                np.degrees(
                    np.arctan2(
                        -(xx - prep.nucleus_yx[1]),
                        -(yy - prep.nucleus_yx[0]),
                    )
                )
                + 360
            ) % 360
            all_angles.append(angles)

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
            recovered = [
                RecoveredPathSegmentObservation(
                    id=f"{candidate_id}N{recovered_index}",
                    centroid_xy=(item.x, item.y),
                    pixel_count=item.pixel_count,
                    integrated_snr=item.integrated_snr,
                    angular_extent_deg=item.angular_extent_deg,
                )
                for recovered_index, item in enumerate(candidate.recovered_segments, start=1)
            ]
            seed_ids = [f"S{item + 1}" for item in candidate.seed_indices]
            linked_ids = [f"S{item + 1}" for item in candidate.linked_structure_indices]
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
                    recovered_segments=recovered,
                    measurement_ids=measurement_ids,
                    flags=flags,
                )
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
                "Number of significant residual structures in the configured annulus",
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
                "Minimum circular span containing all detected arc pixels",
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
                "residual_structure_mask": residual_structure_mask.astype(np.uint8),
                # Compatibility alias for v0.5 and earlier analysis notebooks.
                "arc_mask": residual_structure_mask.astype(np.uint8),
                "threshold_level_map": level_map,
                "nucleus_yx": np.asarray(prep.nucleus_yx),
            }
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
                prep, residual_structure_mask, segments, runtime_config, output
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
                    residual_structure_mask=residual_structure_mask,
                    segments=segments,
                    geometric_paths=geometric_paths_raw,
                    foreground_comparison=comparison,
                    output=output,
                )
            )

        finite_fraction = image.finite_fraction
        flags = list(prep.flags)
        if not structures:
            flags.append("no_significant_annular_structure")
        limitations = [
            "All residual detectors share the same galaxy-subtracted image and are correlated.",
            "A single-band run cannot test source-color consistency.",
            "Disk connectivity, multiple-core count, and residual asymmetry are screening proxies rather than calibrated morphology classifiers.",
            "Geometric paths only guide a local pixel search; unfilled path regions are not treated as rings or missing evidence against lensing.",
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
            "measured residual-structure pixels",
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
