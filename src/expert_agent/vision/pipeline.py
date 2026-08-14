from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..core.config import VisionConfig
from ..schemas.observation import (
    AlternativeIndicators,
    DataQuality,
    ForegroundGalaxyObservation,
    Measurement,
    ObservationDocument,
    Provenance,
    SceneGeometryObservation,
    StructureObservation,
)
from .detectors import (
    ArcSegment,
    StabilityDetection,
    circular_span,
    extract_arc_segments,
    sep_consensus,
    threshold_stability,
)
from .diagnostics import write_visual_audit_bundle
from .preprocess import load_image, preprocess


def _nearest(
    segment: ArcSegment, detections: list[StabilityDetection], max_distance: float = 4.0
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
                arc_snr_low=max(3.0, self.config.arc_snr_low),
            )
        prep = preprocess(image, runtime_config)
        threshold_detections, level_map = threshold_stability(prep, runtime_config)
        consensus_detections = sep_consensus(prep, runtime_config)
        segments, residual_segmentation, footprint, arc_mask = extract_arc_segments(
            prep, runtime_config
        )

        measurements: list[Measurement] = []
        structures: list[StructureObservation] = []
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
                ("integrated_snr", float(segment.snr), None, "arc_pixels", base_reliability),
                ("radius", float(segment.median_radius_px), "px", "arc_pixels", base_reliability),
                ("angular_extent", float(segment.angular_extent_deg), "deg", "arc_pixels", base_reliability),
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
                "arc_pixel_union",
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
                False,
                None,
                1.0,
                "input_metadata_check",
                "Whether a validated PSF model was supplied to this pipeline",
                ["not_measured"],
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
            np.savez_compressed(
                artifact_path,
                data=prep.image.data,
                rms=prep.image.rms,
                model=prep.model,
                residual=prep.residual,
                residual_snr=prep.residual_snr,
                source_mask=prep.source_mask.astype(np.uint8),
                fit_mask=prep.fit_mask.astype(np.uint8),
                initial_segmentation=prep.segmentation,
                residual_segmentation=residual_segmentation,
                sep_footprint=footprint.astype(np.uint8),
                arc_mask=arc_mask.astype(np.uint8),
                threshold_level_map=level_map,
                nucleus_yx=np.asarray(prep.nucleus_yx),
            )
            artifacts["vision_arrays"] = str(artifact_path)
            diagnostic_path = self._write_diagnostic(prep, arc_mask, output)
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
                    arc_mask=arc_mask,
                    output=output,
                )
            )

        finite_fraction = image.finite_fraction
        flags = list(prep.flags)
        if not structures:
            flags.append("no_significant_annular_structure")
        limitations = [
            "All residual detectors share the same galaxy-subtracted image and are correlated.",
            "Angular measurements are in pixel geometry unless a validated pixel scale is supplied.",
            "A single-band run cannot test source-color consistency.",
            "No validated PSF model is supplied; PSF-specific artifact rejection is unavailable.",
            "Disk connectivity, multiple-core count, and residual asymmetry are screening proxies rather than calibrated morphology classifiers.",
        ]
        if image.input_format == "png":
            limitations.append("PNG noise was estimated from image pixels; no supplied RMS map was available.")
        return ObservationDocument(
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
                psf_information_available=False,
                measurement_ids=[item[0] for item in alternative_entries],
                limitations=[
                    "Connectivity and asymmetry are residual-image proxies.",
                    "No PSF-specific measurement is available.",
                ],
            ),
            structures=structures,
            measurements=measurements,
            artifacts=artifacts,
            limitations=limitations,
            provenance=Provenance(parameters=runtime_config.to_dict()),
        )

    @staticmethod
    def _write_diagnostic(prep, arc_mask: np.ndarray, output: Path) -> Path | None:
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
        panels = [prep.image.data, prep.model, prep.residual_snr, arc_mask]
        titles = ["input", "galaxy model", "residual S/N", "measured arc pixels"]
        cmaps = ["gray", "gray", "coolwarm", "gray"]
        for axis, panel, title, cmap in zip(axes, panels, titles, cmaps):
            axis.imshow(panel[view], origin="upper", cmap=cmap)
            axis.plot(half, half, "c+", ms=10, mew=1.5)
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
        figure.tight_layout()
        path = output / "vision_diagnostic.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        return path
