from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..core.config import VisionConfig
from ..schemas.observation import (
    DataQuality,
    Measurement,
    ObservationDocument,
    Provenance,
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

        finite_fraction = float(np.isfinite(prep.image.data).mean())
        flags = list(prep.flags)
        if not structures:
            flags.append("no_significant_annular_structure")
        limitations = [
            "All residual detectors share the same galaxy-subtracted image and are correlated.",
            "Angular measurements are in pixel geometry unless a validated pixel scale is supplied.",
            "A single-band run cannot test source-color consistency.",
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
