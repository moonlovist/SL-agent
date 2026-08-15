from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.config import VisionConfig
from .astrometry import (
    axial_misalignment_deg,
    pixel_to_sky,
    sky_axis_pa,
    sky_separation_and_pa,
)
from .preprocess import PreprocessedImage, VisionDependencyError


@dataclass
class StabilityDetection:
    x: float
    y: float
    radius_px: float
    position_angle_deg: float
    survival: int
    total_runs: int
    axis_ratio: float | None = None
    tangential_misalignment_deg: float | None = None
    radius_arcsec: float | None = None
    sky_position_angle_deg: float | None = None


@dataclass
class ResidualStructure:
    label: int
    x: float
    y: float
    pixel_count: int
    flux: float
    snr: float
    median_radius_px: float
    position_angle_deg: float
    angular_extent_deg: float
    axis_ratio: float
    tangential_misalignment_deg: float
    inner_connection_fraction: float
    connects_to_inner_region: bool
    pixels_y: np.ndarray
    pixels_x: np.ndarray
    centroid_ra_deg: float | None
    centroid_dec_deg: float | None
    radius_arcsec: float | None
    sky_position_angle_deg: float | None


def position_angle(x: float, y: float, nucleus_yx: tuple[int, int]) -> float:
    return float(
        (np.degrees(np.arctan2(-(x - nucleus_yx[1]), -(y - nucleus_yx[0]))) + 360)
        % 360
    )


def circular_span(angles: np.ndarray) -> float:
    if angles.size <= 1:
        return 0.0
    ordered = np.sort(np.mod(angles, 360.0))
    gaps = np.diff(np.concatenate([ordered, ordered[:1] + 360.0]))
    return float(360.0 - gaps.max())


def _ring_geometry(prep: PreprocessedImage, config: VisionConfig):
    yy, xx = np.indices(prep.image.data.shape)
    radius = np.hypot(xx - prep.nucleus_yx[1], yy - prep.nucleus_yx[0])
    pa = (
        np.degrees(
            np.arctan2(-(xx - prep.nucleus_yx[1]), -(yy - prep.nucleus_yx[0]))
        )
        + 360
    ) % 360
    ring = (radius > config.ring_inner_px) & (radius < config.ring_outer_px)
    return yy, xx, radius, pa, ring


def _cluster_detections(
    runs: list[list[dict[str, float]]], distance_px: float
) -> list[dict[str, object]]:
    clusters: list[dict[str, object]] = []
    for run_index, detections in enumerate(runs):
        for detection in detections:
            eligible = []
            for cluster in clusters:
                if run_index in cluster["runs"]:
                    continue
                distance = np.hypot(
                    float(cluster["x"]) - detection["x"],
                    float(cluster["y"]) - detection["y"],
                )
                if distance <= distance_px:
                    eligible.append((float(distance), cluster))
            if eligible:
                cluster = min(eligible, key=lambda item: item[0])[1]
                samples = cluster["samples"]
                samples.append(detection)
                cluster["runs"].add(run_index)
                cluster["x"] = float(np.mean([sample["x"] for sample in samples]))
                cluster["y"] = float(np.mean([sample["y"] for sample in samples]))
            else:
                clusters.append(
                    {
                        "x": detection["x"],
                        "y": detection["y"],
                        "runs": {run_index},
                        "samples": [detection],
                    }
                )
    return clusters


def threshold_masks(
    prep: PreprocessedImage, config: VisionConfig
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return the exact annular masks used by threshold stability detection."""
    _, _, _, _, ring = _ring_geometry(prep, config)
    masks = [
        (prep.residual_snr >= threshold) & ring & ~prep.source_mask
        for threshold in config.snr_thresholds
    ]
    return masks, ring


def threshold_stability(
    prep: PreprocessedImage, config: VisionConfig
) -> tuple[list[StabilityDetection], np.ndarray]:
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise VisionDependencyError("threshold detection requires scipy") from exc
    masks, _ = threshold_masks(prep, config)
    runs: list[list[dict[str, float]]] = []
    level_map = np.zeros(prep.image.data.shape, dtype=np.uint8)
    for active in masks:
        labels, count = ndimage.label(active)
        detections: list[dict[str, float]] = []
        for label in range(1, count + 1):
            pixels = labels == label
            area = int(pixels.sum())
            if area < config.snr_min_area:
                continue
            y, x = ndimage.center_of_mass(pixels)
            detections.append({"x": float(x), "y": float(y)})
            level_map[pixels] += 1
        runs.append(detections)
    clusters = _cluster_detections(runs, config.cluster_distance_px)
    results = []
    for cluster in clusters:
        x, y = float(cluster["x"]), float(cluster["y"])
        radius_arcsec, sky_pa = sky_separation_and_pa(
            prep.image,
            (float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0])),
            (x, y),
        )
        results.append(
            StabilityDetection(
                x=x,
                y=y,
                radius_px=float(np.hypot(x - prep.nucleus_yx[1], y - prep.nucleus_yx[0])),
                position_angle_deg=position_angle(x, y, prep.nucleus_yx),
                survival=len(cluster["runs"]),
                total_runs=len(config.snr_thresholds),
                radius_arcsec=radius_arcsec,
                sky_position_angle_deg=sky_pa,
            )
        )
    return sorted(results, key=lambda item: item.survival, reverse=True), level_map


def sep_consensus(
    prep: PreprocessedImage, config: VisionConfig
) -> list[StabilityDetection]:
    try:
        import sep
    except ImportError as exc:
        raise VisionDependencyError("SEP consensus requires sep") from exc
    sep.set_sub_object_limit(100000)
    _, _, _, _, ring = _ring_geometry(prep, config)
    residual = np.ascontiguousarray(
        np.where(prep.source_mask, 0.0, prep.residual), dtype="=f8"
    )
    error = np.ascontiguousarray(prep.image.rms, dtype="=f8")
    runs: list[list[dict[str, float]]] = []
    for nthresh in config.sep_nthresh:
        for deblend_cont in config.sep_deblend_cont:
            objects = sep.extract(
                residual,
                config.sep_detect_sigma,
                err=error,
                minarea=config.sep_min_area,
                deblend_nthresh=nthresh,
                deblend_cont=deblend_cont,
            )
            detections = []
            for item in objects:
                x, y = float(item["x"]), float(item["y"])
                ix = min(max(int(round(x)), 0), ring.shape[1] - 1)
                iy = min(max(int(round(y)), 0), ring.shape[0] - 1)
                if not ring[iy, ix]:
                    continue
                radius_arcsec, sky_pa = sky_separation_and_pa(
                    prep.image,
                    (float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0])),
                    (x, y),
                )
                axis_sky_pa = sky_axis_pa(
                    prep.image, x, y, float(item["theta"])
                )
                if sky_pa is not None and axis_sky_pa is not None:
                    misalignment = axial_misalignment_deg(
                        axis_sky_pa, (sky_pa + 90.0) % 180.0
                    )
                else:
                    radial_angle = np.degrees(
                        np.arctan2(y - prep.nucleus_yx[0], x - prep.nucleus_yx[1])
                    )
                    misalignment = float(
                        abs((np.degrees(item["theta"]) - (radial_angle + 90) + 90) % 180 - 90)
                    )
                detections.append(
                    {
                        "x": x,
                        "y": y,
                        "axis_ratio": float(item["a"] / max(item["b"], 1e-9)),
                        "misalignment": misalignment,
                        "radius_arcsec": radius_arcsec,
                        "sky_pa": sky_pa,
                    }
                )
            runs.append(detections)
    clusters = _cluster_detections(runs, config.cluster_distance_px)
    total_runs = len(runs)
    results = []
    for cluster in clusters:
        samples = cluster["samples"]
        x, y = float(cluster["x"]), float(cluster["y"])
        radius_arcsec, sky_pa = sky_separation_and_pa(
            prep.image,
            (float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0])),
            (x, y),
        )
        results.append(
            StabilityDetection(
                x=x,
                y=y,
                radius_px=float(np.hypot(x - prep.nucleus_yx[1], y - prep.nucleus_yx[0])),
                position_angle_deg=position_angle(x, y, prep.nucleus_yx),
                survival=len(cluster["runs"]),
                total_runs=total_runs,
                axis_ratio=float(np.median([item["axis_ratio"] for item in samples])),
                tangential_misalignment_deg=float(
                    np.median([item["misalignment"] for item in samples])
                ),
                radius_arcsec=radius_arcsec,
                sky_position_angle_deg=sky_pa,
            )
        )
    return sorted(results, key=lambda item: item.survival, reverse=True)


def extract_residual_structures(
    prep: PreprocessedImage, config: VisionConfig
) -> tuple[list[ResidualStructure], np.ndarray, np.ndarray, np.ndarray]:
    try:
        import sep
        from scipy import ndimage
    except ImportError as exc:
        raise VisionDependencyError("arc extraction requires scipy and sep") from exc
    sep.set_sub_object_limit(100000)
    _, _, radius, pa, ring = _ring_geometry(prep, config)
    residual = np.ascontiguousarray(
        np.where(prep.source_mask, 0.0, prep.residual), dtype="=f8"
    )
    objects, segmentation = sep.extract(
        residual,
        config.sep_detect_sigma,
        err=np.ascontiguousarray(prep.image.rms, dtype="=f8"),
        minarea=config.sep_min_area,
        deblend_nthresh=32,
        deblend_cont=0.005,
        segmentation_map=True,
    )
    snr_mask = prep.residual_snr >= config.structure_snr_low
    footprint = np.zeros(prep.image.data.shape, dtype=bool)
    segments: list[ResidualStructure] = []
    for index, item in enumerate(objects):
        x, y = float(item["x"]), float(item["y"])
        object_radius = float(np.hypot(x - prep.nucleus_yx[1], y - prep.nucleus_yx[0]))
        if not (config.ring_inner_px < object_radius < config.ring_outer_px):
            continue
        label = index + 1
        object_footprint = ndimage.binary_dilation(segmentation == label, iterations=1)
        footprint |= object_footprint
        structure_pixels = object_footprint & snr_mask & ring
        if int(structure_pixels.sum()) < config.snr_min_area:
            continue
        ys, xs = np.where(structure_pixels)
        flux = float(prep.residual[structure_pixels].sum())
        noise = float(np.sqrt(np.square(prep.image.rms[structure_pixels]).sum()))
        radius_arcsec, sky_pa = sky_separation_and_pa(
            prep.image,
            (float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0])),
            (x, y),
        )
        axis_sky_pa = sky_axis_pa(prep.image, x, y, float(item["theta"]))
        if sky_pa is not None and axis_sky_pa is not None:
            misalignment = axial_misalignment_deg(
                axis_sky_pa, (sky_pa + 90.0) % 180.0
            )
        else:
            radial_angle = np.degrees(
                np.arctan2(y - prep.nucleus_yx[0], x - prep.nucleus_yx[1])
            )
            misalignment = float(
                abs((np.degrees(item["theta"]) - (radial_angle + 90) + 90) % 180 - 90)
            )
        inner_connection_region = radius <= (config.ring_inner_px + 1.0)
        inner_connection_fraction = float(
            np.count_nonzero(object_footprint & inner_connection_region)
            / max(np.count_nonzero(object_footprint), 1)
        )
        centroid_x, centroid_y = float(xs.mean()), float(ys.mean())
        centroid_ra, centroid_dec = pixel_to_sky(
            prep.image, centroid_x, centroid_y
        )
        _, centroid_sky_pa = sky_separation_and_pa(
            prep.image,
            (float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0])),
            (centroid_x, centroid_y),
        )
        centroid_radius_arcsec = (
            float(np.median(radius[structure_pixels]) * prep.image.pixel_scale_arcsec)
            if prep.image.pixel_scale_arcsec is not None
            else None
        )
        segments.append(
            ResidualStructure(
                label=label,
                x=centroid_x,
                y=centroid_y,
                pixel_count=int(structure_pixels.sum()),
                flux=flux,
                snr=flux / noise if noise > 0 else 0.0,
                median_radius_px=float(np.median(radius[structure_pixels])),
                position_angle_deg=position_angle(float(xs.mean()), float(ys.mean()), prep.nucleus_yx),
                angular_extent_deg=circular_span(pa[structure_pixels]),
                axis_ratio=float(item["a"] / max(item["b"], 1e-9)),
                tangential_misalignment_deg=misalignment,
                inner_connection_fraction=inner_connection_fraction,
                connects_to_inner_region=inner_connection_fraction > 0.0,
                pixels_y=ys,
                pixels_x=xs,
                centroid_ra_deg=centroid_ra,
                centroid_dec_deg=centroid_dec,
                radius_arcsec=centroid_radius_arcsec,
                sky_position_angle_deg=centroid_sky_pa,
            )
        )
    residual_structure_mask = footprint & snr_mask & ring
    return segments, segmentation, footprint, residual_structure_mask


# Import compatibility for v0.5-era callers. New code must use neutral residual
# structure terminology because detection does not establish lensing identity.
ArcSegment = ResidualStructure
extract_arc_segments = extract_residual_structures
