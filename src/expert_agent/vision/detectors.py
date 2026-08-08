from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.config import VisionConfig
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


@dataclass
class ArcSegment:
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


def threshold_stability(
    prep: PreprocessedImage, config: VisionConfig
) -> tuple[list[StabilityDetection], np.ndarray]:
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise VisionDependencyError("threshold detection requires scipy") from exc
    _, _, radius, _, ring = _ring_geometry(prep, config)
    runs: list[list[dict[str, float]]] = []
    level_map = np.zeros(prep.image.data.shape, dtype=np.uint8)
    for threshold in config.snr_thresholds:
        active = (
            (prep.residual_snr >= threshold)
            & ring
            & ~prep.source_mask
        )
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
        results.append(
            StabilityDetection(
                x=x,
                y=y,
                radius_px=float(np.hypot(x - prep.nucleus_yx[1], y - prep.nucleus_yx[0])),
                position_angle_deg=position_angle(x, y, prep.nucleus_yx),
                survival=len(cluster["runs"]),
                total_runs=len(config.snr_thresholds),
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
                    }
                )
            runs.append(detections)
    clusters = _cluster_detections(runs, config.cluster_distance_px)
    total_runs = len(runs)
    results = []
    for cluster in clusters:
        samples = cluster["samples"]
        x, y = float(cluster["x"]), float(cluster["y"])
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
            )
        )
    return sorted(results, key=lambda item: item.survival, reverse=True)


def extract_arc_segments(
    prep: PreprocessedImage, config: VisionConfig
) -> tuple[list[ArcSegment], np.ndarray, np.ndarray, np.ndarray]:
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
    snr_mask = prep.residual_snr >= config.arc_snr_low
    footprint = np.zeros(prep.image.data.shape, dtype=bool)
    segments: list[ArcSegment] = []
    for index, item in enumerate(objects):
        x, y = float(item["x"]), float(item["y"])
        object_radius = float(np.hypot(x - prep.nucleus_yx[1], y - prep.nucleus_yx[0]))
        if not (config.ring_inner_px < object_radius < config.ring_outer_px):
            continue
        label = index + 1
        object_footprint = ndimage.binary_dilation(segmentation == label, iterations=1)
        footprint |= object_footprint
        arc_pixels = object_footprint & snr_mask & ring
        if int(arc_pixels.sum()) < config.snr_min_area:
            continue
        ys, xs = np.where(arc_pixels)
        flux = float(prep.residual[arc_pixels].sum())
        noise = float(np.sqrt(np.square(prep.image.rms[arc_pixels]).sum()))
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
        segments.append(
            ArcSegment(
                label=label,
                x=float(xs.mean()),
                y=float(ys.mean()),
                pixel_count=int(arc_pixels.sum()),
                flux=flux,
                snr=flux / noise if noise > 0 else 0.0,
                median_radius_px=float(np.median(radius[arc_pixels])),
                position_angle_deg=position_angle(float(xs.mean()), float(ys.mean()), prep.nucleus_yx),
                angular_extent_deg=circular_span(pa[arc_pixels]),
                axis_ratio=float(item["a"] / max(item["b"], 1e-9)),
                tangential_misalignment_deg=misalignment,
                inner_connection_fraction=inner_connection_fraction,
                connects_to_inner_region=inner_connection_fraction > 0.0,
                pixels_y=ys,
                pixels_x=xs,
            )
        )
    arc_mask = footprint & snr_mask & ring
    return segments, segmentation, footprint, arc_mask
