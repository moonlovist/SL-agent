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
    pixels_y: np.ndarray | None = None
    pixels_x: np.ndarray | None = None


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


@dataclass
class RecoveredPathSegment:
    """Real residual pixels found inside a proposed geometric search band."""

    x: float
    y: float
    pixel_count: int
    integrated_snr: float
    angular_extent_deg: float
    pixels_y: np.ndarray
    pixels_x: np.ndarray


@dataclass
class GeometricPathCandidate:
    """A search route proposed from an arc seed or the foreground centre.

    The ellipse itself is not an observed structure. Only recovered_segments
    contain newly measured image pixels.
    """

    mode: str
    seed_indices: list[int]
    linked_structure_indices: list[int]
    center_x: float
    center_y: float
    semi_major_radius_px: float
    axis_ratio: float
    position_angle_deg: float
    half_width_px: float
    fit_scatter_px: float | None
    median_on_path_snr: float
    on_minus_off_path_snr: float
    above_local_background_fraction: float
    longest_above_local_background_deg: float
    recovered_segments: list[RecoveredPathSegment]
    route_mask: np.ndarray
    recovered_mask: np.ndarray


def path_pixel_consistency(
    candidate: GeometricPathCandidate,
    pixels_y: np.ndarray,
    pixels_x: np.ndarray,
) -> tuple[float, float, bool]:
    """Measure whether an observed pixel component follows a search route.

    Consistency uses the route's already recorded half-width: the component is
    consistent when its median radial deviation is inside that band. The
    overlap fraction remains a descriptive continuous measurement rather than
    a separate calibrated pass percentage.
    """
    if not pixels_y.size:
        return 0.0, float("inf"), False
    elliptical_radius = _ellipse_radius(
        pixels_y.astype(float),
        pixels_x.astype(float),
        candidate.center_x,
        candidate.center_y,
        candidate.axis_ratio,
        candidate.position_angle_deg,
    )
    deviation = np.abs(elliptical_radius - candidate.semi_major_radius_px)
    median_deviation = float(np.median(deviation))
    overlap_fraction = float(np.mean(deviation <= candidate.half_width_px))
    return (
        overlap_fraction,
        median_deviation,
        median_deviation <= candidate.half_width_px,
    )


def path_support_relations(
    candidates: list[GeometricPathCandidate],
    segments: list[ResidualStructure],
) -> tuple[
    dict[int, dict[int, tuple[float, float, bool]]],
    dict[int, list[int]],
]:
    """Relate raw SEP components to routes without letting a seed prove itself.

    A route supplies additional structural support only when it is brighter
    than its adjacent control route. An arc-seeded route additionally needs an
    independent component: either a route-masked SEP recovery or another raw
    SEP component consistent with the same route.
    """
    metrics_by_segment: dict[int, dict[int, tuple[float, float, bool]]] = {
        index: {} for index in range(len(segments))
    }
    consistent_by_path: dict[int, list[int]] = {}
    for path_index, candidate in enumerate(candidates):
        consistent_indices: list[int] = []
        for segment_index, segment in enumerate(segments):
            metrics = path_pixel_consistency(
                candidate, segment.pixels_y, segment.pixels_x
            )
            metrics_by_segment[segment_index][path_index] = metrics
            if metrics[2]:
                consistent_indices.append(segment_index)
        consistent_by_path[path_index] = consistent_indices

    supported_by_path: dict[int, list[int]] = {}
    for path_index, candidate in enumerate(candidates):
        consistent_indices = consistent_by_path[path_index]
        independent_component = bool(candidate.recovered_segments) or any(
            index not in candidate.seed_indices for index in consistent_indices
        )
        supported: list[int] = []
        if candidate.on_minus_off_path_snr > 0.0:
            for segment_index in consistent_indices:
                self_seeded = (
                    candidate.mode == "arc_seeded"
                    and segment_index in candidate.seed_indices
                )
                if self_seeded and not independent_component:
                    continue
                supported.append(segment_index)
        supported_by_path[path_index] = supported
    return metrics_by_segment, supported_by_path


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


def _longest_circular_run(values: np.ndarray) -> int:
    """Length of the longest True run in a circular boolean array."""
    if not values.size or not np.any(values):
        return 0
    if np.all(values):
        return int(values.size)
    doubled = np.concatenate([values, values])
    best = current = 0
    for value in doubled:
        current = current + 1 if value else 0
        best = max(best, current)
    return int(min(best, values.size))


def _annulus_measurement(
    residual_snr: np.ndarray,
    source_mask: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    center_x: float,
    center_y: float,
    semi_major_radius_px: float,
    axis_ratio: float,
    position_angle_deg: float,
    width_px: float,
    angle_bins: int,
) -> tuple[float, float, float, float, np.ndarray] | None:
    """Measure weak positive residuals along one circular/elliptical band."""
    angle = np.deg2rad(position_angle_deg)
    dx = xx - center_x
    dy = yy - center_y
    # Image y grows downward, which only changes the PA convention, not the
    # annulus itself.  The convention is recorded as pixel geometry.
    x_rot = dx * np.cos(angle) + dy * np.sin(angle)
    y_rot = -dx * np.sin(angle) + dy * np.cos(angle)
    elliptical_radius = np.hypot(x_rot, y_rot / max(axis_ratio, 1e-6))
    elliptical_angle = (np.arctan2(y_rot / max(axis_ratio, 1e-6), x_rot) + 2 * np.pi) % (2 * np.pi)
    valid = np.isfinite(residual_snr) & ~source_mask
    on_ring = valid & (np.abs(elliptical_radius - semi_major_radius_px) <= width_px)
    if int(on_ring.sum()) < max(12, angle_bins // 2):
        return None
    bin_index = np.minimum(
        (elliptical_angle / (2 * np.pi) * angle_bins).astype(int), angle_bins - 1
    )
    # A per-bin median would repeatedly materialize 72 full-size boolean
    # images for every template. A bin mean is adequate for a descriptive
    # template measurement and keeps the 300x300 Euclid scan practical.
    selected_bins = bin_index[on_ring]
    selected_snr = residual_snr[on_ring]
    counts = np.bincount(selected_bins, minlength=angle_bins)
    sums = np.bincount(selected_bins, weights=selected_snr, minlength=angle_bins)
    bin_values = np.divide(
        sums, counts, out=np.full(angle_bins, np.nan), where=counts > 0
    )
    available = np.isfinite(bin_values)
    if int(available.sum()) < max(8, angle_bins // 3):
        return None
    median_on = float(np.median(residual_snr[on_ring]))
    # Adjacent bands establish whether the candidate is locally ring-enhanced,
    # rather than a smooth foreground-model residual at all radii.
    gap = 2.5 * width_px
    off_ring = valid & (
        (np.abs(elliptical_radius - (semi_major_radius_px - gap)) <= width_px)
        | (np.abs(elliptical_radius - (semi_major_radius_px + gap)) <= width_px)
    )
    median_off = float(np.median(residual_snr[off_ring])) if np.any(off_ring) else 0.0
    # This is a descriptive comparison with the *measured* adjacent annuli,
    # not a fixed S/N threshold. It is never used as an acceptance decision.
    above_local_background = available & (bin_values > median_off)
    fraction = float(np.count_nonzero(above_local_background) / np.count_nonzero(available))
    longest = float(_longest_circular_run(above_local_background) * 360.0 / angle_bins)
    return fraction, longest, median_on, median_on - median_off, on_ring


def _psf_half_width_px(prep: PreprocessedImage) -> float:
    if prep.image.psf is None:
        return 0.0
    psf = np.asarray(prep.image.psf, dtype=float)
    weights = np.clip(psf - np.nanmin(psf), 0.0, None)
    total = float(np.nansum(weights))
    if not np.isfinite(total) or total <= 0:
        return 0.0
    yy, xx = np.indices(psf.shape)
    cx = float(np.nansum(xx * weights) / total)
    cy = float(np.nansum(yy * weights) / total)
    second_moment = float(np.nansum(((xx - cx) ** 2 + (yy - cy) ** 2) * weights) / total)
    return float(np.sqrt(max(second_moment, 0.0) / 2.0))


def _ellipse_radius(
    yy: np.ndarray,
    xx: np.ndarray,
    center_x: float,
    center_y: float,
    axis_ratio: float,
    position_angle_deg: float,
) -> np.ndarray:
    angle = np.deg2rad(position_angle_deg)
    dx = xx - center_x
    dy = yy - center_y
    x_rot = dx * np.cos(angle) + dy * np.sin(angle)
    y_rot = -dx * np.sin(angle) + dy * np.cos(angle)
    return np.hypot(x_rot, y_rot / max(axis_ratio, 1e-6))


def _recover_segments_on_path(
    prep: PreprocessedImage,
    config: VisionConfig,
    route_mask: np.ndarray,
    known_mask: np.ndarray,
) -> tuple[list[RecoveredPathSegment], np.ndarray]:
    """Run SEP inside a geometric band and return only newly found pixels."""
    try:
        import sep
        from scipy import ndimage
    except ImportError as exc:
        raise VisionDependencyError("geometry-guided SEP search requires scipy and sep") from exc
    exclusion = ndimage.binary_dilation(known_mask, iterations=1)
    yy, xx = np.indices(route_mask.shape)
    center_radius = np.hypot(
        xx - float(prep.nucleus_yx[1]), yy - float(prep.nucleus_yx[0])
    )
    configured_search_region = (
        (center_radius >= config.ring_inner_px)
        & (center_radius <= config.ring_outer_px)
    )
    search_mask = (
        route_mask
        & configured_search_region
        & ~exclusion
        & ~prep.source_mask
    )
    sep_mask = ~search_mask
    residual = np.ascontiguousarray(prep.residual, dtype="=f8")
    error = np.ascontiguousarray(prep.image.rms, dtype="=f8")
    objects, segmentation = sep.extract(
        residual,
        config.sep_detect_sigma,
        err=error,
        mask=np.ascontiguousarray(sep_mask),
        minarea=config.sep_min_area,
        deblend_nthresh=32,
        deblend_cont=0.005,
        segmentation_map=True,
    )
    recovered: list[RecoveredPathSegment] = []
    recovered_mask = np.zeros_like(route_mask, dtype=bool)
    for index, _ in enumerate(objects, start=1):
        mask = (
            (segmentation == index)
            & search_mask
            & np.isfinite(prep.residual_snr)
            & (prep.residual_snr >= config.structure_snr_low)
        )
        if int(mask.sum()) < config.snr_min_area:
            continue
        ys, xs = np.where(mask)
        angles = (
            np.degrees(np.arctan2(-(xs - prep.nucleus_yx[1]), -(ys - prep.nucleus_yx[0])))
            + 360.0
        ) % 360.0
        recovered.append(
            RecoveredPathSegment(
                x=float(xs.mean()),
                y=float(ys.mean()),
                pixel_count=int(mask.sum()),
                integrated_snr=float(prep.residual_snr[mask].sum() / np.sqrt(mask.sum())),
                angular_extent_deg=circular_span(angles),
                pixels_y=ys,
                pixels_x=xs,
            )
        )
        recovered_mask |= mask
    return recovered, recovered_mask


def geometric_path_search(
    prep: PreprocessedImage,
    config: VisionConfig,
    segments: list[ResidualStructure],
) -> list[GeometricPathCandidate]:
    """Use detected structures to propose paths and recover missed pixels.

    If no seed exists, the same module uses centre-constrained fallback paths.
    A path is a region to inspect, not evidence that a complete ring exists.
    """
    candidates: list[GeometricPathCandidate] = []
    offsets = np.linspace(
        -config.path_center_offset_px,
        config.path_center_offset_px,
        max(1, config.path_center_steps),
    )
    center_x, center_y = float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0])
    yy, xx = np.indices(prep.residual_snr.shape)
    known_mask = np.zeros_like(prep.source_mask, dtype=bool)
    for segment in segments:
        known_mask[segment.pixels_y, segment.pixels_x] = True
    geometries: list[tuple[float, float, float, float]] = []
    for offset_x in offsets:
        for offset_y in offsets:
            geometries.append(
                (center_x + float(offset_x), center_y + float(offset_y), 1.0, 0.0)
            )
    # Elliptical paths retain the measured foreground centre in this first
    # implementation. Combining every centre offset with every q/PA would add
    # many poorly constrained paths from a short seed and inflate runtime.
    for axis_ratio in config.path_axis_ratios:
        if axis_ratio >= 0.999:
            continue
        for position_angle in config.path_position_angles_deg:
            geometries.append(
                (center_x, center_y, float(axis_ratio), float(position_angle))
            )

    psf_width = _psf_half_width_px(prep)
    seeded_rows: list[tuple[float, int, float, float, float, float, float]] = []
    if segments:
        proposals: list[tuple[float, int, float, float, float, float, float]] = []
        for seed_index, segment in enumerate(segments):
            seed_y, seed_x = segment.pixels_y, segment.pixels_x
            for candidate_x, candidate_y, axis_ratio, position_angle in geometries:
                seed_radius = _ellipse_radius(
                    seed_y.astype(float), seed_x.astype(float), candidate_x, candidate_y,
                    axis_ratio, position_angle,
                )
                radius = float(np.median(seed_radius))
                if not (config.ring_inner_px < radius < config.ring_outer_px):
                    continue
                fit_error = float(
                    np.median(np.abs(seed_radius - radius))
                    / max(config.path_min_half_width_px, psf_width, 1e-6)
                )
                proposals.append((fit_error, seed_index, candidate_x, candidate_y, axis_ratio, position_angle, radius))
        proposals.sort(key=lambda item: item[0])
        used_seeds: set[int] = set()
        seeded_limit = max(1, config.path_max_candidates // 2)
        for proposal in proposals:
            if proposal[1] in used_seeds:
                continue
            seeded_rows.append(proposal)
            used_seeds.add(proposal[1])
            if len(seeded_rows) >= seeded_limit:
                break

    # Centre-based fallback proposals are always retained alongside seeded
    # routes. A stable but unrelated SEP island must not prevent recovery of a
    # low-surface-brightness ring that SEP missed globally.
    radii = np.arange(
        np.ceil(config.ring_inner_px + config.path_min_half_width_px),
        np.floor(config.ring_outer_px - config.path_min_half_width_px) + 1,
        1.0,
    )
    fallback: list[tuple[float, tuple[float, int, float, float, float, float, float]]] = []
    for candidate_x, candidate_y, axis_ratio, position_angle in geometries:
        for radius in radii:
            measured = _annulus_measurement(
                prep.residual_snr, prep.source_mask, yy, xx, candidate_x, candidate_y,
                float(radius), axis_ratio, position_angle,
                max(config.path_min_half_width_px, psf_width), config.path_angle_bins,
            )
            if measured is None:
                continue
            _, _, _, contrast, _ = measured
            row = (0.0, -1, candidate_x, candidate_y, axis_ratio, position_angle, float(radius))
            fallback.append((contrast, row))
    fallback.sort(key=lambda item: item[0], reverse=True)
    distinct_fallback: list[tuple[float, int, float, float, float, float, float]] = []
    fallback_limit = max(0, config.path_max_candidates - len(seeded_rows))
    for _, row in fallback:
        if fallback_limit <= 0:
            break
        _, _, row_x, row_y, row_q, row_pa, row_radius = row
        comparison_rows = seeded_rows + distinct_fallback
        duplicate = any(
            np.hypot(row_x - old[2], row_y - old[3]) < 1.0
            and abs(row_radius - old[6]) < max(1.0, config.path_min_half_width_px)
            and abs(row_q - old[4]) < 0.06
            and abs(((row_pa - old[5] + 90.0) % 180.0) - 90.0) < 15.0
            for old in comparison_rows
        )
        if not duplicate:
            distinct_fallback.append(row)
        if len(distinct_fallback) >= fallback_limit:
            break
    proposal_rows = [
        *[("arc_seeded", row) for row in seeded_rows],
        *[("center_fallback", row) for row in distinct_fallback],
    ]

    for mode, row in proposal_rows:
        fit_error, seed_index, candidate_x, candidate_y, axis_ratio, position_angle, radius = row
        if seed_index >= 0:
            seed_radii = _ellipse_radius(
                segments[seed_index].pixels_y.astype(float),
                segments[seed_index].pixels_x.astype(float),
                candidate_x, candidate_y, axis_ratio, position_angle,
            )
            scatter = float(1.4826 * np.median(np.abs(seed_radii - radius)))
        else:
            scatter = None
        width = max(config.path_min_half_width_px, psf_width, scatter or 0.0)
        measured = _annulus_measurement(
            prep.residual_snr, prep.source_mask, yy, xx, candidate_x, candidate_y,
            float(radius), axis_ratio, position_angle, width, config.path_angle_bins,
        )
        if measured is None:
            continue
        fraction, longest, median_on, contrast, route_mask = measured
        recovered, recovered_mask = _recover_segments_on_path(
            prep, config, route_mask, known_mask
        )
        linked = [
            index for index, segment in enumerate(segments)
            if np.any(route_mask[segment.pixels_y, segment.pixels_x])
        ]
        candidates.append(
            GeometricPathCandidate(
                mode=mode,
                seed_indices=[] if seed_index < 0 else [seed_index],
                linked_structure_indices=linked,
                center_x=candidate_x,
                center_y=candidate_y,
                semi_major_radius_px=float(radius),
                axis_ratio=axis_ratio,
                position_angle_deg=position_angle,
                half_width_px=width,
                fit_scatter_px=scatter,
                median_on_path_snr=median_on,
                on_minus_off_path_snr=contrast,
                above_local_background_fraction=fraction,
                longest_above_local_background_deg=longest,
                recovered_segments=recovered,
                route_mask=route_mask,
                recovered_mask=recovered_mask,
            )
        )
        # A real pixel can be newly recovered only once. Later overlapping
        # routes may still be recorded, but cannot duplicate earlier segments.
        known_mask |= recovered_mask
    return candidates


# Compatibility aliases for v0.8 callers. New code uses geometric path terms.
RingModelCandidate = GeometricPathCandidate


def posterior_ring_candidates(
    prep: PreprocessedImage,
    config: VisionConfig,
    segments: list[ResidualStructure],
) -> list[GeometricPathCandidate]:
    return geometric_path_search(prep, config, segments)


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
        from scipy import ndimage
    except ImportError as exc:
        raise VisionDependencyError("SEP consensus requires scipy and sep") from exc
    sep.set_sub_object_limit(100000)
    _, _, _, _, ring = _ring_geometry(prep, config)
    residual = np.ascontiguousarray(
        np.where(prep.source_mask, 0.0, prep.residual), dtype="=f8"
    )
    error = np.ascontiguousarray(prep.image.rms, dtype="=f8")
    clusters: list[dict[str, object]] = []
    run_index = 0
    for nthresh in config.sep_nthresh:
        for deblend_cont in config.sep_deblend_cont:
            objects, segmentation = sep.extract(
                residual,
                config.sep_detect_sigma,
                err=error,
                minarea=config.sep_min_area,
                deblend_nthresh=nthresh,
                deblend_cont=deblend_cont,
                segmentation_map=True,
            )
            for object_index, item in enumerate(objects, start=1):
                x, y = float(item["x"]), float(item["y"])
                ix = min(max(int(round(x)), 0), ring.shape[1] - 1)
                iy = min(max(int(round(y)), 0), ring.shape[0] - 1)
                if not ring[iy, ix]:
                    continue
                object_mask = (
                    (segmentation == object_index)
                    & ring
                    & ~prep.source_mask
                )
                if not np.any(object_mask):
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
                sample = {
                    "x": x,
                    "y": y,
                    "axis_ratio": float(item["a"] / max(item["b"], 1e-9)),
                    "misalignment": misalignment,
                }
                eligible: list[tuple[int, dict[str, object]]] = []
                for cluster in clusters:
                    cluster_mask = np.asarray(cluster["mask"], dtype=bool)
                    overlap = int(
                        np.count_nonzero(
                            ndimage.binary_dilation(cluster_mask, iterations=1)
                            & object_mask
                        )
                    )
                    if overlap > 0:
                        eligible.append((overlap, cluster))
                if eligible:
                    cluster = max(eligible, key=lambda value: value[0])[1]
                    cluster["mask"] = np.asarray(cluster["mask"], dtype=bool) | object_mask
                    cluster["runs"].add(run_index)
                    cluster["samples"].append(sample)
                else:
                    clusters.append(
                        {
                            "mask": object_mask.copy(),
                            "runs": {run_index},
                            "samples": [sample],
                        }
                    )
            run_index += 1
    total_runs = run_index
    results = []
    for cluster in clusters:
        samples = cluster["samples"]
        cluster_mask = np.asarray(cluster["mask"], dtype=bool)
        pixels_y, pixels_x = np.where(cluster_mask)
        x = float(np.mean([item["x"] for item in samples]))
        y = float(np.mean([item["y"] for item in samples]))
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
                pixels_y=pixels_y,
                pixels_x=pixels_x,
            )
        )
    return sorted(results, key=lambda item: item.survival, reverse=True)


def consensus_residual_structures(
    prep: PreprocessedImage,
    config: VisionConfig,
    detections: list[StabilityDetection],
) -> tuple[list[ResidualStructure], np.ndarray, np.ndarray, np.ndarray]:
    """Create internal geometry seeds from exact multi-run SEP pixel unions.

    These seeds are not formal Observation structures. They must pass through
    the geometry/model chain before any downstream exposure.
    """
    _, _, radius, _, ring = _ring_geometry(prep, config)
    segmentation = np.zeros(prep.image.data.shape, dtype=np.int32)
    footprint = np.zeros(prep.image.data.shape, dtype=bool)
    seed_mask = np.zeros(prep.image.data.shape, dtype=bool)
    segments: list[ResidualStructure] = []
    for raw_index, detection in enumerate(detections, start=1):
        if detection.pixels_y is None or detection.pixels_x is None:
            continue
        raw_mask = np.zeros(prep.image.data.shape, dtype=bool)
        raw_mask[detection.pixels_y, detection.pixels_x] = True
        footprint |= raw_mask
        segmentation[raw_mask] = raw_index
        if detection.survival < config.sep_min_survival:
            continue
        structure_pixels = (
            raw_mask
            & ring
            & ~prep.source_mask
            & np.isfinite(prep.residual_snr)
            & (prep.residual_snr >= config.structure_snr_low)
        )
        if int(structure_pixels.sum()) < config.snr_min_area:
            continue
        ys, xs = np.where(structure_pixels)
        flux = float(prep.residual[structure_pixels].sum())
        noise = float(np.sqrt(np.square(prep.image.rms[structure_pixels]).sum()))
        centroid_x, centroid_y = float(xs.mean()), float(ys.mean())
        centroid_ra, centroid_dec = pixel_to_sky(
            prep.image, centroid_x, centroid_y
        )
        radius_arcsec, centroid_sky_pa = sky_separation_and_pa(
            prep.image,
            (float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0])),
            (centroid_x, centroid_y),
        )
        angles = (
            np.degrees(
                np.arctan2(
                    -(xs - prep.nucleus_yx[1]),
                    -(ys - prep.nucleus_yx[0]),
                )
            )
            + 360.0
        ) % 360.0
        inner_connection_region = radius <= (config.ring_inner_px + 1.0)
        inner_connection_fraction = float(
            np.count_nonzero(raw_mask & inner_connection_region)
            / max(np.count_nonzero(raw_mask), 1)
        )
        seed_mask |= structure_pixels
        segments.append(
            ResidualStructure(
                label=raw_index,
                x=centroid_x,
                y=centroid_y,
                pixel_count=int(structure_pixels.sum()),
                flux=flux,
                snr=flux / noise if noise > 0 else 0.0,
                median_radius_px=float(np.median(radius[structure_pixels])),
                position_angle_deg=position_angle(
                    centroid_x, centroid_y, prep.nucleus_yx
                ),
                angular_extent_deg=circular_span(angles),
                axis_ratio=float(detection.axis_ratio or 1.0),
                tangential_misalignment_deg=float(
                    detection.tangential_misalignment_deg or 0.0
                ),
                inner_connection_fraction=inner_connection_fraction,
                connects_to_inner_region=inner_connection_fraction > 0.0,
                pixels_y=ys,
                pixels_x=xs,
                centroid_ra_deg=centroid_ra,
                centroid_dec_deg=centroid_dec,
                radius_arcsec=radius_arcsec,
                sky_position_angle_deg=centroid_sky_pa,
            )
        )
    return segments, segmentation, footprint, seed_mask


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
