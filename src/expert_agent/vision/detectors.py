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


@dataclass
class RingModelCandidate:
    """Posterior circular/elliptical annulus measurement.

    It is intentionally a geometric measurement only. It contains no
    pre-calibrated acceptance cut and does not identify an Einstein ring or a
    gravitational lens.
    """

    center_x: float
    center_y: float
    semi_major_radius_px: float
    axis_ratio: float
    position_angle_deg: float
    width_px: float
    above_local_background_fraction: float
    longest_above_local_background_deg: float
    median_on_ring_snr: float
    on_minus_off_ring_snr: float
    proposal_sources: list[str]
    on_ring_mask: np.ndarray


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


def posterior_ring_candidates(
    prep: PreprocessedImage,
    config: VisionConfig,
    segments: list[ResidualStructure],
) -> list[RingModelCandidate]:
    """Search posterior circular/elliptical templates after SEP extraction.

    The foreground centre supplies the physically motivated starting point.
    Small circular centre offsets protect against a slightly imperfect light
    centre; elliptical templates are tried at the measured centre.  SEP
    residual structures are used only to annotate template support, never as a
    requirement, because a low-surface-brightness ring may be invisible to SEP.
    """
    candidates: list[RingModelCandidate] = []
    offsets = np.linspace(
        -config.ring_model_center_offset_px,
        config.ring_model_center_offset_px,
        max(1, config.ring_model_center_steps),
    )
    center_x, center_y = float(prep.nucleus_yx[1]), float(prep.nucleus_yx[0])
    yy, xx = np.indices(prep.residual_snr.shape)
    radii = np.arange(
        np.ceil(config.ring_inner_px + config.ring_model_width_px),
        np.floor(config.ring_outer_px - config.ring_model_width_px) + 1,
        1.0,
    )
    geometries: list[tuple[float, float, float, float, str]] = []
    for offset_x in offsets:
        for offset_y in offsets:
            geometries.append((center_x + float(offset_x), center_y + float(offset_y), 1.0, 0.0, "foreground_center_circle_grid"))
    for axis_ratio in config.ring_model_axis_ratios:
        if axis_ratio >= 0.999:
            continue
        for position_angle in config.ring_model_position_angles_deg:
            geometries.append((center_x, center_y, float(axis_ratio), float(position_angle), "foreground_center_ellipse_grid"))

    for candidate_x, candidate_y, axis_ratio, position_angle, source in geometries:
        for radius in radii:
            measured = _annulus_measurement(
                prep.residual_snr,
                prep.source_mask,
                yy,
                xx,
                candidate_x,
                candidate_y,
                float(radius),
                axis_ratio,
                position_angle,
                config.ring_model_width_px,
                config.ring_model_angle_bins,
            )
            if measured is None:
                continue
            fraction, longest, median_on, contrast, on_ring = measured
            proposal_sources = [source]
            if any(
                abs(np.hypot(item.x - candidate_x, item.y - candidate_y) - radius)
                <= 2.0 * config.ring_model_width_px
                for item in segments
            ):
                proposal_sources.append("sep_residual_seed")
            candidates.append(
                RingModelCandidate(
                    center_x=candidate_x,
                    center_y=candidate_y,
                    semi_major_radius_px=float(radius),
                    axis_ratio=axis_ratio,
                    position_angle_deg=position_angle,
                    width_px=config.ring_model_width_px,
                    above_local_background_fraction=fraction,
                    longest_above_local_background_deg=longest,
                    median_on_ring_snr=median_on,
                    on_minus_off_ring_snr=contrast,
                    proposal_sources=proposal_sources,
                    on_ring_mask=on_ring,
                )
            )

    # Keep spatially distinct templates ordered by their descriptive local
    # contrast. This selects a manageable audit display, not accepted rings.
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.on_minus_off_ring_snr,
            item.above_local_background_fraction,
            item.longest_above_local_background_deg,
        ),
        reverse=True,
    )
    selected: list[RingModelCandidate] = []
    for item in ranked:
        duplicate = any(
            np.hypot(item.center_x - old.center_x, item.center_y - old.center_y) < 1.0
            and abs(item.semi_major_radius_px - old.semi_major_radius_px) < max(1.0, item.width_px)
            and abs(item.axis_ratio - old.axis_ratio) < 0.06
            and abs(((item.position_angle_deg - old.position_angle_deg + 90.0) % 180.0) - 90.0) < 15.0
            for old in selected
        )
        if not duplicate:
            selected.append(item)
        if len(selected) >= config.ring_model_max_candidates:
            break
    return selected


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
