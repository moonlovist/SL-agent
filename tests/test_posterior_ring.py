from pathlib import Path

import numpy as np

from expert_agent.core.config import VisionConfig
from expert_agent.vision.detectors import (
    GeometricPathCandidate,
    RecoveredPathSegment,
    ResidualStructure,
    geometric_path_search,
    partition_sep_segments_by_routes,
    path_support_relations,
    sep_consensus,
)
from expert_agent.vision.preprocess import ImageData, PreprocessedImage


def _prep_from_residual_snr(residual_snr: np.ndarray) -> PreprocessedImage:
    data = np.asarray(residual_snr, dtype=float)
    zeros = np.zeros_like(data)
    false_mask = np.zeros_like(data, dtype=bool)
    size = data.shape[0]
    image = ImageData(
        path=Path("synthetic.fits"), object_id="synthetic", input_format="fits",
        band="VIS", data=data, rms=np.ones_like(data), supplied_rms=True,
        finite_fraction=1.0, wcs=None, pixel_scale_arcsec=None, psf=None,
    )
    return PreprocessedImage(
        image=image, model=zeros, residual=data, residual_snr=data,
        source_mask=false_mask, fit_mask=false_mask,
        segmentation=np.zeros_like(data, dtype=np.int32),
        nucleus_yx=(size // 2, size // 2), background_median=0.0,
        background_rms=1.0, residual_median=0.0, residual_robust_rms=1.0,
        model_method="synthetic", model_success=True, foreground_detected=True,
        foreground_peak_value=1.0, foreground_integrated_flux=1.0,
        foreground_ellipticity=0.0, foreground_position_angle_deg=0.0,
        foreground_axis_ratio=1.0, foreground_reliability=1.0,
        multiple_core_count=0, tidal_asymmetry=0.0,
        core_residual_fraction=0.0, flags=[],
    )


def _seed_from_mask(mask: np.ndarray, prep: PreprocessedImage) -> ResidualStructure:
    ys, xs = np.where(mask)
    angles = (
        np.degrees(np.arctan2(-(xs - prep.nucleus_yx[1]), -(ys - prep.nucleus_yx[0])))
        + 360.0
    ) % 360.0
    ordered = np.sort(angles)
    gaps = np.diff(np.concatenate([ordered, ordered[:1] + 360.0]))
    return ResidualStructure(
        label=1, x=float(xs.mean()), y=float(ys.mean()), pixel_count=int(mask.sum()),
        flux=float(prep.residual[mask].sum()), snr=8.0,
        median_radius_px=float(np.median(np.hypot(xs - prep.nucleus_yx[1], ys - prep.nucleus_yx[0]))),
        position_angle_deg=90.0, angular_extent_deg=float(360.0 - gaps.max()),
        axis_ratio=2.0, tangential_misalignment_deg=0.0,
        inner_connection_fraction=0.0, connects_to_inner_region=False,
        pixels_y=ys, pixels_x=xs, centroid_ra_deg=None, centroid_dec_deg=None,
        radius_arcsec=None, sky_position_angle_deg=None,
    )


def test_arc_guided_path_recovers_a_separate_arc_segment() -> None:
    size = 61
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - 30.0, yy - 30.0)
    angle = np.arctan2(yy - 30.0, xx - 30.0)
    radial_band = np.abs(radius - 12.0) <= 1.0
    seed_mask = radial_band & (np.abs(angle) < 0.35)
    second_arc = radial_band & (np.abs(np.abs(angle) - np.pi) < 0.35)
    residual_snr = np.zeros((size, size), dtype=float)
    residual_snr[seed_mask] = 4.0
    residual_snr[second_arc] = 3.0
    prep = _prep_from_residual_snr(residual_snr)
    seed = _seed_from_mask(seed_mask, prep)
    config = VisionConfig(
        ring_inner_px=4.0, ring_outer_px=20.0,
        path_center_offset_px=0.0, path_center_steps=1,
        path_axis_ratios=(1.0,), path_position_angles_deg=(0.0,),
        path_max_candidates=3,
    )

    candidates = geometric_path_search(prep, config, [seed])

    assert candidates
    best = candidates[0]
    assert best.mode == "arc_seeded"
    assert best.seed_indices == [0]
    assert abs(best.semi_major_radius_px - 12.0) <= 1.0
    assert best.recovered_segments
    assert any(item.x < 30.0 for item in best.recovered_segments)
    assert not np.any(best.recovered_mask[seed_mask])


def test_center_fallback_can_recover_ring_pixels_without_a_seed() -> None:
    size = 61
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - 30.0, yy - 30.0)
    ring = np.abs(radius - 12.0) <= 1.0
    residual_snr = np.zeros((size, size), dtype=float)
    residual_snr[ring] = 3.0
    prep = _prep_from_residual_snr(residual_snr)
    config = VisionConfig(
        ring_inner_px=4.0, ring_outer_px=20.0,
        path_center_offset_px=0.0, path_center_steps=1,
        path_axis_ratios=(1.0,), path_position_angles_deg=(0.0,),
        path_max_candidates=1,
    )

    candidates = geometric_path_search(prep, config, [])

    assert candidates
    assert candidates[0].mode == "center_fallback"
    assert not candidates[0].seed_indices
    assert candidates[0].recovered_segments
    assert abs(candidates[0].semi_major_radius_px - 12.0) <= 1.0


def test_sep_consensus_preserves_exact_detected_pixel_union() -> None:
    residual_snr = np.zeros((61, 61), dtype=float)
    residual_snr[29:32, 41:44] = 8.0
    prep = _prep_from_residual_snr(residual_snr)
    config = VisionConfig(
        ring_inner_px=4.0,
        ring_outer_px=20.0,
        sep_nthresh=(16, 32),
        sep_deblend_cont=(0.01, 0.001),
        sep_detect_sigma=2.0,
        sep_min_area=5,
        sep_min_survival=2,
    )

    detections = sep_consensus(prep, config)

    assert detections
    detection = detections[0]
    assert detection.pixels_y is not None
    assert detection.pixels_x is not None
    exact_pixels = set(zip(detection.pixels_y.tolist(), detection.pixels_x.tolist()))
    assert (30, 42) in exact_pixels
    assert (30, 18) not in exact_pixels
    assert detection.survival == detection.total_runs == 4


def test_center_fallback_remains_when_an_unrelated_sep_seed_exists() -> None:
    size = 61
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - 30.0, yy - 30.0)
    ring = np.abs(radius - 12.0) <= 1.0
    residual_snr = np.zeros((size, size), dtype=float)
    residual_snr[ring] = 3.0
    prep = _prep_from_residual_snr(residual_snr)
    unrelated_mask = np.zeros_like(ring)
    unrelated_mask[20:23, 30:33] = True
    seed = _seed_from_mask(unrelated_mask, prep)
    config = VisionConfig(
        ring_inner_px=4.0,
        ring_outer_px=20.0,
        path_center_offset_px=0.0,
        path_center_steps=1,
        path_axis_ratios=(1.0,),
        path_position_angles_deg=(0.0,),
        path_max_candidates=3,
    )

    candidates = geometric_path_search(prep, config, [seed])

    assert any(candidate.mode == "arc_seeded" for candidate in candidates)
    fallback = [candidate for candidate in candidates if candidate.mode == "center_fallback"]
    assert fallback
    assert any(
        abs(candidate.semi_major_radius_px - 12.0) <= 1.0
        and candidate.recovered_segments
        for candidate in fallback
    )


def test_path_recovery_respects_configured_inner_search_boundary() -> None:
    size = 61
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - 30.0, yy - 30.0)
    valid_arc = (np.abs(radius - 8.0) <= 1.0) & (xx < 30)
    central_residual = radius <= 2.5
    residual_snr = np.zeros((size, size), dtype=float)
    residual_snr[valid_arc] = 4.0
    residual_snr[central_residual] = 10.0
    prep = _prep_from_residual_snr(residual_snr)
    config = VisionConfig(
        ring_inner_px=4.0,
        ring_outer_px=20.0,
        path_center_offset_px=0.0,
        path_center_steps=1,
        path_axis_ratios=(1.0,),
        path_position_angles_deg=(0.0,),
        path_max_candidates=1,
    )

    candidates = geometric_path_search(prep, config, [])

    assert candidates
    recovered_pixels = [
        (int(y), int(x))
        for candidate in candidates
        for segment in candidate.recovered_segments
        for y, x in zip(segment.pixels_y, segment.pixels_x)
    ]
    assert recovered_pixels
    assert all(np.hypot(x - 30.0, y - 30.0) >= 4.0 for y, x in recovered_pixels)


def test_seeded_route_needs_independent_component_before_supporting_its_seed() -> None:
    size = 61
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - 30.0, yy - 30.0)
    angle = np.arctan2(yy - 30.0, xx - 30.0)
    seed_mask = (np.abs(radius - 12.0) <= 1.0) & (np.abs(angle) < 0.3)
    recovered_mask = (np.abs(radius - 12.0) <= 1.0) & (
        np.abs(np.abs(angle) - np.pi) < 0.3
    )
    prep = _prep_from_residual_snr(np.zeros((size, size), dtype=float))
    seed = _seed_from_mask(seed_mask, prep)
    recovered_y, recovered_x = np.where(recovered_mask)

    def route(recovered: list[RecoveredPathSegment]) -> GeometricPathCandidate:
        return GeometricPathCandidate(
            mode="arc_seeded",
            seed_indices=[0],
            linked_structure_indices=[0],
            center_x=30.0,
            center_y=30.0,
            semi_major_radius_px=12.0,
            axis_ratio=1.0,
            position_angle_deg=0.0,
            half_width_px=1.5,
            fit_scatter_px=0.5,
            median_on_path_snr=1.0,
            on_minus_off_path_snr=0.5,
            above_local_background_fraction=0.5,
            longest_above_local_background_deg=30.0,
            recovered_segments=recovered,
            route_mask=np.abs(radius - 12.0) <= 1.5,
            recovered_mask=recovered_mask if recovered else np.zeros_like(recovered_mask),
        )

    _, unsupported = path_support_relations([route([])], [seed])
    assert unsupported[0] == []

    recovered_component = RecoveredPathSegment(
        x=float(recovered_x.mean()),
        y=float(recovered_y.mean()),
        pixel_count=int(recovered_mask.sum()),
        integrated_snr=5.0,
        angular_extent_deg=30.0,
        pixels_y=recovered_y,
        pixels_x=recovered_x,
    )
    _, supported = path_support_relations(
        [route([recovered_component])], [seed]
    )
    assert supported[0] == [0]


def test_center_fallback_can_supply_geometry_support_to_existing_sep_component() -> None:
    size = 61
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - 30.0, yy - 30.0)
    seed_mask = (np.abs(radius - 12.0) <= 1.0) & (xx > 30)
    prep = _prep_from_residual_snr(np.zeros((size, size), dtype=float))
    seed = _seed_from_mask(seed_mask, prep)
    route = GeometricPathCandidate(
        mode="center_fallback",
        seed_indices=[],
        linked_structure_indices=[0],
        center_x=30.0,
        center_y=30.0,
        semi_major_radius_px=12.0,
        axis_ratio=1.0,
        position_angle_deg=0.0,
        half_width_px=1.5,
        fit_scatter_px=None,
        median_on_path_snr=1.0,
        on_minus_off_path_snr=0.5,
        above_local_background_fraction=0.5,
        longest_above_local_background_deg=30.0,
        recovered_segments=[],
        route_mask=np.abs(radius - 12.0) <= 1.5,
        recovered_mask=np.zeros_like(seed_mask),
    )

    _, supported = path_support_relations([route], [seed])

    assert supported[0] == [0]


def test_local_partition_separates_route_pixels_from_attached_sep_contamination() -> None:
    size = 61
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - 30.0, yy - 30.0)
    merged_mask = np.zeros((size, size), dtype=bool)
    merged_mask[29:32, 39:51] = True
    prep = _prep_from_residual_snr(np.zeros((size, size), dtype=float))
    seed = _seed_from_mask(merged_mask, prep)
    route_mask = np.abs(radius - 10.0) <= 1.5
    route = GeometricPathCandidate(
        mode="center_fallback",
        seed_indices=[],
        linked_structure_indices=[0],
        center_x=30.0,
        center_y=30.0,
        semi_major_radius_px=10.0,
        axis_ratio=1.0,
        position_angle_deg=0.0,
        half_width_px=1.5,
        fit_scatter_px=None,
        median_on_path_snr=1.0,
        on_minus_off_path_snr=0.5,
        above_local_background_fraction=0.5,
        longest_above_local_background_deg=30.0,
        recovered_segments=[],
        route_mask=route_mask,
        recovered_mask=np.zeros_like(route_mask),
    )

    partitions, supported = partition_sep_segments_by_routes(
        [route], [seed], min_supported_area=3
    )

    supported_mask = np.zeros_like(merged_mask)
    off_route_mask = np.zeros_like(merged_mask)
    for partition in partitions:
        target = (
            supported_mask
            if partition.status == "sep_geometry_consistent"
            else off_route_mask
        )
        target[partition.pixels_y, partition.pixels_x] = True
    assert supported[0] == [0]
    assert np.any(supported_mask)
    assert np.any(off_route_mask)
    assert not np.any(supported_mask & off_route_mask)
    assert np.array_equal(supported_mask | off_route_mask, merged_mask)
    assert np.all(route_mask[supported_mask])
    assert not np.any(route_mask[off_route_mask])


def test_local_partition_does_not_let_arc_seed_route_certify_itself() -> None:
    size = 61
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - 30.0, yy - 30.0)
    seed_mask = (np.abs(radius - 12.0) <= 1.0) & (xx > 30)
    prep = _prep_from_residual_snr(np.zeros((size, size), dtype=float))
    seed = _seed_from_mask(seed_mask, prep)
    route = GeometricPathCandidate(
        mode="arc_seeded",
        seed_indices=[0],
        linked_structure_indices=[0],
        center_x=30.0,
        center_y=30.0,
        semi_major_radius_px=12.0,
        axis_ratio=1.0,
        position_angle_deg=0.0,
        half_width_px=1.5,
        fit_scatter_px=0.5,
        median_on_path_snr=1.0,
        on_minus_off_path_snr=0.5,
        above_local_background_fraction=0.5,
        longest_above_local_background_deg=30.0,
        recovered_segments=[],
        route_mask=np.abs(radius - 12.0) <= 1.5,
        recovered_mask=np.zeros_like(seed_mask),
    )

    partitions, supported = partition_sep_segments_by_routes(
        [route], [seed], min_supported_area=3
    )

    assert supported[0] == []
    assert {partition.status for partition in partitions} == {"sep_only"}
    assert sum(partition.pixels_y.size for partition in partitions) == int(seed_mask.sum())
