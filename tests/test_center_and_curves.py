import numpy as np

from expert_agent.core.config import VisionConfig
from expert_agent.vision.preprocess import select_foreground_center
from expert_agent.vision.curved_geometry import measure_curved_geometry


def scene(points):
    data = np.zeros((100, 100))
    segmentation = np.zeros((100, 100), dtype=int)
    objects = np.zeros(len(points), dtype=[("x", float), ("y", float)])
    for i, (x, y, brightness) in enumerate(points):
        objects[i] = (x, y)
        segmentation[y-1:y+2, x-1:x+2] = i + 1
        data[y, x] = brightness
    return data, objects, segmentation


def test_bright_neighbour_cannot_steal_target_center():
    data, objects, segmentation = scene([(50, 50, 1), (70, 20, 1000)])
    center, index, audit = select_foreground_center(data, objects, segmentation, VisionConfig())
    assert center == (50, 50) and index == 0
    assert audit["status"] == "selected" and len(audit["candidates"]) == 2


def test_near_equal_center_candidates_still_choose_exactly_one():
    center, index, audit = select_foreground_center(*scene([(46, 50, 1), (54, 50, 100)]), VisionConfig())
    assert audit["status"] == "selected"
    assert index in {0, 1}
    assert audit["routing_only"] and not audit["affects_grading"]


def test_extended_shortlist_is_capped_and_produces_one_target():
    data, _, segmentation = scene([(50, 50, 2), (75, 35, 6)])
    objects = np.array([(50., 50., 8., 6., 20.), (75., 35., 14., 10., 60.)],
                       dtype=[("x", float), ("y", float), ("a", float), ("b", float), ("flux", float)])
    center, index, audit = select_foreground_center(data, objects, segmentation, VisionConfig())
    assert index in {0, 1}
    assert audit["status"] == "selected"
    assert len(audit["candidates"]) <= 4
    assert audit["selection_basis"] == "raw_morphology_among_extended_shortlist"


def test_remote_morphology_winner_cannot_override_a_valid_central_target(monkeypatch):
    data, _, segmentation = scene([(50, 50, 2), (75, 35, 6)])
    objects = np.array([(50., 50., 8., 6., 20.), (75., 35., 14., 10., 60.)],
                       dtype=[("x", float), ("y", float), ("a", float), ("b", float), ("flux", float)])

    def metrics(_data, _rms, obj, _config):
        if obj["x"] == 50:
            return {"angular_coverage": 0.2, "multiplicity": 0.25, "annular_contrast": 0.1}
        return {"angular_coverage": 1.0, "multiplicity": 1.0, "annular_contrast": 1.0}

    monkeypatch.setattr("expert_agent.vision.preprocess._target_morphology_metrics", metrics)
    center, index, audit = select_foreground_center(
        data, objects, segmentation, VisionConfig(center_blend_radius_px=20)
    )
    assert center == (50, 50) and index == 0
    assert audit["central_prior_gate"]["offcenter_override_accepted"] is False


def test_larger_brighter_but_nonoverlapping_object_cannot_replace_center(monkeypatch):
    data, _, segmentation = scene([(50, 50, 2), (82, 50, 100)])
    objects = np.array(
        [(50.0, 50.0, 8.0, 6.0, 20.0), (82.0, 50.0, 15.0, 10.0, 200.0)],
        dtype=[("x", float), ("y", float), ("a", float), ("b", float), ("flux", float)],
    )

    def metrics(_data, _rms, obj, _config):
        if obj["x"] == 50:
            return {"angular_coverage": 0.2, "multiplicity": 0.25, "annular_contrast": 0.1}
        return {"angular_coverage": 1.0, "multiplicity": 1.0, "annular_contrast": 1.0}

    monkeypatch.setattr("expert_agent.vision.preprocess._target_morphology_metrics", metrics)
    center, index, audit = select_foreground_center(data, objects, segmentation, VisionConfig())
    assert center == (50, 50) and index == 0
    assert audit["central_prior_gate"]["larger"]
    assert audit["central_prior_gate"]["brighter"]
    assert not audit["central_prior_gate"]["light_profiles_overlap"]
    assert not audit["central_prior_gate"]["offcenter_override_accepted"]


def test_explicit_target_prevents_photometric_dominance_switch():
    data, _, segmentation = scene([(50, 50, 2), (75, 35, 6)])
    objects = np.array([(50., 50., 8., 6., 20.), (75., 35., 14., 10., 60.)],
                       dtype=[("x", float), ("y", float), ("a", float), ("b", float), ("flux", float)])
    center, index, audit = select_foreground_center(
        data, objects, segmentation, VisionConfig(target_center_xy=(50, 50))
    )
    assert center == (50, 50) and index == 0
    assert audit["reference_source"] == "explicit"


def test_remote_only_detection_is_still_the_single_selected_object():
    center, index, audit = select_foreground_center(*scene([(75, 50, 10)]), VisionConfig())
    assert center == (50, 75) and index == 0
    assert audit["status"] == "selected"


def test_high_sigma_central_deblend_recovers_target_from_bridged_remote_island():
    yy, xx = np.indices((100, 100))
    data = 20 * np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / 18)
    objects = np.array(
        [(15.0, 50.0, 12.0, 4.0, 100.0)],
        dtype=[("x", float), ("y", float), ("a", float), ("b", float), ("flux", float)],
    )
    segmentation = np.zeros((100, 100), dtype=int)
    segmentation[48:53, 10:55] = 1
    center, index, audit = select_foreground_center(
        data,
        objects,
        segmentation,
        VisionConfig(target_candidate_min_area_px=2),
        rms=np.ones_like(data),
    )
    assert index is None
    assert np.hypot(center[1] - 50, center[0] - 50) <= 2
    assert audit["selection_basis"] == "high_sigma_central_deblend_after_low_threshold_bridge"


def test_explicit_offcenter_target_supported():
    center, index, audit = select_foreground_center(*scene([(50, 50, 100), (70, 20, 1)]), VisionConfig(target_center_xy=(70, 20)))
    assert center == (20, 70) and index == 1
    assert audit["reference_source"] == "explicit"


def test_ring_local_tangents_not_global_axis():
    y, x = np.indices((100, 100))
    r = np.hypot(x - 50, y - 50)
    result = measure_curved_geometry((r >= 19) & (r <= 21), (50, 50))
    assert result["angular_occupied_deg"] >= 300
    assert result["local_tangential_fraction"] > 0.8
    assert result["radial_fractional_width"] < 0.2


def test_filled_disk_not_narrow_ring():
    y, x = np.indices((100, 100))
    result = measure_curved_geometry(np.hypot(x - 50, y - 50) <= 20, (50, 50))
    assert result["radial_fractional_width"] > 0.5


def test_sparse_pixels_do_not_turn_span_into_coverage():
    mask = np.zeros((100, 100), dtype=bool)
    mask[30, 50] = mask[70, 50] = mask[50, 30] = mask[50, 70] = True
    result = measure_curved_geometry(mask, (50, 50))
    assert result["angular_occupied_deg"] == 20
    assert result["local_shape_count"] == 0
