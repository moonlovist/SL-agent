"""The trial preserves pixels and uncertainty, never promotes by test label."""
import copy

import numpy as np
import pytest

from expert_agent.agents.decision_audit import DecisionAudit
from expert_agent.cli import build_parser
from expert_agent.core.config import VisionConfig
from expert_agent.schemas.observation import (
    CandidateStructureAuditObservation, GeometricPathObservation,
    ModelCandidateObservation, ObservationDocument,
)
from expert_agent.vision.candidate_handoff import attach_candidate_review
from tests.test_reasoning import make_observation, run


def scene(*, comparison_value=0.8, origin="path_recovery", has_path=True, survival=0):
    doc = make_observation()
    mask = np.zeros(doc.image_shape, dtype=bool)
    mask[142:159, 162:164] = True  # Measured pixels, not a filled route template.
    candidate = ModelCandidateObservation(
        id="C1", origin=origin,
        status="geometry_recovered" if origin == "path_recovery" else "sep_only",
        geometry_consistent_path_ids=["P1"] if has_path else [],
        support_sources=["route_masked_sep"] if origin == "path_recovery" else [],
        centroid_xy=(162.5, 150), radius_px=12.5, position_angle_deg=270,
        pixel_count=int(mask.sum()), integrated_snr=17.49, angular_extent_deg=70,
        snr_threshold_survival_count=survival, snr_threshold_total_count=8,
        near_core=False, foreground_comparison_survival_fraction=0,
    )
    doc.model_candidates = [candidate]
    doc.candidate_structure_audits = [CandidateStructureAuditObservation(
        id="CSA1", candidate_id="C1", verdict="hold_for_review",
        handoff_class="foreground_contaminated", rationale="legacy gate",
    )]
    if has_path:
        doc.geometric_paths = [GeometricPathObservation(
            id="P1", mode="arc_seeded", center_xy=doc.nucleus_xy,
            semi_major_radius_px=14, axis_ratio=.7, position_angle_deg=20,
            half_width_px=2, median_on_path_snr=3, on_minus_off_path_snr=1,
            above_local_background_fraction=.8, longest_above_local_background_deg=80,
        )]
    arrays = dict(residual=np.full(doc.image_shape, 3.0), rms=np.ones(doc.image_shape),
                  model_candidate_mask_C1=mask,
                  comparison_psf_mge_residual=np.full(doc.image_shape, comparison_value))
    return doc, arrays


def test_zero_threshold_occupancy_is_not_zero_flux_or_proven_contamination():
    doc, arrays = scene()
    result = attach_candidate_review(doc, arrays)
    entry = result.candidate_handoff_review.entries[0]
    assert entry.status == "measured_for_review"
    assert entry.comparison_detection == "below_threshold"
    assert entry.physical_origin == "unresolved"
    assert entry.comparison_aperture_flux == pytest.approx(0.8 * 34)
    assert entry.measurement.foreground_comparison_survival_fraction == 0
    assert entry.legacy_handoff_class == "foreground_contaminated"  # Provenance only.
    assert entry.measurement.axis_ratio > 1
    assert entry.measurement.integrated_snr == pytest.approx(3 * np.sqrt(34))


def test_input_visibility_and_psf_normalized_width_are_measured_not_labels():
    doc, arrays = scene()
    arrays["data"] = np.zeros(doc.image_shape)
    arrays["data"][arrays["model_candidate_mask_C1"]] = 3.0
    yy, xx = np.indices((9, 9))
    arrays["psf"] = np.exp(-((xx - 4) ** 2 + (yy - 4) ** 2) / 2.0)

    entry = attach_candidate_review(doc, arrays).candidate_handoff_review.entries[0]
    measured = entry.measurement
    assert measured.input_radial_contrast_snr == pytest.approx(3 * np.sqrt(34))
    assert measured.input_radial_positive_fraction == 1.0
    assert measured.input_radial_control_coverage == 1.0
    assert measured.radial_width_px > 0
    assert measured.radial_width_psf > 0
    assert entry.physical_origin == "unresolved"


@pytest.mark.parametrize("survival", [0, 1, 2, 3, 8])
def test_threshold_persistence_does_not_delete_measurable_candidate(survival):
    doc, arrays = scene(survival=survival)
    entry = attach_candidate_review(doc, arrays).candidate_handoff_review.entries[0]
    assert entry.measurement.pixel_count == 34
    assert entry.measurement.threshold_survival_fraction == survival / 8


@pytest.mark.parametrize("comparison", [0.0, -4.0, 0.8, 2.0, 4.0])
def test_comparison_states_do_not_assign_physical_origin(comparison):
    doc, arrays = scene(comparison_value=comparison)
    entry = attach_candidate_review(doc, arrays).candidate_handoff_review.entries[0]
    assert entry.physical_origin == "unresolved"
    assert entry.comparison_detection == ("detected" if comparison >= 2 else "below_threshold")
    assert entry.measurement.comparison_negative_fraction == (1 if comparison <= -2 else 0)


def test_missing_comparison_is_unknown_not_erasure():
    doc, arrays = scene()
    del arrays["comparison_psf_mge_residual"]
    entry = attach_candidate_review(doc, arrays).candidate_handoff_review.entries[0]
    assert entry.comparison_detection == "unavailable"
    assert entry.measurement.foreground_comparison_survival_fraction is None


@pytest.mark.parametrize("failure", ["missing", "shape", "count", "nonbinary", "nan", "zero_rms", "wrong_center"])
def test_invalid_pixel_sources_remain_explicitly_unmeasurable(failure):
    doc, arrays = scene()
    if failure == "missing":
        arrays.pop("model_candidate_mask_C1")
    elif failure == "shape":
        arrays["residual"] = np.ones((3, 3))
    elif failure == "count":
        doc.model_candidates[0].pixel_count += 1
    elif failure == "nonbinary":
        arrays["model_candidate_mask_C1"] = arrays["model_candidate_mask_C1"].astype(float) * .5
    elif failure == "nan":
        arrays["residual"][142, 162] = np.nan
    elif failure == "zero_rms":
        arrays["rms"][142, 162] = 0
    else:
        arrays["nucleus_yx"] = [12, 15]
    entry = attach_candidate_review(doc, arrays).candidate_handoff_review.entries[0]
    assert entry.status == "unmeasurable"
    assert entry.measurement is None
    assert entry.physical_origin == "unresolved"


def test_integrated_review_does_not_mutate_observation_or_scene_statistics():
    doc, arrays = scene()
    before = doc.model_dump(mode="json")
    baseline = run(doc)
    trial_doc = attach_candidate_review(doc, arrays)
    trial = run(ObservationDocument.model_validate_json(trial_doc.model_dump_json()))
    assert doc.model_dump(mode="json") == before
    assert trial_doc.structures == doc.structures
    assert trial_doc.scene_geometry == doc.scene_geometry
    assert trial.expert_assessment.vote_category == baseline.expert_assessment.vote_category
    assert len(trial.require_observation().structures) == len(doc.structures) + 1
    assert trial.decision_audit.audit_passed
    assert len(trial.evidence_dossier.candidate_review_facts) == 1
    group = trial.evidence_dossier.candidate_configuration_review[0]
    assert group.candidate_ids == ["C1"]
    assert group.path_id == "P1"
    assert not group.used_for_grading
    assert "唯一正式评级" in trial.final_report_markdown


def test_offroute_candidate_gets_review_without_diluting_existing_strong_arc():
    doc, arrays = scene(origin="sep_seed", has_path=False)
    baseline = run(doc)
    trial = run(attach_candidate_review(doc, arrays))
    assert trial.expert_assessment.vote_category == baseline.expert_assessment.vote_category
    group = trial.evidence_dossier.candidate_configuration_review[0]
    assert group.path_id is None and group.candidate_ids == ["C1"]


def test_existing_formal_candidate_is_not_added_twice():
    doc, arrays = scene()
    doc.structures[0].source_candidate_ids = ["C1"]
    result = attach_candidate_review(doc, arrays)
    entry = result.candidate_handoff_review.entries[0]
    assert entry.status == "formal_reference"
    assert entry.formal_structure_id == "S1"
    assert len(result.structures) == len(doc.structures)


def test_audit_rejects_review_as_decisive_vote_and_invalid_group_ref():
    doc, arrays = scene()
    state = run(attach_candidate_review(doc, arrays))
    state.expert_assessment.decisive_evidence.append("CRF_C1")
    state.evidence_dossier.candidate_configuration_review[0].candidate_ids.append("C999")
    result = DecisionAudit().run(state)
    assert not result.decision_audit.audit_passed
    assert result.decision_audit.measurement_used_as_proof
    assert result.decision_audit.hallucinated_facts


def test_duplicate_candidates_raise_instead_of_silent_merge():
    doc, arrays = scene()
    doc.model_candidates.append(copy.deepcopy(doc.model_candidates[0]))
    with pytest.raises(ValueError, match="duplicate"):
        attach_candidate_review(doc, arrays)


def test_default_cli_has_one_integrated_grading_mode():
    args = build_parser().parse_args(["analyze", "test.fits"])
    assert args.command == "analyze"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["analyze", "test.fits", "--candidate-review-grading"])


@pytest.mark.parametrize("legacy_policy", ["baseline_unchanged", "provisional_candidate_inclusive_v1"])
def test_saved_trial_observations_upgrade_to_integrated_policy(legacy_policy):
    doc, arrays = scene()
    payload = attach_candidate_review(doc, arrays).model_dump(mode="json")
    payload["candidate_handoff_review"]["grading_policy"] = legacy_policy
    restored = ObservationDocument.model_validate(payload)
    assert restored.candidate_handoff_review.grading_policy == "integrated_candidate_evidence_v1"


def test_observation_without_pixel_handoff_omits_review_serialization_keys():
    doc = make_observation()
    assert "candidate_handoff_review" not in doc.model_dump(mode="json")
    assert "candidate_handoff_review" not in VisionConfig().to_dict()
    assert "candidate_review_grading" not in VisionConfig().to_dict()
    dossier = run(doc).evidence_dossier.model_dump(mode="json")
    assert "candidate_review_facts" not in dossier
    assert "candidate_configuration_review" not in dossier


def test_failed_comparison_model_cannot_supply_positive_review_evidence():
    doc, arrays = scene(comparison_value=4)
    doc.foreground_model_comparison.comparison_success = False
    entry = attach_candidate_review(doc, arrays).candidate_handoff_review.entries[0]
    assert entry.comparison_detection == "unavailable"
    assert entry.comparison_aperture_flux is None


def test_audit_rejects_invalid_formal_structure_reference():
    doc, arrays = scene()
    state = run(attach_candidate_review(doc, arrays))
    state.observation_document.candidate_handoff_review.entries[0].formal_structure_id = "S999"
    assert not DecisionAudit().run(state).decision_audit.audit_passed


def _polar_mask(shape, center_xy, radius, start, stop, count):
    mask = np.zeros(shape, dtype=bool)
    angles = np.linspace(start, stop, count)
    xs = np.rint(center_xy[0] + radius * np.cos(np.radians(angles))).astype(int)
    ys = np.rint(center_xy[1] + radius * np.sin(np.radians(angles))).astype(int)
    mask[ys, xs] = True
    return mask


def _grading_scene(*, comparison_clear=0.0, multiple_cores=0, tiny=False, second_radius=20):
    doc = make_observation(structure_count=0)
    doc.alternative_indicators.multiple_core_count = multiple_cores
    center = doc.nucleus_xy
    masks = {
        "C1": _polar_mask(doc.image_shape, center, 10, -35, 35, 25),
        "C2": _polar_mask(doc.image_shape, center, second_radius, 145, 170, 12),
    }
    if tiny:
        masks["C1"][:] = False
        masks["C1"][140, 140:143] = True
        masks.pop("C2")
    candidates, audits = [], []
    for index, (cid, mask) in enumerate(masks.items(), 1):
        ys, xs = np.where(mask)
        candidates.append(ModelCandidateObservation(
            id=cid, origin="path_recovery", status="geometry_recovered",
            support_sources=["route_masked_sep"], geometry_consistent_path_ids=["P1"],
            centroid_xy=(float(xs.mean()), float(ys.mean())), radius_px=10,
            position_angle_deg=0, pixel_count=int(mask.sum()), integrated_snr=8,
            angular_extent_deg=60, snr_threshold_survival_count=1,
            snr_threshold_total_count=8, near_core=False,
            foreground_comparison_survival_fraction=0,
        ))
        audits.append(CandidateStructureAuditObservation(
            id=f"CSA{index}", candidate_id=cid, verdict="hold_for_review",
            handoff_class="recovery_candidate", rationale="unit-test held candidate",
        ))
    doc.model_candidates, doc.candidate_structure_audits = candidates, audits
    residual = np.zeros(doc.image_shape)
    rms = np.ones(doc.image_shape)
    comparison = np.zeros(doc.image_shape)
    for cid, mask in masks.items():
        residual[mask] = 3
        comparison[mask] = comparison_clear if cid == "C1" else 3
    arrays = {"residual": residual, "rms": rms,
              "comparison_psf_mge_residual": comparison,
              **{f"model_candidate_mask_{cid}": mask for cid, mask in masks.items()}}
    return doc, arrays


def test_integrated_grading_uses_one_evidence_union_without_second_vote():
    doc, arrays = _grading_scene(second_radius=20, comparison_clear=3)
    baseline = run(doc).expert_assessment.vote_category
    state = run(attach_candidate_review(doc, arrays))
    assert baseline in {"X", "C"}
    assert state.expert_assessment.vote_category == "A"
    assert state.candidate_evidence is None
    assert len(state.require_observation().structures) == 2
    assert all(i["included"] for i in state.evidence_preparation["items"])
    assert state.expert_assessment.provisional_expert_score is None
    assert state.decision_audit.audit_passed


def test_below_threshold_positive_flux_is_not_erasure():
    # All pixels below 2 sigma, but summed aperture flux is measurable.
    zero_doc, zero_arrays = _grading_scene(comparison_clear=1.5)
    high_doc, high_arrays = _grading_scene(comparison_clear=3)
    zero = run(attach_candidate_review(zero_doc, zero_arrays))
    high = run(attach_candidate_review(high_doc, high_arrays))
    assert zero.expert_assessment.vote_category == "A"
    assert high.expert_assessment.vote_category == "A"


def test_extra_brightness_peak_does_not_establish_merger_or_cap_grade():
    doc, arrays = _grading_scene(multiple_cores=1, comparison_clear=3)
    state = run(attach_candidate_review(doc, arrays))
    assert next(e for e in state.evidence_dossier.competing_explanations if e.id == "merger_or_interaction").status == "unknown"
    assert state.expert_assessment.vote_category == "A"


def test_three_pixel_trace_cannot_upgrade_grade():
    doc, arrays = _grading_scene(tiny=True)
    baseline = run(doc).expert_assessment.vote_category
    state = run(attach_candidate_review(doc, arrays))
    assert state.expert_assessment.vote_category == baseline
    assert not state.evidence_preparation["items"][0]["included"]


def test_implausibly_different_radii_do_not_form_fold_but_single_arc_remains():
    doc, arrays = _grading_scene(second_radius=40)
    state = run(attach_candidate_review(doc, arrays))
    assert state.expert_assessment.vote_category == "A"
    assert not any(c.status == "supported" for c in state.evidence_dossier.configuration_hypotheses if c.id in {"fold_arc", "cusp_arc"})
    assert next(
        c for c in state.evidence_dossier.configuration_hypotheses
        if c.id == "tangential_arc"
    ).status == "supported"


def test_legacy_centroid_core_flag_is_remeasured_from_pixels():
    doc, arrays = _grading_scene()
    doc.model_candidates[0].near_core = True
    state = run(attach_candidate_review(doc, arrays))
    item = next(item for item in state.evidence_preparation["items"] if item["candidate_id"] == "C1")
    assert item["included"]
    assert item["measurements"]["subtraction_core_overlap_fraction"] == 0
