import copy
import numpy as np
import pytest

from expert_agent.agents.decision_audit import DecisionAudit
from expert_agent.agents.expert_assessment_agent import ExpertAssessmentAgent
from expert_agent.agents.report_writer import ReportWriter
from expert_agent.core.state import ExpertState
from expert_agent.schemas.observation import ModelCandidateObservation
from expert_agent.vision.candidate_handoff import attach_candidate_review
from tests.test_reasoning import make_observation, run
from tests.test_candidate_handoff import _grading_scene


@pytest.mark.parametrize("half", [False, True])
def test_annulus_pixel_core_geometry_not_centroid(half):
    doc = make_observation(structure_count=0)
    yy, xx = np.indices(doc.image_shape)
    x, y = doc.nucleus_xy
    radius = np.hypot(xx-x, yy-y)
    mask = (radius >= 13.3) & (radius <= 14.7)
    if half:
        mask &= yy >= y
    ys, xs = np.where(mask)
    doc.model_candidates = [ModelCandidateObservation(
        id="C1", origin="path_recovery", status="geometry_recovered",
        centroid_xy=(float(xs.mean()), float(ys.mean())), radius_px=0,
        position_angle_deg=0, pixel_count=int(mask.sum()), integrated_snr=20,
        angular_extent_deg=180 if half else 360, near_core=True,
        snr_threshold_survival_count=1, snr_threshold_total_count=8)]
    arrays = {"residual": mask.astype(float)*3, "rms": np.ones(mask.shape),
              "model_candidate_mask_C1": mask,
              "comparison_psf_mge_residual": mask.astype(float)*1.5}
    reviewed = attach_candidate_review(doc, arrays)
    item = reviewed.candidate_handoff_review.entries[0].measurement
    assert item.subtraction_core_overlap_fraction == 0
    assert not item.near_core
    assert item.median_radius_px > 13
    if not half:
        assert item.radius_px == pytest.approx(0)
    # Same aperture remains significant even with zero super-threshold pixels.
    assert item.foreground_comparison_survival_fraction == 0
    assert item.comparison_integrated_snr > 4
    state = run(reviewed)
    assert state.expert_assessment.vote_category == "A"
    assert state.decision_audit.audit_passed


def test_negative_model_conflict_keeps_possible_morphology_without_calling_artifact():
    doc, arrays = _grading_scene(comparison_clear=-4)
    # Both candidate apertures conflict, so the whole proposed system is unresolved.
    arrays["comparison_psf_mge_residual"][arrays["model_candidate_mask_C2"]] = -4
    state = run(attach_candidate_review(doc, arrays))
    assert state.expert_assessment.vote_category == "C"
    assert state.final_decision["published_grade"] == "C"
    assert state.decision_audit.audit_passed
    explanation = next(e for e in state.evidence_dossier.competing_explanations if e.id == "foreground_subtraction_or_psf_artifact")
    assert explanation.status == "unknown"


def test_original_is_immutable_and_duplicate_mask_is_not_another_image():
    doc, arrays = _grading_scene(comparison_clear=3)
    doc.model_candidates[1].pixel_count = doc.model_candidates[0].pixel_count
    arrays["model_candidate_mask_C2"] = arrays["model_candidate_mask_C1"].copy()
    review = attach_candidate_review(doc, arrays)
    before = review.model_dump(mode="json")
    result = run(review)
    assert before == review.model_dump(mode="json")
    assert len(result.require_observation().structures) == 1
    assert "duplicate_pixel_mask" in str(result.evidence_preparation["items"])


def test_tampered_grade_is_rejected_and_not_published_as_abc():
    state = ExpertState(observation_document=make_observation())
    ExpertAssessmentAgent().run(state)
    state.expert_assessment = state.expert_assessment.model_copy(update={
        "vote_category": "C", "provisional_grade": "C", "plausible_vote_categories": ("C",)})
    DecisionAudit().run(state)
    ReportWriter().run(state)
    assert not state.decision_audit.audit_passed
    assert state.final_decision["published_grade"] == "unknown"
    assert state.final_decision["publication_reason"] == "audit_failed"


def test_no_lens_x_is_distinguished_from_processing_unknown():
    state = run(make_observation(structure_count=0))
    assert state.final_decision["published_grade"] == "unknown"
    assert state.final_decision["publication_reason"] == "non_lens_vote_X"
    assert state.expert_assessment.vote_category == "X"


def test_repeat_and_label_path_invariance():
    doc, arrays = _grading_scene(comparison_clear=3)
    doc = attach_candidate_review(doc, arrays)
    first, second = run(doc), run(doc)
    assert first.to_dict() == second.to_dict()
    changed = doc.model_copy(deep=True)
    changed.input_path = "B/not_a_label.fits"
    assert run(changed).expert_assessment == first.expert_assessment
    assert all(first.expert_assessment.model_dump()[name] is None for name in [
        "provisional_expert_score", "vote_code", "expert_vote_anchor", "official_expert_score"])
