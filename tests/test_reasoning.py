from __future__ import annotations

import pytest

from expert_agent.agents.decision_audit import DecisionAudit
from expert_agent.agents.expert_assessment_agent import ExpertAssessmentAgent
from expert_agent.core.grading import build_q1_paper_grade, map_q1_paper_grade
from expert_agent.core.state import ExpertState
from expert_agent.core.workflow import ExpertWorkflow
from expert_agent.schemas.observation import (
    AlternativeIndicators,
    CandidateStructureAuditObservation,
    DataQuality,
    ForegroundModelComparisonObservation,
    GeometricPathObservation,
    Measurement,
    ModelCandidateObservation,
    ObservationDocument,
    Provenance,
    SceneGeometryObservation,
    StructureObservation,
)
from expert_agent.schemas.reasoning import TraceableFact


def measurement(
    mid: str,
    kind: str,
    value,
    reliability: float = 0.9,
    group: str = "shared_residual_pixels",
    structure_id: str | None = None,
) -> Measurement:
    return Measurement(
        id=mid,
        kind=kind,
        value=value,
        reliability=reliability,
        method="unit_test",
        description=kind,
        structure_id=structure_id,
        correlation_group=group,
    )


def make_structure(index: int, *, misalignment: float = 10.0, snr: float = 20.0) -> tuple[StructureObservation, list[Measurement]]:
    sid = f"S{index}"
    values = [
        ("integrated_snr", snr, "shared_residual_pixels"),
        ("radius", 10.0 + 0.1 * index, "scene_geometry"),
        ("angular_extent", 60.0, "scene_geometry"),
        ("axis_ratio", 2.5, "shared_residual_pixels"),
        ("tangential_misalignment", misalignment, "scene_geometry"),
        ("threshold_survival_fraction", 0.8, "shared_residual_pixels"),
        ("sep_survival_fraction", 1.0, "shared_residual_pixels"),
    ]
    measurements = [
        measurement(f"M_{sid}_{kind}", kind, value, group=group, structure_id=sid)
        for kind, value, group in values
    ]
    structure = StructureObservation(
        id=sid,
        centroid_xy=(140.0 + index * 5, 145.0 + index * 3),
        radius_px=10.0 + 0.1 * index,
        position_angle_deg=float(index * 110 % 360),
        pixel_count=35,
        integrated_flux=1.0,
        integrated_snr=snr,
        median_radius_px=10.0,
        angular_extent_deg=60.0,
        axis_ratio=2.5,
        tangential_misalignment_deg=misalignment,
        threshold_survival_fraction=0.8,
        sep_survival_fraction=1.0,
        inner_connection_fraction=0.0,
        connects_to_inner_region=False,
        near_core=False,
        measurement_ids=[item.id for item in measurements],
    )
    return structure, measurements


def make_observation(
    *,
    structure_count: int = 3,
    coherent: bool = True,
    usable: bool = True,
    model_success: bool = True,
    competitor: bool = False,
) -> ObservationDocument:
    structures: list[StructureObservation] = []
    measurements: list[Measurement] = []
    misalignment = 10.0 if coherent else 60.0
    for index in range(1, structure_count + 1):
        structure, structure_measurements = make_structure(index, misalignment=misalignment)
        structures.append(structure)
        measurements.extend(structure_measurements)

    scene = SceneGeometryObservation(
        structure_count=structure_count,
        common_radius_cv=0.05 if coherent and structure_count >= 2 else 0.9 if structure_count >= 2 else None,
        total_angular_coverage_deg=180.0 if coherent and structure_count >= 2 else 20.0 * structure_count,
        mean_tangential_misalignment_deg=misalignment if structure_count else None,
        tangential_alignment_fraction=1.0 if coherent and structure_count >= 2 else 0.0 if structure_count >= 2 else None,
        measurement_ids=[],
    )
    alternative = AlternativeIndicators(
        inner_connected_structure_count=1 if competitor else 0,
        multiple_core_count=0,
        tidal_asymmetry=0.8 if competitor else 0.1,
        core_residual_fraction=0.1,
        bad_pixel_fraction=0.0,
        psf_information_available=True,
    )
    comparison = ForegroundModelComparisonObservation(
        primary_method="unit_test",
        comparison_method="independent_unit_test",
        comparison_success=True,
        psf_applied=True,
        primary_residual_robust_rms=1.0,
        comparison_residual_robust_rms=1.0,
        measured_structure_pixel_survival_fraction=1.0,
        residual_correlation=0.9,
    )
    return ObservationDocument(
        object_id="synthetic",
        input_path="synthetic.fits",
        input_format="fits",
        band="VIS",
        image_shape=(300, 300),
        nucleus_xy=(150.0, 150.0),
        data_quality=DataQuality(
            usable=usable,
            supplied_rms=True,
            finite_fraction=1.0,
            background_median=0.0,
            background_rms=1.0,
            model_method="unit_test",
            model_success=model_success,
            residual_median=0.0,
            residual_robust_rms=1.0,
        ),
        scene_geometry=scene,
        alternative_indicators=alternative,
        foreground_model_comparison=comparison,
        structures=structures,
        measurements=measurements,
        limitations=["single band"],
        provenance=Provenance(pipeline="unit_test"),
    )


def run(observation: ObservationDocument):
    return ExpertWorkflow().run(ExpertState(observation_document=observation))


def test_high_snr_alone_cannot_reach_a_or_b():
    observation = make_observation(structure_count=1, coherent=False)
    observation.structures[0].integrated_snr = 1000.0
    result = run(observation)
    assert result.expert_assessment.provisional_grade not in {"A", "B"}


def test_sep_stability_alone_is_not_lens_proof():
    observation = make_observation(structure_count=1, coherent=False)
    observation.structures[0].sep_survival_fraction = 1.0
    result = run(observation)
    assert result.expert_assessment.vote_category == "C"
    assert result.expert_assessment.vote_code == 1
    assert "cannot identify lensing physics" in next(
        item.rationale for item in result.evidence_dossier.competing_explanations
        if item.id == "noise_bad_pixel_or_projected_source"
    )


def test_ellipse_route_consistency_alone_is_not_lens_proof():
    observation = make_observation(structure_count=1, coherent=False)
    observation.geometric_paths = [
        GeometricPathObservation(
            id="G1",
            mode="arc_seeded",
            center_xy=(150.0, 150.0),
            semi_major_radius_px=10.0,
            axis_ratio=0.8,
            position_angle_deg=30.0,
            half_width_px=2.0,
            median_on_path_snr=20.0,
            on_minus_off_path_snr=10.0,
            above_local_background_fraction=1.0,
            longest_above_local_background_deg=300.0,
            seed_structure_ids=["S1"],
            linked_structure_ids=["S1"],
        )
    ]
    result = run(observation)
    assert result.expert_assessment.provisional_grade not in {"A", "B"}
    assert any(item.field_path == "geometric_paths" for item in result.evidence_dossier.unused_facts)


def test_recovered_structure_is_not_automatically_downgraded():
    observation = make_observation()
    observation.model_candidates = [
        ModelCandidateObservation(
            id="C-recovered",
            origin="path_recovery",
            status="geometry_recovered",
            structural_support_level="route_recovered",
            pixel_partition="route_recovery",
            support_sources=["geometric_route"],
            path_ids=["G1"],
            centroid_xy=(155.0, 150.0),
            radius_px=10.0,
            position_angle_deg=90.0,
            pixel_count=20,
            integrated_snr=15.0,
            angular_extent_deg=30.0,
            near_core=False,
        )
    ]
    result = run(observation)
    assert result.expert_assessment.provisional_grade == "A"
    recovery = next(item for item in result.evidence_dossier.processing_cautions if "recovery" in item.id)
    assert recovery.effect == "uncertainty_only"
    assert result.decision_audit.recovery_auto_penalized == []


def test_missing_color_remains_not_measured_not_inconsistent():
    result = run(make_observation())
    color = next(item for item in result.evidence_dossier.missing_information if "color" in item.topic)
    assert color.status == "not_measured"
    assert all("color" not in fact.statement.lower() for fact in result.evidence_dossier.contradicting_evidence)


def test_foreground_model_failure_has_no_fixed_grade_cap():
    result = run(make_observation(model_success=False))
    assert result.expert_assessment.provisional_grade == "A"
    assert any("does not impose a grade cap" in item.description for item in result.evidence_dossier.processing_cautions)


def test_many_random_structures_are_not_multiple_images():
    result = run(make_observation(structure_count=6, coherent=False))
    assert result.expert_assessment.provisional_grade not in {"A", "B"}
    assert result.evidence_dossier.lens_specificity.status == "weak"


def test_clear_lens_geometry_with_competitor_keeps_b_to_a_vote_set():
    result = run(make_observation(competitor=True))
    assert result.expert_assessment.plausible_vote_categories == ("B", "A")
    assert result.expert_assessment.provisional_score_range == (2.0, 3.0)
    assert result.expert_assessment.provisional_grade is None


def test_tidal_asymmetry_proxy_alone_does_not_force_a_downgrade():
    observation = make_observation()
    observation.alternative_indicators.tidal_asymmetry = 0.9
    result = run(observation)
    assert result.expert_assessment.vote_category == "A"


def test_two_image_geometry_can_reach_a_without_three_promoted_structures():
    result = run(make_observation(structure_count=2))
    assert result.expert_assessment.vote_category == "A"


def test_robust_single_tangential_arc_keeps_b_to_a_vote_set():
    observation = make_observation(structure_count=1)
    observation.scene_geometry.total_angular_coverage_deg = 70.0
    result = run(observation)
    assert result.expert_assessment.plausible_vote_categories == ("B", "A")


def test_common_radius_cv_just_above_old_cut_remains_eligible():
    observation = make_observation(structure_count=2)
    observation.scene_geometry.common_radius_cv = 0.25
    result = run(observation)
    assert result.expert_assessment.vote_category == "A"


def test_strong_held_candidate_widens_vote_set_instead_of_forcing_x():
    observation = make_observation(structure_count=0, coherent=False, model_success=False)
    observation.candidate_structure_audits = [
        CandidateStructureAuditObservation(
            id="CSA-review",
            candidate_id="C-review",
            verdict="hold_for_review",
            handoff_class="foreground_contaminated",
            discovery_context="global_sep_plus_geometry",
            checks={
                "existing_detector_acceptance": True,
                "connected_signal_present": True,
                "positive_integrated_signal": True,
                "snr_multi_threshold_stable": True,
                "comparison_model_does_not_fully_erase": True,
                "geometry_route_relation_recorded": True,
            },
            rationale="Synthetic extraction conflict.",
        )
    ]
    result = run(observation)
    assert result.expert_assessment.plausible_vote_categories == ("C", "B", "A")
    assert result.expert_assessment.provisional_score_range == (1.0, 3.0)
    assert result.decision_audit.audit_passed is True


def test_unusable_data_abstains_instead_of_x():
    result = run(make_observation(usable=False))
    assessment = result.expert_assessment
    assert assessment.provisional_grade == "insufficient_for_grading"
    assert assessment.expert_vote_anchor is None
    assert assessment.provisional_expert_score is None
    assert result.decision_audit.abstention_recommended is True


def test_coherent_ring_arc_or_multiple_image_system_can_reach_a_anchor():
    result = run(make_observation())
    assert result.expert_assessment.expert_vote_anchor == 3
    assert result.expert_assessment.provisional_grade == "A"
    assert result.evidence_dossier.lens_specificity.status == "credible"


def test_correlated_evidence_is_grouped_not_accumulated():
    result = run(make_observation())
    assert result.evidence_dossier.correlated_evidence_groups
    assert result.decision_audit.correlated_evidence_double_counted == []
    trace = next(item for item in result.reasoning_trace if item["agent"] == "ExpertAssessmentAgent")
    assert trace["content"]["subscore_aggregation_used"] is False


def test_official_and_zenodo_answers_do_not_enter_scoring_input():
    observation = make_observation()
    observation.artifacts.update({"official_grade": "A", "zenodo_label": "known_lens", "pyautolens_result": "source_reconstruction"})
    result = run(observation)
    assert result.decision_audit.external_answer_leakage == []
    unused = {item.field_path for item in result.evidence_dossier.unused_facts}
    assert {"artifacts", "euclid_q1_input"}.issubset(unused)


def test_decision_audit_is_lightweight_and_does_not_rerun_hypotheses():
    result = run(make_observation())
    agents = [item["agent"] for item in result.reasoning_trace]
    assert agents.count("ExpertAssessmentAgent") == 1
    assert agents.count("DecisionAudit") == 1
    assert not any(name in agents for name in ("HypothesisAgent", "EvidenceAgent", "CounterEvidenceAgent", "ReasoningAuditor"))
    audit_trace = next(item for item in result.reasoning_trace if item["agent"] == "DecisionAudit")
    assert audit_trace["content"]["full_hypothesis_reasoning_rerun"] is False


def test_decision_audit_detects_hallucinated_observation_reference():
    state = ExpertState(observation_document=make_observation())
    ExpertAssessmentAgent().run(state)
    state.evidence_dossier.observed_facts.append(
        TraceableFact(id="F-injected", statement="Injected unsupported fact.", observation_refs=["not_in_observation.value"], source_types=["primary_residual_measurement"])
    )
    DecisionAudit().run(state)
    assert state.decision_audit.audit_passed is False
    assert state.decision_audit.hallucinated_facts


def test_official_expert_score_remains_null_before_calibration():
    result = run(make_observation())
    assessment = result.expert_assessment
    assert assessment.official_expert_score is None
    assert assessment.score_status == "provisional_uncalibrated"
    assert assessment.calibration_required is True


def test_every_used_fact_is_traceable_to_observation():
    result = run(make_observation())
    assert result.decision_audit.audit_passed is True
    assert all(fact.observation_refs for fact in result.evidence_dossier.observed_facts)
    assert any(ref.startswith("M_") for fact in result.evidence_dossier.observed_facts for ref in fact.observation_refs)


def test_observation_1_0_remains_readable():
    payload = make_observation().model_dump(mode="json")
    payload["schema_version"] = "1.0"
    payload.pop("scene_geometry", None)
    payload.pop("alternative_indicators", None)
    observation = ObservationDocument.model_validate(payload)
    result = run(observation)
    assert observation.schema_version == "1.0"
    assert result.expert_assessment.score_status == "provisional_uncalibrated"


def test_q1_paper_grade_mapping_boundaries():
    assert map_q1_paper_grade(0.0) == "below_B_cut"
    assert map_q1_paper_grade(1.5) == "below_B_cut"
    assert map_q1_paper_grade(1.5001) == "B"
    assert map_q1_paper_grade(2.0) == "B"
    assert map_q1_paper_grade(2.0001) == "A"
    with pytest.raises(ValueError):
        map_q1_paper_grade(-0.01)
    with pytest.raises(ValueError):
        map_q1_paper_grade(3.01)


def test_a_plus_is_separate_from_grade_mapping():
    result = build_q1_paper_grade(2.7, rationale="Calibrated test fixture.", a_plus=True, a_plus_reason="Independent scientific-value review.")
    assert result.grade == "A"
    assert result.a_plus is True


def test_below_b_cut_does_not_manufacture_a_continuous_c_grade():
    result = build_q1_paper_grade(1.2, rationale="Calibrated low-score fixture.")
    assert result.catalog_bin == "below_B_cut"
    assert result.grade is None
