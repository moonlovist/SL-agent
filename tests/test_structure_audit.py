from expert_agent.schemas.observation import ModelCandidateObservation
from expert_agent.vision.structure_audit import audit_structure_candidates


def _candidate(
    *,
    review_priority: str = "priority_keep",
    comparison_survival: float | None = 0.5,
    threshold_survival_count: int = 4,
    near_core: bool = False,
    core_overlap: float = 0.0,
) -> ModelCandidateObservation:
    return ModelCandidateObservation(
        id="C1",
        origin="sep_seed",
        status=(
            "sep_geometry_consistent"
            if review_priority == "priority_keep"
            else "sep_only"
        ),
        structural_support_level=(
            "sep_plus_geometry"
            if review_priority == "priority_keep"
            else "raw_sep_only"
        ),
        pixel_partition=(
            "route_supported"
            if review_priority == "priority_keep"
            else "off_route_remainder"
        ),
        review_priority=review_priority,
        support_sources=(
            ["sep_parameter_consensus", "geometric_route_consistency"]
            if review_priority == "priority_keep"
            else ["sep_parameter_consensus"]
        ),
        geometry_consistent_path_ids=(
            ["G1"] if review_priority == "priority_keep" else []
        ),
        centroid_xy=(10.0, 12.0),
        radius_px=5.0,
        position_angle_deg=45.0,
        pixel_count=8,
        integrated_snr=9.0,
        angular_extent_deg=20.0,
        snr_threshold_survival_count=threshold_survival_count,
        snr_threshold_total_count=8,
        snr_threshold_survival_fraction=threshold_survival_count / 8.0,
        sep_survival_fraction=0.75,
        foreground_comparison_survival_fraction=comparison_survival,
        near_core=near_core,
        subtraction_core_overlap_fraction=core_overlap,
        flags=["not_scoring_input"],
    )


def test_route_supported_stable_candidate_is_primary() -> None:
    result = audit_structure_candidates(
        [_candidate()], data_usable=True, foreground_model_success=True
    )[0]

    assert result.verdict == "promoted"
    assert result.handoff_class == "primary_candidate"
    assert result.checks["snr_multi_threshold_stable"] is True
    assert result.geometry_used_for_verdict is False
    assert result.geometry_used_for_handoff is True


def test_off_route_stable_candidate_is_promoted_with_recovery_provenance() -> None:
    result = audit_structure_candidates(
        [_candidate(review_priority="downgraded_trace")],
        data_usable=True,
        foreground_model_success=True,
    )[0]

    assert result.verdict == "promoted"
    assert result.handoff_class == "recovery_candidate"
    assert result.discovery_context == "global_sep"
    assert result.geometry_used_for_handoff is False


def test_off_route_weak_candidate_is_trace_only() -> None:
    result = audit_structure_candidates(
        [
            _candidate(
                review_priority="downgraded_trace",
                threshold_survival_count=1,
            )
        ],
        data_usable=True,
        foreground_model_success=True,
    )[0]

    assert result.verdict == "trace_only"
    assert result.handoff_class == "trace_candidate"


def test_comparison_model_erasure_is_foreground_contaminated() -> None:
    result = audit_structure_candidates(
        [_candidate(comparison_survival=0.0)],
        data_usable=True,
        foreground_model_success=True,
    )[0]

    assert result.verdict == "hold_for_review"
    assert result.handoff_class == "foreground_contaminated"


def test_fallback_foreground_core_overlap_is_caution_not_erasure() -> None:
    result = audit_structure_candidates(
        [_candidate(core_overlap=0.5)],
        data_usable=True,
        foreground_model_success=False,
    )[0]

    assert result.verdict == "promoted"
    assert result.handoff_class == "primary_candidate"
    assert "primary_foreground_fit_used_fallback" in result.caution_flags
    assert "touches_subtraction_sensitive_core" in result.caution_flags


def test_fallback_foreground_far_route_candidate_can_remain_primary() -> None:
    result = audit_structure_candidates(
        [_candidate()], data_usable=True, foreground_model_success=False
    )[0]

    assert result.handoff_class == "primary_candidate"
    assert "primary_foreground_fit_used_fallback" in result.caution_flags


def test_route_supported_weak_candidate_is_recovery_not_deleted() -> None:
    result = audit_structure_candidates(
        [_candidate(threshold_survival_count=1)],
        data_usable=True,
        foreground_model_success=True,
    )[0]

    assert result.handoff_class == "recovery_candidate"


def test_unusable_data_is_not_handed_to_reasoning() -> None:
    result = audit_structure_candidates(
        [_candidate()], data_usable=False, foreground_model_success=True
    )[0]

    assert result.verdict == "hold_for_review"
    assert result.handoff_class == "trace_candidate"
