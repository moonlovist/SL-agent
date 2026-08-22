from expert_agent.schemas.observation import ModelCandidateObservation
from expert_agent.vision.structure_audit import audit_structure_candidates


def _candidate(
    *,
    review_priority: str = "priority_keep",
    comparison_survival: float | None = 0.5,
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
        support_sources=["sep_consensus", "geometry_route"],
        centroid_xy=(10.0, 12.0),
        radius_px=5.0,
        position_angle_deg=45.0,
        pixel_count=8,
        integrated_snr=9.0,
        angular_extent_deg=20.0,
        sep_survival_fraction=0.75,
        foreground_comparison_survival_fraction=comparison_survival,
        near_core=False,
        flags=["not_scoring_input"],
    )


def test_supported_candidate_is_promoted_to_measured_structure() -> None:
    result = audit_structure_candidates(
        [_candidate()], data_usable=True, foreground_model_success=True
    )[0]

    assert result.verdict == "promoted"
    assert result.checks["additional_structural_support"] is True


def test_off_route_sep_remainder_stays_trace_only() -> None:
    result = audit_structure_candidates(
        [_candidate(review_priority="downgraded_trace")],
        data_usable=True,
        foreground_model_success=True,
    )[0]

    assert result.verdict == "trace_only"


def test_comparison_model_erasure_holds_supported_candidate() -> None:
    result = audit_structure_candidates(
        [_candidate(comparison_survival=0.0)],
        data_usable=True,
        foreground_model_success=True,
    )[0]

    assert result.verdict == "hold_for_review"


def test_fallback_foreground_model_is_caution_not_veto() -> None:
    result = audit_structure_candidates(
        [_candidate()], data_usable=True, foreground_model_success=False
    )[0]

    assert result.verdict == "promoted"
    assert "primary_foreground_fit_used_fallback" in result.caution_flags


def test_unusable_data_holds_candidate() -> None:
    result = audit_structure_candidates(
        [_candidate()], data_usable=False, foreground_model_success=True
    )[0]

    assert result.verdict == "hold_for_review"
