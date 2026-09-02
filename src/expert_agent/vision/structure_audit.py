from __future__ import annotations

from ..schemas.observation import (
    CandidateStructureAuditObservation,
    ModelCandidateObservation,
)


def audit_structure_candidates(
    candidates: list[ModelCandidateObservation],
    *,
    data_usable: bool,
    foreground_model_success: bool,
    snr_min_survival: int = 3,
) -> list[CandidateStructureAuditObservation]:
    """Audit signal reality, then route candidates for downstream reasoning.

    Signal validation and lens-system relevance are deliberately separate.
    Geometry is never used to claim that a signal is real.  Stable off-route
    signals are promoted with recovery provenance so that the expert layer can
    evaluate their morphology instead of losing them at the handoff boundary.
    Proximity to the subtraction core is a caution, not an exclusion: only
    affirmative erasure by the independent foreground comparison isolates a
    candidate as foreground-contaminated.
    """

    results: list[CandidateStructureAuditObservation] = []
    for index, candidate in enumerate(candidates, start=1):
        audit_id = f"CSA{index}"
        geometry_support = bool(candidate.geometry_consistent_path_ids)
        global_sep_support = (
            candidate.origin == "sep_seed"
            and candidate.sep_survival_fraction is not None
            and candidate.sep_survival_fraction > 0.0
        )
        route_masked_sep_support = (
            candidate.origin == "path_recovery"
            and "route_masked_sep" in candidate.support_sources
        )
        detector_support = global_sep_support or route_masked_sep_support
        connected_signal = candidate.pixel_count > 0
        positive_integrated_signal = candidate.integrated_snr > 0.0

        threshold_count = candidate.snr_threshold_survival_count
        threshold_total = candidate.snr_threshold_total_count
        threshold_available = (
            threshold_count is not None
            and threshold_total is not None
            and threshold_total > 0
        )
        threshold_stable = (
            threshold_count >= snr_min_survival
            if threshold_available
            else None
        )

        comparison_survival = candidate.foreground_comparison_survival_fraction
        comparison_not_erased = (
            None if comparison_survival is None else comparison_survival > 0.0
        )
        core_overlap = candidate.subtraction_core_overlap_fraction
        touches_subtraction_core = bool(
            candidate.near_core
            or (core_overlap is not None and core_overlap > 0.0)
        )
        foreground_contamination_risk = comparison_not_erased is False
        discovery_context = (
            "route_recovery"
            if candidate.origin == "path_recovery"
            else "global_sep_plus_geometry"
            if geometry_support
            else "global_sep"
        )

        sections: dict[str, dict[str, bool | None]] = {
            "signal_quality": {
                "existing_detector_acceptance": detector_support,
                "connected_signal_present": connected_signal,
                "positive_integrated_signal": positive_integrated_signal,
                "snr_multi_threshold_stable": threshold_stable,
            },
            "processing_robustness": {
                "data_usable": data_usable,
                "primary_foreground_fit_success": foreground_model_success,
                "comparison_model_does_not_fully_erase": comparison_not_erased,
                "touches_subtraction_sensitive_core": touches_subtraction_core,
                "foreground_contamination_risk": foreground_contamination_risk,
            },
            "discovery_provenance": {
                "global_sep_consensus_support": global_sep_support,
                "route_masked_sep_support": route_masked_sep_support,
                "geometry_route_relation_recorded": geometry_support,
            },
        }
        checks: dict[str, bool | None] = {
            "data_usable": data_usable,
            "foreground_model_success": foreground_model_success,
            "existing_detector_acceptance": detector_support,
            "connected_signal_present": connected_signal,
            "positive_integrated_signal": positive_integrated_signal,
            "snr_multi_threshold_stable": threshold_stable,
            "geometry_route_relation_recorded": geometry_support,
            "comparison_model_does_not_fully_erase": comparison_not_erased,
            "touches_subtraction_sensitive_core": touches_subtraction_core,
            "foreground_contamination_risk": foreground_contamination_risk,
        }

        supporting_facts = list(candidate.support_sources)
        if global_sep_support:
            supporting_facts.append("global_sep_consensus_signal")
        if route_masked_sep_support:
            supporting_facts.append("route_masked_sep_signal")
        if threshold_stable is True:
            supporting_facts.append("snr_multi_threshold_stable")
        elif threshold_stable is False:
            supporting_facts.append("snr_multi_threshold_weak")
        else:
            supporting_facts.append("snr_multi_threshold_unavailable")
        supporting_facts.append("geometry_not_used_for_signal_reality")

        caution_flags = [
            flag
            for flag in candidate.flags
            if flag not in {"not_scoring_input", "model_chain_unvalidated"}
        ]
        if not foreground_model_success:
            caution_flags.append("primary_foreground_fit_used_fallback")
        if comparison_survival is None:
            caution_flags.append("foreground_comparison_not_available")
        if candidate.origin == "path_recovery":
            caution_flags.append("search_conditioned_route_recovery")
        if not geometry_support:
            caution_flags.append("geometry_route_support_absent")
        if touches_subtraction_core:
            caution_flags.append("touches_subtraction_sensitive_core")

        base_signal_ok = bool(
            detector_support and connected_signal and positive_integrated_signal
        )
        geometry_used_for_handoff = False
        if not data_usable:
            verdict = "hold_for_review"
            handoff_class = "trace_candidate"
            rationale = (
                "The image is not sufficiently usable for automatic candidate "
                "handoff; the detection remains traceable."
            )
        elif not base_signal_ok:
            verdict = "trace_only"
            handoff_class = "trace_candidate"
            rationale = (
                "The candidate lacks the detector-level signal facts required "
                "for downstream handoff."
            )
        elif foreground_contamination_risk:
            verdict = "hold_for_review"
            handoff_class = "foreground_contaminated"
            rationale = (
                "The signal is measurable but lies in a subtraction-sensitive "
                "configuration or is erased by the comparison foreground model; "
                "it is isolated from positive lens evidence."
            )
        elif threshold_stable is True and geometry_support:
            verdict = "promoted"
            handoff_class = "primary_candidate"
            geometry_used_for_handoff = True
            rationale = (
                "The candidate has detector support, survives the configured S/N "
                "threshold grid, and is associated with a recorded geometric "
                "search route. It enters the primary reasoning structure list; "
                "this is not a lens classification."
            )
        elif threshold_stable is True:
            verdict = "promoted"
            handoff_class = "recovery_candidate"
            rationale = (
                "The off-route signal is stable across the configured S/N grid. "
                "It is promoted with recovery provenance and enters downstream "
                "reasoning with complete measurements; route membership is not a "
                "signal-reality or lens-grade gate."
            )
        elif geometry_support:
            verdict = "hold_for_review"
            handoff_class = "recovery_candidate"
            geometry_used_for_handoff = True
            rationale = (
                "The candidate has geometric support but weak or unavailable S/N "
                "threshold persistence. It is preserved for recovery review rather "
                "than promoted or deleted."
            )
        else:
            verdict = "trace_only"
            handoff_class = "trace_candidate"
            rationale = (
                "The off-route candidate does not survive enough configured S/N "
                "levels to enter the recovery handoff; it remains fully traceable."
            )

        results.append(
            CandidateStructureAuditObservation(
                id=audit_id,
                candidate_id=candidate.id,
                verdict=verdict,
                handoff_class=handoff_class,
                discovery_context=discovery_context,
                geometry_used_for_verdict=False,
                geometry_used_for_handoff=geometry_used_for_handoff,
                sections=sections,
                checks=checks,
                supporting_facts=list(dict.fromkeys(supporting_facts)),
                caution_flags=list(dict.fromkeys(caution_flags)),
                rationale=rationale,
            )
        )
    return results
