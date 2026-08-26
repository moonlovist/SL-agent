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
) -> list[CandidateStructureAuditObservation]:
    """Audit candidate signal reality without using lens geometry as a gate.

    Every recorded model-chain candidate is audited.  A geometric route may
    discover a candidate or describe its relation to other candidates, but it
    cannot by itself promote or reject the signal.  The audit only checks
    whether an existing detector measurement is usable and whether a wholly
    different foreground subtraction erases it.  It does not classify arcs,
    rings, multiple images, or lenses, and it introduces no new S/N threshold.
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
        comparison_survival = candidate.foreground_comparison_survival_fraction
        comparison_not_erased = (
            None if comparison_survival is None else comparison_survival > 0.0
        )
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
            },
            "processing_robustness": {
                "data_usable": data_usable,
                "primary_foreground_fit_success": foreground_model_success,
                "comparison_model_does_not_fully_erase": comparison_not_erased,
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
            "geometry_route_relation_recorded": geometry_support,
            "comparison_model_does_not_fully_erase": comparison_not_erased,
        }
        supporting_facts = list(candidate.support_sources)
        if global_sep_support:
            supporting_facts.append("global_sep_consensus_signal")
        if route_masked_sep_support:
            supporting_facts.append("route_masked_sep_signal")
        supporting_facts.append("geometry_relation_not_used_for_signal_verdict")
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
            caution_flags.append("geometry_route_support_absent_not_a_signal_veto")

        if not data_usable:
            verdict = "hold_for_review"
            rationale = (
                "The candidate is retained, but the image is not sufficiently "
                "usable for automatic structure promotion."
            )
        elif not (detector_support and connected_signal and positive_integrated_signal):
            verdict = "trace_only"
            rationale = (
                "The candidate remains traceable, but the recorded candidate does "
                "not contain the minimum signal-level detector facts required for "
                "a measured residual structure. Geometry is not used in this decision."
            )
        elif comparison_not_erased is False:
            verdict = "hold_for_review"
            rationale = (
                "The candidate has detector support, but no candidate pixel "
                "survives the audit-only comparison foreground model."
            )
        else:
            verdict = "promoted"
            fallback_note = (
                " The primary fit used a fallback foreground model, which remains "
                "an explicit caution rather than a veto."
                if not foreground_model_success
                else ""
            )
            rationale = (
                "The candidate passed an existing SEP detector channel, contains "
                "connected positive residual signal, and is not fully erased by the "
                "comparison foreground model. Route membership is recorded only as "
                "discovery context and was not used for promotion. Promotion means "
                "measurable residual structure only, not a lens interpretation."
                + fallback_note
            )

        results.append(
            CandidateStructureAuditObservation(
                id=audit_id,
                candidate_id=candidate.id,
                verdict=verdict,
                discovery_context=discovery_context,
                geometry_used_for_verdict=False,
                sections=sections,
                checks=checks,
                supporting_facts=list(dict.fromkeys(supporting_facts)),
                caution_flags=list(dict.fromkeys(caution_flags)),
                rationale=rationale,
            )
        )
    return results
