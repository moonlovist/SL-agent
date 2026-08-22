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
    """Audit whether visual candidates may enter the measured structure list.

    This gate tests measurement usability and independent failure modes. It
    does not decide whether a candidate is an arc, ring, multiple image, or
    strong lens. No aggregate score or newly tuned scientific threshold is
    introduced here.
    """
    results: list[CandidateStructureAuditObservation] = []
    for index, candidate in enumerate(candidates, start=1):
        audit_id = f"CSA{index}"
        supported = candidate.review_priority == "priority_keep"
        comparison_survival = candidate.foreground_comparison_survival_fraction
        comparison_not_erased = (
            None if comparison_survival is None else comparison_survival > 0.0
        )
        checks: dict[str, bool | None] = {
            "data_usable": data_usable,
            "foreground_model_success": foreground_model_success,
            "minimum_connected_area_already_passed": candidate.pixel_count > 0,
            "additional_structural_support": supported,
            "comparison_model_does_not_fully_erase": comparison_not_erased,
        }
        supporting_facts = list(candidate.support_sources)
        caution_flags = [
            flag
            for flag in candidate.flags
            if flag not in {"not_scoring_input", "model_chain_unvalidated"}
        ]
        if not foreground_model_success:
            caution_flags.append("primary_foreground_fit_used_fallback")
        if comparison_survival is None:
            caution_flags.append("foreground_comparison_not_available")

        if not data_usable:
            verdict = "hold_for_review"
            rationale = (
                "The candidate is retained, but the image is not sufficiently "
                "usable for automatic structure promotion."
            )
        elif not supported:
            verdict = "trace_only"
            rationale = (
                "The candidate remains a real SEP trace, but currently has no "
                "additional route or recovery support for structure promotion."
            )
        elif comparison_not_erased is False:
            verdict = "hold_for_review"
            rationale = (
                "The candidate has structural support, but no candidate pixel "
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
                "The candidate passed the existing connected-pixel detector, has "
                "additional structural support, and is not fully erased by the "
                "comparison foreground model. Promotion means measurable residual "
                "structure only, not a lens interpretation."
                + fallback_note
            )

        results.append(
            CandidateStructureAuditObservation(
                id=audit_id,
                candidate_id=candidate.id,
                verdict=verdict,
                checks=checks,
                supporting_facts=supporting_facts,
                caution_flags=list(dict.fromkeys(caution_flags)),
                rationale=rationale,
            )
        )
    return results
