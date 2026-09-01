from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class ReportWriter(BaseAgent):
    """Render the already-audited assessment without adding scientific claims."""

    def __init__(self):
        super().__init__("ReportWriter")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("将已审查的专家判断写成可追溯报告")
        dossier = state.evidence_dossier
        assessment = state.expert_assessment
        audit = state.decision_audit
        if dossier is None or assessment is None or audit is None:
            raise ValueError("ReportWriter requires assessment and decision audit")

        def fact_lines(items):
            return [
                f"- `{item.id}`: {item.statement} (Observation refs: {', '.join(item.observation_refs)})"
                for item in items
            ] or ["- None recorded."]

        missing_lines = [
            f"- {item.topic}: **{item.status}** - {item.reason}"
            for item in dossier.missing_information
        ] or ["- None recorded."]
        alternative_lines = [
            f"- `{item.id}`: **{item.status}** - {item.rationale}"
            for item in dossier.competing_explanations
        ]
        range_text = (
            f"[{assessment.provisional_score_range[0]:.1f}, {assessment.provisional_score_range[1]:.1f}]"
            if assessment.provisional_score_range
            else "null"
        )
        compatibility_text = (
            str(assessment.provisional_expert_score)
            if assessment.provisional_expert_score is not None
            else range_text
        )
        markdown = "\n".join(
            [
                f"# Strong-lens expert assessment: {state.object_id}",
                "",
                "## Evidence dossier",
                "",
                f"- Signal reality: **{dossier.signal_reality.status}** - {dossier.signal_reality.rationale}",
                f"- Lens specificity: **{dossier.lens_specificity.status}** - {dossier.lens_specificity.rationale}",
                "",
                "### Observed facts used",
                "",
                *fact_lines(dossier.observed_facts),
                "",
                "### Competing explanations",
                "",
                *alternative_lines,
                "",
                "### Missing information",
                "",
                *missing_lines,
                "",
                "## Provisional expert assessment",
                "",
                f"- Vote category: **{assessment.vote_category}**",
                f"- Plausible vote categories: **{list(assessment.plausible_vote_categories)}**",
                f"- Vote code: **{assessment.vote_code}**",
                f"- Compatibility anchor (deprecated): **{assessment.expert_vote_anchor}**",
                f"- Compatibility score/range (not an expert_score): **{compatibility_text}**",
                f"- Score status: **{assessment.score_status}**",
                "- Official expert score: **null**",
                f"- Rationale: {assessment.assessment_rationale}",
                "",
                "## Independent decision audit",
                "",
                f"- Passed: **{audit.audit_passed}**",
                f"- Abstention recommended: **{audit.abstention_recommended}**",
                f"- Required corrections: {audit.required_corrections or 'none'}",
                "",
                "## Grading contract",
                "",
                "The current output is a categorical single-agent vote or a contiguous plausible "
                "vote set. Its compatibility code/range is not calibrated to the aggregate human "
                "expert_score and is not a probability. "
                "A formal official_expert_score remains null until labelled Q1 calibration and "
                "independent validation are completed.",
                "",
                "The visual layer remains frozen. No FITS reading, foreground subtraction, SEP, "
                "S/N-threshold, geometry-route, recovery, handoff, or diagnostic logic was rerun here.",
            ]
        )
        state.final_report_markdown = markdown
        state.final_decision = state.to_dict()
        state.final_decision["report_markdown"] = markdown
        self.record_reasoning(
            state,
            "report_generation",
            {"scientific_reasoning_added": False, "audit_passed": audit.audit_passed},
        )
        state.final_decision["reasoning_trace"] = state.reasoning_trace
        return state
