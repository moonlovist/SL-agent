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
        configuration_lines = [
            f"- `{item.id}`: **{item.status}** - {item.rationale}"
            for item in dossier.configuration_hypotheses
        ] or ["- None evaluated."]
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
                "### Lens-configuration hypotheses",
                "",
                *configuration_lines,
                "",
                "### Missing information",
                "",
                *missing_lines,
                "",
                "## Provisional expert assessment",
                "",
                f"- Vote category: **{assessment.vote_category}**",
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
                "The current output is one categorical single-agent vote. It does not manufacture "
                "a numeric expert_score from one agent. "
                "A formal official_expert_score remains null until labelled Q1 calibration and "
                "independent validation are completed.",
                "",
                "FITS reading, foreground subtraction, SEP, S/N-threshold detection, and geometry-route "
                "generation remain unchanged. The candidate handoff is recall-preserving: stable "
                "off-route structures enter reasoning with recovery provenance, and near-core/model-fit "
                "cautions do not erase independently surviving signals.",
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
