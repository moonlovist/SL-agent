from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class ReportWriter(BaseAgent):
    def __init__(self):
        super().__init__("ReportWriter")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("生成可审计专家报告")
        self.load_skill("report_generation_skill")
        document = state.require_observation()
        decision = state.decision_result
        confidence = state.confidence_assessment
        audit = state.audit_result
        if decision is None or confidence is None or audit is None:
            raise ValueError("ReportWriter requires decision, confidence, and audit results")
        effective_classification = decision.classification if audit.valid else "indeterminate"
        top_rows = decision.ranking
        structure_lines = [
            (
                f"- {item.id}: radius={item.radius_px:.2f} px, PA={item.position_angle_deg:.1f}°, "
                f"S/N={item.integrated_snr:.1f}, extent={item.angular_extent_deg:.1f}°, "
                f"tangential misalignment={item.tangential_misalignment_deg:.1f}°"
            )
            for item in document.structures
        ] or ["- No significant annular structure was measured."]
        hypothesis_lines = [
            (
                f"- {row.hypothesis_id} `{row.hypothesis}`: evidence={row.evidence_score:.3f}, "
                f"contradiction={row.contradiction_penalty:.3f}, final={row.final_score:.3f}"
            )
            for row in top_rows
        ]
        selected_id = top_rows[0].hypothesis_id if decision.selected_hypothesis else None
        support_lines = [
            f"- {item.id}: {item.claim} (refs: {', '.join(item.observation_refs)})"
            for item in state.evidence.get(selected_id or "", [])
            if item.status == "evaluated" and item.polarity == "support"
        ] or ["- No hypothesis-specific supporting evidence cleared evaluation."]
        contradiction_lines = [
            f"- {item.id} [{item.hypothesis_id}]: {item.description} "
            f"(refs: {', '.join(item.observation_refs)})"
            for items in state.contradictions.values()
            for item in items
        ] or ["- No explicit contradictory measurement was identified."]
        limitation_lines = [f"- {item}" for item in confidence.remaining_uncertainties]
        markdown = "\n".join(
            [
                f"# Strong-lens expert report: {document.object_id}",
                "",
                "## Data reliability",
                "",
                f"- Usable: {document.data_quality.usable}",
                f"- Band: {document.band}",
                f"- Foreground model: {document.data_quality.model_method} "
                f"(success={document.data_quality.model_success})",
                f"- Supplied RMS map: {document.data_quality.supplied_rms}",
                "",
                "## Structured observations",
                "",
                *structure_lines,
                "",
                "## Competing hypotheses",
                "",
                *hypothesis_lines,
                "",
                "## Supporting evidence for the preferred explanation",
                "",
                *support_lines,
                "",
                "## Contradictions by hypothesis",
                "",
                *contradiction_lines,
                "",
                "## Expert judgement",
                "",
                f"- Classification: **{effective_classification}**",
                f"- Preferred explanation: {decision.selected_hypothesis or 'none (abstained)'}",
                f"- Confidence: **{confidence.level}** ({confidence.score:.3f})",
                f"- Decision basis: {decision.decision_reasoning}",
                f"- Audit valid: {audit.valid}",
                "",
                "## Remaining uncertainty",
                "",
                *limitation_lines,
                "",
                "This report is an evidence-based candidate assessment, not a calibrated lens probability or a substitute for expert review.",
            ]
        )
        state.final_report_markdown = markdown
        state.final_decision = {
            "object_id": document.object_id,
            "classification": effective_classification,
            "decision": decision.model_dump(mode="json"),
            "confidence": confidence.model_dump(mode="json"),
            "audit": audit.model_dump(mode="json"),
            "report_markdown": markdown,
        }
        self.record_reasoning(
            state,
            "report_generation",
            {"classification": effective_classification, "audit_valid": audit.valid},
        )
        return state
