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
        decision_audit = state.decision_audit
        if decision is None or confidence is None or audit is None or decision_audit is None:
            raise ValueError("ReportWriter requires decision, confidence, and all audit results")
        effective_classification = (
            decision_audit.effective_classification if audit.valid else "indeterminate"
        )
        effective_confidence = (
            decision_audit.effective_confidence_level if audit.valid else "insufficient"
        )
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
        hypothesis_case_lines: list[str] = []
        for hypothesis in state.hypotheses:
            analysis = state.hypothesis_analyses[hypothesis.id]
            missing = ", ".join(item.question_id for item in analysis.missing_evidence) or "none"
            hypothesis_case_lines.extend(
                [
                    f"### {hypothesis.id}: {hypothesis.name}",
                    "",
                    f"- Current status: {analysis.status}",
                    f"- Supporting evidence: {', '.join(analysis.supporting_evidence_refs) or 'none'}",
                    f"- Opposing evidence: {', '.join(analysis.opposing_evidence_refs) or 'none'}",
                    f"- Missing evidence questions: {missing}",
                    f"- Unexplained relevant measurements: {', '.join(analysis.unexplained_relevant_observation_refs) or 'none'}",
                    "",
                ]
            )
        comparison_lines = [
            (
                f"- {item.id}: {item.observed_fact} | "
                + "; ".join(
                    f"{effect.hypothesis_id}={effect.effect}"
                    for effect in item.effects
                )
                + f" | discriminating={item.discriminating}"
            )
            for item in state.evidence_comparisons
        ] or ["- No cross-hypothesis comparison was available."]
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
        trace_labels = {
            "overview_diagnostic": "overview diagnostic",
            "snr_threshold_process": "S/N threshold process",
            "sep_consensus_diagnostic": "SEP consensus process",
            "arc_extraction_diagnostic": "arc-pixel extraction process",
            "vision_manifest": "trace manifest",
            "vision_arrays": "numerical trace arrays",
        }
        trace_lines = [
            f"- {trace_labels[key]}: `{document.artifacts[key]}`"
            for key in trace_labels
            if key in document.artifacts
        ] or ["- No visual trace bundle was written for this run."]
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
                f"- Observation audit: {state.observation_audit.quality_level} "
                f"(valid={state.observation_audit.valid})",
                "",
                "## Vision processing trace",
                "",
                *trace_lines,
                "",
                "## Structured observations",
                "",
                *structure_lines,
                "",
                "## Competing hypotheses",
                "",
                *hypothesis_lines,
                "",
                "## Per-hypothesis analysis",
                "",
                *hypothesis_case_lines,
                "## Cross-hypothesis evidence comparison",
                "",
                *comparison_lines,
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
                f"- Confidence: **{effective_confidence}** (raw score={confidence.score:.3f})",
                f"- Decision basis: {decision.decision_reasoning}",
                f"- Decision audit: {decision_audit.verdict}",
                f"- Structural audit valid: {audit.valid}",
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
            "hypothesis_analyses": {
                key: value.model_dump(mode="json")
                for key, value in state.hypothesis_analyses.items()
            },
            "evidence_comparisons": [
                item.model_dump(mode="json") for item in state.evidence_comparisons
            ],
            "confidence": confidence.model_dump(mode="json"),
            "observation_audit": state.observation_audit.model_dump(mode="json"),
            "evidence_audit": state.evidence_audit.model_dump(mode="json"),
            "decision_audit": decision_audit.model_dump(mode="json"),
            "audit": audit.model_dump(mode="json"),
            "report_markdown": markdown,
        }
        self.record_reasoning(
            state,
            "report_generation",
            {
                "classification": effective_classification,
                "effective_confidence": effective_confidence,
                "decision_audit_verdict": decision_audit.verdict,
                "audit_valid": audit.valid,
            },
        )
        return state
