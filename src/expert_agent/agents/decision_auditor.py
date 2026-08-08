from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.config import ReasoningConfig
from ..core.state import ExpertState
from ..schemas.reasoning import AuditFinding, DecisionAuditResult


class DecisionAuditor(BaseAgent):
    """Challenge the proposed decision and apply downgrade/abstention gates."""

    _LEVELS = ["insufficient", "low", "medium", "high"]

    def __init__(self, config: ReasoningConfig | None = None):
        super().__init__("DecisionAuditor")
        self.config = config or ReasoningConfig()

    def run(self, state: ExpertState) -> ExpertState:
        self.log("审计最终结论并主动寻找降级或弃权理由")
        self.load_skill("decision_audit_skill")
        topics = ["lens_geometry", "intrinsic_galaxy_morphology", "merger_interaction", "artifacts_subtraction"]
        self.load_knowledge_bundle(topics)
        decision = state.decision_result
        confidence = state.confidence_assessment
        if decision is None or confidence is None or state.observation_audit is None or state.evidence_audit is None:
            raise ValueError("DecisionAuditor requires decision, confidence, and preceding scientific audits")
        findings: list[AuditFinding] = []
        alternatives_considered = len(decision.remaining_competitors) >= 3 if decision.selected_hypothesis else len(decision.ranking) >= 3
        selected_id = decision.ranking[0].hypothesis_id if decision.selected_hypothesis and decision.ranking else None
        selected_analysis = state.hypothesis_analyses.get(selected_id or "")
        high_priority_questions = {q.id for q in state.evidence_plan if q.priority == "high"}
        high_missing = [m.question_id for m in (selected_analysis.missing_evidence if selected_analysis else []) if m.question_id in high_priority_questions]
        has_counterevidence = bool(state.contradictions.get(selected_id or ""))
        scientific_inputs_valid = state.observation_audit.valid and state.evidence_audit.valid
        threshold_consistent = decision.selected_hypothesis is None or (bool(decision.ranking) and decision.ranking[0].final_score >= self.config.decision_min_score and decision.score_margin >= self.config.decision_min_margin)
        if not alternatives_considered:
            findings.append(AuditFinding(id="DA-ALTERNATIVES", severity="critical", category="alternative_explanations", description="The decision did not retain enough competing explanations.", recommended_action="Re-run synthesis with all four domain hypotheses."))
        if not scientific_inputs_valid:
            findings.append(AuditFinding(id="DA-UPSTREAM", severity="critical", category="upstream_quality", description="Observation or evidence audit contains a blocking failure.", recommended_action="Abstain until the upstream scientific audit passes."))
        if not threshold_consistent:
            findings.append(AuditFinding(id="DA-THRESHOLD", severity="critical", category="decision_consistency", description="A hypothesis was selected without clearing declared score and margin thresholds.", recommended_action="Abstain or correct the decision synthesis."))
        if high_missing:
            findings.append(AuditFinding(id="DA-MISSING", severity="warning", category="missing_discriminating_evidence", description="High-priority evidence remains missing for the preferred hypothesis.", hypothesis_ids=[selected_id] if selected_id else [], recommended_action="Downgrade confidence and obtain the missing measurements."))
        if has_counterevidence:
            findings.append(AuditFinding(id="DA-COUNTER", severity="warning", category="counterexample", description="The preferred explanation retains explicit counter-evidence.", hypothesis_ids=[selected_id] if selected_id else [], evidence_refs=[item.id for item in state.contradictions.get(selected_id or "", [])], recommended_action="Keep the contradiction visible and avoid high confidence unless independently resolved."))
        low_completeness = confidence.completeness < self.config.decision_downgrade_completeness
        if low_completeness:
            findings.append(AuditFinding(id="DA-COVERAGE", severity="warning", category="evidence_coverage", description="Evidence-plan coverage is below the decision-audit threshold.", recommended_action="Downgrade confidence until more planned checks are answered."))

        blocking = not scientific_inputs_valid or not alternatives_considered or not threshold_consistent
        if blocking:
            verdict = "abstain"
            effective_classification = "indeterminate"
            effective_level = "insufficient"
        else:
            effective_classification = decision.classification
            downgrade = bool(high_missing or has_counterevidence or low_completeness)
            verdict = "downgrade" if downgrade else "confirm"
            level_index = self._LEVELS.index(confidence.level)
            effective_level = self._LEVELS[max(0, level_index - 1)] if downgrade else confidence.level
        checks = {"alternatives_considered": alternatives_considered, "upstream_scientific_audits": scientific_inputs_valid, "selection_thresholds": threshold_consistent, "high_priority_evidence_complete": not high_missing, "evidence_coverage": not low_completeness}
        state.decision_audit = DecisionAuditResult(verdict=verdict, effective_classification=effective_classification, effective_confidence_level=effective_level, findings=findings, checks=checks, recommendation=("The proposed classification survives adversarial review." if verdict == "confirm" else "Report the classification with reduced confidence." if verdict == "downgrade" else "Abstain from scientific classification until blocking audit findings are resolved."))
        self.record_reasoning(state, "decision_audit", {**state.decision_audit.model_dump(mode="json"), "knowledge_topics": topics})
        return state
