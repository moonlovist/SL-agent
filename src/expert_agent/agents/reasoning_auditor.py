from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState
from ..schemas.reasoning import AuditResult


class ReasoningAuditor(BaseAgent):
    def __init__(self):
        super().__init__("ReasoningAuditor")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("审计证据来源与决策约束")
        document = state.require_observation()
        issues: list[str] = []
        measurement_ids = [item.id for item in document.measurements]
        evidence_ids = [item.id for items in state.evidence.values() for item in items]
        contradiction_ids = [
            item.id for items in state.contradictions.values() for item in items
        ]
        unique_measurements = len(measurement_ids) == len(set(measurement_ids))
        if not unique_measurements:
            issues.append("Observation contains duplicate measurement IDs.")
        known_measurements = set(measurement_ids)
        valid_evidence_refs = all(
            set(item.observation_refs).issubset(known_measurements)
            and (item.observation_refs or item.status == "not_available")
            for items in state.evidence.values()
            for item in items
        )
        if not valid_evidence_refs:
            issues.append("One or more evidence claims lack valid observation references.")
        valid_contradiction_refs = all(
            item.observation_refs
            and set(item.observation_refs).issubset(known_measurements)
            for items in state.contradictions.values()
            for item in items
        )
        if not valid_contradiction_refs:
            issues.append("One or more contradictions lack valid observation references.")
        has_competition = len(state.hypotheses) >= 3
        if not has_competition:
            issues.append("Too few competing hypotheses were considered.")
        has_evaluated_evidence = any(
            item.status == "evaluated"
            for items in state.evidence.values()
            for item in items
        )
        if not has_evaluated_evidence:
            issues.append("No observation was converted into evaluated evidence.")
        decision = state.decision_result
        decision_valid = decision is not None
        if not decision_valid:
            issues.append("Decision stage did not run.")
        elif decision.selected_hypothesis is not None:
            valid_decision_refs = set(decision.evidence_refs).issubset(
                set(evidence_ids) | set(contradiction_ids)
            ) and bool(decision.evidence_refs)
            if not valid_decision_refs:
                decision_valid = False
                issues.append("Selected hypothesis lacks valid evidence references.")
            if decision.score_margin <= 0:
                decision_valid = False
                issues.append("A tied hypothesis was selected instead of abstaining.")
        confidence_valid = state.confidence_assessment is not None
        if not confidence_valid:
            issues.append("Confidence calibration did not run.")
        checks = {
            "observation_schema": True,
            "unique_measurement_ids": unique_measurements,
            "competing_hypotheses": has_competition,
            "evaluated_evidence": has_evaluated_evidence,
            "evidence_provenance": valid_evidence_refs,
            "contradiction_provenance": valid_contradiction_refs,
            "decision_constraints": decision_valid,
            "confidence_present": confidence_valid,
        }
        valid = all(checks.values())
        state.audit_result = AuditResult(
            valid=valid,
            issues=issues,
            checks=checks,
            recommendation=(
                "Reasoning is traceable to measurements; retain stated limitations."
                if valid
                else "Treat the result as indeterminate until the audit issues are resolved."
            ),
        )
        self.record_reasoning(
            state, "reasoning_audit", state.audit_result.model_dump(mode="json")
        )
        return state
