from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState
from ..schemas.reasoning import HypothesisAnalysis, MissingEvidence


class HypothesisAnalysisAgent(BaseAgent):
    """Build one explicit case file for every competing explanation."""

    def __init__(self):
        super().__init__("HypothesisAnalysisAgent")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("分别汇总每个假设的支持、反对、缺失与不确定性")
        document = state.require_observation()
        measurement_by_kind: dict[str, list] = {}
        for measurement in document.measurements:
            measurement_by_kind.setdefault(measurement.kind, []).append(measurement)

        analyses: dict[str, HypothesisAnalysis] = {}
        for hypothesis in state.hypotheses:
            items = state.evidence.get(hypothesis.id, [])
            supporting = [
                item.id
                for item in items
                if item.status == "evaluated" and item.polarity == "support"
            ]
            opposing = [
                item.id
                for item in items
                if item.status == "evaluated" and item.polarity == "oppose"
            ]
            questions = [
                item
                for item in state.evidence_plan
                if hypothesis.id in item.target_hypotheses
            ]
            missing: list[MissingEvidence] = []
            relevant_measurement_ids: set[str] = set()
            for question in questions:
                candidates = [
                    measurement
                    for kind in question.measurement_kinds
                    for measurement in measurement_by_kind.get(kind, [])
                ]
                relevant_measurement_ids.update(item.id for item in candidates)
                available = any(
                    item.value is not None and item.reliability > 0 for item in candidates
                )
                if not available:
                    missing.append(
                        MissingEvidence(
                            question_id=question.id,
                            question=question.question,
                            required_measurement_kinds=question.measurement_kinds,
                            reason=question.reason,
                        )
                    )
            not_available_claims = [
                item.claim for item in items if item.status == "not_available"
            ]
            explained = {
                ref
                for item in items
                if item.status == "evaluated"
                for ref in item.observation_refs
            }
            if not supporting and not opposing:
                status = "insufficient"
            elif opposing and not supporting:
                status = "weakened"
            else:
                status = "viable"
            uncertainties = list(dict.fromkeys(not_available_claims + document.limitations))
            analyses[hypothesis.id] = HypothesisAnalysis(
                hypothesis_id=hypothesis.id,
                hypothesis=hypothesis.name,
                status=status,
                supporting_evidence_refs=supporting,
                opposing_evidence_refs=opposing,
                missing_evidence=missing,
                uncertainties=uncertainties,
                explained_observation_refs=sorted(explained),
                unexplained_relevant_observation_refs=sorted(
                    relevant_measurement_ids - explained
                ),
            )

        state.hypothesis_analyses = analyses
        self.record_reasoning(
            state,
            "multiple_hypothesis_analysis",
            {
                hypothesis_id: {
                    "status": analysis.status,
                    "support_count": len(analysis.supporting_evidence_refs),
                    "opposition_count": len(analysis.opposing_evidence_refs),
                    "missing_count": len(analysis.missing_evidence),
                }
                for hypothesis_id, analysis in analyses.items()
            },
        )
        return state
