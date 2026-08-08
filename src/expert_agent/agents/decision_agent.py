from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..core.base_agent import BaseAgent
from ..core.config import ReasoningConfig
from ..core.state import ExpertState
from ..schemas.reasoning import DecisionResult, HypothesisScore


class DecisionAgent(BaseAgent):
    def __init__(self, config: ReasoningConfig | None = None):
        super().__init__("DecisionAgent")
        self.config = config or ReasoningConfig()

    @staticmethod
    def _mean_by_group(values: dict[str, list[float]]) -> float:
        if not values:
            return 0.0
        group_scores = [float(np.mean(group)) for group in values.values() if group]
        return float(np.mean(group_scores)) if group_scores else 0.0

    def run(self, state: ExpertState) -> ExpertState:
        self.log("比较假设并允许证据不足时弃权")
        self.load_skill("decision_making_skill")
        if set(state.hypothesis_analyses) != {item.id for item in state.hypotheses}:
            raise ValueError("DecisionAgent requires one completed analysis per hypothesis")
        ranking: list[HypothesisScore] = []
        for hypothesis in state.hypotheses:
            supporting: dict[str, list[float]] = defaultdict(list)
            for item in state.evidence.get(hypothesis.id, []):
                if item.status == "evaluated" and item.polarity == "support":
                    supporting[item.correlation_group].append(max(0.0, item.weighted_score))
            contradicting: dict[str, list[float]] = defaultdict(list)
            for item in state.contradictions.get(hypothesis.id, []):
                contradicting[item.correlation_group].append(
                    item.severity * item.reliability
                )
            evidence_score = self._mean_by_group(supporting)
            contradiction_penalty = self._mean_by_group(contradicting)
            final_score = float(np.clip(evidence_score - contradiction_penalty, -1.0, 1.0))
            ranking.append(
                HypothesisScore(
                    hypothesis_id=hypothesis.id,
                    hypothesis=hypothesis.name,
                    evidence_score=evidence_score,
                    contradiction_penalty=contradiction_penalty,
                    final_score=final_score,
                )
            )
        ranking.sort(key=lambda item: item.final_score, reverse=True)
        top = ranking[0] if ranking else None
        second = ranking[1] if len(ranking) > 1 else None
        margin = max(0.0, (top.final_score - second.final_score) if top and second else 0.0)
        selected = None
        if (
            top
            and top.final_score >= self.config.decision_min_score
            and margin >= self.config.decision_min_margin
        ):
            selected = top.hypothesis
        signature_gate_passed = True
        if selected == "strong_gravitational_lens":
            signature_gate_passed = any(
                item.status == "evaluated"
                and item.polarity == "support"
                and item.specificity >= 0.6
                and item.correlation_group in {"lens_geometry", "scene_geometry"}
                for item in state.evidence.get("H1", [])
            )
            if not signature_gate_passed:
                selected = None
        if selected == "strong_gravitational_lens":
            classification = "strong_lens_candidate"
        elif selected is None:
            classification = "indeterminate"
        else:
            classification = "not_lens_preferred"

        selected_id = top.hypothesis_id if selected and top else None
        evidence_refs = []
        if selected_id:
            evidence_refs.extend(
                item.id
                for item in state.evidence.get(selected_id, [])
                if item.status == "evaluated" and item.polarity == "support"
            )
            evidence_refs.extend(
                item.id for item in state.contradictions.get(selected_id, [])
            )
        if selected:
            reasoning = (
                f"{selected} is preferred with score {top.final_score:.3f} and "
                f"margin {margin:.3f}; correlated evidence groups were aggregated once."
            )
        elif top and top.hypothesis == "strong_gravitational_lens" and not signature_gate_passed:
            reasoning = (
                f"The lens hypothesis led numerically (score {top.final_score:.3f}) but lacked "
                "a sufficiently specific geometry measurement; non-specific S/N and extent "
                "evidence cannot trigger a lens decision, so the system abstains."
            )
        else:
            top_score = top.final_score if top else 0.0
            reasoning = (
                f"No hypothesis cleared both the minimum score and separation thresholds "
                f"(top score {top_score:.3f}, margin {margin:.3f}); the system abstains."
            )
        state.decision_result = DecisionResult(
            classification=classification,
            selected_hypothesis=selected,
            ranking=ranking,
            score_margin=margin,
            decision_reasoning=reasoning,
            evidence_refs=evidence_refs,
            remaining_competitors=[
                item.hypothesis for item in ranking if item.hypothesis != selected
            ],
        )
        self.record_reasoning(
            state,
            "hypothesis_synthesis_and_decision",
            {
                "classification": classification,
                "selected_hypothesis": selected,
                "top_score": top.final_score if top else None,
                "margin": margin,
                "abstained": selected is None,
                "signature_gate_passed": signature_gate_passed,
            },
        )
        return state
