from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState
from ..schemas.reasoning import (
    CrossHypothesisEvidenceComparison,
    HypothesisEvidenceEffect,
)


class EvidenceComparisonAgent(BaseAgent):
    """Express how the same measured fact affects every competing hypothesis."""

    def __init__(self):
        super().__init__("EvidenceComparisonAgent")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("构建跨假设证据比较矩阵")
        document = state.require_observation()
        measurement_map = document.measurement_map()
        grouped: dict[tuple[tuple[str, ...], str], list] = defaultdict(list)
        for items in state.evidence.values():
            for item in items:
                if item.status != "evaluated" or not item.observation_refs:
                    continue
                key = (tuple(sorted(item.observation_refs)), item.correlation_group)
                grouped[key].append(item)

        comparisons: list[CrossHypothesisEvidenceComparison] = []
        for serial, ((observation_refs, group), items) in enumerate(
            sorted(grouped.items()), start=1
        ):
            descriptions = [
                measurement_map[ref].description
                for ref in observation_refs
                if ref in measurement_map
            ]
            observed_fact = "; ".join(descriptions) or ", ".join(observation_refs)
            effects: list[HypothesisEvidenceEffect] = []
            represented_effects: set[str] = set()
            for hypothesis in state.hypotheses:
                hypothesis_items = [
                    item for item in items if item.hypothesis_id == hypothesis.id
                ]
                if not hypothesis_items:
                    effects.append(
                        HypothesisEvidenceEffect(
                            hypothesis_id=hypothesis.id,
                            effect="unknown",
                            weighted_effect=0.0,
                            explanation="No current reasoning rule maps this fact to the hypothesis.",
                        )
                    )
                    continue
                polarities = {item.polarity for item in hypothesis_items}
                if len(polarities) == 1:
                    effect = next(iter(polarities))
                else:
                    effect = "neutral"
                represented_effects.add(effect)
                effects.append(
                    HypothesisEvidenceEffect(
                        hypothesis_id=hypothesis.id,
                        effect=effect,
                        evidence_refs=[item.id for item in hypothesis_items],
                        weighted_effect=float(
                            np.clip(
                                np.mean([item.weighted_score for item in hypothesis_items]),
                                -1.0,
                                1.0,
                            )
                        ),
                        explanation=" ".join(item.rationale for item in hypothesis_items),
                    )
                )
            comparisons.append(
                CrossHypothesisEvidenceComparison(
                    id=f"X{serial}",
                    observation_refs=list(observation_refs),
                    observed_fact=observed_fact,
                    correlation_group=group,
                    effects=effects,
                    discriminating=(
                        len({effect for effect in represented_effects if effect != "neutral"})
                        >= 2
                    ),
                )
            )

        state.evidence_comparisons = comparisons
        self.record_reasoning(
            state,
            "cross_hypothesis_evidence_comparison",
            {
                "comparison_count": len(comparisons),
                "discriminating_count": sum(item.discriminating for item in comparisons),
                "all_hypotheses_represented": all(
                    len(item.effects) == len(state.hypotheses) for item in comparisons
                ),
            },
        )
        return state
