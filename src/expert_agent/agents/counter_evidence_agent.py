from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState
from ..schemas.reasoning import ContradictionItem


class CounterEvidenceAgent(BaseAgent):
    def __init__(self):
        super().__init__("CounterEvidenceAgent")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("将不利观测转化为显式反证")
        self.load_skill("contradiction_skill")
        contradictions: dict[str, list[ContradictionItem]] = {
            hypothesis.id: [] for hypothesis in state.hypotheses
        }
        serial = 0
        for hypothesis_id, items in state.evidence.items():
            for item in items:
                if item.status != "evaluated" or item.polarity != "oppose":
                    continue
                serial += 1
                contradictions[hypothesis_id].append(
                    ContradictionItem(
                        id=f"C{serial}",
                        hypothesis_id=hypothesis_id,
                        description=item.claim,
                        observation_refs=item.observation_refs,
                        severity=item.strength * item.specificity,
                        reliability=item.reliability,
                        correlation_group=item.correlation_group,
                        rationale=item.rationale,
                    )
                )
        state.contradictions = contradictions
        self.record_reasoning(
            state,
            "counter_evidence_analysis",
            {
                "contradiction_counts": {
                    key: len(items) for key, items in contradictions.items()
                }
            },
        )
        return state
