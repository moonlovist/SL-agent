from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState
from ..schemas.reasoning import EvidenceQuestion


class EvidencePlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("EvidencePlannerAgent")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("规划区分性证据")
        self.load_skill("evidence_planning_skill")
        document = state.require_observation()
        available = {item.kind for item in document.measurements}
        candidates = [
            EvidenceQuestion(
                id="Q1",
                question="Do residual structures share a common radius around the central source?",
                target_hypotheses=["H1", "H2"],
                measurement_kinds=["common_radius_cv", "radius", "structure_count"],
                priority="high",
                reason="A common lens center and radius is more characteristic of lens geometry than an arbitrary internal feature.",
            ),
            EvidenceQuestion(
                id="Q2",
                question="Are the structures extended and tangentially aligned?",
                target_hypotheses=["H1", "H2"],
                measurement_kinds=["tangential_misalignment", "angular_extent", "axis_ratio"],
                priority="high",
                reason="Tangential elongation is a discriminating geometric signature, while radial or compact structure favors alternatives.",
            ),
            EvidenceQuestion(
                id="Q3",
                question="Are the structures significant and stable across detection settings?",
                target_hypotheses=["H1", "H4"],
                measurement_kinds=[
                    "integrated_snr",
                    "threshold_survival_fraction",
                    "sep_survival_fraction",
                    "galaxy_model_success",
                ],
                priority="high",
                reason="Repeatability weakens a pure-noise explanation, but shared preprocessing keeps these measurements correlated.",
            ),
            EvidenceQuestion(
                id="Q4",
                question="Are multiple cores or tidal structures present?",
                target_hypotheses=["H3"],
                measurement_kinds=["multiple_core_count", "tidal_asymmetry"],
                priority="medium",
                reason="These measurements discriminate an interaction from lensing.",
            ),
            EvidenceQuestion(
                id="Q5",
                question="Do candidate images have consistent colors across bands?",
                target_hypotheses=["H1", "H2", "H3"],
                measurement_kinds=["color_consistency"],
                priority="medium",
                reason="Consistent source colors can link multiple images, but the current single-band pipeline may not provide this test.",
            ),
        ]
        state.evidence_plan = sorted(
            candidates,
            key=lambda item: (
                0 if any(kind in available for kind in item.measurement_kinds) else 1,
                {"high": 0, "medium": 1, "low": 2}[item.priority],
            ),
        )
        self.record_reasoning(
            state,
            "evidence_planning",
            {
                "question_ids": [item.id for item in state.evidence_plan],
                "available_measurement_kinds": sorted(available),
            },
        )
        return state
