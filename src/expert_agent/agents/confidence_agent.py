from __future__ import annotations

import numpy as np

from ..core.base_agent import BaseAgent
from ..core.config import ReasoningConfig
from ..core.state import ExpertState
from ..schemas.reasoning import ConfidenceAssessment


class ConfidenceAgent(BaseAgent):
    def __init__(self, config: ReasoningConfig | None = None):
        super().__init__("ConfidenceAgent")
        self.config = config or ReasoningConfig()

    def run(self, state: ExpertState) -> ExpertState:
        self.log("按证据强度、覆盖度和质量校准可信度")
        self.load_skill("confidence_calibration_skill")
        document = state.require_observation()
        decision = state.decision_result
        if decision is None:
            raise ValueError("ConfidenceAgent requires a decision")
        available_kinds = {item.kind for item in document.measurements if item.value is not None}
        answered = sum(
            any(kind in available_kinds for kind in question.measurement_kinds)
            for question in state.evidence_plan
        )
        completeness = answered / len(state.evidence_plan) if state.evidence_plan else 0.0
        quality = 1.0 if document.data_quality.usable else 0.4
        if not document.data_quality.model_success:
            quality *= 0.75
        top_score = decision.ranking[0].final_score if decision.ranking else 0.0
        positive_score = max(0.0, top_score)
        calibrated = float(
            np.clip(
                0.45 * positive_score
                + 0.25 * min(1.0, decision.score_margin * 3.0)
                + 0.20 * completeness
                + 0.10 * quality,
                0.0,
                1.0,
            )
        )
        if decision.selected_hypothesis is None:
            level = "insufficient" if completeness < 0.6 or positive_score < 0.1 else "low"
        elif calibrated >= self.config.high_confidence_score and completeness >= 0.8:
            level = "high"
        elif calibrated >= self.config.medium_confidence_score:
            level = "medium"
        else:
            level = "low"
        selected_id = decision.ranking[0].hypothesis_id if decision.selected_hypothesis else None
        support_refs = [
            item.id
            for item in state.evidence.get(selected_id or "", [])
            if item.status == "evaluated" and item.polarity == "support"
        ]
        limiting = list(document.data_quality.flags)
        if not document.data_quality.supplied_rms:
            limiting.append("noise_map_was_estimated")
        limiting.extend(
            item.description
            for item in state.contradictions.get(selected_id or "", [])
        )
        uncertainties = list(document.limitations)
        unanswered = [
            question.question
            for question in state.evidence_plan
            if not any(kind in available_kinds for kind in question.measurement_kinds)
        ]
        uncertainties.extend(f"Unanswered: {question}" for question in unanswered)
        state.confidence_assessment = ConfidenceAssessment(
            level=level,
            score=calibrated,
            completeness=completeness,
            reasoning=(
                f"Confidence combines the preferred score, score margin, evidence-plan "
                f"coverage ({completeness:.0%}), and data quality; it is not a calibrated lens probability."
            ),
            supporting_evidence_refs=support_refs,
            limiting_factors=limiting,
            remaining_uncertainties=uncertainties,
        )
        self.record_reasoning(
            state,
            "confidence_calibration",
            {"level": level, "score": calibrated, "completeness": completeness},
        )
        return state
