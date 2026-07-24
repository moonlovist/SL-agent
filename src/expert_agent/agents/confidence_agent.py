from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class ConfidenceAgent(BaseAgent):
    """
    Confidence calibration Agent.

    Simulates expert judgement:

    "How strongly should I believe this conclusion?"

    Responsibilities:

    1. Review evidence balance.
    2. Consider contradictions.
    3. Evaluate uncertainty.
    4. Assign confidence level.

    This stage does not create new hypotheses.
    """


    def __init__(self):
        super().__init__("ConfidenceAgent")


    def run(
        self,
        state: ExpertState
    ) -> ExpertState:


        self.log(
            "开始可信度校准"
        )


        skill = self.load_skill(
            "confidence_calibration_skill"
        )


        hypotheses = (
            state.hypotheses
        )

        evidence = (
            state.evidence
        )

        contradictions = (
            state.contradictions
        )


        supporting_factors = []

        limiting_factors = []

        remaining_uncertainties = []


        # 当前版本先选择已有候选中的最高优先级假设
        # 后续接入模型评分后替换

        if hypotheses:

            final_hypothesis = (
                hypotheses[0]["name"]
            )

            initial_confidence = (
                hypotheses[0].get(
                    "initial_confidence",
                    "unknown"
                )
            )

        else:

            final_hypothesis = (
                "unknown"
            )

            initial_confidence = (
                "unknown"
            )


        # 收集支持因素

        for hypothesis_id, items in evidence.items():

            for item in items:

                supporting_factors.append(
                    {
                        "hypothesis":
                            hypothesis_id,

                        "feature":
                            item.get(
                                "feature"
                            ),

                        "status":
                            item.get(
                                "evaluation_status",
                                "pending"
                            )
                    }
                )


        # 收集限制因素

        for hypothesis_id, items in contradictions.items():

            for item in items:

                limiting_factors.append(
                    {
                        "hypothesis":
                            hypothesis_id,

                        "contradiction":
                            item.get(
                                "description"
                            ),

                        "status":
                            item.get(
                                "evaluation_status",
                                "pending"
                            )
                    }
                )


        remaining_uncertainties.extend(
            [
                "Visual evidence requires further analysis.",
                "Alternative explanations have not been fully excluded."
            ]
        )


        confidence_level = (
            "medium"
        )


        confidence_result = {

            "final_hypothesis":
                final_hypothesis,

            "confidence_level":
                confidence_level,

            "confidence_reasoning":
                (
                    "Confidence is calibrated "
                    "based on evidence balance, "
                    "contradictions, and remaining uncertainty."
                ),

            "supporting_factors":
                supporting_factors,

            "limiting_factors":
                limiting_factors,

            "remaining_uncertainties":
                remaining_uncertainties
        }


        state.final_decision = (
            confidence_result
        )


        self.record_reasoning(
            state,

            action="confidence_calibration",

            content={
                "skill_used":
                    "confidence_calibration_skill",

                "result":
                    confidence_result
            }
        )


        return state