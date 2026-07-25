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


        self.load_skill(
            "confidence_calibration_skill"
        )


        hypotheses = state.hypotheses

        evidence = state.evidence

        contradictions = state.contradictions



        supporting_factors: List[Dict[str, Any]] = []

        limiting_factors: List[Dict[str, Any]] = []

        remaining_uncertainties: List[str] = []



        # =================================
        # v2.1:
        # 根据证据和反证数量选择候选
        #
        # 当前为基础评分框架。
        #
        # 未来：
        # 使用模型根据证据质量动态评分。
        # =================================


        hypothesis_scores = {}



        for hypothesis in hypotheses:


            hypothesis_id = (
                hypothesis["id"]
            )


            support_count = len(
                evidence.get(
                    hypothesis_id,
                    []
                )
            )


            contradiction_count = len(
                contradictions.get(
                    hypothesis_id,
                    []
                )
            )


            score = (
                support_count
                -
                contradiction_count
            )


            hypothesis_scores[
                hypothesis_id
            ] = score



        if hypotheses:


            best_hypothesis_id = max(
                hypothesis_scores,
                key=hypothesis_scores.get
            )


            selected_hypothesis = next(

                h for h in hypotheses

                if h["id"] == best_hypothesis_id

            )


            final_hypothesis = (
                selected_hypothesis.get(
                    "name",
                    "unknown"
                )
            )


            previous_confidence = (
                selected_hypothesis.get(
                    "initial_confidence",
                    "unknown"
                )
            )


        else:


            final_hypothesis = "unknown"

            previous_confidence = "unknown"



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



        # 当前阶段：
        # 由于真实证据尚未接入，
        # 保持中等可信度。

        confidence_level = (
            "medium"
        )



        confidence_result = {


            "final_hypothesis":
                final_hypothesis,


            "previous_confidence":
                previous_confidence,


            "hypothesis_scores":
                hypothesis_scores,


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



        state.confidence_assessment = (
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