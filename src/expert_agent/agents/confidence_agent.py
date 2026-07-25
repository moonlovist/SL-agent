from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class ConfidenceAgent(BaseAgent):
    """
    Confidence calibration Agent.

    Simulates expert judgement:

    "How strongly should I believe this conclusion?"

    Responsibilities:

    1. Read the selected decision.
    2. Evaluate evidence balance.
    3. Consider contradiction strength.
    4. Preserve uncertainty.

    This stage does not select hypotheses.
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


        evidence = (
            state.evidence
        )


        contradictions = (
            state.contradictions
        )


        decision_result = (
            state.decision_result
        )



        supporting_factors: List[Dict[str, Any]] = []

        limiting_factors: List[Dict[str, Any]] = []

        remaining_uncertainties: List[str] = []



        # =================================
        # V2.3 Decision Separation
        #
        # DecisionAgent:
        # 负责选择最佳解释
        #
        # ConfidenceAgent:
        # 负责判断可信程度
        # =================================


        if decision_result:


            final_hypothesis = (
                decision_result.get(
                    "selected_hypothesis",
                    "unknown"
                )
            )


        else:


            final_hypothesis = (
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


                        "support_score":
                            item.get(
                                "support_score"
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


                        "severity":
                            item.get(
                                "severity"
                            ),


                        "status":
                            item.get(
                                "evaluation_status",
                                "pending"
                            )

                    }

                )



        # 当前版本保留评分摘要

        evidence_score_breakdown = {}



        for hypothesis_id, items in evidence.items():


            total_score = 0


            for item in items:


                score = (
                    item.get(
                        "support_score"
                    )
                )


                if score is not None:

                    total_score += score



            evidence_score_breakdown[

                hypothesis_id

            ] = {


                "support_score":
                    total_score

            }



        remaining_uncertainties.extend(

            [

                "Evidence scores require further evaluation.",

                "Alternative explanations have not been fully excluded."

            ]

        )



        # 当前阶段：
        # 没有真实模型评分时保持框架状态

        confidence_level = (
            "medium"
        )



        confidence_result = {


            "final_hypothesis":
                final_hypothesis,


            "evidence_score_breakdown":
                evidence_score_breakdown,


            "confidence_level":
                confidence_level,


            "confidence_reasoning":
                (
                    "Confidence is calibrated "
                    "after decision selection, "
                    "based on evidence balance, "
                    "contradictions, and uncertainty."
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