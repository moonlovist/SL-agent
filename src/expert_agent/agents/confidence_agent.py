from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class ConfidenceAgent(BaseAgent):
    """
    Confidence calibration Agent.

    Simulates expert judgement:

    "How strongly should I believe this conclusion?"

    Responsibilities:

    1. Evaluate evidence balance.
    2. Consider contradiction strength.
    3. Calculate hypothesis confidence.
    4. Preserve uncertainty.

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
        # v2.2 Evidence Scoring
        #
        # 使用证据权重计算候选评分。
        #
        # 当前：
        # 使用support_score。
        #
        # 未来：
        # 使用:
        # strength *
        # reliability *
        # specificity
        # =================================


        hypothesis_scores = {}

        evidence_score_breakdown = {}



        for hypothesis in hypotheses:


            hypothesis_id = (
                hypothesis["id"]
            )


            positive_score = 0

            negative_score = 0



            # 计算支持证据

            for item in evidence.get(
                hypothesis_id,
                []
            ):


                support_score = (
                    item.get(
                        "support_score"
                    )
                )


                if support_score is not None:

                    positive_score += (
                        support_score
                    )



            # 计算反证影响

            for item in contradictions.get(
                hypothesis_id,
                []
            ):


                severity = (
                    item.get(
                        "severity"
                    )
                )


                if severity is not None:

                    negative_score += (
                        severity
                    )



            final_score = (
                positive_score
                -
                negative_score
            )


            hypothesis_scores[
                hypothesis_id
            ] = final_score



            evidence_score_breakdown[
                hypothesis_id
            ] = {


                "positive_evidence_score":
                    positive_score,


                "negative_contradiction_score":
                    negative_score,


                "final_score":
                    final_score

            }



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



        remaining_uncertainties.extend(

            [

                "Evidence scores require further evaluation.",

                "Alternative explanations have not been fully excluded."

            ]

        )



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


            "evidence_score_breakdown":
                evidence_score_breakdown,


            "confidence_level":
                confidence_level,


            "confidence_reasoning":
                (
                    "Confidence is calibrated "
                    "based on weighted evidence, "
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