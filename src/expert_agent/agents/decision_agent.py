from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState



class DecisionAgent(BaseAgent):
    """
    Decision Agent.

    Selects the currently best supported hypothesis.

    Responsibilities:

    1. Compare candidate hypotheses.
    2. Use evidence scoring results.
    3. Consider contradictions.
    4. Produce a ranked decision.

    This stage does not evaluate confidence.
    """



    def __init__(self):
        super().__init__("DecisionAgent")



    def run(
        self,
        state: ExpertState
    ) -> ExpertState:


        self.log(
            "开始选择最佳解释"
        )


        self.load_skill(
            "decision_making_skill"
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



        ranking: List[Dict[str, Any]] = []



        # =================================
        # 根据证据评分进行假设比较
        #
        # 当前版本：
        # 使用已有support_score
        #
        # 未来：
        # 接入LLM/Vision评分
        # =================================


        for hypothesis in hypotheses:


            hypothesis_id = (
                hypothesis["id"]
            )


            hypothesis_name = (
                hypothesis["name"]
            )


            support_score = 0

            contradiction_score = 0



            # 支持证据

            for item in evidence.get(
                hypothesis_id,
                []
            ):


                score = (
                    item.get(
                        "support_score"
                    )
                )


                if score is not None:

                    support_score += score



            # 反证影响

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

                    contradiction_score += severity



            final_score = (
                support_score
                -
                contradiction_score
            )



            ranking.append(

                {

                    "hypothesis_id":
                        hypothesis_id,


                    "hypothesis":
                        hypothesis_name,


                    "support_score":
                        support_score,


                    "contradiction_score":
                        contradiction_score,


                    "final_score":
                        final_score

                }

            )



        # 排序

        ranking = sorted(

            ranking,

            key=lambda x:
                x["final_score"],

            reverse=True

        )



        if ranking:


            selected_hypothesis = (
                ranking[0]["hypothesis"]
            )


        else:


            selected_hypothesis = (
                "unknown"
            )



        decision_result = {


            "selected_hypothesis":
                selected_hypothesis,


            "ranking":
                ranking,


            "decision_reasoning":
                (
                    "The selected hypothesis "
                    "is currently the best supported "
                    "explanation based on evidence "
                    "and contradiction analysis."
                ),


            "remaining_competitors":
                [
                    item["hypothesis"]

                    for item in ranking[1:]
                ]

        }



        state.decision_result = (
            decision_result
        )



        self.record_reasoning(

            state,

            action="decision_making",

            content={

                "skill_used":
                    "decision_making_skill",


                "result":
                    decision_result

            }

        )



        return state