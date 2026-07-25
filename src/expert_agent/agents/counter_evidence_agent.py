from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class CounterEvidenceAgent(BaseAgent):
    """
    Counter evidence analysis Agent.

    Simulates expert reasoning:

    "Why might my current judgement be wrong?"

    Responsibilities:

    1. Search for important contradictions.
    2. Identify risks for current hypotheses.
    3. Reduce confirmation bias.

    This stage does not make final decisions.
    """


    def __init__(self):
        super().__init__("CounterEvidenceAgent")



    def run(
        self,
        state: ExpertState
    ) -> ExpertState:


        self.log(
            "开始寻找反证"
        )


        hypotheses = (
            state.hypotheses
        )


        evidence_plan = (
            state.evidence_plan
        )


        contradictions: Dict[
            str,
            List[Dict[str, Any]]
        ] = {}



        planned_questions = []


        if evidence_plan:

            planned_questions = (
                evidence_plan.get(
                    "priority_questions",
                    []
                )
            )



        for hypothesis in hypotheses:


            hypothesis_id = (
                hypothesis["id"]
            )


            hypothesis_name = (
                hypothesis["name"]
            )


            contradiction_items = []



            # =========================
            # v2.1:
            # 根据优先证据规划寻找风险
            # =========================

            if planned_questions:


                for question in planned_questions:


                    if hypothesis_id in (
                        question.get(
                            "target_hypotheses",
                            []
                        )
                    ):


                        contradiction_items.append(

                            {
                                "type":
                                    "planned_contradiction",


                                "related_hypothesis":
                                    hypothesis_name,


                                "description":
                                    (
                                        "Failure of planned evidence: "
                                        +
                                        question["question"]
                                    ),


                                "reasoning":
                                    (
                                        "This missing or negative "
                                        "evidence may weaken "
                                        "the hypothesis."
                                    ),


                                "severity":
                                    None,


                                "confidence":
                                    None,


                                "evaluation_status":
                                    "pending"

                            }

                        )



            # =========================
            # fallback:
            # 保留v2.0行为
            # =========================

            else:


                possible_contradictions = (
                    hypothesis.get(
                        "possible_contradictions",
                        []
                    )
                )


                for item in possible_contradictions:


                    contradiction_items.append(

                        {
                            "type":
                                "potential_contradiction",


                            "related_hypothesis":
                                hypothesis_name,


                            "description":
                                item,


                            "reasoning":
                                (
                                    "This observation "
                                    "may weaken the hypothesis "
                                    "if confirmed."
                                ),


                            "severity":
                                None,


                            "confidence":
                                None,


                            "evaluation_status":
                                "pending"
                        }

                    )



            contradictions[hypothesis_id] = (
                contradiction_items
            )



        state.contradictions = contradictions



        self.record_reasoning(

            state,

            action="counter_evidence_analysis",

            content={

                "used_evidence_plan":
                    bool(evidence_plan),

                "number_of_hypotheses":
                    len(hypotheses),

                "contradictions":
                    contradictions
            }
        )


        return state