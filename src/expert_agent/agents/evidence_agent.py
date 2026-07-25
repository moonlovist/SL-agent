from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class EvidenceAgent(BaseAgent):
    """
    Evidence evaluation Agent.

    Simulates expert evidence analysis.

    Responsibilities:

    1. Read evidence planning results.
    2. Generate evidence evaluation tasks.
    3. Record why evidence is relevant.

    This stage does not make final judgement.
    """


    def __init__(self):
        super().__init__("EvidenceAgent")


    def run(
        self,
        state: ExpertState
    ) -> ExpertState:


        self.log(
            "开始专家证据评估阶段"
        )


        self.load_skill(
            "evidence_evaluation_skill"
        )


        observations = (
            state.observations
        )


        hypotheses = (
            state.hypotheses
        )


        evidence_plan = (
            state.evidence_plan
        )


        evidence: Dict[str, List[Dict[str, Any]]] = {}



        # v2.1:
        # 优先使用 EvidencePlanner 选择的问题。
        #
        # 如果没有 evidence_plan，
        # 则回退到旧版全部检查逻辑。


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


            evidence_items = []



            # =========================
            # v2.1 Evidence Planning模式
            # =========================

            if planned_questions:


                for question in planned_questions:


                    if hypothesis_id in (
                        question.get(
                            "target_hypotheses",
                            []
                        )
                    ):


                        evidence_items.append(

                            {
                                "feature":
                                    question["question"],


                                "evidence_type":
                                    "planned_evidence",


                                "observation":
                                    observations,


                                "related_hypothesis":
                                    hypothesis_name,


                                "description":
                                    question["reason"],


                                "priority":
                                    question.get(
                                        "priority",
                                        "unknown"
                                    ),


                                "reliability":
                                    None,


                                "specificity":
                                    None,


                                "strength":
                                    None,


                                "evaluation_status":
                                    "pending",


                                "reasoning":
                                    (
                                        "Evidence selected "
                                        "by EvidencePlannerAgent."
                                    )
                            }

                        )



            # =========================
            # fallback:
            # 保留v2.0行为
            # =========================

            else:


                required_evidence = (
                    hypothesis.get(
                        "required_evidence",
                        []
                    )
                )


                for item in required_evidence:


                    evidence_items.append(

                        {
                            "feature":
                                item,


                            "evidence_type":
                                "required_evidence",


                            "observation":
                                observations,


                            "related_hypothesis":
                                hypothesis_name,


                            "description":
                                (
                                    "This evidence item "
                                    "should be evaluated "
                                    "to determine whether "
                                    "it supports the hypothesis."
                                ),


                            "expected_result":
                                item,


                            "reliability":
                                None,


                            "specificity":
                                None,


                            "strength":
                                None,


                            "evaluation_status":
                                "pending",


                            "reasoning":
                                (
                                    "Evidence evaluation "
                                    "requires analysis of "
                                    "observed astronomical features."
                                )
                        }

                    )



            evidence[hypothesis_id] = evidence_items



        state.evidence = evidence



        self.record_reasoning(

            state,

            action="evidence_analysis",

            content={

                "skill_used":
                    "evidence_evaluation_skill",


                "used_evidence_plan":
                    bool(evidence_plan),


                "number_of_hypotheses":
                    len(hypotheses),


                "evidence":
                    evidence
            }
        )


        return state