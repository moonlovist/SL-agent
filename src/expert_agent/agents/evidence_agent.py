from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class EvidenceAgent(BaseAgent):
    """
    Evidence evaluation Agent.

    Simulates expert evidence analysis.

    Responsibilities:

    1. Read hypothesis requirements.
    2. Create evidence evaluation tasks.
    3. Record why evidence supports a hypothesis.

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


        skill = self.load_skill(
            "evidence_evaluation_skill"
        )


        observations = (
            state.observations
        )


        hypotheses = (
            state.hypotheses
        )


        evidence: Dict[str, List[Dict[str, Any]]] = {}


        for hypothesis in hypotheses:


            hypothesis_id = (
                hypothesis["id"]
            )


            hypothesis_name = (
                hypothesis["name"]
            )


            required_evidence = (
                hypothesis.get(
                    "required_evidence",
                    []
                )
            )


            evidence_items = []


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


            evidence[hypothesis_id] = (
                evidence_items
            )


        state.evidence = evidence


        self.record_reasoning(
            state,

            action="evidence_analysis",

            content={

                "skill_used":
                    "evidence_evaluation_skill",

                "number_of_hypotheses":
                    len(hypotheses),

                "evidence":
                    evidence
            }
        )


        return state