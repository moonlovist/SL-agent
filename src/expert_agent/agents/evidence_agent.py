from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class EvidenceAgent(BaseAgent):
    """
    Evidence evaluation Agent.

    Simulates expert evaluation:

    "Why does this hypothesis have support?"

    This stage does not make final judgement.
    It only evaluates supporting evidence.
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


        # Load expert evidence rules
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

            hypothesis_id = hypothesis["id"]


            evidence[hypothesis_id] = [

                {
                    "feature": "observation_based_feature",

                    "observation": observations,

                    "related_hypothesis":
                        hypothesis["name"],

                    "description":
                        (
                            "Evidence evaluation requires "
                            "visual analysis of observed features."
                        ),

                    "reliability": None,

                    "specificity": None,

                    "strength": None,

                    "reasoning":
                        (
                            "Evidence should be evaluated "
                            "according to expert evidence rules."
                        )
                }

            ]


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