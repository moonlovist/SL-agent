from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class CounterEvidenceAgent(BaseAgent):
    """
    Counter evidence analysis Agent.

    Simulates expert reasoning:

    "Why might my current judgement be wrong?"

    Responsibilities:

    1. Search for contradictions.
    2. Identify evidence that weakens hypotheses.
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


        contradictions: Dict[
            str,
            List[Dict[str, Any]]
        ] = {}


        for hypothesis in state.hypotheses:

            hypothesis_id = (
                hypothesis["id"]
            )


            hypothesis_name = (
                hypothesis["name"]
            )


            possible_contradictions = (
                hypothesis.get(
                    "possible_contradictions",
                    []
                )
            )


            contradiction_items = []


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
                "number_of_hypotheses":
                    len(state.hypotheses),

                "contradictions":
                    contradictions
            }
        )


        return state