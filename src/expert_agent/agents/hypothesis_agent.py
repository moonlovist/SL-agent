from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class HypothesisAgent(BaseAgent):
    """
    Hypothesis generation Agent.

    Simulates expert reasoning after observation.

    Responsibilities:
    1. Generate limited competing hypotheses.
    2. Maintain uncertainty.
    3. Identify discriminative questions.
    4. Avoid premature conclusions.
    """

    def __init__(self):
        super().__init__("HypothesisAgent")


    def run(
        self,
        state: ExpertState
    ) -> ExpertState:

        self.log(
            "根据观察生成竞争假设"
        )


        # Load expert hypothesis rules
        skill = self.load_skill(
            "hypothesis_skill"
        )


        observations = (
            state.observations
        )


        task = {
            "role":
                "hypothesis_generation",

            "skill":
                "hypothesis_skill",

            "observation_input":
                observations
        }


        self.record_reasoning(
            state,
            action="hypothesis_reasoning_setup",
            content=task
        )


        hypotheses: List[Dict[str, Any]] = [

            {
                "id":
                    "H1",

                "category":
                    "strong_gravitational_lens",

                "name":
                    "strong_gravitational_lens",

                "description":
                    "The observed structure may be produced by strong gravitational lensing.",

                "initial_confidence":
                    "medium",

                "supporting_observations":
                    observations,

                "required_evidence":
                    [
                        "common lens center",
                        "tangential alignment",
                        "consistent source morphology"
                    ],

                "possible_contradictions":
                    [
                        "structure connected to galaxy disk",
                        "lack of lensing geometry"
                    ]
            },


            {
                "id":
                    "H2",

                "category":
                    "galaxy_morphology",

                "name":
                    "intrinsic_galaxy_structure",

                "description":
                    "The structure may originate from intrinsic galaxy morphology.",

                "initial_confidence":
                    "medium",

                "supporting_observations":
                    observations,

                "required_evidence":
                    [
                        "connection to galaxy center",
                        "spiral or disk pattern"
                    ],

                "possible_contradictions":
                    [
                        "independent arc-like structure",
                        "multiple consistent images"
                    ]
            },


            {
                "id":
                    "H3",

                "category":
                    "galaxy_interaction",

                "name":
                    "merger_or_interaction",

                "description":
                    "The structure may be caused by galaxy interaction or merger.",

                "initial_confidence":
                    "low",

                "supporting_observations":
                    observations,

                "required_evidence":
                    [
                        "multiple cores",
                        "tidal features"
                    ],

                "possible_contradictions":
                    [
                        "clear lensing geometry"
                    ]
            },


            {
                "id":
                    "H4",

                "category":
                    "observation_artifact",

                "name":
                    "imaging_artifact",

                "description":
                    "The structure may be caused by observational artifacts.",

                "initial_confidence":
                    "low",

                "supporting_observations":
                    observations,

                "required_evidence":
                    [
                        "instrumental effect",
                        "noise pattern"
                    ],

                "possible_contradictions":
                    [
                        "continuous astronomical structure"
                    ]
            }

        ]


        discrimination_questions = [

            {
                "question":
                    "Does the structure share a common center with the lens candidate?",

                "purpose":
                    "Distinguish gravitational lensing from intrinsic galaxy morphology.",

                "related_hypotheses":
                    [
                        "H1",
                        "H2"
                    ]
            },


            {
                "question":
                    "Is the structure connected to the galaxy disk?",

                "purpose":
                    "Test spiral arm or intrinsic morphology explanation.",

                "related_hypotheses":
                    [
                        "H1",
                        "H2"
                    ]
            },


            {
                "question":
                    "Are multiple bright components physically consistent?",

                "purpose":
                    "Distinguish multiple images from separate objects.",

                "related_hypotheses":
                    [
                        "H1",
                        "H3"
                    ]
            }

        ]


        state.hypotheses.extend(
            hypotheses
        )


        self.record_reasoning(
            state,
            action="hypothesis_generation",

            content={

                "skill_used":
                    "hypothesis_skill",

                "number_of_hypotheses":
                    len(hypotheses),

                "hypotheses":
                    hypotheses,

                "discrimination_questions":
                    discrimination_questions
            }
        )


        return state