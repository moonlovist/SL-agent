from typing import Dict, Any

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class ObserverAgent(BaseAgent):
    """
    Expert observation agent.

    Responsibilities:
    - follow observation skill
    - describe observable facts
    - avoid premature classification
    - preserve uncertainty
    """

    def __init__(self):
        super().__init__("ObserverAgent")


    def run(
        self,
        state: ExpertState
    ) -> ExpertState:

        self.log(
            "开始专家观察阶段"
        )


        # Load expert observation rules
        skill = self.load_skill(
            "observation_skill"
        )


        task = """
Analyze the astronomical target.

Follow the observation procedure.

Only describe observable facts.

Do not classify the object.

Record:
- morphology
- possible lens candidate
- possible source candidate
- geometric features
- uncertainty
"""


        prompt = self.build_prompt(
            skill,
            task
        )


        # Current stage:
        # LLM/VLM connection will be added later.
        # Keep reasoning trace.

        self.record_reasoning(
            state,
            action="observation_prompt_generation",
            content={
                "skill": "observation_skill",
                "prompt": prompt
            }
        )


        observation = {
            "type": "expert_observation",
            "description": (
                "Observation task generated. "
                "Waiting for vision model."
            ),
            "confidence": None,
            "source": self.name
        }


        state.observations.append(
            observation
        )


        return state