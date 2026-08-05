from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState
from ..schemas.reasoning import Hypothesis


class HypothesisAgent(BaseAgent):
    def __init__(self):
        super().__init__("HypothesisAgent")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("建立竞争解释")
        self.load_skill("hypothesis_skill")
        state.require_observation()
        state.hypotheses = [
            Hypothesis(
                id="H1",
                name="strong_gravitational_lens",
                description="A foreground mass produces tangentially aligned, geometrically related images of a background source.",
            ),
            Hypothesis(
                id="H2",
                name="intrinsic_galaxy_morphology",
                description="Spiral, ring, disk, or star-forming structure belongs to the observed galaxy itself.",
            ),
            Hypothesis(
                id="H3",
                name="merger_or_interaction",
                description="Multiple galaxies, tidal debris, or an interaction produces the observed structures.",
                prior="low",
            ),
            Hypothesis(
                id="H4",
                name="observation_or_subtraction_artifact",
                description="Noise, PSF effects, detector behavior, or foreground subtraction creates the residual structure.",
                prior="low",
            ),
        ]
        self.record_reasoning(
            state,
            "hypothesis_generation",
            {"hypothesis_ids": [item.id for item in state.hypotheses]},
        )
        return state
