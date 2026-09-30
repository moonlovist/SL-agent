from __future__ import annotations

from ..agents.decision_audit import DecisionAudit
from ..agents.expert_assessment_agent import ExpertAssessmentAgent
from ..agents.report_writer import ReportWriter
from .config import ReasoningConfig
from .state import ExpertState


class ExpertWorkflow:
    def __init__(self, config: ReasoningConfig | None = None):
        reasoning_config = config or ReasoningConfig()
        self.expert_assessment_agent = ExpertAssessmentAgent(reasoning_config)
        self.decision_audit = DecisionAudit(reasoning_config)
        self.agents = [
            self.expert_assessment_agent,
            self.decision_audit,
            ReportWriter(),
        ]

    def run(self, state: ExpertState) -> ExpertState:
        state.require_observation()
        for agent in self.agents:
            state = agent.run(state)
        return state
