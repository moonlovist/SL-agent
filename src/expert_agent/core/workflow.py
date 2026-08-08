from __future__ import annotations

from ..agents.confidence_agent import ConfidenceAgent
from ..agents.counter_evidence_agent import CounterEvidenceAgent
from ..agents.decision_agent import DecisionAgent
from ..agents.evidence_agent import EvidenceAgent
from ..agents.evidence_comparison_agent import EvidenceComparisonAgent
from ..agents.evidence_planner import EvidencePlannerAgent
from ..agents.hypothesis_agent import HypothesisAgent
from ..agents.hypothesis_analysis_agent import HypothesisAnalysisAgent
from ..agents.observer_agent import ObserverAgent
from ..agents.reasoning_auditor import ReasoningAuditor
from ..agents.report_writer import ReportWriter
from .config import ReasoningConfig
from .state import ExpertState


class ExpertWorkflow:
    def __init__(self, config: ReasoningConfig | None = None):
        reasoning_config = config or ReasoningConfig()
        self.agents = [
            ObserverAgent(),
            HypothesisAgent(),
            EvidencePlannerAgent(),
            EvidenceAgent(),
            EvidenceComparisonAgent(),
            CounterEvidenceAgent(),
            HypothesisAnalysisAgent(),
            DecisionAgent(reasoning_config),
            ConfidenceAgent(reasoning_config),
            ReasoningAuditor(),
            ReportWriter(),
        ]

    def run(self, state: ExpertState) -> ExpertState:
        state.require_observation()
        for agent in self.agents:
            state = agent.run(state)
        return state
