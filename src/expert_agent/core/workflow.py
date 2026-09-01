from __future__ import annotations

from ..agents.counter_evidence_agent import CounterEvidenceAgent
from ..agents.evidence_agent import EvidenceAgent
from ..agents.evidence_auditor import EvidenceAuditor
from ..agents.evidence_comparison_agent import EvidenceComparisonAgent
from ..agents.evidence_planner import EvidencePlannerAgent
from ..agents.hypothesis_agent import HypothesisAgent
from ..agents.hypothesis_analysis_agent import HypothesisAnalysisAgent
from ..agents.grading_preparation_agent import GradingPreparationAgent
from ..agents.observer_agent import ObserverAgent
from ..agents.observation_auditor import ObservationAuditor
from ..agents.reasoning_auditor import ReasoningAuditor
from ..agents.report_writer import ReportWriter
from .config import ReasoningConfig
from .state import ExpertState


class ExpertWorkflow:
    def __init__(self, config: ReasoningConfig | None = None):
        reasoning_config = config or ReasoningConfig()
        self.agents = [
            ObserverAgent(),
            ObservationAuditor(reasoning_config),
            HypothesisAgent(),
            EvidencePlannerAgent(),
            EvidenceAgent(),
            EvidenceComparisonAgent(),
            CounterEvidenceAgent(),
            HypothesisAnalysisAgent(),
            EvidenceAuditor(reasoning_config),
            ReasoningAuditor(),
            GradingPreparationAgent(),
            ReportWriter(),
        ]

    def run(self, state: ExpertState) -> ExpertState:
        state.require_observation()
        for agent in self.agents:
            state = agent.run(state)
        return state
