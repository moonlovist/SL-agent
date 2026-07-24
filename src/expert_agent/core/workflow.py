from .state import ExpertState

from ..agents.observer_agent import ObserverAgent
from ..agents.hypothesis_agent import HypothesisAgent
from ..agents.evidence_agent import EvidenceAgent
from ..agents.counter_evidence_agent import CounterEvidenceAgent
from ..agents.confidence_agent import ConfidenceAgent
from ..agents.reasoning_auditor import ReasoningAuditor
from ..agents.report_writer import ReportWriter


class ExpertWorkflow:
    """
    专家推理工作流。

    控制整个专家分析流程。

    流程：

    观察
      ↓
    假设生成
      ↓
    证据评估
      ↓
    反证分析
      ↓
    可信度校准
      ↓
    推理审计
      ↓
    最终报告

    对应 Expert Reasoning Framework:
    
    Observation
        ↓
    Hypothesis Generation
        ↓
    Evidence Evaluation
        ↓
    Contradiction Analysis
        ↓
    Confidence Calibration
        ↓
    Reasoning Audit
        ↓
    Final Report
    """


    def __init__(self):

        self.agents = [

            ObserverAgent(),

            HypothesisAgent(),

            EvidenceAgent(),

            CounterEvidenceAgent(),

            ConfidenceAgent(),

            ReasoningAuditor(),

            ReportWriter()

        ]


    def run(
        self,
        state: ExpertState
    ) -> ExpertState:


        for agent in self.agents:

            state = agent.run(state)


        return state