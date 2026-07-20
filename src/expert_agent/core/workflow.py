from .state import ExpertState

from ..agents.observer_agent import ObserverAgent
from ..agents.hypothesis_agent import HypothesisAgent
from ..agents.evidence_agent import EvidenceAgent
from ..agents.counter_evidence_agent import CounterEvidenceAgent
from ..agents.reasoning_auditor import ReasoningAuditor
from ..agents.report_writer import ReportWriter



class ExpertWorkflow:
    """
    专家推理工作流。

    控制整个专家分析流程。

    顺序：

    观察
      ↓
    假设
      ↓
    证据
      ↓
    反证
      ↓
    审计
      ↓
    报告
    """


    def __init__(self):

        self.agents = [

            ObserverAgent(),

            HypothesisAgent(),

            EvidenceAgent(),

            CounterEvidenceAgent(),

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