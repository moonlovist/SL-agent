from typing import Dict, Any

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class ReportWriter(BaseAgent):
    """
    专家报告生成 Agent。

    将完整推理过程整理成最终报告。

    不负责重新判断。
    只负责表达和总结。
    """

    def __init__(self):
        super().__init__("ReportWriter")


    def run(
        self,
        state: ExpertState
    ) -> ExpertState:

        self.log("生成专家分析报告")


        report = {

            "observations":
                state.observations,

            "hypotheses":
                state.hypotheses,

            "evidence":
                state.evidence,

            "contradictions":
                state.contradictions,

            "audit":
                state.audit_result,

            "final_statement":
                "基于当前证据生成专家评估"

        }


        state.final_decision = report


        self.record_reasoning(
            state,
            action="report_generation",
            content=report
        )


        return state