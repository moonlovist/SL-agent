from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class EvidenceAgent(BaseAgent):
    """
    证据分析 Agent。

    模拟专家验证：

    "为什么这个假设可能成立？"

    注意：
    不负责最终判断。
    只负责整理支持证据。
    """

    def __init__(self):
        super().__init__("EvidenceAgent")


    def run(
        self,
        state: ExpertState
    ) -> ExpertState:

        self.log("开始分析支持证据")


        evidence = {}


        for hypothesis in state.hypotheses:

            hypothesis_id = hypothesis["id"]


            evidence[hypothesis_id] = [

                {
                    "feature": "待视觉分析",
                    "description":
                        "需要根据观测数据寻找支持该假设的特征",
                    "strength": None,
                    "confidence": None
                }

            ]


        state.evidence = evidence


        self.record_reasoning(
            state,
            action="evidence_analysis",
            content={
                "evidence": evidence
            }
        )


        return state