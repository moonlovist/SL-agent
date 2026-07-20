from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class CounterEvidenceAgent(BaseAgent):
    """
    反证分析 Agent。

    模拟专家：

    "为什么我的判断可能是错的？"

    职责：

    1. 不寻找支持证据
    2. 专门寻找反例
    3. 提醒其他 Agent 注意风险
    """

    def __init__(self):
        super().__init__("CounterEvidenceAgent")


    def run(
        self,
        state: ExpertState
    ) -> ExpertState:

        self.log("开始寻找反证")


        contradictions = {}


        for hypothesis in state.hypotheses:

            hypothesis_id = hypothesis["id"]


            contradictions[hypothesis_id] = [

                {
                    "type": "unknown",
                    "description":
                        "需要进一步检查该假设可能存在的问题",
                    "severity": None,
                    "confidence": None
                }

            ]


        state.contradictions = contradictions


        self.record_reasoning(
            state,
            action="counter_evidence_analysis",
            content={
                "contradictions": contradictions
            }
        )


        return state