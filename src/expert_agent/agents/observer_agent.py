from typing import Dict, Any

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class ObserverAgent(BaseAgent):
    """
    专家观察 Agent。

    模拟人类专家拿到天文图像后的第一轮观察。

    规则：
    1. 只描述事实
    2. 不进行分类
    3. 不提前给结论
    4. 保留不确定性
    """

    def __init__(self):
        super().__init__("ObserverAgent")


    def run(
        self,
        state: ExpertState
    ) -> ExpertState:

        self.log("开始观察目标")


        # 第一版先建立框架。
        # 后续这里接入：
        # - 图像模型
        # - 视觉LLM
        # - 天文测量工具


        observation = {
            "type": "morphology",
            "description": "待视觉模型分析",
            "confidence": None,
            "source": "ObserverAgent"
        }


        state.observations.append(
            observation
        )


        self.record_reasoning(
            state,
            action="observation",
            content=observation
        )


        return state