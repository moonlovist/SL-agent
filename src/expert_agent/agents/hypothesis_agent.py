from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class HypothesisAgent(BaseAgent):
    """
    假设生成 Agent。

    模拟专家看到初步观察结果后的思考过程。

    核心原则：

    1. 不立即下结论
    2. 保留多个解释
    3. 每个假设都需要后续验证
    4. 避免确认偏差
    """

    def __init__(self):
        super().__init__("HypothesisAgent")


    def run(
        self,
        state: ExpertState
    ) -> ExpertState:

        self.log("根据观察生成候选假设")


        hypotheses: List[Dict[str, Any]] = [
            {
                "id": "H1",
                "name": "strong_gravitational_lens",
                "description":
                    "观测结构可能由强引力透镜效应产生",
                "status": "candidate"
            },
            {
                "id": "H2",
                "name": "galaxy_morphology",
                "description":
                    "观测结构可能是星系自身形态结构",
                "status": "candidate"
            },
            {
                "id": "H3",
                "name": "merger_or_interaction",
                "description":
                    "观测结构可能来自星系相互作用",
                "status": "candidate"
            },
            {
                "id": "H4",
                "name": "imaging_artifact",
                "description":
                    "观测结构可能由数据或成像误差造成",
                "status": "candidate"
            }
        ]


        state.hypotheses.extend(
            hypotheses
        )


        self.record_reasoning(
            state,
            action="hypothesis_generation",
            content={
                "number_of_hypotheses":
                    len(hypotheses),
                "hypotheses":
                    hypotheses
            }
        )


        return state