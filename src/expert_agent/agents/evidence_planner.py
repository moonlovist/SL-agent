from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class EvidencePlannerAgent(BaseAgent):
    """
    Evidence Planning Agent.

    Simulates expert decision making:

    "Which evidence should be checked first?"

    Responsibilities:

    1. Review observations.
    2. Review competing hypotheses.
    3. Select high-value evidence questions.
    4. Avoid unnecessary exhaustive checking.

    This stage does not evaluate evidence.
    It only plans evidence collection.
    """


    def __init__(self):
        super().__init__("EvidencePlannerAgent")


    def run(
        self,
        state: ExpertState
    ) -> ExpertState:


        self.log(
            "规划优先验证证据"
        )


        self.load_skill(
            "evidence_planning_skill"
        )


        observations = (
            state.observations
        )

        hypotheses = (
            state.hypotheses
        )


        priority_questions: List[Dict[str, Any]] = []


        # 当前版本：
        # 使用规则模板生成证据规划。
        #
        # 后续版本：
        # 使用LLM根据观察动态选择
        # 最有信息价值的问题。


        if hypotheses:


            # 强引力透镜 vs 星系自身结构
            # 是当前最重要区分问题

            priority_questions.append(

                {
                    "question":
                        "是否存在共同透镜中心？",

                    "reason":
                        (
                            "该证据可以区分 "
                            "引力透镜结构与星系自身形态结构。"
                        ),

                    "target_hypotheses":
                        [
                            "H1",
                            "H2"
                        ],

                    "priority":
                        "high"
                }

            )



            priority_questions.append(

                {
                    "question":
                        "是否存在切向排列或透镜几何结构？",

                    "reason":
                        (
                            "该证据能够提高或降低 "
                            "强引力透镜解释的可信度。"
                        ),

                    "target_hypotheses":
                        [
                            "H1"
                        ],

                    "priority":
                        "high"
                }

            )



            priority_questions.append(

                {
                    "question":
                        "是否存在明显并合或扰动结构？",

                    "reason":
                        (
                            "用于判断星系相互作用 "
                            "是否为替代解释。"
                        ),

                    "target_hypotheses":
                        [
                            "H3"
                        ],

                    "priority":
                        "medium"
                }

            )



        evidence_plan = {


            "planning_strategy":
                (
                    "Select discriminative evidence "
                    "rather than evaluating all evidence."
                ),


            "observation_input":
                observations,


            "priority_questions":
                priority_questions,


            "number_of_selected_questions":
                len(priority_questions)

        }



        state.evidence_plan = evidence_plan



        self.record_reasoning(

            state,

            action="evidence_planning",

            content=evidence_plan

        )


        return state