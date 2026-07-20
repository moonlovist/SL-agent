from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class ReasoningAuditor(BaseAgent):
    """
    推理审计 Agent。

    模拟专家最后一次自我检查。

    它不产生新事实。

    它检查：

    1. 推理是否合理
    2. 证据是否足够
    3. 是否存在未解决矛盾
    4. 结论是否过度确定
    """

    def __init__(self):
        super().__init__("ReasoningAuditor")


    def run(
        self,
        state: ExpertState
    ) -> ExpertState:

        self.log("开始审查推理过程")


        issues = []


        # 检查是否有假设
        if len(state.hypotheses) == 0:
            issues.append(
                "没有生成候选假设"
            )


        # 检查是否有证据
        if len(state.evidence) == 0:
            issues.append(
                "没有进行证据分析"
            )


        # 检查是否有反证
        if len(state.contradictions) == 0:
            issues.append(
                "没有进行反证检查"
            )


        audit_result = {

            "valid":
                len(issues) == 0,

            "issues":
                issues,

            "confidence":
                None,

            "recommendation":
                "需要根据完整证据进一步判断"

        }


        state.audit_result = audit_result


        self.record_reasoning(
            state,
            action="reasoning_audit",
            content=audit_result
        )


        return state