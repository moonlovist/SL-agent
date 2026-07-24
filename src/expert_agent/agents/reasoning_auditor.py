from typing import Dict, Any, List

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class ReasoningAuditor(BaseAgent):
    """
    Reasoning audit Agent.

    Simulates the final expert self-review.

    It does not create new facts.

    It checks:

    1. Whether reasoning follows the expert workflow.
    2. Whether evidence supports hypotheses.
    3. Whether contradictions are considered.
    4. Whether confidence and uncertainty are properly recorded.
    """


    def __init__(self):
        super().__init__("ReasoningAuditor")


    def run(
        self,
        state: ExpertState
    ) -> ExpertState:


        self.log(
            "开始审查推理过程"
        )


        issues = []


        # Check hypothesis generation

        if len(state.hypotheses) == 0:

            issues.append(
                "没有生成候选假设"
            )


        # Check evidence evaluation

        if len(state.evidence) == 0:

            issues.append(
                "没有进行证据分析"
            )


        # Check contradiction analysis

        if len(state.contradictions) == 0:

            issues.append(
                "没有进行反证检查"
            )


        # Check confidence calibration

        confidence_result = (
            state.final_decision
        )


        if not confidence_result:

            issues.append(
                "没有进行可信度校准"
            )


        else:

            if (
                "confidence_level"
                not in confidence_result
            ):

                issues.append(
                    "缺少可信度等级"
                )


            if (
                "remaining_uncertainties"
                not in confidence_result
            ):

                issues.append(
                    "没有记录不确定性"
                )


        # Check evidence completion

        pending_evidence = []


        for hypothesis_id, items in state.evidence.items():

            for item in items:

                if (
                    item.get(
                        "evaluation_status"
                    )
                    ==
                    "pending"
                ):

                    pending_evidence.append(
                        hypothesis_id
                    )


        if pending_evidence:

            issues.append(
                "部分证据仍未完成评估"
            )


        audit_result = {


            "valid":
                len(issues) == 0,


            "issues":
                issues,


            "workflow_check":

                {
                    "hypothesis":
                        len(state.hypotheses) > 0,

                    "evidence":
                        len(state.evidence) > 0,

                    "contradiction":
                        len(state.contradictions) > 0,

                    "confidence":
                        bool(confidence_result)
                },


            "recommendation":
                (
                    "推理流程完整，但需要根据 "
                    "证据质量和不确定性进一步判断。"
                )

        }


        state.audit_result = audit_result


        self.record_reasoning(
            state,

            action="reasoning_audit",

            content=audit_result
        )


        return state