from typing import Dict, Any

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class ReportWriter(BaseAgent):
    """
    Expert report generation Agent.

    Converts the reasoning process into a final report.

    Does not make new judgement.

    Only summarizes the complete reasoning process.
    """


    def __init__(self):
        super().__init__("ReportWriter")



    def run(
        self,
        state: ExpertState
    ) -> ExpertState:


        self.log(
            "生成专家分析报告"
        )


        confidence_assessment = (
            state.confidence_assessment
            if state.confidence_assessment
            else {}
        )


        evidence_scoring = (
            confidence_assessment.get(
                "evidence_score_breakdown",
                {}
            )
        )


        report = {


            "observations":
                state.observations,


            "hypotheses":
                state.hypotheses,


            "evidence_plan":
                state.evidence_plan,


            "evidence":
                state.evidence,


            "evidence_scoring":
                evidence_scoring,


            "contradictions":
                state.contradictions,


            "confidence_assessment":
                confidence_assessment,


            "audit":
                state.audit_result,


            "final_statement":
                (
                    "Expert judgement generated "
                    "from evidence planning, "
                    "weighted evidence evaluation, "
                    "contradiction analysis, "
                    "confidence calibration, "
                    "and reasoning audit."
                )

        }



        state.final_decision = report



        self.record_reasoning(

            state,

            action="report_generation",

            content=report

        )



        return state