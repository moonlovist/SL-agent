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


        report = {


            "observations":
                state.observations,


            "hypotheses":
                state.hypotheses,


            "evidence":
                state.evidence,


            "contradictions":
                state.contradictions,


            "confidence_assessment":
                state.confidence_assessment,


            "audit":
                state.audit_result,


            "final_statement":
                (
                    "Expert judgement generated "
                    "from evidence balance, "
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