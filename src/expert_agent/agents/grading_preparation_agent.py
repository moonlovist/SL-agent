from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState
from ..schemas.reasoning import GradeAssessment


class GradingPreparationAgent(BaseAgent):
    """Expose grading readiness without inventing an uncalibrated expert score."""

    def __init__(self):
        super().__init__("GradingPreparationAgent")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("检查是否具备经过标定后进行Q1评级的条件")
        observation_audit = state.observation_audit
        evidence_audit = state.evidence_audit
        reasoning_audit = state.audit_result
        if observation_audit is None or evidence_audit is None or reasoning_audit is None:
            raise ValueError(
                "GradingPreparationAgent requires observation, evidence, and reasoning audits"
            )

        if observation_audit.quality_level == "unusable":
            assessment = GradeAssessment(
                status="insufficient_data",
                rationale=(
                    "The observation is scientifically unusable. No X grade is issued, "
                    "because insufficient data is not evidence that the object is not a lens."
                ),
                missing_requirements=[
                    item.description
                    for item in observation_audit.findings
                    if item.severity == "critical"
                ]
                or ["A scientifically usable observation"],
            )
        elif not evidence_audit.valid or not reasoning_audit.valid:
            assessment = GradeAssessment(
                status="blocked_by_audit",
                rationale=(
                    "A structural or evidence-provenance audit failed. Grading is blocked "
                    "until the trace is internally consistent."
                ),
                missing_requirements=list(reasoning_audit.issues)
                + [
                    item.description
                    for item in evidence_audit.findings
                    if item.severity == "critical"
                ],
            )
        else:
            assessment = GradeAssessment(
                status="calibration_pending",
                rationale=(
                    "The observation and reasoning trace are ready for a calibrated grading "
                    "stage. The former provisional score and confidence formulas have been "
                    "retired, so no A/B/C label is manufactured before Q1 calibration."
                ),
                missing_requirements=[
                    "A labelled Euclid Q1 calibration table containing expert_score",
                    "A validated pre-grading evidence-audit rubric",
                    "A calibrated and independently tested expert-score estimator",
                ],
            )

        state.grade_assessment = assessment
        self.record_reasoning(
            state,
            "grading_preparation",
            assessment.model_dump(mode="json"),
        )
        return state
