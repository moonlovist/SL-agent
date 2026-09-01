from __future__ import annotations

from ..schemas.reasoning import GradeAssessment


Q1_PAPER_GRADING_SCHEME = "euclid_q1_paper"


def map_q1_paper_grade(expert_score: float) -> str:
    """Map a calibrated 0--3 expert score to the Q1 paper grade bins."""

    score = float(expert_score)
    if not 0.0 <= score <= 3.0:
        raise ValueError("expert_score must be within [0, 3]")
    if score > 2.0:
        return "A"
    if score > 1.5:
        return "B"
    return "C"


def build_q1_paper_grade(
    expert_score: float,
    *,
    rationale: str,
    a_plus: bool = False,
    a_plus_reason: str | None = None,
) -> GradeAssessment:
    """Build a validated graded result after an external calibrated estimator runs."""

    grade = map_q1_paper_grade(expert_score)
    return GradeAssessment(
        status="graded",
        expert_score=float(expert_score),
        grade=grade,
        a_plus=a_plus,
        a_plus_reason=a_plus_reason,
        rationale=rationale,
    )
