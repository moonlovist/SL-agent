from __future__ import annotations

from ..schemas.reasoning import GradeAssessment


Q1_PAPER_GRADING_SCHEME = "euclid_q1_paper"


def map_q1_catalog_bin(expert_score: float) -> str:
    """Q1 Engine A sections 6.2 and 8.3; input must be a calibrated aggregate."""

    score = float(expert_score)
    if not 0.0 <= score <= 3.0:
        raise ValueError("expert_score must be within [0, 3]")
    if score > 2.0:
        return "A"
    if score > 1.5:
        return "B"
    if score > 1.0:
        return "C"
    return "below_C_cut"


def map_q1_paper_grade(expert_score: float) -> str:
    """Compatibility function name for :func:`map_q1_catalog_bin`."""

    return map_q1_catalog_bin(expert_score)


def build_q1_paper_grade(
    expert_score: float,
    *,
    rationale: str,
    a_plus: bool = False,
    a_plus_reason: str | None = None,
) -> GradeAssessment:
    """Build a validated graded result after an external calibrated estimator runs."""

    catalog_bin = map_q1_catalog_bin(expert_score)
    return GradeAssessment(
        status="graded",
        expert_score=float(expert_score),
        catalog_bin=catalog_bin,
        grade=catalog_bin if catalog_bin in {"A", "B", "C"} else None,
        a_plus=a_plus,
        a_plus_reason=a_plus_reason,
        rationale=rationale,
    )
