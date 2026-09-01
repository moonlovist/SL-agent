from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Hypothesis(StrictModel):
    id: Literal["H1", "H2", "H3", "H4"]
    name: str
    description: str
    prior: Literal["low", "medium", "high"] = "medium"


class EvidenceQuestion(StrictModel):
    id: str
    question: str
    target_hypotheses: list[str]
    measurement_kinds: list[str]
    priority: Literal["high", "medium", "low"]
    reason: str


class EvidenceItem(StrictModel):
    id: str
    hypothesis_id: str
    claim: str
    observation_refs: list[str]
    polarity: Literal["support", "oppose", "neutral"]
    strength: float = Field(ge=0.0, le=1.0)
    reliability: float = Field(ge=0.0, le=1.0)
    specificity: float = Field(ge=0.0, le=1.0)
    correlation_group: str
    rationale: str
    status: Literal["evaluated", "not_available"]


class ContradictionItem(StrictModel):
    id: str
    hypothesis_id: str
    description: str
    observation_refs: list[str]
    severity: float = Field(ge=0.0, le=1.0)
    reliability: float = Field(ge=0.0, le=1.0)
    correlation_group: str
    rationale: str


class HypothesisEvidenceEffect(StrictModel):
    hypothesis_id: Literal["H1", "H2", "H3", "H4"]
    effect: Literal["support", "oppose", "neutral", "unknown"]
    evidence_refs: list[str] = Field(default_factory=list)
    explanation: str


class CrossHypothesisEvidenceComparison(StrictModel):
    id: str
    observation_refs: list[str]
    observed_fact: str
    correlation_group: str
    effects: list[HypothesisEvidenceEffect]
    discriminating: bool


class MissingEvidence(StrictModel):
    question_id: str
    question: str
    required_measurement_kinds: list[str]
    reason: str


class HypothesisAnalysis(StrictModel):
    hypothesis_id: Literal["H1", "H2", "H3", "H4"]
    hypothesis: str
    status: Literal["viable", "weakened", "insufficient"]
    supporting_evidence_refs: list[str]
    opposing_evidence_refs: list[str]
    missing_evidence: list[MissingEvidence]
    uncertainties: list[str]
    explained_observation_refs: list[str]
    unexplained_relevant_observation_refs: list[str]


class GradeAssessment(StrictModel):
    """Versioned grading contract with no uncalibrated fallback score."""

    grading_scheme: Literal["euclid_q1_paper"] = "euclid_q1_paper"
    status: Literal[
        "graded", "calibration_pending", "insufficient_data", "blocked_by_audit"
    ]
    expert_score: float | None = Field(default=None, ge=0.0, le=3.0)
    grade: Literal["A", "B", "C"] | None = None
    a_plus: bool | None = None
    a_plus_reason: str | None = None
    rationale: str
    missing_requirements: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_grading_state(self) -> "GradeAssessment":
        if self.status == "graded":
            if self.expert_score is None or self.grade is None:
                raise ValueError("A graded result requires expert_score and grade")
            if self.a_plus is None:
                raise ValueError("A graded result must explicitly evaluate A+")
            expected_grade = (
                "A"
                if self.expert_score > 2.0
                else "B"
                if self.expert_score > 1.5
                else "C"
            )
            if self.grade != expected_grade:
                raise ValueError(
                    "grade is inconsistent with the Euclid Q1 paper score mapping"
                )
            if self.a_plus and self.grade != "A":
                raise ValueError("A+ may only be attached to an A-grade result")
            if self.a_plus and not self.a_plus_reason:
                raise ValueError("A+ requires a scientific-value rationale")
        elif (
            self.expert_score is not None
            or self.grade is not None
            or self.a_plus is not None
            or self.a_plus_reason is not None
        ):
            raise ValueError(
                "Ungraded results must not contain expert_score, grade, or A+ claims"
            )
        return self


class AuditFinding(StrictModel):
    id: str
    severity: Literal["info", "warning", "critical"]
    category: str
    description: str
    observation_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    hypothesis_ids: list[str] = Field(default_factory=list)
    recommended_action: str


class ObservationAuditResult(StrictModel):
    valid: bool
    quality_level: Literal["usable", "limited", "unusable"]
    findings: list[AuditFinding]
    checks: dict[str, bool]
    reliability_summary: str
    recommendation: str


class EvidenceAuditResult(StrictModel):
    valid: bool
    findings: list[AuditFinding]
    checks: dict[str, bool]
    correlated_groups: dict[str, list[str]]
    low_specificity_evidence_refs: list[str]
    recommendation: str


class AuditResult(StrictModel):
    valid: bool
    issues: list[str]
    checks: dict[str, bool]
    recommendation: str
