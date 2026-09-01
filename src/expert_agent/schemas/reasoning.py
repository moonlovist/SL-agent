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


ObservationSourceType = Literal[
    "raw_image_observation",
    "primary_residual_measurement",
    "independent_model_check",
    "multiband_corroboration",
    "external_catalog_posthoc",
]


class TraceableFact(StrictModel):
    """An interpretation-ready fact that still points back to Observation."""

    id: str
    statement: str
    observation_refs: list[str]
    source_types: list[ObservationSourceType]
    reliability: Literal["low", "medium", "high", "unknown"] = "unknown"


class EvidenceDimension(StrictModel):
    status: Literal["credible", "mixed", "weak", "unavailable", "unknown"]
    fact_refs: list[str] = Field(default_factory=list)
    rationale: str


ExplanationId = Literal[
    "strong_gravitational_lens",
    "intrinsic_galaxy_morphology",
    "merger_or_interaction",
    "foreground_subtraction_or_psf_artifact",
    "noise_bad_pixel_or_projected_source",
]


class CompetingExplanation(StrictModel):
    id: ExplanationId
    status: Literal["favored", "viable", "weakened", "unsupported", "unknown"]
    supporting_fact_refs: list[str] = Field(default_factory=list)
    contradicting_fact_refs: list[str] = Field(default_factory=list)
    rationale: str


class MissingInformationItem(StrictModel):
    topic: str
    status: Literal["unknown", "not_measured", "unavailable"]
    reason: str
    observation_refs: list[str] = Field(default_factory=list)


class ProcessingCaution(StrictModel):
    id: str
    description: str
    observation_refs: list[str]
    effect: Literal["uncertainty_only", "supports_artifact_alternative"]


class CorrelatedEvidenceGroup(StrictModel):
    id: str
    fact_refs: list[str]
    observation_refs: list[str]
    rationale: str


class UnusedFact(StrictModel):
    field_path: str
    reason: str


class RequestedTest(StrictModel):
    id: str
    purpose: str
    status: Literal["not_measured", "unavailable", "future"]


class EvidenceDossier(StrictModel):
    observed_facts: list[TraceableFact]
    signal_reality: EvidenceDimension
    lens_specificity: EvidenceDimension
    supporting_evidence: list[TraceableFact]
    contradicting_evidence: list[TraceableFact]
    competing_explanations: list[CompetingExplanation]
    missing_information: list[MissingInformationItem]
    processing_cautions: list[ProcessingCaution]
    correlated_evidence_groups: list[CorrelatedEvidenceGroup]
    unused_facts: list[UnusedFact]
    null_tests: list[RequestedTest]
    future_tool_requests: list[RequestedTest]


class HypothesisComparison(StrictModel):
    hypothesis_id: ExplanationId
    status: Literal["favored", "viable", "weakened", "unsupported", "unknown"]
    supporting_fact_refs: list[str]
    contradicting_fact_refs: list[str]
    missing_information_topics: list[str]
    rationale: str


class ProvisionalExpertAssessment(StrictModel):
    expert_vote_anchor: Literal[0, 1, 2, 3] | None
    provisional_expert_score: float | None = Field(default=None, ge=0.0, le=3.0)
    provisional_score_range: tuple[float, float] | None = None
    provisional_grade: Literal["X", "C", "B", "A", "insufficient_for_grading"] | None
    score_status: Literal["provisional_uncalibrated"] = "provisional_uncalibrated"
    calibration_required: Literal[True] = True
    official_expert_score: None = None
    scientific_value_flags: list[str] = Field(default_factory=list)
    decisive_evidence: list[str] = Field(default_factory=list)
    decisive_counterevidence: list[str] = Field(default_factory=list)
    assessment_rationale: str
    assessment_questions: dict[str, str]

    @model_validator(mode="after")
    def validate_provisional_state(self) -> "ProvisionalExpertAssessment":
        if self.provisional_score_range is not None:
            low, high = self.provisional_score_range
            if not 0.0 <= low <= high <= 3.0:
                raise ValueError("provisional_score_range must be ordered within [0, 3]")
        if self.provisional_grade == "insufficient_for_grading":
            if any(
                value is not None
                for value in (
                    self.expert_vote_anchor,
                    self.provisional_expert_score,
                    self.provisional_score_range,
                )
            ):
                raise ValueError("insufficient cases must abstain from all score fields")
            return self
        if self.provisional_score_range is not None:
            if self.expert_vote_anchor is not None or self.provisional_expert_score is not None:
                raise ValueError("an interval assessment must not manufacture a point score")
            if self.provisional_grade is not None:
                raise ValueError("an interval assessment must not manufacture a grade")
            return self
        if self.expert_vote_anchor is None or self.provisional_expert_score is None:
            raise ValueError("a point assessment requires an anchor and provisional score")
        expected = {0: "X", 1: "C", 2: "B", 3: "A"}[self.expert_vote_anchor]
        if self.provisional_grade != expected:
            raise ValueError("provisional grade is inconsistent with the expert vote anchor")
        return self


class DecisionAuditResult(StrictModel):
    audit_passed: bool
    hallucinated_facts: list[str]
    missing_as_negative: list[str]
    correlated_evidence_double_counted: list[str]
    measurement_used_as_proof: list[str]
    recovery_auto_penalized: list[str]
    alternative_ignored: list[str]
    external_answer_leakage: list[str]
    rationale_score_mismatch: list[str]
    abstention_recommended: bool
    required_corrections: list[str]


class GradingSchemeStatus(StrictModel):
    name: Literal["euclid_q1_paper"] = "euclid_q1_paper"
    status: Literal["provisional_mapping_only"] = "provisional_mapping_only"
