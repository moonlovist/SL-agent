from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
    weighted_score: float = Field(ge=-1.0, le=1.0)
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
    weighted_effect: float = Field(ge=-1.0, le=1.0)
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


class HypothesisScore(StrictModel):
    hypothesis_id: str
    hypothesis: str
    evidence_score: float = Field(ge=-1.0, le=1.0)
    contradiction_penalty: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=-1.0, le=1.0)


class DecisionResult(StrictModel):
    classification: Literal[
        "strong_lens_candidate", "not_lens_preferred", "indeterminate"
    ]
    selected_hypothesis: str | None
    ranking: list[HypothesisScore]
    score_margin: float = Field(ge=0.0, le=2.0)
    decision_reasoning: str
    evidence_refs: list[str]
    remaining_competitors: list[str]


class ConfidenceAssessment(StrictModel):
    level: Literal["high", "medium", "low", "insufficient"]
    score: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    reasoning: str
    supporting_evidence_refs: list[str]
    limiting_factors: list[str]
    remaining_uncertainties: list[str]


class AuditResult(StrictModel):
    valid: bool
    issues: list[str]
    checks: dict[str, bool]
    recommendation: str
