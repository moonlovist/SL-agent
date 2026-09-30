from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schemas.observation import ObservationDocument
from ..schemas.reasoning import (
    AuditResult,
    EvidenceAuditResult,
    ContradictionItem,
    EvidenceItem,
    CrossHypothesisEvidenceComparison,
    EvidenceQuestion,
    GradeAssessment,
    Hypothesis,
    HypothesisAnalysis,
    ObservationAuditResult,
    DecisionAuditResult,
    CandidateEvidenceIntegration,
    EvidenceDossier,
    GradingSchemeStatus,
    HypothesisComparison,
    ProvisionalExpertAssessment,
)


@dataclass
class ExpertState:
    image_path: str | None = None
    object_id: str | None = None
    observation_document: ObservationDocument | None = None
    source_observation_document: ObservationDocument | None = None
    evidence_preparation: dict[str, Any] = field(default_factory=dict)
    observations: list[dict[str, Any]] = field(default_factory=list)
    observation_audit: ObservationAuditResult | None = None
    hypotheses: list[Hypothesis] = field(default_factory=list)
    evidence_plan: list[EvidenceQuestion] = field(default_factory=list)
    evidence: dict[str, list[EvidenceItem]] = field(default_factory=dict)
    evidence_comparisons: list[CrossHypothesisEvidenceComparison] = field(
        default_factory=list
    )
    contradictions: dict[str, list[ContradictionItem]] = field(default_factory=dict)
    hypothesis_analyses: dict[str, HypothesisAnalysis] = field(default_factory=dict)
    evidence_audit: EvidenceAuditResult | None = None
    grade_assessment: GradeAssessment | None = None
    audit_result: AuditResult | None = None
    final_decision: dict[str, Any] | None = None
    final_report_markdown: str | None = None
    reasoning_trace: list[dict[str, Any]] = field(default_factory=list)
    evidence_dossier: EvidenceDossier | None = None
    hypothesis_comparison: list[HypothesisComparison] = field(default_factory=list)
    expert_assessment: ProvisionalExpertAssessment | None = None
    decision_audit: DecisionAuditResult | None = None
    candidate_evidence: CandidateEvidenceIntegration | None = None
    grading_scheme: GradingSchemeStatus = field(default_factory=GradingSchemeStatus)
    llm_assessment: dict[str, Any] | None = None
    llm_input: dict[str, Any] | None = None

    def require_observation(self) -> ObservationDocument:
        if self.observation_document is None:
            raise ValueError("Expert reasoning requires an ObservationDocument")
        return self.observation_document

    def to_dict(self) -> dict[str, Any]:
        def dump(value: Any) -> Any:
            if hasattr(value, "model_dump"):
                return value.model_dump(mode="json")
            if isinstance(value, list):
                return [dump(item) for item in value]
            if isinstance(value, dict):
                return {key: dump(item) for key, item in value.items()}
            return value

        active = {
            "object_id": self.object_id,
            "evidence_preparation": dump(self.evidence_preparation),
            "evidence_dossier": dump(self.evidence_dossier),
            "hypothesis_comparison": dump(self.hypothesis_comparison),
            "expert_assessment": dump(self.expert_assessment),
            "decision_audit": dump(self.decision_audit),
            "candidate_evidence": dump(self.candidate_evidence),
            "grading_scheme": dump(self.grading_scheme),
            "reasoning_trace": dump(self.reasoning_trace),
        }
        if self.candidate_evidence is None:
            active.pop("candidate_evidence")
        if self.llm_assessment is not None:
            active["llm_assessment"] = dump(self.llm_assessment)
        return active
