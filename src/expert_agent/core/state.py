from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schemas.observation import ObservationDocument
from ..schemas.reasoning import (
    AuditResult,
    DecisionAuditResult,
    EvidenceAuditResult,
    ConfidenceAssessment,
    ContradictionItem,
    DecisionResult,
    EvidenceItem,
    CrossHypothesisEvidenceComparison,
    EvidenceQuestion,
    Hypothesis,
    HypothesisAnalysis,
    ObservationAuditResult,
)


@dataclass
class ExpertState:
    image_path: str | None = None
    object_id: str | None = None
    observation_document: ObservationDocument | None = None
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
    decision_result: DecisionResult | None = None
    confidence_assessment: ConfidenceAssessment | None = None
    decision_audit: DecisionAuditResult | None = None
    audit_result: AuditResult | None = None
    final_decision: dict[str, Any] | None = None
    final_report_markdown: str | None = None
    reasoning_trace: list[dict[str, Any]] = field(default_factory=list)

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

        return {
            "object_id": self.object_id,
            "observation": dump(self.observation_document),
            "observation_summary": dump(self.observations),
            "observation_audit": dump(self.observation_audit),
            "hypotheses": dump(self.hypotheses),
            "evidence_plan": dump(self.evidence_plan),
            "evidence": dump(self.evidence),
            "evidence_comparisons": dump(self.evidence_comparisons),
            "contradictions": dump(self.contradictions),
            "hypothesis_analyses": dump(self.hypothesis_analyses),
            "evidence_audit": dump(self.evidence_audit),
            "decision": dump(self.decision_result),
            "confidence": dump(self.confidence_assessment),
            "decision_audit": dump(self.decision_audit),
            "audit": dump(self.audit_result),
            "reasoning_trace": dump(self.reasoning_trace),
        }
