from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schemas.observation import ObservationDocument
from ..schemas.reasoning import (
    AuditResult,
    ConfidenceAssessment,
    ContradictionItem,
    DecisionResult,
    EvidenceItem,
    EvidenceQuestion,
    Hypothesis,
)


@dataclass
class ExpertState:
    image_path: str | None = None
    object_id: str | None = None
    observation_document: ObservationDocument | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    evidence_plan: list[EvidenceQuestion] = field(default_factory=list)
    evidence: dict[str, list[EvidenceItem]] = field(default_factory=dict)
    contradictions: dict[str, list[ContradictionItem]] = field(default_factory=dict)
    decision_result: DecisionResult | None = None
    confidence_assessment: ConfidenceAssessment | None = None
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
            "hypotheses": dump(self.hypotheses),
            "evidence_plan": dump(self.evidence_plan),
            "evidence": dump(self.evidence),
            "contradictions": dump(self.contradictions),
            "decision": dump(self.decision_result),
            "confidence": dump(self.confidence_assessment),
            "audit": dump(self.audit_result),
            "reasoning_trace": dump(self.reasoning_trace),
        }
