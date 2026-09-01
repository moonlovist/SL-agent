from __future__ import annotations

from collections import defaultdict

from ..core.base_agent import BaseAgent
from ..core.config import ReasoningConfig
from ..core.state import ExpertState
from ..schemas.reasoning import AuditFinding, EvidenceAuditResult


class EvidenceAuditor(BaseAgent):
    """Check provenance, dependence, specificity, and over-interpretation."""

    def __init__(self, config: ReasoningConfig | None = None):
        super().__init__("EvidenceAuditor")
        self.config = config or ReasoningConfig()

    def run(self, state: ExpertState) -> ExpertState:
        self.log("审计证据独立性、区分力与解释强度")
        self.load_skill("evidence_audit_skill")
        topics = ["lens_geometry", "arc_morphology", "intrinsic_galaxy_morphology", "merger_interaction", "artifacts_subtraction", "color_multiband"]
        self.load_knowledge_bundle(topics)
        document = state.require_observation()
        findings: list[AuditFinding] = []
        known_measurements = {m.id for m in document.measurements}
        all_evidence = [item for items in state.evidence.values() for item in items]
        evidence_ids = [item.id for item in all_evidence]
        unique_ids = len(evidence_ids) == len(set(evidence_ids))
        if not unique_ids:
            findings.append(AuditFinding(id="EA-DUP-ID", severity="critical", category="duplication", description="Evidence IDs are not unique.", recommended_action="Regenerate evidence with unique stable IDs."))
        refs_valid = all(set(item.observation_refs).issubset(known_measurements) and (item.observation_refs or item.status == "not_available") for item in all_evidence)
        if not refs_valid:
            findings.append(AuditFinding(id="EA-PROVENANCE", severity="critical", category="provenance", description="One or more evidence items lack valid measurement provenance.", recommended_action="Remove unsupported claims or restore their measurement references."))
        neutral_missing = all(
            item.status != "not_available" or item.polarity == "neutral"
            for item in all_evidence
        )
        if not neutral_missing:
            findings.append(AuditFinding(id="EA-MISSING", severity="critical", category="missing_evidence", description="Unavailable evidence was assigned a directional score.", recommended_action="Represent missing evidence as neutral rather than negative or positive evidence."))
        signatures: dict[tuple[object, ...], list[str]] = defaultdict(list)
        groups: dict[str, list[str]] = defaultdict(list)
        for item in all_evidence:
            if item.status == "evaluated":
                signatures[(item.hypothesis_id, tuple(sorted(item.observation_refs)), item.polarity, item.correlation_group)].append(item.id)
                groups[item.correlation_group].append(item.id)
        duplicates = [refs for refs in signatures.values() if len(refs) > 1]
        for index, refs in enumerate(duplicates, 1):
            findings.append(AuditFinding(id=f"EA-DUP-{index}", severity="warning", category="double_counting", description="Multiple evidence items encode the same observation effect.", evidence_refs=refs, recommended_action="Aggregate this effect once within its correlation group."))
        low_specificity = [item.id for item in all_evidence if item.status == "evaluated" and item.specificity < self.config.evidence_low_specificity_threshold]
        if low_specificity:
            findings.append(AuditFinding(id="EA-LOW-SPEC", severity="info", category="specificity", description="Some evidence is shared by several hypotheses and has low discriminatory power.", evidence_refs=low_specificity, recommended_action="Use it as context, not as a decisive signature."))
        comparison_complete = bool(state.evidence_comparisons) and all({e.hypothesis_id for e in c.effects} == {h.id for h in state.hypotheses} for c in state.evidence_comparisons)
        if not comparison_complete:
            findings.append(AuditFinding(id="EA-COMPARE", severity="critical", category="hypothesis_coverage", description="Cross-hypothesis comparison is incomplete.", recommended_action="Compare each observed fact against every active hypothesis."))
        checks = {
            "unique_evidence_ids": unique_ids,
            "measurement_provenance": refs_valid,
            "missing_evidence_is_neutral": neutral_missing,
            "cross_hypothesis_coverage": comparison_complete,
            "low_specificity_evidence_exposed": True,
        }
        valid = all(checks.values())
        state.evidence_audit = EvidenceAuditResult(valid=valid, findings=findings, checks=checks, correlated_groups={k: v for k, v in groups.items() if len(v) > 1}, low_specificity_evidence_refs=low_specificity, recommendation=("Evidence may proceed to synthesis; correlation groups must be aggregated once." if valid else "Do not trust hypothesis scores until critical evidence-audit findings are corrected."))
        self.record_reasoning(state, "evidence_quality_audit", {**state.evidence_audit.model_dump(mode="json"), "knowledge_topics": topics})
        return state
