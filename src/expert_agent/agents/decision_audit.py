from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState
from ..schemas.reasoning import DecisionAuditResult


class DecisionAudit(BaseAgent):
    """Lightweight integrity review; it never performs hypothesis synthesis."""

    REQUIRED_ALTERNATIVES = {
        "strong_gravitational_lens",
        "intrinsic_galaxy_morphology",
        "merger_or_interaction",
        "foreground_subtraction_or_psf_artifact",
        "noise_bad_pixel_or_projected_source",
    }
    LEAKAGE_TOKENS = (
        "official_expert_score",
        "official_grade",
        "zenodo_label",
        "known_lens",
        "pyautolens",
        "source_reconstruction",
    )

    def __init__(self):
        super().__init__("DecisionAudit")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("轻量检查事实引用、证据独立性、替代解释和评分一致性")
        document = state.require_observation()
        dossier = state.evidence_dossier
        assessment = state.expert_assessment
        if dossier is None or assessment is None:
            raise ValueError("DecisionAudit requires one ExpertAssessmentAgent result")

        known_ids = {item.id for item in document.measurements}
        for collection in (
            document.structures,
            document.ring_candidates,
            document.geometric_paths,
            document.model_candidates,
            document.candidate_systems,
            document.candidate_structure_audits,
        ):
            known_ids.update(item.id for item in collection)
        known_roots = set(type(document).model_fields)

        def ref_exists(ref: str) -> bool:
            if ref in known_ids:
                return True
            root = ref.split(".", 1)[0]
            if root not in known_roots:
                return False
            if "." not in ref:
                return True
            parts = ref.split(".")
            entity_roots = {
                "structures": {item.id for item in document.structures},
                "ring_candidates": {item.id for item in document.ring_candidates},
                "model_candidates": {item.id for item in document.model_candidates},
            }
            if root in entity_roots:
                return len(parts) > 1 and parts[1] in entity_roots[root]
            return True

        hallucinated: list[str] = []
        all_facts = dossier.observed_facts + dossier.supporting_evidence + dossier.contradicting_evidence
        for fact in all_facts:
            missing_refs = [ref for ref in fact.observation_refs if not ref_exists(ref)]
            if missing_refs:
                hallucinated.append(f"{fact.id}: {', '.join(missing_refs)}")

        missing_topics = {item.topic.lower() for item in dossier.missing_information}
        missing_as_negative = [
            fact.id
            for fact in dossier.contradicting_evidence
            if any(topic in fact.statement.lower() for topic in missing_topics)
        ]

        decisive = assessment.decisive_evidence
        duplicate_decisive = sorted({item for item in decisive if decisive.count(item) > 1})
        fact_to_groups: dict[str, list[str]] = {}
        for group in dossier.correlated_evidence_groups:
            for fact_ref in group.fact_refs:
                fact_to_groups.setdefault(fact_ref, []).append(group.id)
        group_counts: dict[str, int] = {}
        for fact_ref in decisive:
            for group in fact_to_groups.get(fact_ref, []):
                group_counts[group] = group_counts.get(group, 0) + 1
        correlated = duplicate_decisive + [
            f"{group}: presented as repeated decisive votes"
            for group, count in group_counts.items()
            if count > 1 and "add" in assessment.assessment_rationale.lower()
        ]

        specificity_refs = set(dossier.lens_specificity.fact_refs)
        measurement_as_proof: list[str] = []
        high_vote_possible = any(
            item in {"B", "A"} for item in assessment.plausible_vote_categories
        )
        if high_vote_possible and not specificity_refs.intersection(decisive):
            measurement_as_proof.append("B/A vote lacks decisive lens-specific relational evidence.")

        recovery_penalized = [
            item.id
            for item in dossier.processing_cautions
            if "recovery" in item.id.lower() and item.effect != "uncertainty_only"
        ]
        alternatives_present = {item.id for item in dossier.competing_explanations}
        alternatives_ignored = sorted(self.REQUIRED_ALTERNATIVES - alternatives_present)

        serialized = " ".join(
            [assessment.assessment_rationale]
            + [fact.statement for fact in dossier.observed_facts]
            + [" ".join(fact.observation_refs) for fact in dossier.observed_facts]
        ).lower()
        leakage = [token for token in self.LEAKAGE_TOKENS if token in serialized]

        mismatch: list[str] = []
        expected = {0: "X", 1: "C", 2: "B", 3: "A"}
        if assessment.expert_vote_anchor is not None and assessment.provisional_grade != expected[assessment.expert_vote_anchor]:
            mismatch.append("Expert vote anchor and provisional grade disagree.")
        if assessment.vote_code != assessment.expert_vote_anchor:
            mismatch.append("Vote code and compatibility anchor disagree.")
        if assessment.vote_category not in {None, "insufficient_for_grading"} and assessment.provisional_grade != assessment.vote_category:
            mismatch.append("Vote category and compatibility grade disagree.")
        if assessment.provisional_score_range is not None and assessment.provisional_grade is not None:
            mismatch.append("An ambiguous score range was forced into a point grade.")

        abstention = not document.data_quality.usable
        if abstention and assessment.vote_category != "insufficient_for_grading":
            mismatch.append("Unusable data should trigger abstention, not X or a lens grade.")

        categories = [
            hallucinated,
            missing_as_negative,
            correlated,
            measurement_as_proof,
            recovery_penalized,
            alternatives_ignored,
            leakage,
            mismatch,
        ]
        corrections = [item for category in categories for item in category]
        state.decision_audit = DecisionAuditResult(
            audit_passed=not corrections,
            hallucinated_facts=hallucinated,
            missing_as_negative=missing_as_negative,
            correlated_evidence_double_counted=correlated,
            measurement_used_as_proof=measurement_as_proof,
            recovery_auto_penalized=recovery_penalized,
            alternative_ignored=alternatives_ignored,
            external_answer_leakage=leakage,
            rationale_score_mismatch=mismatch,
            abstention_recommended=abstention,
            required_corrections=corrections,
        )
        self.record_reasoning(
            state,
            "decision_audit",
            {
                "audit_passed": state.decision_audit.audit_passed,
                "full_hypothesis_reasoning_rerun": False,
                "required_corrections": corrections,
            },
        )
        return state
