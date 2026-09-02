from __future__ import annotations

from itertools import combinations

from ..core.base_agent import BaseAgent
from ..core.config import ReasoningConfig
from ..core.state import ExpertState
from ..schemas.reasoning import (
    CompetingExplanation,
    CorrelatedEvidenceGroup,
    EvidenceDimension,
    EvidenceDossier,
    HypothesisComparison,
    LensingConfigurationAssessment,
    MissingInformationItem,
    ProcessingCaution,
    ProvisionalExpertAssessment,
    RequestedTest,
    TraceableFact,
    UnusedFact,
)


class ExpertAssessmentAgent(BaseAgent):
    """The sole active layer that interprets an Observation as a lens expert.

    This is a transparent, provisional rule layer. It compares alternative
    lens configurations and casts one expert category, never a sum or average
    of evidence sub-scores.
    """

    def __init__(self, config: ReasoningConfig | None = None):
        super().__init__("ExpertAssessmentAgent")
        self.config = config or ReasoningConfig()

    def run(self, state: ExpertState) -> ExpertState:
        self.log("建立证据档案并完成唯一一次多假设专家综合判断")
        document = state.require_observation()
        state.object_id = document.object_id
        state.image_path = document.input_path
        measurements = document.measurement_map()
        facts: list[TraceableFact] = []

        def add_fact(
            fid: str,
            statement: str,
            refs: list[str],
            *,
            source: str = "primary_residual_measurement",
            reliability: str = "unknown",
        ) -> str:
            facts.append(
                TraceableFact(
                    id=fid,
                    statement=statement,
                    observation_refs=list(dict.fromkeys(refs)),
                    source_types=[source],
                    reliability=reliability,
                )
            )
            return fid

        add_fact(
            "F-data-quality",
            (
                f"Observation usability is {document.data_quality.usable}; finite fraction "
                f"is {document.data_quality.finite_fraction:.4f} and positive residual noise "
                f"availability is {document.data_quality.residual_robust_rms > 0}."
            ),
            [
                "data_quality.usable",
                "data_quality.finite_fraction",
                "data_quality.residual_robust_rms",
            ],
            reliability="high",
        )
        model_id = add_fact(
            "F-foreground-model",
            (
                f"The primary foreground model reports success={document.data_quality.model_success} "
                f"using {document.data_quality.model_method}."
            ),
            ["data_quality.model_success", "data_quality.model_method"],
            reliability="high",
        )

        structure_fact_ids: list[str] = []
        for structure in document.structures:
            known_measurements = [mid for mid in structure.measurement_ids if mid in measurements]
            rel_values = [measurements[mid].reliability for mid in known_measurements]
            reliability = (
                "high"
                if rel_values and min(rel_values) >= 0.75
                else "medium"
                if rel_values and min(rel_values) >= self.config.observation_min_reliability
                else "low"
            )
            sid = add_fact(
                f"F-structure-{structure.id}",
                (
                    f"Structure {structure.id} has integrated S/N={structure.integrated_snr:.2f}, "
                    f"radius={structure.radius_px:.2f} px, angular extent={structure.angular_extent_deg:.2f} deg, "
                    f"tangential misalignment={structure.tangential_misalignment_deg if structure.tangential_misalignment_deg is not None else 'unknown'} deg, "
                    f"and near_core={structure.near_core}."
                ),
                [f"structures.{structure.id}", *known_measurements],
                reliability=reliability,
            )
            structure_fact_ids.append(sid)

        review_fact_ids: list[str] = []
        strong_review_fact_ids: list[str] = []
        for audit in document.candidate_structure_audits:
            if audit.verdict != "hold_for_review":
                continue
            checks = audit.checks
            measurable = bool(
                checks.get("existing_detector_acceptance")
                and checks.get("connected_signal_present")
                and checks.get("positive_integrated_signal")
            )
            if not measurable:
                continue
            stable = checks.get("snr_multi_threshold_stable") is True
            survives_comparison = checks.get("comparison_model_does_not_fully_erase") is True
            geometry_recorded = bool(
                checks.get("geometry_route_relation_recorded")
                or audit.discovery_context == "global_sep_plus_geometry"
            )
            fid = add_fact(
                f"F-review-candidate-{audit.candidate_id}",
                (
                    f"Held candidate {audit.candidate_id} contains connected positive signal; "
                    f"multi-threshold stable={stable}, independent-model survival={survives_comparison}, "
                    f"and geometry relation recorded={geometry_recorded}. It remains visible to the "
                    "expert as candidate context but is not equal-weight promoted evidence."
                ),
                [audit.id],
                source="primary_residual_measurement",
                reliability="medium" if stable and survives_comparison else "low",
            )
            review_fact_ids.append(fid)
            if stable and survives_comparison and geometry_recorded:
                strong_review_fact_ids.append(fid)

        scene_id: str | None = None
        if document.scene_geometry is not None:
            scene = document.scene_geometry
            scene_id = add_fact(
                "F-scene-geometry",
                (
                    f"The measured scene contains {scene.structure_count} promoted structures; "
                    f"common-radius CV={scene.common_radius_cv}, total angular coverage={scene.total_angular_coverage_deg:.2f} deg, "
                    f"and mean tangential misalignment={scene.mean_tangential_misalignment_deg}."
                ),
                ["scene_geometry", *scene.measurement_ids],
                reliability="high" if scene.measurement_ids else "medium",
            )

        alternative_id: str | None = None
        if document.alternative_indicators is not None:
            alt = document.alternative_indicators
            alternative_id = add_fact(
                "F-alternative-indicators",
                (
                    f"Residual proxies report inner-connected structures={alt.inner_connected_structure_count}, "
                    f"multiple cores={alt.multiple_core_count}, tidal asymmetry={alt.tidal_asymmetry}, "
                    f"core residual fraction={alt.core_residual_fraction}, and bad-pixel fraction={alt.bad_pixel_fraction}."
                ),
                ["alternative_indicators", *alt.measurement_ids],
                reliability="medium",
            )

        comparison_id: str | None = None
        comparison = document.foreground_model_comparison
        if comparison is not None:
            comparison_id = add_fact(
                "F-independent-foreground-check",
                (
                    f"An audit-only independent foreground model reports success={comparison.comparison_success}; "
                    f"measured-structure pixel survival={comparison.measured_structure_pixel_survival_fraction} "
                    f"and residual correlation={comparison.residual_correlation}."
                ),
                ["foreground_model_comparison"],
                source="independent_model_check",
                reliability="medium",
            )

        ring_fact_ids: list[str] = []
        for ring in document.ring_candidates:
            ring_fact_ids.append(
                add_fact(
                    f"F-ring-candidate-{ring.id}",
                    (
                        f"Annular candidate {ring.id} has longest above-background span={ring.longest_above_local_background_deg:.2f} deg "
                        f"and on-minus-off S/N={ring.on_minus_off_ring_snr:.2f}; it remains an annular residual measurement, not a lens model."
                    ),
                    [f"ring_candidates.{ring.id}", *ring.measurement_ids],
                    reliability="medium",
                )
            )

        usable = (
            document.data_quality.usable
            and document.data_quality.finite_fraction >= self.config.observation_min_finite_fraction
            and document.data_quality.background_rms > 0
            and document.data_quality.residual_robust_rms > 0
        )
        independent_survival = (
            comparison.measured_structure_pixel_survival_fraction
            if comparison and comparison.comparison_success
            else None
        )
        persistent_structures = [
            item
            for item in document.structures
            if (item.threshold_survival_fraction is not None and item.threshold_survival_fraction > 0)
            or (item.sep_survival_fraction is not None and item.sep_survival_fraction > 0)
        ]
        promoted_count = len(document.structures)
        signal_credible = bool(
            usable
            and promoted_count
            and (
                (
                    independent_survival is not None
                    and independent_survival >= self.config.independent_signal_survival_min
                )
                or len(persistent_structures) >= max(1, promoted_count // 2)
            )
        )
        review_signal_credible = bool(usable and strong_review_fact_ids)
        signal_status = (
            "unavailable"
            if not usable
            else "credible"
            if signal_credible
            else "mixed"
            if promoted_count or review_signal_credible
            else "weak"
        )
        signal_refs = [*structure_fact_ids, *review_fact_ids]
        if comparison_id:
            signal_refs.append(comparison_id)
        signal_reality = EvidenceDimension(
            status=signal_status,
            fact_refs=signal_refs,
            rationale=(
                "Signal reality is evaluated from promoted residual structures and available persistence or independent-model checks. "
                "High S/N and SEP survival establish detectability only; neither is treated as lens proof."
            ),
        )

        scene = document.scene_geometry
        common_radius = bool(
            scene
            and promoted_count >= 2
            and scene.common_radius_cv is not None
            and scene.common_radius_cv <= self.config.common_radius_cv_max
        )
        tangential = bool(
            scene
            and promoted_count >= 2
            and (
                (
                    scene.mean_tangential_misalignment_deg is not None
                    and scene.mean_tangential_misalignment_deg
                    <= self.config.tangential_misalignment_max_deg
                )
                or (scene.tangential_alignment_fraction is not None and scene.tangential_alignment_fraction >= 0.67)
            )
        )
        angular_coverage = scene.total_angular_coverage_deg if scene else 0.0
        unified_system = (
            common_radius
            and tangential
            and angular_coverage >= self.config.unified_system_min_coverage_deg
        )
        non_core_fraction = sum(not item.near_core for item in document.structures) / promoted_count if promoted_count else 0.0
        coherent_ring = any(
            ring.longest_above_local_background_deg >= 180.0 and ring.on_minus_off_ring_snr > 0
            for ring in document.ring_candidates
        ) and promoted_count >= 2
        conventional_clear_system = bool(
            signal_credible
            and unified_system
            and (
                promoted_count >= 2
                and angular_coverage >= self.config.clear_system_min_coverage_deg
                or coherent_ring
            )
            and non_core_fraction >= 0.5
        )

        tangential_structures = [
            item
            for item in document.structures
            if item.tangential_misalignment_deg is not None
            and item.tangential_misalignment_deg
            <= self.config.tangential_compatible_max_deg
            and item.axis_ratio is not None
            and item.axis_ratio >= self.config.morphology_min_axis_ratio
            and item.angular_extent_deg >= self.config.short_arc_min_extent_deg
        ]
        radial_structures = [
            item
            for item in document.structures
            if item.tangential_misalignment_deg is not None
            and item.tangential_misalignment_deg
            >= self.config.radial_alignment_min_deg
            and item.axis_ratio is not None
            and item.axis_ratio >= self.config.morphology_min_axis_ratio
        ]
        robust_arc_structures = [
            item
            for item in tangential_structures
            if not item.near_core
            and item.angular_extent_deg >= self.config.robust_arc_min_extent_deg
            and (
                item.threshold_survival_fraction is not None
                and item.threshold_survival_fraction > 0.0
            )
        ]
        independently_robust = bool(
            independent_survival is not None
            and independent_survival >= self.config.independent_signal_robust_min
        )

        def angle_distance(first: float, second: float) -> float:
            return abs((first - second + 180.0) % 360.0 - 180.0)

        def radius_compatible(first, second) -> bool:
            mean_radius = (first.radius_px + second.radius_px) / 2.0
            return bool(
                mean_radius > 0
                and abs(first.radius_px - second.radius_px) / mean_radius
                <= self.config.common_radius_cv_max
            )

        fold_pairs = [
            (first, second)
            for first, second in combinations(tangential_structures, 2)
            if radius_compatible(first, second)
            and angle_distance(first.position_angle_deg, second.position_angle_deg)
            <= self.config.fold_pair_max_separation_deg
        ]
        cusp_groups = [
            group
            for group in combinations(tangential_structures, 3)
            if self._circular_span([item.position_angle_deg for item in group])
            <= self.config.cusp_group_max_span_deg
            and all(radius_compatible(first, second) for first, second in combinations(group, 2))
        ]
        radial_tangential_pair = bool(radial_structures and tangential_structures)
        compact_structures = [
            item
            for item in document.structures
            if item.angular_extent_deg <= self.config.compact_image_max_extent_deg
        ]
        compact_multiple = bool(
            len(compact_structures) >= 4
            and common_radius
            and self._circular_span(
                [item.position_angle_deg for item in compact_structures]
            )
            >= 180.0
        )
        robust_single_arc = bool(
            signal_credible and robust_arc_structures and independently_robust
        )
        fold_supported = bool(signal_credible and fold_pairs and independently_robust)
        cusp_supported = bool(signal_credible and cusp_groups and independently_robust)
        radial_supported = bool(
            signal_credible and radial_tangential_pair and independently_robust
        )
        compact_supported = bool(
            signal_credible and compact_multiple and independently_robust
        )
        clear_system = bool(
            conventional_clear_system
            or robust_single_arc
            or fold_supported
            or cusp_supported
            or radial_supported
            or compact_supported
        )

        configuration_hypotheses = self._configuration_hypotheses(
            document=document,
            coherent_ring=coherent_ring,
            conventional_clear_system=conventional_clear_system,
            tangential_structures=tangential_structures,
            robust_arc_structures=robust_arc_structures,
            fold_pairs=fold_pairs,
            cusp_groups=cusp_groups,
            radial_structures=radial_structures,
            radial_supported=radial_supported,
            compact_structures=compact_structures,
            compact_supported=compact_supported,
            strong_review_fact_ids=strong_review_fact_ids,
        )
        supported_configuration_ids = {
            item.id for item in configuration_hypotheses if item.status == "supported"
        }
        lens_status = (
            "unavailable"
            if not usable
            else "credible"
            if clear_system
            else "mixed"
            if unified_system
            or tangential
            or coherent_ring
            or tangential_structures
            or radial_structures
            or strong_review_fact_ids
            else "weak"
        )
        lens_fact_refs = (
            ([scene_id] if scene_id and (unified_system or tangential or supported_configuration_ids) else [])
            + ring_fact_ids
            + (structure_fact_ids if supported_configuration_ids else [])
            + strong_review_fact_ids
        )
        lens_specificity = EvidenceDimension(
            status=lens_status,
            fact_refs=lens_fact_refs,
            rationale=(
                "Lens specificity is evaluated across annular, tangential, fold, cusp, radial, compact-multiple, and asymmetric-image configurations. "
                "No single arc-length, tangential-angle, component-count, or route-membership threshold assigns a grade."
            ),
        )

        alt = document.alternative_indicators
        artifact_strong = bool(
            promoted_count
            and (
                (
                    independent_survival is not None
                    and independent_survival < self.config.independent_signal_survival_min
                )
                or non_core_fraction < 0.5
            )
        )
        if not document.data_quality.model_success:
            artifact_strong = artifact_strong or not (
                comparison
                and comparison.comparison_success
                and independent_survival is not None
                and independent_survival >= self.config.independent_signal_robust_min
            )
        tidal_proxy_high = bool(
            alt
            and alt.tidal_asymmetry is not None
            and alt.tidal_asymmetry >= self.config.tidal_asymmetry_proxy_high
        )
        intrinsic_viable = bool(
            alt and (alt.inner_connected_structure_count > 0 or tidal_proxy_high)
        )
        intrinsic_strong = bool(
            alt
            and alt.inner_connected_structure_count >= 2
            and tidal_proxy_high
            and alt.core_residual_fraction >= 0.5
        )
        merger_viable = bool(alt and alt.multiple_core_count is not None and alt.multiple_core_count > 0)
        bad_pixel_viable = bool(alt and alt.bad_pixel_fraction > 0)
        equally_strong_alternative = artifact_strong or intrinsic_strong or merger_viable

        support_ids = list(dict.fromkeys(signal_refs + lens_fact_refs))
        support_facts = [item for item in facts if item.id in support_ids]
        contradiction_facts: list[TraceableFact] = []
        if artifact_strong and comparison_id:
            contradiction_facts.append(next(item for item in facts if item.id == comparison_id))
        if (intrinsic_strong or merger_viable) and alternative_id:
            contradiction_facts.append(next(item for item in facts if item.id == alternative_id))

        alternatives = [
            CompetingExplanation(id="strong_gravitational_lens", status="favored" if clear_system and not equally_strong_alternative else "viable" if clear_system or unified_system or tangential_structures or radial_structures or strong_review_fact_ids else "weakened", supporting_fact_refs=lens_fact_refs, contradicting_fact_refs=[item.id for item in contradiction_facts], rationale="Evaluated from supported lens-configuration hypotheses and signal reality, not from candidate count, S/N, or one universal arc threshold."),
            CompetingExplanation(id="intrinsic_galaxy_morphology", status="viable" if intrinsic_viable else "unknown" if alternative_id is None else "weakened", supporting_fact_refs=[alternative_id] if intrinsic_viable and alternative_id else [], contradicting_fact_refs=lens_fact_refs if clear_system else [], rationale="Residual connectivity and tidal-asymmetry proxies may support ordinary galaxy morphology. Tidal asymmetry alone is non-specific and cannot become an equally strong alternative."),
            CompetingExplanation(id="merger_or_interaction", status="viable" if merger_viable else "unknown" if not alt or alt.multiple_core_count is None else "weakened", supporting_fact_refs=[alternative_id] if merger_viable and alternative_id else [], contradicting_fact_refs=lens_fact_refs if clear_system else [], rationale="This explanation is kept viable only when the Observation contains a multiple-core or related proxy."),
            CompetingExplanation(id="foreground_subtraction_or_psf_artifact", status="viable" if artifact_strong else "weakened" if comparison_id else "unknown", supporting_fact_refs=[item for item in [model_id, comparison_id] if item] if artifact_strong else [], contradicting_fact_refs=[comparison_id] if comparison_id and independent_survival is not None and independent_survival >= self.config.independent_signal_robust_min else [], rationale="Foreground-model failure, near-core location, and model sensitivity increase uncertainty; none imposes a fixed grade ceiling."),
            CompetingExplanation(id="noise_bad_pixel_or_projected_source", status="viable" if bad_pixel_viable or not signal_credible else "weakened", supporting_fact_refs=[alternative_id] if bad_pixel_viable and alternative_id else [], contradicting_fact_refs=signal_refs if signal_credible else [], rationale="Persistence and independent-model survival can weaken a noise explanation but cannot identify lensing physics."),
        ]

        missing = self._missing_information(document)
        cautions = self._processing_cautions(document)
        correlated = self._correlated_groups(document, structure_fact_ids)
        unused = [
            UnusedFact(field_path="object_id", reason="Identifier only; never a grading feature."),
            UnusedFact(field_path="input_path", reason="File path only; never a grading feature."),
            UnusedFact(field_path="artifacts", reason="Diagnostic locations are report links, not scientific evidence."),
            UnusedFact(field_path="euclid_q1_input", reason="Deprecated compatibility metadata and potential answer-adjacent context are excluded."),
            UnusedFact(field_path="geometric_paths", reason="Search routes are provenance; route agreement is not direct lens evidence."),
            UnusedFact(field_path="candidate_systems", reason="Route-associated groups are unvalidated and cannot turn candidate count into multiple images."),
        ]

        if not usable:
            vote_category = "insufficient_for_grading"
            categories: tuple[str, ...] = ()
            rationale = "Observation usability or essential noise integrity is insufficient; the expert abstains rather than assigning X."
        elif clear_system and not equally_strong_alternative:
            vote_category, categories = "A", ("A",)
            rationale = "At least one supported lens configuration is clear in the available image evidence, and no equally strong measured non-lens explanation remains."
        elif clear_system and equally_strong_alternative:
            vote_category, categories = "B", ("B",)
            rationale = "A lens-specific configuration is present, but an Observation-supported non-lens explanation remains competitive enough to require confirmation."
        elif signal_credible and (
            unified_system or tangential_structures or radial_tangential_pair
        ) and not equally_strong_alternative:
            vote_category, categories = "B", ("B",)
            rationale = "Credible structures have directional or relational lens specificity, but the current image evidence does not yet form a clear high-confidence configuration."
        elif review_signal_credible:
            vote_category, categories = "C", ("C",)
            rationale = "A measurable held candidate remains lens-like, but the extraction conflict prevents a probable or confident lens vote."
        elif promoted_count and (tangential_structures or radial_structures):
            vote_category, categories = "C", ("C",)
            rationale = "The candidate has a lens-like morphology, but its system-level specificity is incomplete or a non-lens explanation remains comparably plausible."
        elif promoted_count and signal_credible:
            vote_category, categories = "C", ("C",)
            rationale = "A credible residual signal is present, but no lens-specific spatial relationship supports a B or A vote."
        else:
            vote_category, categories = "X", ("X",)
            rationale = "The usable Observation does not contain a credible lens-specific structural relationship and is better described by non-lens or noise explanations."

        # One Q1-style expert chooses a category.  The paper's 0/1/2/3 values
        # encode aggregated human votes and are not a meaningful agent score.
        vote_code = None
        anchor = None
        score = None
        score_range = None
        grade = vote_category

        high_vote_possible = vote_category in {"B", "A"}
        decisive = list(dict.fromkeys(lens_fact_refs if high_vote_possible else signal_refs[:1]))
        decisive_counter = list(dict.fromkeys(item.id for item in contradiction_facts))
        assessment = ProvisionalExpertAssessment(
            vote_category=vote_category,
            plausible_vote_categories=categories,
            vote_code=vote_code,
            expert_vote_anchor=anchor,
            provisional_expert_score=score,
            provisional_score_range=score_range,
            provisional_grade=grade,
            scientific_value_flags=[],
            decisive_evidence=decisive,
            decisive_counterevidence=decisive_counter,
            assessment_rationale=rationale,
            assessment_questions={
                "signal_credible": signal_status,
                "lens_specific_relationship": lens_status,
                "unified_lens_system": "yes" if clear_system else "partial" if unified_system else "no",
                "equally_strong_non_lens_explanation": "yes" if equally_strong_alternative else "no",
                "data_sufficient_for_current_anchor": "no" if not usable else "yes",
                "conclusion_survives_removing_key_evidence": "no" if len(decisive) <= 1 and high_vote_possible else "yes" if vote_category is not None else "unknown",
            },
        )

        state.evidence_dossier = EvidenceDossier(
            observed_facts=facts,
            signal_reality=signal_reality,
            lens_specificity=lens_specificity,
            supporting_evidence=support_facts,
            contradicting_evidence=contradiction_facts,
            competing_explanations=alternatives,
            missing_information=missing,
            processing_cautions=cautions,
            correlated_evidence_groups=correlated,
            unused_facts=unused,
            null_tests=[],
            future_tool_requests=self._future_tests(),
            configuration_hypotheses=configuration_hypotheses,
        )
        state.hypothesis_comparison = [
            HypothesisComparison(hypothesis_id=item.id, status=item.status, supporting_fact_refs=item.supporting_fact_refs, contradicting_fact_refs=item.contradicting_fact_refs, missing_information_topics=[entry.topic for entry in missing], rationale=item.rationale)
            for item in alternatives
        ]
        state.expert_assessment = assessment
        self.record_reasoning(state, "expert_assessment", {"single_primary_reasoning_pass": True, "subscore_aggregation_used": False, "category_only_vote": True, "vote_category": vote_category, "plausible_vote_categories": categories})
        return state

    @staticmethod
    def _circular_span(angles: list[float]) -> float:
        if len(angles) < 2:
            return 0.0
        ordered = sorted(float(item) % 360.0 for item in angles)
        gaps = [second - first for first, second in zip(ordered, ordered[1:])]
        gaps.append(ordered[0] + 360.0 - ordered[-1])
        return 360.0 - max(gaps)

    def _configuration_hypotheses(
        self,
        *,
        document,
        coherent_ring: bool,
        conventional_clear_system: bool,
        tangential_structures: list,
        robust_arc_structures: list,
        fold_pairs: list,
        cusp_groups: list,
        radial_structures: list,
        radial_supported: bool,
        compact_structures: list,
        compact_supported: bool,
        strong_review_fact_ids: list[str],
    ) -> list[LensingConfigurationAssessment]:
        def refs(items: list) -> list[str]:
            return [f"structures.{item.id}" for item in items]

        fold_members = list(dict.fromkeys(item.id for pair in fold_pairs for item in pair))
        cusp_members = list(dict.fromkeys(item.id for group in cusp_groups for item in group))
        asymmetric_possible = bool(
            robust_arc_structures
            and (len(document.structures) >= 2 or strong_review_fact_ids)
        )
        isolated = bool(
            len(document.structures) == 1
            and not tangential_structures
            and not radial_structures
        )
        return [
            LensingConfigurationAssessment(
                id="annular_or_partial_ring",
                status="supported" if coherent_ring or conventional_clear_system else "not_supported",
                structure_ids=[item.id for item in document.structures] if conventional_clear_system else [],
                observation_refs=["ring_candidates", "scene_geometry"],
                high_confidence_capable=True,
                rationale="Tests an annulus or partial ring as one configuration; total coverage is descriptive and is never a universal grade cut.",
            ),
            LensingConfigurationAssessment(
                id="tangential_arc",
                status="supported" if robust_arc_structures else "possible" if tangential_structures else "not_supported",
                structure_ids=[item.id for item in tangential_structures],
                observation_refs=refs(tangential_structures),
                high_confidence_capable=True,
                rationale="Allows a robust single or partial tangential arc to support A without requiring a second promoted component.",
            ),
            LensingConfigurationAssessment(
                id="fold_arc",
                status="supported" if fold_pairs else "not_supported",
                structure_ids=fold_members,
                observation_refs=[f"structures.{item}" for item in fold_members],
                high_confidence_capable=True,
                rationale="Looks for nearby, radius-compatible tangential components that may be an unresolved or segmented merging image pair.",
            ),
            LensingConfigurationAssessment(
                id="cusp_arc",
                status="supported" if cusp_groups else "not_supported",
                structure_ids=cusp_members,
                observation_refs=[f"structures.{item}" for item in cusp_members],
                high_confidence_capable=True,
                rationale="Looks for three same-side, radius-compatible tangential components without demanding a symmetric counterarc.",
            ),
            LensingConfigurationAssessment(
                id="radial_arc",
                status="supported" if radial_supported else "possible" if radial_structures else "not_supported",
                structure_ids=[item.id for item in radial_structures],
                observation_refs=refs(radial_structures),
                high_confidence_capable=radial_supported,
                rationale="Treats radial elongation as a legitimate lens configuration when accompanied by a credible tangential structure; it is not rejected by a tangential-angle gate.",
            ),
            LensingConfigurationAssessment(
                id="compact_multiple_image",
                status="supported" if compact_supported else "possible" if len(compact_structures) >= 2 else "not_supported",
                structure_ids=[item.id for item in compact_structures],
                observation_refs=refs(compact_structures),
                high_confidence_capable=compact_supported,
                rationale="Represents doubles, quads, or Einstein-cross-like compact images that need not contain a long arc.",
            ),
            LensingConfigurationAssessment(
                id="asymmetric_multiple_image",
                status="possible" if asymmetric_possible else "not_supported",
                structure_ids=[item.id for item in robust_arc_structures],
                observation_refs=refs(robust_arc_structures),
                high_confidence_capable=False,
                rationale="Keeps a strong image plus weaker, obscured, or held counterpart available without requiring equal brightness or equal-radius segmentation.",
            ),
            LensingConfigurationAssessment(
                id="single_image",
                status="possible" if isolated else "not_supported",
                structure_ids=[item.id for item in document.structures] if isolated else [],
                observation_refs=refs(document.structures) if isolated else [],
                high_confidence_capable=False,
                rationale="A lone non-distinctive image is not promoted to A from morphology alone.",
            ),
        ]

    @staticmethod
    def _missing_information(document) -> list[MissingInformationItem]:
        kinds = {item.kind.lower() for item in document.measurements}
        items: list[MissingInformationItem] = []
        if not any("color" in kind or "colour" in kind for kind in kinds):
            items.append(MissingInformationItem(topic="same-source color consistency", status="not_measured", reason="Observation contains no calibrated multiband color measurement."))
        if not (document.alternative_indicators and document.alternative_indicators.psf_information_available):
            items.append(MissingInformationItem(topic="reliable PSF comparison", status="unavailable", reason="Observation explicitly reports that PSF information is unavailable.", observation_refs=["alternative_indicators.psf_information_available"]))
        if not any("counterimage" in kind or "counter_image" in kind for kind in kinds):
            items.append(MissingInformationItem(topic="counterimage search", status="not_measured", reason="No dedicated counterimage measurement is present; absence is not treated as evidence against lensing."))
        if not document.data_quality.model_success:
            items.append(MissingInformationItem(topic="successful primary foreground model", status="unavailable", reason="The primary foreground fit failed; the independent comparison is retained separately.", observation_refs=["data_quality.model_success"]))
        items.extend([
            MissingInformationItem(topic="spectroscopy or source/lens redshift", status="not_measured", reason="Observation schema contains no spectroscopy or redshift measurement."),
            MissingInformationItem(topic="mass model and common source-plane reconstruction", status="not_measured", reason="No lens mass model or source-plane reconstruction is part of the current visual Observation."),
        ])
        return items

    @staticmethod
    def _processing_cautions(document) -> list[ProcessingCaution]:
        cautions: list[ProcessingCaution] = []
        if not document.data_quality.model_success:
            cautions.append(ProcessingCaution(id="PC-primary-model", description="Primary foreground modelling failed; this increases model-dependence uncertainty but does not impose a grade cap.", observation_refs=["data_quality.model_success"], effect="supports_artifact_alternative"))
        for candidate in document.model_candidates:
            if candidate.origin == "path_recovery":
                cautions.append(ProcessingCaution(id=f"PC-recovery-{candidate.id}", description=f"Candidate {candidate.id} entered through route recovery. Recovery is provenance and is not itself negative evidence.", observation_refs=[f"model_candidates.{candidate.id}"], effect="uncertainty_only"))
            if candidate.near_core:
                cautions.append(ProcessingCaution(id=f"PC-core-{candidate.id}", description=f"Candidate {candidate.id} is near the subtraction-sensitive core.", observation_refs=[f"model_candidates.{candidate.id}"], effect="supports_artifact_alternative"))
            if len(candidate.path_ids) == 1:
                cautions.append(ProcessingCaution(id=f"PC-single-route-{candidate.id}", description=f"Candidate {candidate.id} is supported by one geometric search route; that route is not independent lens evidence.", observation_refs=[f"model_candidates.{candidate.id}"], effect="uncertainty_only"))
        return cautions

    @staticmethod
    def _correlated_groups(document, fact_ids: list[str]) -> list[CorrelatedEvidenceGroup]:
        groups: dict[str, set[str]] = {}
        refs: dict[str, set[str]] = {}
        measurement_map = document.measurement_map()
        for structure, fact_id in zip(document.structures, fact_ids):
            for mid in structure.measurement_ids:
                measurement = measurement_map.get(mid)
                if measurement is None:
                    continue
                groups.setdefault(measurement.correlation_group, set()).add(fact_id)
                refs.setdefault(measurement.correlation_group, set()).add(mid)
        return [
            CorrelatedEvidenceGroup(id=group, fact_refs=sorted(groups[group]), observation_refs=sorted(refs[group]), rationale="Measurements in this group may derive from the same pixels or geometry and are interpreted jointly, never added as independent votes.")
            for group in sorted(groups)
            if len(refs[group]) > 1
        ]

    @staticmethod
    def _future_tests() -> list[RequestedTest]:
        return [
            RequestedTest(id="rotational_null_test", purpose="Compare the candidate aperture with rotated locations at the same radius.", status="future"),
            RequestedTest(id="independent_radial_foreground_subtraction", purpose="Test residual persistence under an independent radial model.", status="future"),
            RequestedTest(id="counterimage_aperture_search", purpose="Measure plausible counterimage locations without assuming non-detection is negative.", status="future"),
            RequestedTest(id="psf_wing_comparison", purpose="Compare candidate morphology with measured PSF wings.", status="future"),
            RequestedTest(id="same_source_color_consistency", purpose="Test whether candidate components have mutually consistent calibrated colors.", status="future"),
            RequestedTest(id="wcs_physical_scale_check", purpose="Translate image geometry to a physically interpretable angular scale.", status="future"),
        ]
