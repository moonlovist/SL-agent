from __future__ import annotations

from itertools import combinations
import numpy as np

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
from .physical_image_groups import (
    build_physical_image_groups,
    cross_quad_is_supported,
)


def _summarize_colour_configuration_groups(
    pair_configuration_ids: dict[tuple[str, str], set[str]],
    component_pairs: list[dict],
) -> list[dict]:
    """Aggregate pairwise colour tests into auditable multi-image systems.

    A group is fully supported only when high-confidence consistent edges
    connect every candidate required by that geometric configuration. This
    uses a spanning network rather than requiring every possible pair.
    """

    def connected(nodes: set[str], edges: list[tuple[str, str]]) -> bool:
        if len(nodes) < 2:
            return False
        neighbours = {node: set() for node in nodes}
        for first, second in edges:
            neighbours[first].add(second)
            neighbours[second].add(first)
        reached = set()
        pending = [next(iter(nodes))]
        while pending:
            node = pending.pop()
            if node in reached:
                continue
            reached.add(node)
            pending.extend(neighbours[node] - reached)
        return reached == nodes

    records = {
        tuple(sorted((item.get("first"), item.get("second")))): item
        for item in component_pairs
        if item.get("first") and item.get("second")
    }
    configuration_pairs: dict[str, set[tuple[str, str]]] = {}
    for pair, configuration_ids in pair_configuration_ids.items():
        for configuration_id in configuration_ids:
            configuration_pairs.setdefault(configuration_id, set()).add(pair)

    summaries = []
    for configuration_id, expected_pairs in sorted(configuration_pairs.items()):
        nodes = {candidate for pair in expected_pairs for candidate in pair}
        high_consistent_edges = []
        high_conflict_edges = []
        inconclusive_edges = []
        suggestive_consistent_edges = []
        statuses = []
        measured_pair_count = 0
        for pair in sorted(expected_pairs):
            record = records.get(pair)
            status = record.get("status") if record else "unknown"
            statuses.append(status)
            if record is not None:
                measured_pair_count += 1
            if status == "consistent_with_achromatic_scaling":
                high_consistent_edges.append(pair)
            elif status == "strongly_inconsistent":
                high_conflict_edges.append(pair)
            elif status == "inconclusive":
                inconclusive_edges.append(pair)
            elif status == "suggestive_consistent_with_achromatic_scaling":
                suggestive_consistent_edges.append(pair)

        high_connected = connected(nodes, high_consistent_edges)
        corroborating_connected = connected(
            nodes, high_consistent_edges + suggestive_consistent_edges
        )
        if high_conflict_edges and high_consistent_edges:
            status = "group_mixed"
        elif high_conflict_edges:
            status = "group_conflict"
        elif high_connected:
            status = "group_consistent"
        elif high_consistent_edges or suggestive_consistent_edges:
            status = "group_partially_consistent"
        elif inconclusive_edges:
            status = "group_colour_tension"
        else:
            status = "group_unknown"
        summaries.append({
            "configuration_id": configuration_id,
            "status": status,
            "candidate_ids": sorted(nodes),
            "candidate_count": len(nodes),
            "expected_pair_count": len(expected_pairs),
            "measured_pair_count": measured_pair_count,
            "high_consistent_pair_count": len(high_consistent_edges),
            "high_conflict_pair_count": len(high_conflict_edges),
            "inconclusive_pair_count": len(inconclusive_edges),
            "suggestive_consistent_pair_count": len(suggestive_consistent_edges),
            "high_confidence_network_connected": high_connected,
            "corroborating_network_connected": corroborating_connected,
            "pair_statuses": statuses,
        })
    return summaries


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
        from .evidence_preparation import prepare_evidence
        source = state.source_observation_document or state.require_observation()
        state.source_observation_document = source
        document, state.evidence_preparation = prepare_evidence(source, self.config)
        state.observation_document = document
        self.record_reasoning(state, "prepare_pixel_evidence", state.evidence_preparation)
        from ..vision.diagnostics import rewrite_overview_with_grading_evidence
        rewrite_overview_with_grading_evidence(source, state.evidence_preparation)
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
                    f" Comparison threshold occupancy={structure.foreground_comparison_survival_fraction}; "
                    f"comparison nominal aperture SNR={structure.comparison_integrated_snr}; "
                    f"pixel core overlap={structure.subtraction_core_overlap_fraction}; "
                    f"occupied angle={structure.angular_occupied_deg}, radial fractional width={structure.radial_fractional_width}, "
                    f"local tangential fraction={structure.local_tangential_fraction}."
                ),
                [f"structures.{structure.id}", *known_measurements],
                reliability=reliability,
            )
            structure_fact_ids.append(sid)

        review_fact_ids: list[str] = []
        strong_review_fact_ids: list[str] = []
        weak_geometry_review_fact_ids: list[str] = []
        for audit in document.candidate_structure_audits:
            if document.candidate_handoff_review is not None:
                # All measurable candidates were evaluated before this pass.
                # Historical hold/route checks must not supply a second vote.
                continue
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
            if audit.handoff_class == "recovery_candidate" and survives_comparison and geometry_recorded:
                weak_geometry_review_fact_ids.append(fid)
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
                    f"nearby brightness-peak proxy count={alt.multiple_core_count} (not confirmed nuclei), tidal asymmetry={alt.tidal_asymmetry}, "
                    f"core residual fraction={alt.core_residual_fraction}, and bad-pixel fraction={alt.bad_pixel_fraction}."
                ),
                ["alternative_indicators", *alt.measurement_ids],
                reliability="medium",
            )

        comparison_id: str | None = None
        colour = document.supplementary_diagnostics.get("multiband", {})
        colour_fact_id: str | None = None
        if colour:
            colour_fact_id = add_fact("F-multiband-aperture-colours",
                f"Common-PSF calibrated aperture-colour status={colour.get('status')}; "
                f"decontaminated aperture count={len(colour.get('decontaminated_apertures', {}))}. "
                "Colour relations are used only after quality controls; raw contaminated colours cast no vote.",
                ["supplementary_diagnostics"], reliability="medium")
        comparison = document.foreground_model_comparison
        if comparison is not None:
            comparison_id = add_fact(
                "F-independent-foreground-check",
                (
                    f"A same-data foreground model comparison reports success={comparison.comparison_success}; "
                    f"threshold occupancy in the historical structure mask={comparison.measured_structure_pixel_survival_fraction} "
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
            and "foreground_center_ambiguous" not in document.data_quality.flags
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
        signal_credible = bool(usable and any(
            item.integrated_snr >= self.config.candidate_grade_min_snr
            and item.integrated_flux > 0 for item in document.structures
        ))
        review_signal_credible = bool(usable and strong_review_fact_ids)
        signal_status = (
            "unavailable"
            if not usable
            else "credible"
            if signal_credible
            else "mixed"
            if promoted_count or review_signal_credible or weak_geometry_review_fact_ids
            else "weak"
        )
        signal_refs = [*structure_fact_ids, *review_fact_ids]
        if comparison_id:
            signal_refs.append(comparison_id)
        signal_reality = EvidenceDimension(
            status=signal_status,
            fact_refs=signal_refs,
            rationale=(
                "Measurable positive residual signal is distinct from physical origin. Nominal SNR is a detection heuristic, "
                "not calibrated significance; route provenance and same-data persistence are not independent evidence."
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
        ) and promoted_count >= 1
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

        def structure_survival(item):
            if not comparison or not comparison.comparison_success:
                return None
            if item.foreground_comparison_survival_fraction is not None:
                return item.foreground_comparison_survival_fraction
            # Old observations stored this as an atomic measurement, not a
            # structure field. Never substitute a multi-structure pooled value.
            mid = f"M_{item.id}_foreground_comparison_survival_fraction"
            if mid in measurements:
                return measurements[mid].value
            return independent_survival if promoted_count == 1 else None

        def independently_robust_group(items):
            # Morphology eligibility is separate from subtraction sensitivity.
            # Model disagreement is handled explicitly below, never by deleting pixels.
            return bool(items) and all(
                item.integrated_flux > 0
                and item.integrated_snr >= self.config.candidate_grade_min_snr
                for item in items
            )

        curved_structures = [
            item for item in document.structures
            if item.angular_occupied_deg is not None
            and item.angular_occupied_deg >= self.config.curved_min_coverage_deg
            and item.radial_fractional_width is not None
            and item.radial_fractional_width <= self.config.curved_radial_width_max
            and item.local_shape_count >= 3
            and item.local_tangential_fraction is not None
            and item.local_tangential_fraction >= self.config.curved_local_alignment_min
            and item.subtraction_core_overlap_fraction != 1.0
        ]
        curved_ids = {item.id for item in curved_structures}
        foreground_disc_risk_structures = [
            item
            for item in document.structures
            if item.subtraction_core_overlap_fraction is not None
            and item.subtraction_core_overlap_fraction
            >= self.config.foreground_disc_core_overlap_min
            and item.radial_fractional_width is not None
            and item.radial_fractional_width
            >= self.config.foreground_disc_radial_width_min
            and item.local_tangential_fraction is not None
            and item.local_tangential_fraction
            <= self.config.foreground_disc_local_tangential_max
        ]
        foreground_disc_risk_ids = {
            item.id for item in foreground_disc_risk_structures
        }
        if foreground_disc_risk_structures:
            # The legacy pooled scene summary contains every promoted pixel
            # island and cannot subtract the disc-like members.  Disable only
            # that pooled shortcut; specific surviving arc/fold/cusp/ring
            # configurations below remain fully eligible.
            conventional_clear_system = False
        tangential_structures = [
            item
            for item in document.structures
            if item.id not in foreground_disc_risk_ids
            and (item.id in curved_ids or (item.tangential_misalignment_deg is not None
            and item.tangential_misalignment_deg
            <= self.config.tangential_compatible_max_deg
            and item.axis_ratio is not None
            and item.axis_ratio >= self.config.morphology_min_axis_ratio
            and item.angular_extent_deg >= self.config.short_arc_min_extent_deg))
        ]
        radial_structures = [
            item
            for item in document.structures
            if item.id not in foreground_disc_risk_ids
            and item.tangential_misalignment_deg is not None
            and item.tangential_misalignment_deg
            >= self.config.radial_alignment_min_deg
            and item.axis_ratio is not None
            and item.axis_ratio >= self.config.morphology_min_axis_ratio
        ]
        robust_arc_structures = [
            item
            for item in tangential_structures
            if item.subtraction_core_overlap_fraction != 1.0
            and item.angular_extent_deg >= self.config.robust_arc_min_extent_deg
            and (
                item.threshold_survival_fraction is not None
                and item.threshold_survival_fraction > 0.0
            )
        ]

        def angle_distance(first: float, second: float) -> float:
            return abs((first - second + 180.0) % 360.0 - 180.0)

        def radius_compatible(first, second) -> bool:
            return bool(
                first.median_radius_px > 0
                and second.median_radius_px > 0
                and max(first.median_radius_px, second.median_radius_px)
                / min(first.median_radius_px, second.median_radius_px)
                <= self.config.configuration_max_radius_ratio
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
            and item.axis_ratio is not None and item.axis_ratio <= 1.5
            and document.supplementary_diagnostics.get("psf_fwhm_px") is not None
            and item.pixel_count <= np.pi * document.supplementary_diagnostics["psf_fwhm_px"] ** 2
        ]
        compact_multiple = bool(
            len(compact_structures) >= 4
            and all(
                radius_compatible(first, second)
                for first, second in combinations(compact_structures, 2)
            )
            and self._circular_span(
                [item.position_angle_deg for item in compact_structures]
            )
            >= 180.0
        )
        physical_image_groups = build_physical_image_groups(document, self.config)
        cross_quad_geometry = cross_quad_is_supported(
            physical_image_groups, self.config
        )
        robust_single_arc = bool(
            signal_credible and any(independently_robust_group([item]) for item in robust_arc_structures)
        )
        fold_supported = bool(signal_credible and any(independently_robust_group(pair) for pair in fold_pairs))
        cusp_supported = bool(signal_credible and any(independently_robust_group(group) for group in cusp_groups))
        radial_supported = bool(
            signal_credible and radial_tangential_pair and any(
                independently_robust_group([radial, tangent])
                for radial in radial_structures for tangent in tangential_structures
            )
        )
        compact_supported = bool(
            signal_credible and compact_multiple and independently_robust_group(compact_structures)
        )
        cross_quad_supported = bool(
            signal_credible
            and cross_quad_geometry
            and all(
                independently_robust_group(list(group.members))
                for group in physical_image_groups
            )
            and comparison is not None
            and comparison.comparison_success
            and all(
                any(
                    item.comparison_integrated_snr is not None
                    and item.comparison_integrated_snr
                    >= self.config.candidate_grade_min_snr
                    for item in group.members
                )
                for group in physical_image_groups
            )
        )
        clear_system = bool(
            conventional_clear_system
            or robust_single_arc
            or fold_supported
            or cusp_supported
            or radial_supported
            or cross_quad_supported
            or compact_supported
            or (signal_credible and any(independently_robust_group([item]) for item in curved_structures))
        )
        # No scene-only escape from model-sensitivity checks. A partial ring is
        # evaluated even when segmentation gives only one structure.
        clear_system = clear_system and any(independently_robust_group([item]) for item in tangential_structures or document.structures)

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
            physical_image_groups=physical_image_groups,
            cross_quad_geometry=cross_quad_geometry,
            cross_quad_supported=cross_quad_supported,
            compact_structures=compact_structures,
            compact_supported=compact_supported,
            strong_review_fact_ids=strong_review_fact_ids,
            curved_structures=curved_structures,
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
            or cross_quad_geometry
            or tangential_structures
            or radial_structures
            or strong_review_fact_ids
            else "weak"
        )
        lens_fact_refs = (
            ([scene_id] if scene_id and (unified_system or tangential or supported_configuration_ids) else [])
            + ring_fact_ids
            + (structure_fact_ids if supported_configuration_ids or cross_quad_geometry or tangential_structures or radial_structures else [])
            + strong_review_fact_ids
        )
        lens_specificity = EvidenceDimension(
            status=lens_status,
            fact_refs=lens_fact_refs,
            rationale=(
                "Lens specificity is evaluated across annular, tangential, fold, cusp, radial, compact-multiple, and asymmetric-image configurations. "
                "The recorded fixed morphology conditions are provisional heuristics, not universal physical cuts or calibrated Q1 probabilities."
            ),
        )

        alt = document.alternative_indicators
        # Same-data foreground comparisons measure sensitivity, not origin.
        # Colour is a same-source test, so it may vote only for candidate pairs
        # linked by an actually supported lens configuration.  Comparing every
        # detected island with every other island lets one neighbour or residual
        # veto an otherwise valid lens and is not a physical colour test.
        morphology_relevant = [
            *tangential_structures,
            *radial_structures,
            *curved_structures,
        ]
        relevant = list(
            {
                item.id: item
                for item in (morphology_relevant or document.structures)
            }.values()
        )
        relevant_candidate_ids = {
            candidate_id for item in relevant for candidate_id in item.source_candidate_ids
        }
        relationship = colour.get("relationship_audit", {})
        pair_configuration_ids: dict[tuple[str, str], set[str]] = {}

        def add_cross_structure_pairs(configuration_id: str, structures: list) -> None:
            unique = list({item.id: item for item in structures}.values())
            for first_structure, second_structure in combinations(unique, 2):
                for first in first_structure.source_candidate_ids:
                    for second in second_structure.source_candidate_ids:
                        if first == second:
                            continue
                        key = tuple(sorted((first, second)))
                        pair_configuration_ids.setdefault(key, set()).add(configuration_id)

        def add_cross_image_group_pairs(configuration_id: str) -> None:
            # Compare proposed images with each other, never two segmentation
            # fragments assigned to the same physical-image hypothesis.
            for first_group, second_group in combinations(physical_image_groups, 2):
                for first_structure in first_group.members:
                    for second_structure in second_group.members:
                        for first in first_structure.source_candidate_ids:
                            for second in second_structure.source_candidate_ids:
                                if first == second:
                                    continue
                                key = tuple(sorted((first, second)))
                                pair_configuration_ids.setdefault(key, set()).add(
                                    configuration_id
                                )

        if conventional_clear_system:
            add_cross_structure_pairs("annular_or_common_centre_system", tangential_structures)
        if coherent_ring and len(curved_structures) > 1:
            add_cross_structure_pairs("annular_or_partial_ring", curved_structures)
        for index, pair in enumerate(fold_pairs, 1):
            if independently_robust_group(list(pair)):
                add_cross_structure_pairs(f"fold_pair_{index}", list(pair))
        for index, group in enumerate(cusp_groups, 1):
            if independently_robust_group(list(group)):
                add_cross_structure_pairs(f"cusp_group_{index}", list(group))
        if radial_supported:
            for radial in radial_structures:
                for tangent in tangential_structures:
                    if independently_robust_group([radial, tangent]):
                        add_cross_structure_pairs(
                            f"radial_tangential_{radial.id}_{tangent.id}",
                            [radial, tangent],
                        )
        if compact_supported:
            add_cross_structure_pairs("compact_multiple_image", compact_structures)
        if cross_quad_supported:
            add_cross_image_group_pairs("cross_quad")

        all_component_pairs = relationship.get("component_pairs", [])
        colour_pairs = [
            {
                **item,
                "configuration_ids": sorted(
                    pair_configuration_ids[tuple(sorted((item.get("first"), item.get("second"))))]
                ),
            }
            for item in all_component_pairs
            if tuple(sorted((item.get("first"), item.get("second")))) in pair_configuration_ids
        ]
        colour_configuration_groups = _summarize_colour_configuration_groups(
            pair_configuration_ids, all_component_pairs
        )
        consistent_colour_groups = [
            item for item in colour_configuration_groups
            if item["status"] == "group_consistent"
        ]
        partial_colour_groups = [
            item for item in colour_configuration_groups
            if item["status"] == "group_partially_consistent"
        ]
        mixed_colour_groups = [
            item for item in colour_configuration_groups
            if item["status"] == "group_mixed"
        ]
        conflicting_colour_groups = [
            item for item in colour_configuration_groups
            if item["status"] == "group_conflict"
        ]
        tension_colour_groups = [
            item for item in colour_configuration_groups
            if item["status"] == "group_colour_tension"
        ]
        foreground_colours = [
            item for item in relationship.get("candidate_vs_foreground", [])
            if item.get("candidate") in relevant_candidate_ids
        ]
        inconsistent_colour_pairs = [
            item for item in colour_pairs if item.get("status") == "strongly_inconsistent"
        ]
        consistent_colour_pairs = [
            item
            for item in colour_pairs
            if item.get("status") == "consistent_with_achromatic_scaling"
        ]
        suggestive_colour_pairs = [
            item
            for item in colour_pairs
            if item.get("status") in {
                "suggestive_consistent_with_achromatic_scaling",
                "suggestive_colour_conflict",
                "suggestive_inconclusive",
            }
        ]
        reliable_foreground_colours = [
            item for item in foreground_colours
            if item.get("status") in {"distinct_from_foreground", "indistinguishable_from_foreground"}
        ]
        foreground_colour_ambiguity = bool(reliable_foreground_colours) and all(
            item["status"] == "indistinguishable_from_foreground"
            for item in reliable_foreground_colours
        )
        # Candidate-to-candidate colour consistency is the direct lensing test.
        # A candidate resembling the foreground is non-discriminating rather
        # than contradictory: unrelated galaxies can share broad-band colours,
        # and residual lens light biases the aperture in the same direction.
        # Keep that ambiguity visible, but only a measured disagreement between
        # proposed multiple images is allowed to cap A.
        foreground_only_colour_ambiguity = bool(
            foreground_colour_ambiguity
            and not consistent_colour_pairs
            and not any(
                item.get("status") == "distinct_from_foreground"
                for item in foreground_colours
            )
        )
        colour_caps_A = bool(mixed_colour_groups or conflicting_colour_groups)
        foreground_distinct_colours = [
            item
            for item in foreground_colours
            if item.get("status") == "distinct_from_foreground"
        ]
        low_snr_flux_pair_diagnostics = [
            item
            for item in colour_pairs
            if item.get("status") == "unknown"
            and item.get("flux_scale_diagnostic", {}).get("status")
            == "measured_not_calibrated_for_grading"
        ]
        colour_support = bool(
            consistent_colour_groups
            or partial_colour_groups
            or foreground_distinct_colours
        )
        if colour.get("status") != "measured":
            colour_evidence_state = "not_measured"
        elif not pair_configuration_ids:
            colour_evidence_state = (
                "foreground_ambiguous" if foreground_only_colour_ambiguity
                else "foreground_distinct_support" if foreground_distinct_colours
                else "no_testable_same_source_pair"
            )
        elif mixed_colour_groups or (
            conflicting_colour_groups and consistent_colour_groups
        ):
            colour_evidence_state = "mixed"
        elif conflicting_colour_groups:
            colour_evidence_state = "same_source_conflict"
        elif consistent_colour_groups:
            colour_evidence_state = "supports_common_source"
        elif partial_colour_groups:
            colour_evidence_state = "partially_supports_common_source"
        elif tension_colour_groups:
            colour_evidence_state = "same_source_colour_tension"
        elif suggestive_colour_pairs:
            statuses = {item.get("status") for item in suggestive_colour_pairs}
            colour_evidence_state = (
                "suggestive_mixed_colour_relation"
                if len(statuses) > 1
                else next(iter(statuses))
            )
        elif low_snr_flux_pair_diagnostics:
            colour_evidence_state = "low_snr_flux_shape_diagnostic_only"
        else:
            colour_evidence_state = "unknown"
        colour_relation_id: str | None = None
        if relationship:
            colour_relation_id = add_fact(
                "F-colour-relationship-audit",
                f"Colour evidence state={colour_evidence_state}; configuration-linked pairs={len(colour_pairs)} "
                f"of {len(all_component_pairs)} measured candidate pairs; "
                f"multi-image colour groups={len(colour_configuration_groups)} "
                f"(fully consistent={len(consistent_colour_groups)}, partial={len(partial_colour_groups)}, "
                f"tension={len(tension_colour_groups)}, mixed={len(mixed_colour_groups)}, "
                f"conflict={len(conflicting_colour_groups)}); "
                f"strongly inconsistent configuration-linked pairs={len(inconsistent_colour_pairs)}; "
                f"consistent configuration-linked pairs={len(consistent_colour_pairs)}; "
                f"suggestive non-decisive configuration-linked pairs={len(suggestive_colour_pairs)}; "
                f"low-S/N flux-shape-only configuration pairs={len(low_snr_flux_pair_diagnostics)}; "
                f"reliable candidate/foreground relations={len(reliable_foreground_colours)}; "
                f"candidate/foreground relations distinct in colour={len(foreground_distinct_colours)}; "
                f"all reliable candidates foreground-like={foreground_colour_ambiguity}; "
                f"foreground-only ambiguity={foreground_only_colour_ambiguity} (recorded, not an A cap). "
                "Foreground colour separation is corroboration only. Low-S/N flux-shape fits are retained for audit but cast no grade vote. "
                "Reliable comparisons allow an achromatic brightness offset and include a provisional systematic floor.",
                ["supplementary_diagnostics"], source="multiband_corroboration",
                reliability="medium",
            )
        foreground_disc_risk_fact_id: str | None = None
        if foreground_disc_risk_structures:
            foreground_disc_risk_fact_id = add_fact(
                "F-foreground-disc-residual-risk",
                "Structures "
                + ", ".join(item.id for item in foreground_disc_risk_structures)
                + " simultaneously cross the fitted foreground core, span a broad radial range, "
                "and lack local tangential alignment. They remain measured signals but are not "
                "used as arc evidence because an edge-on foreground disc or foreground-light "
                "subtraction mismatch is a viable origin.",
                [f"structures.{item.id}" for item in foreground_disc_risk_structures],
                reliability="medium",
            )
        artifact_strong = False
        # Neither zero threshold occupancy nor fit failure identifies an artifact.
        # Significant opposite-signed aperture flux records unresolved model conflict.
        model_conflict = bool(relevant) and all(
            item.comparison_integrated_snr is not None
            and item.comparison_integrated_snr <= -self.config.candidate_grade_min_snr
            for item in relevant
        )
        tidal_proxy_high = bool(
            alt
            and alt.tidal_asymmetry is not None
            and alt.tidal_asymmetry >= self.config.tidal_asymmetry_proxy_high
        )
        intrinsic_viable = bool(
            alt and (alt.inner_connected_structure_count > 0 or tidal_proxy_high)
        )
        # Correlated subtraction residual proxies cannot establish an origin.
        intrinsic_strong = False
        merger_viable = bool(alt and alt.multiple_core_count is not None and alt.multiple_core_count > 0)
        bad_pixel_viable = bool(alt and alt.bad_pixel_fraction > 0)
        equally_strong_alternative = intrinsic_strong

        support_ids = list(dict.fromkeys(signal_refs + lens_fact_refs +
                                         ([colour_relation_id] if colour_support and colour_relation_id else [])))
        support_facts = [item for item in facts if item.id in support_ids]
        contradiction_facts: list[TraceableFact] = []
        if artifact_strong and comparison_id:
            contradiction_facts.append(next(item for item in facts if item.id == comparison_id))
        if intrinsic_strong and alternative_id:
            contradiction_facts.append(next(item for item in facts if item.id == alternative_id))
        if colour_caps_A and colour_relation_id:
            contradiction_facts.append(next(item for item in facts if item.id == colour_relation_id))

        alternatives = [
            CompetingExplanation(id="strong_gravitational_lens", status="favored" if clear_system and not equally_strong_alternative and not colour_caps_A else "viable" if clear_system or unified_system or cross_quad_geometry or tangential_structures or radial_structures or strong_review_fact_ids else "weakened", supporting_fact_refs=lens_fact_refs + ([colour_relation_id] if colour_support and colour_relation_id else []), contradicting_fact_refs=[item.id for item in contradiction_facts], rationale="Evaluated from supported lens-configuration hypotheses, signal reality and quality-controlled colour relations; colour consistency supports but never creates a lens configuration."),
            CompetingExplanation(id="intrinsic_galaxy_morphology", status="viable" if foreground_disc_risk_fact_id else "unknown", supporting_fact_refs=[foreground_disc_risk_fact_id] if foreground_disc_risk_fact_id else [], contradicting_fact_refs=[], rationale="A core-crossing, radially broad and locally non-tangential residual is compatible with edge-on foreground light, but is not deleted from the evidence. Other residual connectivity and asymmetry proxies remain subtraction-quality indicators rather than an origin diagnosis."),
            CompetingExplanation(id="merger_or_interaction", status="unknown", supporting_fact_refs=[alternative_id] if merger_viable and alternative_id else [], contradicting_fact_refs=[], rationale="Nearby extra brightness peaks are not confirmed nuclei and do not establish that a merger explains the candidate arc. Recorded for follow-up, not a grade veto."),
            CompetingExplanation(id="foreground_subtraction_or_psf_artifact", status="unknown", supporting_fact_refs=[comparison_id] if model_conflict and comparison_id else [], contradicting_fact_refs=[], rationale="No origin-specific PSF/artifact test is implemented. Signed aperture agreement is robustness evidence only; low threshold occupancy or fit failure does not identify contamination."),
            CompetingExplanation(id="noise_bad_pixel_or_projected_source", status="viable" if bad_pixel_viable or not signal_credible else "weakened", supporting_fact_refs=[alternative_id] if bad_pixel_viable and alternative_id else [], contradicting_fact_refs=signal_refs if signal_credible else [], rationale="Persistence and independent-model survival can weaken a noise explanation but cannot identify lensing physics."),
        ]

        missing = self._missing_information(document)
        cautions = self._processing_cautions(document)
        if model_conflict:
            cautions.append(ProcessingCaution(id="PC-signed-model-conflict", description="Relevant apertures have significantly negative nominal flux in the comparison residual; model dependence is unresolved, not proven contamination.", observation_refs=[f"structures.{item.id}" for item in relevant], effect="uncertainty_only"))
        if colour_caps_A and colour_relation_id:
            cautions.append(ProcessingCaution(
                id="PC-colour-A-cap",
                description="Quality-controlled colours of a configuration-linked proposed image pair are strongly inconsistent after allowing an achromatic magnification offset. This caps a clear morphology at B, but is not a non-lens proof because dust, source colour gradients, differential magnification, variability, microlensing and subtraction errors can alter integrated colours.",
                observation_refs=["supplementary_diagnostics"], effect="uncertainty_only"))
        if tension_colour_groups and colour_relation_id:
            cautions.append(ProcessingCaution(
                id="PC-colour-tension",
                description="At least one supported multi-image configuration has quality-controlled colour tension that is neither consistent nor a statistically strong conflict. It can explain retaining an otherwise probable system at B, but cannot by itself cap an independently secure A or establish different source populations.",
                observation_refs=["supplementary_diagnostics"], effect="uncertainty_only"))
        if foreground_only_colour_ambiguity and colour_relation_id:
            cautions.append(ProcessingCaution(
                id="PC-colour-foreground-ambiguity",
                description="Usable candidate colours do not separate the structure from the foreground SED. This is non-discriminating rather than contradictory and therefore does not impose a grade cap.",
                observation_refs=["supplementary_diagnostics"], effect="uncertainty_only"))
        if foreground_disc_risk_fact_id:
            cautions.append(ProcessingCaution(
                id="PC-foreground-disc-residual-risk",
                description="A measured structure has the joint morphology of a core-crossing foreground disc residual. It is retained for audit but cannot by itself support an arc/ring configuration.",
                observation_refs=[f"structures.{item.id}" for item in foreground_disc_risk_structures],
                effect="uncertainty_only",
            ))
        if comparison_id and independent_survival is not None and independent_survival < self.config.independent_signal_robust_min:
            cautions.append(ProcessingCaution(
                id="PC-pooled-model-sensitivity",
                description="Pooled pixel survival is low; inspect structure-specific survival, nearby persistence and negative residuals. The pooled fraction alone is not a non-lens explanation.",
                observation_refs=["foreground_model_comparison"], effect="uncertainty_only",
            ))
        correlated = self._correlated_groups(document, structure_fact_ids)
        unused = [
            UnusedFact(field_path="object_id", reason="Identifier only; never a grading feature."),
            UnusedFact(field_path="input_path", reason="File path only; never a grading feature."),
            UnusedFact(field_path="artifacts", reason="Diagnostic locations are report links, not scientific evidence."),
            UnusedFact(field_path="euclid_q1_input", reason="Deprecated compatibility metadata and potential answer-adjacent context are excluded."),
            UnusedFact(field_path="geometric_paths", reason="Search routes are provenance; route agreement is not direct lens evidence."),
            UnusedFact(field_path="candidate_systems", reason="Route-associated groups are unvalidated and cannot turn candidate count into multiple images."),
        ]

        # Lack of comparison support is uncertainty, not a non-lens verdict.
        comparison_support = any(
            item.comparison_integrated_snr is not None
            and item.comparison_integrated_snr >= self.config.candidate_grade_min_snr
            for item in relevant
        ) if comparison and comparison.comparison_success else False

        # Shadow A-boundary audit -------------------------------------------------
        # Keep candidate discovery permissive, but separately ask whether the
        # evidence is globally specific enough for a high-confidence A vote.
        # This audit is intentionally non-operative until it has been calibrated
        # against held Euclid labels.  It may only suggest A -> B; it never
        # promotes a lower official grade and never reads human annotations.
        psf_fwhm_px = document.supplementary_diagnostics.get("psf_fwhm_px")

        def occupied_bin_gap_deg(first, second) -> float:
            first_bins = (
                first.occupied_angle_bins_5deg
                if first.occupied_angle_bins_5deg is not None
                else [int(first.position_angle_deg // 5)]
            )
            second_bins = (
                second.occupied_angle_bins_5deg
                if second.occupied_angle_bins_5deg is not None
                else [int(second.position_angle_deg // 5)]
            )
            return 5.0 * min(
                min(abs(a - b), 72 - abs(a - b))
                for a in first_bins for b in second_bins
            )

        structure_identity_pairs = []
        for first, second in combinations(tangential_structures, 2):
            angular_gap = occupied_bin_gap_deg(first, second)
            radius_ratio = (
                max(first.median_radius_px, second.median_radius_px)
                / min(first.median_radius_px, second.median_radius_px)
                if min(first.median_radius_px, second.median_radius_px) > 0
                else None
            )
            centroid_separation = float(np.hypot(
                first.centroid_xy[0] - second.centroid_xy[0],
                first.centroid_xy[1] - second.centroid_xy[1],
            ))
            centroid_separation_psf = (
                centroid_separation / psf_fwhm_px
                if psf_fwhm_px is not None and psf_fwhm_px > 0
                else None
            )
            if (
                angular_gap
                <= self.config.identity_ambiguous_max_angular_gap_deg
                and radius_ratio is not None
                and radius_ratio
                <= self.config.identity_ambiguous_max_radius_ratio
            ):
                relation = "fragment_or_merging_images_ambiguous"
            elif angular_gap >= self.config.identity_separated_min_angular_gap_deg:
                relation = "spatially_separated_candidate_images"
            else:
                relation = "physical_image_identity_unresolved"
            structure_identity_pairs.append({
                "first": first.id,
                "second": second.id,
                "relation": relation,
                "angular_footprint_gap_deg": angular_gap,
                "median_radius_ratio": radius_ratio,
                "centroid_separation_px": centroid_separation,
                "centroid_separation_psf": centroid_separation_psf,
                "interpretation_limit": (
                    "Pixel geometry alone cannot distinguish arc fragmentation "
                    "from a close fold/merging-image pair."
                    if relation == "fragment_or_merging_images_ambiguous"
                    else "Separation supports distinct apertures but does not prove a common source."
                    if relation == "spatially_separated_candidate_images"
                    else "Current pixels do not determine the physical image count."
                ),
            })

        structure_observability = []
        for item in document.structures:
            if (
                item.input_radial_control_coverage is None
                or item.input_radial_control_coverage
                < self.config.observability_min_control_coverage
                or item.input_radial_contrast_snr is None
                or item.input_radial_positive_fraction is None
            ):
                observability_state = "control_limited_or_unavailable"
            elif (
                item.input_radial_contrast_snr
                >= self.config.observability_visible_min_contrast_snr
                and item.input_radial_positive_fraction
                >= self.config.observability_visible_min_positive_fraction
            ):
                observability_state = (
                    "input_visible_but_broad_relative_to_psf"
                    if item.radial_width_psf is not None
                    and item.radial_width_psf
                    >= self.config.observability_broad_min_width_psf
                    else "input_visible"
                )
            elif (
                item.input_radial_contrast_snr < 0
                or item.input_radial_positive_fraction < 0.30
            ):
                observability_state = "residual_only_or_input_negative"
            else:
                observability_state = "marginal_input_visibility"
            structure_observability.append({
                "structure_id": item.id,
                "input_radial_contrast_snr": item.input_radial_contrast_snr,
                "input_radial_positive_fraction": item.input_radial_positive_fraction,
                "input_radial_control_coverage": item.input_radial_control_coverage,
                "radial_width_px": item.radial_width_px,
                "radial_width_psf": item.radial_width_psf,
                "source_candidate_ids": list(item.source_candidate_ids),
                "measurement_status": (
                    "measured"
                    if item.input_radial_contrast_snr is not None
                    else "unavailable"
                ),
                "observability_state": observability_state,
                "physical_origin": "unresolved",
            })

        def shadow_radius_compatible(first, second) -> bool:
            return bool(
                first.median_radius_px > 0
                and second.median_radius_px > 0
                and max(first.median_radius_px, second.median_radius_px)
                / min(first.median_radius_px, second.median_radius_px)
                <= self.config.a_shadow_configuration_max_radius_ratio
            )

        def independent_structure_group(items) -> bool:
            provenance_sets = [
                set(item.source_candidate_ids) or {item.id}
                for item in items
            ]
            return all(
                first.isdisjoint(second)
                for first, second in combinations(provenance_sets, 2)
            )

        def shadow_group_is_robust(items) -> bool:
            return bool(
                independently_robust_group(items)
                and independent_structure_group(items)
                and all(
                    item.subtraction_core_overlap_fraction != 1.0
                    for item in items
                )
                and sum(item.angular_extent_deg for item in items)
                >= self.config.a_shadow_multi_min_total_extent_deg
                and all(
                    shadow_radius_compatible(first, second)
                    for first, second in combinations(items, 2)
                )
            )

        shadow_long_arcs = [
            item
            for item in robust_arc_structures
            if item.angular_extent_deg
            >= self.config.a_shadow_long_arc_min_extent_deg
            and item.radial_fractional_width is not None
            and item.radial_fractional_width
            <= self.config.curved_radial_width_max
            and item.tangential_misalignment_deg is not None
            and item.tangential_misalignment_deg
            <= self.config.tangential_misalignment_max_deg
            and item.local_shape_count >= 2
            and item.local_tangential_fraction is not None
            and item.local_tangential_fraction
            >= self.config.curved_local_alignment_min
        ]
        # The local-chain curved detector is stricter than the simple arc
        # detector (>=90 deg occupied angle plus local tangential agreement), so
        # it remains a valid long-arc anchor even if its bounding-span field is
        # conservative.
        shadow_long_arc_ids = sorted({
            item.id for item in [*shadow_long_arcs, *curved_structures]
            if independently_robust_group([item])
        })
        shadow_fold_pairs = [
            pair for pair in fold_pairs if shadow_group_is_robust(pair)
        ]
        shadow_cusp_groups = [
            group
            for group in cusp_groups
            if self._circular_span([item.position_angle_deg for item in group])
            <= self.config.a_shadow_cusp_group_max_span_deg
            and shadow_group_is_robust(group)
        ]
        # Doubles are a real strong-lens configuration and must not be forced
        # into a fold/cusp template.  The two structures must lie on broadly
        # opposite sides of the adopted foreground centre.
        shadow_double_pairs = [
            pair
            for pair in combinations(tangential_structures, 2)
            if angle_distance(
                pair[0].position_angle_deg, pair[1].position_angle_deg
            ) >= self.config.a_shadow_double_min_separation_deg
            and shadow_group_is_robust(pair)
        ]
        # A generic multi-image system is accepted only when at least three
        # independently measured structures occupy separated azimuths.  This
        # closes the gap between fold/cusp/cross templates without counting
        # several fragments of one arc as several images.
        shadow_distributed_groups = [
            group
            for group in combinations(tangential_structures, 3)
            if self._circular_span(
                [item.position_angle_deg for item in group]
            ) >= self.config.a_shadow_distributed_min_span_deg
            and min(
                angle_distance(first.position_angle_deg, second.position_angle_deg)
                for first, second in combinations(group, 2)
            ) >= self.config.a_shadow_distributed_min_pair_separation_deg
            and shadow_group_is_robust(group)
        ]
        shadow_ring_anchor = bool(
            signal_credible
            and coherent_ring
            and any(
                independently_robust_group([item])
                for item in document.structures
            )
        )
        shadow_long_arc_anchor = bool(signal_credible and shadow_long_arc_ids)
        shadow_multi_anchor = bool(
            signal_credible
            and (
                shadow_fold_pairs
                or shadow_cusp_groups
                or shadow_double_pairs
                or shadow_distributed_groups
                or cross_quad_supported
                or compact_supported
                or radial_supported
            )
        )
        shadow_anchor_types = [
            name
            for name, supported in (
                ("coherent_ring", shadow_ring_anchor),
                ("strict_long_arc", shadow_long_arc_anchor),
                ("globally_validated_multiple_image", shadow_multi_anchor),
            )
            if supported
        ]
        shadow_only_weak_multi_anchor = bool(
            shadow_multi_anchor
            and not shadow_ring_anchor
            and not shadow_long_arc_anchor
            and not shadow_double_pairs
            and not shadow_distributed_groups
            and not cross_quad_supported
            and not compact_supported
            and not radial_supported
        )
        shadow_colour_tension_cap = bool(
            colour_evidence_state == "same_source_colour_tension"
            and shadow_only_weak_multi_anchor
        )
        shadow_blocking_reasons: list[str] = []
        if not signal_credible:
            shadow_blocking_reasons.append("signal_not_credible")
        if not shadow_anchor_types:
            shadow_blocking_reasons.append("no_A_strength_lens_specific_anchor")
        if not comparison_support:
            shadow_blocking_reasons.append("no_independent_model_positive_support")
        if model_conflict:
            shadow_blocking_reasons.append("signed_foreground_model_conflict")
        if equally_strong_alternative:
            shadow_blocking_reasons.append("equally_strong_non_lens_alternative")
        if colour_caps_A:
            shadow_blocking_reasons.append("strong_same_source_colour_conflict")
        elif shadow_colour_tension_cap:
            shadow_blocking_reasons.append("colour_tension_in_weak_multi_image_only_case")
        shadow_a_eligible = not shadow_blocking_reasons
        geometry_boundary_shadow = {
            "policy": "diagnostic_only_no_official_grade_change",
            "a_eligible": shadow_a_eligible,
            "anchor_types": shadow_anchor_types,
            "strict_long_arc_structure_ids": shadow_long_arc_ids,
            "validated_fold_pairs": [
                [item.id for item in pair] for pair in shadow_fold_pairs
            ],
            "validated_cusp_groups": [
                [item.id for item in group] for group in shadow_cusp_groups
            ],
            "validated_double_pairs": [
                [item.id for item in pair] for pair in shadow_double_pairs
            ],
            "validated_distributed_groups": [
                [item.id for item in group]
                for group in shadow_distributed_groups
            ],
            "cross_quad_anchor": cross_quad_supported,
            "compact_multiple_anchor": compact_supported,
            "radial_tangential_anchor": radial_supported,
            "colour_tension_caps_weak_multi_only": shadow_colour_tension_cap,
            "blocking_reasons": shadow_blocking_reasons,
            "thresholds": {
                "long_arc_min_extent_deg": self.config.a_shadow_long_arc_min_extent_deg,
                "configuration_max_radius_ratio": self.config.a_shadow_configuration_max_radius_ratio,
                "cusp_group_max_span_deg": self.config.a_shadow_cusp_group_max_span_deg,
                "multi_min_total_extent_deg": self.config.a_shadow_multi_min_total_extent_deg,
                "double_min_separation_deg": self.config.a_shadow_double_min_separation_deg,
                "distributed_min_span_deg": self.config.a_shadow_distributed_min_span_deg,
                "distributed_min_pair_separation_deg": self.config.a_shadow_distributed_min_pair_separation_deg,
            },
            "structure_observability": structure_observability,
            "structure_identity_pairs": structure_identity_pairs,
            "observability_state_counts": {
                state: sum(
                    item["observability_state"] == state
                    for item in structure_observability
                )
                for state in sorted({
                    item["observability_state"]
                    for item in structure_observability
                })
            },
        }
        if not usable:
            vote_category = "insufficient_for_grading"
            categories: tuple[str, ...] = ()
            rationale = "Unresolved signed foreground-model conflict." if model_conflict else "Observation usability or essential noise integrity is insufficient; the expert abstains rather than assigning X."
        elif clear_system and not equally_strong_alternative and comparison_support and not model_conflict and not colour_caps_A:
            vote_category, categories = "A", ("A",)
            rationale = "At least one supported lens configuration is clear in the available image evidence, and no equally strong measured non-lens explanation remains."
        elif clear_system and equally_strong_alternative:
            vote_category, categories = "C", ("C",)
            rationale = "Lens-like morphology and an equally strong measured alternative coexist: possible, not probable."
        elif clear_system and model_conflict:
            vote_category, categories = "C", ("C",)
            rationale = "Lens-like morphology remains available, but opposite-signed subtraction results leave its interpretation unresolved; this is possible-lens evidence, not confirmed contamination or a probable-lens claim."
        elif clear_system and colour_caps_A:
            vote_category, categories = "B", ("B",)
            rationale = "The morphology is clearly lens-like, but quality-controlled colours are strongly inconsistent between proposed images after allowing an achromatic magnification offset. The system remains probable rather than high-confidence A."
        elif clear_system:
            vote_category, categories = "B", ("B",)
            rationale = "Lens-specific morphology favors lensing, but subtraction sensitivity remains unresolved. Comparison failure does not erase the structure or establish contamination."
        elif signal_credible and (
            unified_system or cross_quad_geometry or tangential_structures or radial_tangential_pair
        ) and not equally_strong_alternative:
            vote_category, categories = "B", ("B",)
            rationale = "Credible structures have directional or relational lens specificity, but the current image evidence does not yet form a clear high-confidence configuration."
        elif review_signal_credible or (usable and weak_geometry_review_fact_ids):
            vote_category, categories = "C", ("C",)
            rationale = "Measurable geometry-related held candidates survive comparison, but threshold persistence or extraction is unresolved. They justify only a possible-lens C vote, not X, B, or A."
        elif promoted_count and (tangential_structures or radial_structures):
            vote_category, categories = "C", ("C",)
            rationale = "The candidate has a lens-like morphology, but its system-level specificity is incomplete or a non-lens explanation remains comparably plausible."
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
        geometry_boundary_shadow["official_grade"] = vote_category
        hard_cap_reasons = sorted(set(shadow_blocking_reasons).intersection({
            "signed_foreground_model_conflict",
            "equally_strong_non_lens_alternative",
            "strong_same_source_colour_conflict",
        }))
        geometry_boundary_shadow["hard_cap_reasons"] = hard_cap_reasons
        geometry_boundary_shadow["shadow_grade"] = (
            "B"
            if vote_category == "A" and hard_cap_reasons
            else "review_required"
            if vote_category == "A" and not shadow_a_eligible
            else vote_category
        )
        geometry_boundary_shadow["recommendation_semantics"] = (
            "Missing or weakly measured A anchors request review; only measured positive counterevidence can recommend A-to-B."
        )

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
                "equally_strong_non_lens_explanation": "yes" if equally_strong_alternative else "not_established",
                "colour_relationship_supports_high_confidence": "no" if colour_caps_A else "yes" if colour_support else "unknown",
                "colour_evidence_state": colour_evidence_state,
                "data_sufficient_for_current_anchor": "no" if not usable else "yes",
                "conclusion_survives_removing_key_evidence": "not_tested",
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
        if document.candidate_handoff_review is not None:
            from .candidate_review import review_candidate_handoff

            review_facts, review_groups = review_candidate_handoff(document)
            state.evidence_dossier.candidate_review_facts = review_facts
            state.evidence_dossier.candidate_configuration_review = review_groups
        self.record_reasoning(state, "decision_conditions", {
            "signal_credible": signal_credible, "clear_system": clear_system,
            "physical_image_groups": [
                group.to_trace() for group in physical_image_groups
            ],
            "cross_quad_geometry": cross_quad_geometry,
            "cross_quad_supported": cross_quad_supported,
            "curved_structure_ids": [item.id for item in curved_structures],
            "tangential_structure_ids": [item.id for item in tangential_structures],
            "foreground_disc_residual_risk_ids": sorted(foreground_disc_risk_ids),
            "comparison_supported_ids": [item.id for item in document.structures if independently_robust_group([item])],
            "model_conflict": model_conflict, "multiple_brightness_peaks_only": merger_viable,
            "equally_strong_alternative": equally_strong_alternative,
            "colour_evidence_state": colour_evidence_state,
            "colour_configuration_pair_count": len(colour_pairs),
            "colour_configuration_group_count": len(colour_configuration_groups),
            "colour_configuration_groups": colour_configuration_groups,
            "colour_configuration_pairs": [
                {
                    "first": item.get("first"), "second": item.get("second"),
                    "status": item.get("status"),
                    "evidence_tier": item.get("evidence_tier"),
                    "configuration_ids": item.get("configuration_ids", []),
                    "maximum_color_departure_mag": item.get("maximum_color_departure_mag"),
                    "maximum_departure_sigma": item.get("maximum_departure_sigma"),
                    "reduced_chi2": item.get("reduced_chi2"),
                    "flux_shape_diagnostic_status": item.get("flux_scale_diagnostic", {}).get("status"),
                    "flux_shape_reduced_chi2": item.get("flux_scale_diagnostic", {}).get("same_sed_reduced_chi2"),
                }
                for item in colour_pairs
            ],
            "colour_caps_A": colour_caps_A,
            "colour_support": colour_support,
            "geometry_boundary_shadow": geometry_boundary_shadow,
            "selected_grade": vote_category, "rationale": rationale,
        })
        self.record_reasoning(state, "expert_assessment", {"single_primary_reasoning_pass": True, "subscore_aggregation_used": False, "category_only_vote": True, "vote_category": assessment.vote_category, "plausible_vote_categories": assessment.plausible_vote_categories})
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
        physical_image_groups: list,
        cross_quad_geometry: bool,
        cross_quad_supported: bool,
        compact_structures: list,
        compact_supported: bool,
        strong_review_fact_ids: list[str],
        curved_structures: list,
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
                status="supported" if coherent_ring or conventional_clear_system or curved_structures else "not_supported",
                structure_ids=[item.id for item in document.structures] if conventional_clear_system else [item.id for item in curved_structures],
                observation_refs=["ring_candidates", "scene_geometry", *refs(curved_structures)],
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
                rationale="Looks for nearby tangential components within a broad physically plausible radius envelope. Similar radius strengthens a fold interpretation but exact circular-radius agreement is not required.",
            ),
            LensingConfigurationAssessment(
                id="cusp_arc",
                status="supported" if cusp_groups else "not_supported",
                structure_ids=cusp_members,
                observation_refs=[f"structures.{item}" for item in cusp_members],
                high_confidence_capable=True,
                rationale="Looks for three same-side tangential components within a broad physically plausible radius envelope, without demanding exact equal radii or a symmetric counterarc.",
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
                id="cross_quad",
                status=(
                    "supported"
                    if cross_quad_supported
                    else "possible"
                    if cross_quad_geometry
                    else "not_supported"
                ),
                structure_ids=[
                    structure_id
                    for group in physical_image_groups
                    for structure_id in group.structure_ids
                ] if len(physical_image_groups) == 4 else [],
                observation_refs=[
                    f"structures.{structure_id}"
                    for group in physical_image_groups
                    for structure_id in group.structure_ids
                ] if len(physical_image_groups) == 4 else [],
                high_confidence_capable=cross_quad_supported,
                rationale=(
                    "Tests four conservative PSF-scale physical-image groups in four separated sectors around the foreground centre. "
                    "Grouping never modifies masks, and a cross quad does not require every small image to have a stable fitted axis ratio."
                ),
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
            multiband = document.supplementary_diagnostics.get("multiband", {})
            relationship = multiband.get("relationship_audit", {})
            usable_relations = [
                item for key in ("component_pairs", "candidate_vs_foreground")
                for item in relationship.get(key, []) if item.get("status") not in {"unknown", "inconclusive"}
            ]
            if not usable_relations:
                items.append(MissingInformationItem(
                    topic="quality-controlled source color relationship",
                    status="not_measured" if multiband.get("status") != "measured" else "unknown",
                    reason="No usable calibrated multiband colour relation passed the S/N and local-control checks."))
            else:
                items.append(MissingInformationItem(
                    topic="pure-source color isolation",
                    status="unknown",
                    reason="Colour relationships are measured and used, but residual foreground light, dust, variability and microlensing are not fully separated.",
                    observation_refs=["supplementary_diagnostics"]))
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
            cautions.append(ProcessingCaution(id="PC-primary-model", description="Primary foreground modelling failed; model dependence remains unresolved, not proven artifact.", observation_refs=["data_quality.model_success"], effect="uncertainty_only"))
        reviewed = {e.candidate_id: e.measurement for e in document.candidate_handoff_review.entries} if document.candidate_handoff_review else {}
        for candidate in document.model_candidates:
            if candidate.origin == "path_recovery":
                cautions.append(ProcessingCaution(id=f"PC-recovery-{candidate.id}", description=f"Candidate {candidate.id} entered through route recovery. Recovery is provenance and is not itself negative evidence.", observation_refs=[f"model_candidates.{candidate.id}"], effect="uncertainty_only"))
            item = reviewed.get(candidate.id)
            if (item.near_core if item is not None else candidate.near_core):
                ref = f"candidate_handoff_review.entries.{candidate.id}" if item is not None else f"model_candidates.{candidate.id}"
                cautions.append(ProcessingCaution(id=f"PC-core-{candidate.id}", description=f"Candidate {candidate.id} has core-risk context; measured pixel overlap={item.subtraction_core_overlap_fraction if item is not None else 'unavailable (legacy flag)'}. Not an origin diagnosis.", observation_refs=[ref], effect="uncertainty_only"))
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
            RequestedTest(id="joint_multiband_source_separation", purpose="Replace the current protected radial colour model with joint foreground/source inference and calibrated colour-system validation.", status="future"),
            RequestedTest(id="wcs_physical_scale_check", purpose="Translate image geometry to a physically interpretable angular scale.", status="future"),
        ]
