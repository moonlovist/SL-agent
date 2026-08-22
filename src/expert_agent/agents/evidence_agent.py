from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState
from ..schemas.observation import Measurement
from ..schemas.reasoning import EvidenceItem


class EvidenceAgent(BaseAgent):
    def __init__(self):
        super().__init__("EvidenceAgent")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("评价观测对各假设的支持关系")
        self.load_skill("evidence_evaluation_skill")
        self.load_skill("evidence_scoring_skill")
        knowledge_topics = [
            "lens_geometry",
            "arc_morphology",
            "intrinsic_galaxy_morphology",
            "merger_interaction",
            "artifacts_subtraction",
            "color_multiband",
        ]
        self.load_knowledge_bundle(knowledge_topics)
        document = state.require_observation()
        by_kind: dict[str, list[Measurement]] = defaultdict(list)
        for measurement in document.measurements:
            by_kind[measurement.kind].append(measurement)
        evidence: dict[str, list[EvidenceItem]] = {
            hypothesis.id: [] for hypothesis in state.hypotheses
        }
        serial = 0

        def add(
            hypothesis_id: str,
            claim: str,
            refs: Iterable[Measurement],
            polarity: str,
            strength: float,
            specificity: float,
            group: str,
            rationale: str,
            status: str = "evaluated",
        ) -> None:
            nonlocal serial
            serial += 1
            ref_list = list(refs)
            reliability = (
                float(np.mean([item.reliability for item in ref_list]))
                if ref_list
                else 0.0
            )
            sign = 1.0 if polarity == "support" else -1.0 if polarity == "oppose" else 0.0
            score = float(np.clip(sign * strength * reliability * specificity, -1.0, 1.0))
            evidence[hypothesis_id].append(
                EvidenceItem(
                    id=f"E{serial}",
                    hypothesis_id=hypothesis_id,
                    claim=claim,
                    observation_refs=[item.id for item in ref_list],
                    polarity=polarity,
                    strength=float(np.clip(strength, 0.0, 1.0)),
                    reliability=reliability,
                    specificity=float(np.clip(specificity, 0.0, 1.0)),
                    weighted_score=score,
                    correlation_group=group,
                    rationale=rationale,
                    status=status,
                )
            )

        for structure in document.structures:
            structure_measurements = {
                item.kind: item
                for item in document.measurements
                if item.structure_id == structure.id
            }
            misalignment = structure_measurements.get("tangential_misalignment")
            if misalignment and misalignment.value is not None:
                value = float(misalignment.value)
                if value <= 30:
                    strength = 1.0 - value / 60.0
                    add(
                        "H1",
                        f"{structure.id} is approximately tangential (misalignment {value:.1f} deg).",
                        [misalignment],
                        "support",
                        strength,
                        0.85,
                        "lens_geometry",
                        "Tangential alignment is expected for strongly lensed arcs around a common deflector.",
                    )
                    add(
                        "H2",
                        f"{structure.id}'s tangential alignment is less typical of an arbitrary internal feature.",
                        [misalignment],
                        "oppose",
                        strength,
                        0.65,
                        "lens_geometry",
                        "This geometry weakens, but does not exclude, an intrinsic morphology explanation.",
                    )
                elif value >= 45:
                    strength = min(1.0, (value - 30.0) / 45.0)
                    add(
                        "H1",
                        f"{structure.id} is poorly aligned with the local tangential direction ({value:.1f} deg).",
                        [misalignment],
                        "oppose",
                        strength,
                        0.8,
                        "lens_geometry",
                        "A compact or radially oriented component is less characteristic of a lensed arc.",
                    )
                    add(
                        "H2",
                        f"{structure.id} does not show strong tangential alignment.",
                        [misalignment],
                        "support",
                        strength,
                        0.55,
                        "lens_geometry",
                        "Non-tangential morphology is compatible with intrinsic galaxy structure.",
                    )

            extent = structure_measurements.get("angular_extent")
            if extent and extent.value is not None:
                value = float(extent.value)
                if value >= 35:
                    add(
                        "H1",
                        f"{structure.id} is extended over {value:.1f} deg around the center.",
                        [extent],
                        "support",
                        min(1.0, value / 120.0),
                        0.65,
                        "arc_morphology",
                        "A long curved residual is more arc-like than a compact source, though not unique to lensing.",
                    )
                elif value < 20:
                    add(
                        "H1",
                        f"{structure.id} is compact in angular extent ({value:.1f} deg).",
                        [extent],
                        "oppose",
                        0.35,
                        0.45,
                        "arc_morphology",
                        "Compact residuals are less diagnostic of strong lensing.",
                    )

            snr = structure_measurements.get("integrated_snr")
            if snr and snr.value is not None and float(snr.value) >= 6:
                strength = min(1.0, float(snr.value) / 20.0)
                add(
                    "H1",
                    f"{structure.id} is detected at integrated S/N {float(snr.value):.1f}.",
                    [snr],
                    "support",
                    strength,
                    0.3,
                    "signal_reality",
                    "High S/N supports a real residual structure but is not specific to lensing.",
                )
                add(
                    "H4",
                    f"{structure.id}'s integrated S/N is inconsistent with a pure random-noise fluctuation.",
                    [snr],
                    "oppose",
                    strength,
                    0.55,
                    "signal_reality",
                    "Statistical significance weakens a pure-noise explanation, while subtraction systematics remain possible.",
                )

            stability_refs = [
                item
                for kind in ("threshold_survival_fraction", "sep_survival_fraction")
                if (item := structure_measurements.get(kind)) is not None
                and item.value is not None
            ]
            if stability_refs:
                stability = float(np.mean([float(item.value) for item in stability_refs]))
                if stability >= 0.55:
                    add(
                        "H1",
                        f"{structure.id} persists across detector settings (mean stability {stability:.2f}).",
                        stability_refs,
                        "support",
                        stability,
                        0.35,
                        "shared_residual_detection",
                        "Persistence supports a reproducible structure, but the tests share one residual image and are correlated.",
                    )
                    add(
                        "H4",
                        f"{structure.id} is stable across threshold and deblending settings.",
                        stability_refs,
                        "oppose",
                        stability,
                        0.6,
                        "shared_residual_detection",
                        "Parameter stability weakens tuning-specific detections without excluding model-subtraction artifacts.",
                    )
                elif stability < 0.3:
                    add(
                        "H4",
                        f"{structure.id} is unstable across detection settings.",
                        stability_refs,
                        "support",
                        1.0 - stability,
                        0.6,
                        "shared_residual_detection",
                        "A parameter-sensitive residual is more vulnerable to noise or deblending artifacts.",
                    )

            if structure.near_core:
                radius = structure_measurements.get("radius")
                add(
                    "H4",
                    f"{structure.id} lies close to the fitted galaxy core.",
                    [radius] if radius else [],
                    "support",
                    0.55,
                    0.7,
                    "subtraction_quality",
                    "Near-core residuals are especially sensitive to foreground-model mismatch and PSF effects.",
                )
                add(
                    "H1",
                    f"{structure.id} is close enough to the core to be subtraction-sensitive.",
                    [radius] if radius else [],
                    "oppose",
                    0.4,
                    0.65,
                    "subtraction_quality",
                    "This component should not be treated as secure lens evidence without additional checks.",
                )

            connection = structure_measurements.get("inner_connection_fraction")
            if connection and connection.value is not None:
                value = float(connection.value)
                if value >= 0.02:
                    strength = min(1.0, value / 0.15)
                    add(
                        "H2",
                        f"{structure.id} is connected to the inner light distribution (footprint fraction {value:.3f}).",
                        [connection],
                        "support",
                        strength,
                        0.65,
                        "galaxy_connectivity",
                        "A connected residual is compatible with a spiral arm, ring, or star-forming structure belonging to the foreground galaxy.",
                    )
                    add(
                        "H3",
                        f"{structure.id} has measurable continuity toward the inner system.",
                        [connection],
                        "support",
                        strength,
                        0.35,
                        "galaxy_connectivity",
                        "Connected irregular structure can be compatible with interaction debris, but this proxy is not merger-specific.",
                    )
                    add(
                        "H1",
                        f"{structure.id} is not cleanly detached from the inner galaxy light.",
                        [connection],
                        "oppose",
                        strength,
                        0.6,
                        "galaxy_connectivity",
                        "Continuity with foreground light weakens a background-source interpretation, although blending can mimic a connection.",
                    )

        common_radius = by_kind.get("common_radius_cv", [])
        if common_radius and common_radius[0].value is not None:
            cv = float(common_radius[0].value)
            if cv <= 0.25:
                strength = 1.0 - min(1.0, cv / 0.25)
                add(
                    "H1",
                    f"Detected structures occupy a common radius (CV={cv:.3f}).",
                    common_radius,
                    "support",
                    strength,
                    0.8,
                    "scene_geometry",
                    "A shared radius about one center is expected for images related by one lens potential.",
                )
                add(
                    "H2",
                    "The common-radius geometry is less naturally explained by an arbitrary disk feature.",
                    common_radius,
                    "oppose",
                    strength,
                    0.55,
                    "scene_geometry",
                    "Intrinsic rings remain a viable exception, so this evidence is not conclusive.",
                )

        structure_count = by_kind.get("structure_count", [])
        if structure_count and int(structure_count[0].value or 0) >= 2:
            add(
                "H1",
                f"{int(structure_count[0].value)} distinct audited residual structures were measured.",
                structure_count,
                "support",
                0.45,
                0.45,
                "scene_geometry",
                "Multiple real components can be related lensed images, but their count alone is weak and independent sources remain possible.",
            )

        foreground_detected = by_kind.get("foreground_detected", [])
        if foreground_detected:
            detected = bool(foreground_detected[0].value)
            if detected:
                add(
                    "H1",
                    "A central light concentration provides a measurable common geometric reference.",
                    foreground_detected,
                    "support",
                    0.45,
                    0.35,
                    "foreground_light",
                    "A central mass tracer is expected in galaxy-scale lens candidates, but central light alone is not lens-specific.",
                )
            else:
                add(
                    "H1",
                    "No central light concentration was robustly measured.",
                    foreground_detected,
                    "oppose",
                    0.6,
                    0.55,
                    "foreground_light",
                    "Without a reliable center, the claimed lens geometry is less secure.",
                )

        coverage = by_kind.get("total_angular_coverage", [])
        if coverage and coverage[0].value is not None:
            value = float(coverage[0].value)
            if value >= 90.0:
                add(
                    "H1",
                    f"The residual structures span {value:.1f} deg around the common center.",
                    coverage,
                    "support",
                    min(1.0, value / 180.0),
                    0.7,
                    "scene_geometry",
                    "Broad azimuthal coverage is compatible with multiple images or a partial Einstein ring.",
                )

        multiple_cores = by_kind.get("multiple_core_count", [])
        if multiple_cores and multiple_cores[0].value is not None:
            count = int(multiple_cores[0].value)
            if count >= 1:
                add(
                    "H3",
                    f"{count} secondary central light concentration(s) passed the screening threshold.",
                    multiple_cores,
                    "support",
                    min(1.0, 0.55 + 0.2 * count),
                    0.75,
                    "interaction_morphology",
                    "Multiple central concentrations are compatible with a merger or interaction, though source deblending can fragment one galaxy.",
                )
                add(
                    "H1",
                    "The central region contains more than one plausible light concentration.",
                    multiple_cores,
                    "oppose",
                    0.4,
                    0.45,
                    "interaction_morphology",
                    "Multiple cores introduce an interaction alternative that must be considered before a simple one-deflector interpretation.",
                )
            else:
                add(
                    "H3",
                    "No secondary central light concentration passed the screening threshold.",
                    multiple_cores,
                    "oppose",
                    0.25,
                    0.4,
                    "interaction_morphology",
                    "The absence of a detected secondary core weakens but does not exclude a merger interpretation.",
                )

        asymmetry = by_kind.get("tidal_asymmetry", [])
        if asymmetry and asymmetry[0].value is not None:
            value = float(asymmetry[0].value)
            core_count = (
                int(multiple_cores[0].value)
                if multiple_cores and multiple_cores[0].value is not None
                else 0
            )
            if value >= 0.35 and core_count >= 1:
                add(
                    "H3",
                    f"The residual has elevated 180-degree asymmetry ({value:.3f}) together with a secondary central concentration.",
                    [asymmetry[0], multiple_cores[0]],
                    "support",
                    min(1.0, value / 0.7),
                    0.55,
                    "interaction_morphology",
                    "The combination is compatible with interaction, while each proxy alone remains non-specific and subtraction-sensitive.",
                )

        core_residual = by_kind.get("core_residual_fraction", [])
        if core_residual and core_residual[0].value is not None:
            value = float(core_residual[0].value)
            if value >= 0.35:
                add(
                    "H4",
                    f"A large fraction of absolute residual flux is concentrated in the core ({value:.3f}).",
                    core_residual,
                    "support",
                    min(1.0, value / 0.7),
                    0.7,
                    "subtraction_quality",
                    "Core-concentrated residuals are compatible with foreground-model or PSF mismatch.",
                )
                add(
                    "H1",
                    "The residual field is strongly influenced by the subtraction-sensitive core.",
                    core_residual,
                    "oppose",
                    min(1.0, value / 0.7),
                    0.6,
                    "subtraction_quality",
                    "Lens-specific morphology is less secure when core modeling dominates the residual budget.",
                )

        bad_pixels = by_kind.get("bad_pixel_fraction", [])
        if bad_pixels and bad_pixels[0].value is not None and float(bad_pixels[0].value) > 0.01:
            value = float(bad_pixels[0].value)
            add(
                "H4",
                f"{value:.2%} of input pixels required non-finite-value replacement.",
                bad_pixels,
                "support",
                min(1.0, value / 0.1),
                0.65,
                "input_quality",
                "A substantial invalid-pixel fraction increases artifact risk.",
            )

        model_success = by_kind.get("galaxy_model_success", [])
        if model_success:
            if bool(model_success[0].value):
                add(
                    "H4",
                    "The preferred isophotal foreground model converged.",
                    model_success,
                    "oppose",
                    0.25,
                    0.35,
                    "subtraction_quality",
                    "Convergence modestly lowers model-failure risk but does not prove accurate subtraction.",
                )
            else:
                add(
                    "H4",
                    "The preferred isophotal model failed and a radial fallback was used.",
                    model_success,
                    "support",
                    0.75,
                    0.8,
                    "subtraction_quality",
                    "Residual morphology is less reliable when the foreground model falls back.",
                )

        for hypothesis_id in ("H1", "H2", "H3"):
            if not by_kind.get("color_consistency"):
                add(
                    hypothesis_id,
                    "Cross-band source-color consistency was not measured.",
                    [],
                    "neutral",
                    0.0,
                    0.0,
                    "color_consistency",
                    "The current single-band observation cannot use color to connect or separate components.",
                    status="not_available",
                )
        if not by_kind.get("multiple_core_count"):
            add(
                "H3",
                "Multiple-core and tidal-asymmetry evidence was not measured.",
                [],
                "neutral",
                0.0,
                0.0,
                "interaction_morphology",
                "The merger hypothesis remains insufficiently tested rather than excluded.",
                status="not_available",
            )
        psf_available = by_kind.get("psf_information_available", [])
        if not psf_available or not bool(psf_available[0].value):
            for hypothesis_id in ("H1", "H4"):
                add(
                    hypothesis_id,
                    "PSF-specific artifact evidence was not available.",
                    psf_available,
                    "neutral",
                    0.0,
                    0.0,
                    "psf_quality",
                    "Without a validated PSF model, PSF residuals cannot be confirmed or excluded.",
                    status="not_available",
                )

        state.evidence = evidence
        self.record_reasoning(
            state,
            "evidence_evaluation",
            {
                "evaluated_counts": {
                    key: sum(item.status == "evaluated" for item in items)
                    for key, items in evidence.items()
                },
                "all_claims_reference_measurements": all(
                    item.observation_refs or item.status == "not_available"
                    for items in evidence.values()
                    for item in items
                ),
                "knowledge_topics": knowledge_topics,
            },
        )
        return state
