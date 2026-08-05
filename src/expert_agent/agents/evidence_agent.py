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
                f"{int(structure_count[0].value)} distinct annular structures were measured.",
                structure_count,
                "support",
                0.45,
                0.45,
                "scene_geometry",
                "Multiple components can be related lensed images, but independent sources are also possible.",
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
            },
        )
        return state
