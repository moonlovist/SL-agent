from __future__ import annotations

import unittest

from expert_agent.core.state import ExpertState
from expert_agent.core.workflow import ExpertWorkflow
from expert_agent.schemas.observation import (
    DataQuality,
    Measurement,
    ObservationDocument,
    Provenance,
    StructureObservation,
)


def measurement(
    mid: str,
    kind: str,
    value,
    reliability: float,
    group: str,
    structure_id: str | None = None,
):
    return Measurement(
        id=mid,
        kind=kind,
        value=value,
        reliability=reliability,
        method="unit_test",
        description=kind,
        structure_id=structure_id,
        correlation_group=group,
    )


def make_observation(with_arcs: bool) -> ObservationDocument:
    measurements = [
        measurement("M_count", "structure_count", 2 if with_arcs else 0, 1.0, "scene_geometry"),
        measurement(
            "M_model", "galaxy_model_success", True, 1.0, "subtraction_quality"
        ),
        measurement(
            "M_coverage",
            "total_angular_coverage",
            130.0 if with_arcs else 0.0,
            0.9,
            "scene_geometry",
        ),
        measurement(
            "M_cv",
            "common_radius_cv",
            0.05 if with_arcs else None,
            0.9 if with_arcs else 0.0,
            "scene_geometry",
        ),
    ]
    structures = []
    if with_arcs:
        for index, (pa, misalignment, extent, snr) in enumerate(
            [(10.0, 8.0, 92.0, 24.0), (205.0, 14.0, 48.0, 15.0)], start=1
        ):
            sid = f"S{index}"
            ids = []
            for kind, value, group in [
                ("integrated_snr", snr, "residual_morphology"),
                ("radius", 9.0 + 0.2 * index, "residual_morphology"),
                ("angular_extent", extent, "residual_morphology"),
                ("axis_ratio", 2.5, "residual_morphology"),
                ("tangential_misalignment", misalignment, "residual_morphology"),
                ("threshold_survival_fraction", 0.75, "shared_residual_detection"),
                ("sep_survival_fraction", 1.0, "shared_residual_detection"),
            ]:
                mid = f"M_{sid}_{kind}"
                ids.append(mid)
                measurements.append(measurement(mid, kind, value, 0.9, group, sid))
            structures.append(
                StructureObservation(
                    id=sid,
                    centroid_xy=(150.0, 140.0 + index),
                    radius_px=9.0,
                    position_angle_deg=pa,
                    pixel_count=35,
                    integrated_flux=1.0,
                    integrated_snr=snr,
                    median_radius_px=9.0,
                    angular_extent_deg=extent,
                    axis_ratio=2.5,
                    tangential_misalignment_deg=misalignment,
                    threshold_survival_fraction=0.75,
                    sep_survival_fraction=1.0,
                    near_core=False,
                    measurement_ids=ids,
                )
            )
    return ObservationDocument(
        object_id="synthetic",
        input_path="synthetic.fits",
        input_format="fits",
        band="VIS",
        image_shape=(300, 300),
        nucleus_xy=(150.0, 150.0),
        data_quality=DataQuality(
            usable=True,
            supplied_rms=True,
            finite_fraction=1.0,
            background_median=0.0,
            background_rms=1.0,
            model_method="unit_test",
            model_success=True,
            residual_median=0.0,
            residual_robust_rms=1.0,
        ),
        structures=structures,
        measurements=measurements,
        limitations=["single band"],
        provenance=Provenance(pipeline="unit_test"),
    )


class ReasoningTests(unittest.TestCase):
    def test_arc_geometry_prefers_lens_with_valid_audit(self):
        observation = make_observation(True)
        result = ExpertWorkflow().run(
            ExpertState(observation_document=observation)
        )
        self.assertEqual(result.decision_result.classification, "strong_lens_candidate")
        self.assertEqual(
            result.decision_result.selected_hypothesis, "strong_gravitational_lens"
        )
        self.assertTrue(result.audit_result.valid)
        self.assertTrue(result.decision_result.evidence_refs)
        self.assertEqual(set(result.hypothesis_analyses), {"H1", "H2", "H3", "H4"})
        self.assertTrue(result.evidence_comparisons)
        self.assertTrue(
            all(len(item.effects) == 4 for item in result.evidence_comparisons)
        )
        self.assertTrue(
            any(item.discriminating for item in result.evidence_comparisons)
        )
        self.assertTrue(result.observation_audit.valid)
        self.assertTrue(result.evidence_audit.valid)
        self.assertIn(result.decision_audit.verdict, {"confirm", "downgrade"})
        self.assertEqual(
            result.decision_audit.effective_classification,
            "strong_lens_candidate",
        )

    def test_each_hypothesis_records_missing_evidence_and_uncertainty(self):
        observation = make_observation(True)
        result = ExpertWorkflow().run(
            ExpertState(observation_document=observation)
        )
        merger_analysis = result.hypothesis_analyses["H3"]
        self.assertTrue(merger_analysis.missing_evidence)
        self.assertTrue(merger_analysis.uncertainties)
        self.assertEqual(merger_analysis.status, "insufficient")
        self.assertIn("per_hypothesis_analysis", result.audit_result.checks)
        self.assertTrue(result.audit_result.checks["cross_hypothesis_comparison"])

    def test_empty_observation_abstains(self):
        observation = make_observation(False)
        result = ExpertWorkflow().run(
            ExpertState(observation_document=observation)
        )
        self.assertEqual(result.decision_result.classification, "indeterminate")
        self.assertIsNone(result.decision_result.selected_hypothesis)
        self.assertNotEqual(result.confidence_assessment.level, "medium")

    def test_observation_1_0_remains_readable(self):
        payload = make_observation(True).model_dump(mode="json")
        payload["schema_version"] = "1.0"
        payload.pop("foreground_galaxy", None)
        payload.pop("scene_geometry", None)
        payload.pop("alternative_indicators", None)
        observation = ObservationDocument.model_validate(payload)
        result = ExpertWorkflow().run(
            ExpertState(observation_document=observation)
        )
        self.assertEqual(observation.schema_version, "1.0")
        self.assertEqual(set(result.hypothesis_analyses), {"H1", "H2", "H3", "H4"})

    def test_non_specific_arc_evidence_cannot_force_lens(self):
        observation = make_observation(True)
        observation.structures = observation.structures[:1]
        for item in observation.measurements:
            if item.kind == "structure_count":
                item.value = 1
            elif item.kind == "common_radius_cv":
                item.value = None
                item.reliability = 0.0
            elif item.kind == "tangential_misalignment" and item.structure_id == "S1":
                item.value = 60.0
        observation.structures[0].tangential_misalignment_deg = 60.0
        result = ExpertWorkflow().run(ExpertState(observation_document=observation))
        self.assertNotEqual(result.decision_result.classification, "strong_lens_candidate")

    def test_unusable_observation_forces_audited_abstention(self):
        observation = make_observation(True)
        observation.data_quality.usable = False
        result = ExpertWorkflow().run(ExpertState(observation_document=observation))
        self.assertFalse(result.observation_audit.valid)
        self.assertEqual(result.observation_audit.quality_level, "unusable")
        self.assertEqual(result.decision_audit.verdict, "abstain")
        self.assertEqual(result.decision_audit.effective_classification, "indeterminate")
        self.assertTrue(result.audit_result.valid)
        self.assertEqual(result.final_decision["classification"], "indeterminate")

    def test_evidence_audit_exposes_correlated_groups(self):
        result = ExpertWorkflow().run(
            ExpertState(observation_document=make_observation(True))
        )
        self.assertTrue(result.evidence_audit.correlated_groups)
        self.assertTrue(result.evidence_audit.checks["measurement_provenance"])
        self.assertIn("observation_audit", result.to_dict())
        self.assertIn("decision_audit", result.final_decision)


if __name__ == "__main__":
    unittest.main()
