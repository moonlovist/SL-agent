from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from scipy.signal import fftconvolve
from uuid import uuid4

from expert_agent.vision.supplementary import (
    aperture_measurement, measure_colours, measure_arc_chains, measure_signed_routes,
    _gray_colour_test, _tiered_colour_test,
)
from expert_agent.core.config import VisionConfig
from expert_agent.vision.candidate_handoff import attach_candidate_review
from expert_agent.schemas.observation import ModelCandidateObservation
from tests.test_reasoning import make_observation, run
from expert_agent.agents.expert_assessment_agent import (
    _summarize_colour_configuration_groups,
)


@pytest.fixture
def colour_path():
    # Windows sandbox cannot inspect pytest's mode-0700 temp directories.
    # Inherit workspace permissions; retain fixtures for reproducible inspection.
    path = Path(".tmp") / ("colour-fixture-" + uuid4().hex)
    path.mkdir(parents=True)
    return path


def test_signed_route_includes_negative_and_subthreshold_pixels():
    residual = np.array([[4., -4.], [1., -1.]])
    record = measure_signed_routes({"residual": residual, "rms": np.ones((2, 2)),
        "geometric_path_mask_G1": np.ones((2, 2), bool)})[0]
    assert record["signed_flux"] == 0
    assert record["negative_pixel_fraction"] == 0.5
    assert record["above_2sigma_fraction"] == 0.25
    assert record["status"] == "diagnostic_not_calibrated_detection"


def test_signed_route_never_mutates_original_mask():
    mask = np.ones((3, 3), bool)
    measure_signed_routes({"residual": np.ones((3, 3)), "rms": np.ones((3, 3)),
        "source_mask": np.eye(3, dtype=bool), "geometric_path_mask_G1": mask})
    assert mask.all()


def test_convolved_aperture_adjoint_and_noise():
    rng = np.random.default_rng(220)
    shape = (25, 25)
    aperture = np.zeros(shape, bool)
    aperture[10:14, 10:14] = True
    sky = np.zeros(shape, bool)
    sky[2:5, 2:20] = True
    kernel = np.ones((3, 3))/9
    data = rng.normal(size=shape)
    flux, error = aperture_measurement(data, np.ones(shape), kernel, aperture, sky)
    convolved = fftconvolve(data, kernel, mode="same")
    assert flux == pytest.approx(convolved[aperture].sum()-aperture.sum()*convolved[sky].mean())
    samples = [aperture_measurement(rng.normal(size=shape), np.ones(shape), kernel, aperture, sky)[0]
               for _ in range(1000)]
    assert np.std(samples) == pytest.approx(error, rel=0.08)


def make_colour_file(tmp_path, shift=False, missing_zp=False):
    doc = make_observation(structure_count=0)
    yy, xx = np.indices(doc.image_shape)
    cx, cy = doc.nucleus_xy
    source = 200*np.exp(-((xx-cx)**2+(yy-cy)**2)/18)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [150, 150]
    wcs.wcs.crval = [60, -49]
    wcs.wcs.cdelt = [-0.1/3600, 0.1/3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdus = [fits.PrimaryHDU()]
    for band, factor, zp in [("VIS", 2., 24.6), ("NIR_Y", 1., 29.8)]:
        header = wcs.to_header()
        header["MAGZERO"] = zp
        if band == "NIR_Y" and missing_zp:
            del header["MAGZERO"]
        if band == "NIR_Y" and shift:
            header["CRPIX1"] += 1
        calibration = 10**(-0.4*(zp-25))
        hdus.extend([fits.ImageHDU((source*factor+5)/calibration, header, name=f"{band}_FLUX"),
                     fits.ImageHDU(np.ones(doc.image_shape)/calibration, name=f"{band}_RMS"),
                     fits.ImageHDU(np.ones((3,3))/9, name=f"{band}_PSF")])
    path = tmp_path/"input.fits"
    fits.HDUList(hdus).writeto(path)
    return doc, path


def test_colour_zeropoints_common_psf_and_repeat(colour_path):
    doc, path = make_colour_file(colour_path)
    result = measure_colours(doc, {}, path)
    assert result["status"] == "measured"
    colour = result["colours"]["foreground"]["VIS-NIR_Y"]
    assert colour["value_mag"] == pytest.approx(-2.5*np.log10(2))
    assert result == measure_colours(doc, {}, path)


@pytest.mark.parametrize("shift,missing_zp", [(True, False), (False, True)])
def test_unregistered_or_uncalibrated_colour_is_unknown(colour_path, shift, missing_zp):
    doc, path = make_colour_file(colour_path, shift, missing_zp)
    result = measure_colours(doc, {}, path)
    assert result["status"] == "unavailable"
    assert result["bands"]["NIR_Y"]["status"] == "unavailable"
    assert not result["colours"]


def test_arc_fragments_group_without_gap_filling_or_multiple_votes():
    doc = make_observation(structure_count=0)
    yy, xx = np.indices(doc.image_shape)
    cx, cy = doc.nucleus_xy
    radius = np.hypot(xx-cx, yy-cy)
    angle = np.degrees(np.arctan2(yy-cy, xx-cx)) % 360
    arrays = {"residual": np.zeros(doc.image_shape), "rms": np.ones(doc.image_shape)}
    originals = []
    for i, (lo, hi) in enumerate([(0, 35), (39, 74), (78, 113)], 1):
        mask = (radius >= 13)&(radius <= 15)&(angle >= lo)&(angle <= hi)
        originals.append(mask)
        arrays[f"model_candidate_mask_C{i}"] = mask
        arrays["residual"][mask] = 3
        ys, xs = np.where(mask)
        doc.model_candidates.append(ModelCandidateObservation(id=f"C{i}", origin="path_recovery",
            status="geometry_recovered", centroid_xy=(float(xs.mean()), float(ys.mean())), radius_px=14,
            position_angle_deg=lo, pixel_count=int(mask.sum()), integrated_snr=20,
            angular_extent_deg=hi-lo, near_core=False, snr_threshold_survival_count=1, snr_threshold_total_count=8))
    arrays["comparison_psf_mge_residual"] = arrays["residual"].copy()
    doc = attach_candidate_review(doc, arrays)
    chains, records = measure_arc_chains(doc, arrays, 3)
    assert len(chains) == 1
    assert chains[0].pixel_count == np.logical_or.reduce(originals).sum()
    assert chains[0].source_candidate_ids == ["C1", "C2", "C3"]
    doc.arc_chains = chains
    state = run(doc)
    assert state.decision_audit.audit_passed
    assert len(state.require_observation().structures) == 1
    assert state.to_dict() == run(doc).to_dict()


def test_compact_count_alone_without_psf_is_not_supported_quad():
    doc = make_observation(structure_count=4)
    for s in doc.structures:
        s.angular_extent_deg = 10
        s.axis_ratio = 1.1
    state = run(doc)
    compact = next(c for c in state.evidence_dossier.configuration_hypotheses if c.id == "compact_multiple_image")
    assert compact.status != "supported"


def test_combined_intrinsic_proxies_do_not_establish_origin():
    doc = make_observation()
    doc.alternative_indicators.inner_connected_structure_count = 3
    doc.alternative_indicators.tidal_asymmetry = 0.9
    doc.alternative_indicators.core_residual_fraction = 0.9
    state = run(doc)
    intrinsic = next(c for c in state.evidence_dossier.competing_explanations if c.id == "intrinsic_galaxy_morphology")
    assert intrinsic.status == "unknown"
    assert not intrinsic.supporting_fact_refs


def _sed(magnitudes, error=0.03):
    return {band: {"magnitude": value, "magnitude_error": error}
            for band, value in zip(("VIS", "NIR_Y", "NIR_J", "NIR_H"), magnitudes)}


def _flux_sed(fluxes, errors):
    return {band: {"flux_zp25": flux, "error_including_local_controls": error}
            for band, flux, error in zip(("VIS", "NIR_Y", "NIR_J", "NIR_H"), fluxes, errors)}


def test_colour_test_allows_achromatic_magnification_offset():
    first = _sed([23.0, 22.5, 22.2, 22.0])
    second = _sed([24.0, 23.5, 23.2, 23.0])
    assert _gray_colour_test(first, second)["status"] == "consistent_with_achromatic_scaling"


def test_colour_test_finds_sed_shape_inconsistency_not_brightness_difference():
    first = _sed([23.0, 22.5, 22.2, 22.0])
    second = _sed([24.0, 23.5, 23.2, 22.3])
    result = _gray_colour_test(first, second)
    assert result["status"] == "strongly_inconsistent"
    assert result["maximum_color_departure_mag"] >= 0.35
    assert result["reduced_chi2"] == pytest.approx(
        result["chi2"] / result["degrees_of_freedom"]
    )
    assert set(result["per_band_diagnostics"]) == {"VIS", "NIR_Y", "NIR_J", "NIR_H"}


def test_tiered_colour_keeps_snr_2p5_measurement_as_suggestive_only():
    first = _sed([23.0, 22.5, 22.2, 22.0])
    second = _sed([24.0, 23.5, 23.2, 23.0])
    for records in (first, second):
        for index, record in enumerate(records.values()):
            record["snr"] = 2.7 if index < 3 else 2.0
    result = _tiered_colour_test(first, second, VisionConfig())
    assert result["status"] == "suggestive_consistent_with_achromatic_scaling"
    assert result["evidence_tier"] == "suggestive_non_decisive"
    assert result["reliable_common_band_count"] == 3


def test_tiered_colour_keeps_two_high_snr_bands_as_suggestive_only():
    first = _sed([23.0, 22.5, 22.2, 22.0])
    second = _sed([24.0, 23.5, 23.2, 23.0])
    for records in (first, second):
        for index, record in enumerate(records.values()):
            record["snr"] = 4.0 if index < 2 else 2.0
    result = _tiered_colour_test(first, second, VisionConfig())
    assert result["status"] == "suggestive_consistent_with_achromatic_scaling"
    assert result["evidence_tier"] == "suggestive_non_decisive"
    assert result["reliable_common_band_count"] == 2


def test_flux_scale_diagnostic_keeps_low_snr_and_negative_flux_without_grading_it():
    first = _flux_sed([10.0, 8.0, 1.0, -0.5], [1.0, 1.0, 2.0, 2.0])
    second = _flux_sed([5.0, 4.0, 0.5, -0.25], [0.5, 0.5, 1.0, 1.0])
    result = _gray_colour_test(first, second)
    diagnostic = result["flux_scale_diagnostic"]
    assert result["status"] == "unknown"
    assert diagnostic["status"] == "measured_not_calibrated_for_grading"
    assert diagnostic["band_count"] == 4
    assert diagnostic["best_flux_scale_alpha"] == pytest.approx(2.0, rel=1e-3)
    assert diagnostic["same_sed_chi2"] == pytest.approx(0.0, abs=1e-8)


def test_reliable_colour_inconsistency_caps_clear_A_at_B():
    doc = make_observation()
    for index, structure in enumerate(doc.structures, 1):
        structure.source_candidate_ids = [f"C{index}"]
    doc.supplementary_diagnostics = {"multiband": {
        "status": "measured", "decontaminated_apertures": {"C1": {}, "C2": {}},
        "relationship_audit": {
            "component_pairs": [{"first": "C1", "second": "C2", "status": "strongly_inconsistent"}],
            "candidate_vs_foreground": []}}}
    state = run(doc)
    assert state.expert_assessment.vote_category == "B"
    assert "colours" in state.expert_assessment.assessment_rationale
    assert any(item.id == "PC-colour-A-cap" for item in state.evidence_dossier.processing_cautions)


def test_multi_image_colour_group_requires_connected_high_confidence_network():
    mapping = {
        ("C1", "C2"): {"cusp_group_1"},
        ("C1", "C3"): {"cusp_group_1"},
        ("C2", "C3"): {"cusp_group_1"},
    }
    partial = _summarize_colour_configuration_groups(mapping, [
        {"first": "C1", "second": "C2", "status": "consistent_with_achromatic_scaling"},
        {"first": "C1", "second": "C3", "status": "unknown"},
        {"first": "C2", "second": "C3", "status": "unknown"},
    ])[0]
    assert partial["status"] == "group_partially_consistent"
    assert partial["high_confidence_network_connected"] is False

    complete = _summarize_colour_configuration_groups(mapping, [
        {"first": "C1", "second": "C2", "status": "consistent_with_achromatic_scaling"},
        {"first": "C1", "second": "C3", "status": "unknown"},
        {"first": "C2", "second": "C3", "status": "consistent_with_achromatic_scaling"},
    ])[0]
    assert complete["status"] == "group_consistent"
    assert complete["high_confidence_network_connected"] is True


def test_multi_image_colour_group_suggestive_connectivity_stays_partial():
    mapping = {
        ("C1", "C2"): {"compact_multiple_image"},
        ("C2", "C3"): {"compact_multiple_image"},
    }
    group = _summarize_colour_configuration_groups(mapping, [
        {"first": "C1", "second": "C2", "status": "consistent_with_achromatic_scaling"},
        {"first": "C2", "second": "C3", "status": "suggestive_consistent_with_achromatic_scaling"},
    ])[0]
    assert group["status"] == "group_partially_consistent"
    assert group["corroborating_network_connected"] is True
    assert group["high_confidence_network_connected"] is False


def test_inconclusive_configuration_pair_is_group_colour_tension_not_conflict():
    mapping = {("C1", "C2"): {"fold_pair_1"}}
    group = _summarize_colour_configuration_groups(mapping, [{
        "first": "C1",
        "second": "C2",
        "status": "inconclusive",
        "maximum_color_departure_mag": 0.31,
        "maximum_departure_sigma": 1.45,
    }])[0]
    assert group["status"] == "group_colour_tension"
    assert group["inconclusive_pair_count"] == 1


def test_suggestive_colour_conflict_never_caps_a_grade():
    doc = make_observation()
    for index, structure in enumerate(doc.structures, 1):
        structure.source_candidate_ids = [f"C{index}"]
    doc.supplementary_diagnostics = {"multiband": {
        "status": "measured", "decontaminated_apertures": {"C1": {}, "C2": {}},
        "relationship_audit": {
            "component_pairs": [{
                "first": "C1", "second": "C2",
                "status": "suggestive_colour_conflict",
                "evidence_tier": "suggestive_non_decisive",
            }],
            "candidate_vs_foreground": [],
        }}}
    state = run(doc)
    assert state.expert_assessment.vote_category == "A"
    assert state.expert_assessment.assessment_questions["colour_evidence_state"] == "suggestive_colour_conflict"
    assert not any(
        item.id == "PC-colour-A-cap"
        for item in state.evidence_dossier.processing_cautions
    )


def test_foreground_colour_similarity_is_ambiguous_not_a_grade_cap():
    doc = make_observation()
    for index, structure in enumerate(doc.structures, 1):
        structure.source_candidate_ids = [f"C{index}"]
    doc.supplementary_diagnostics = {"multiband": {
        "status": "measured", "decontaminated_apertures": {"C1": {}, "C2": {}},
        "relationship_audit": {
            "component_pairs": [],
            "candidate_vs_foreground": [
                {"candidate": "C1", "status": "indistinguishable_from_foreground"},
                {"candidate": "C2", "status": "indistinguishable_from_foreground"},
            ]}}}
    state = run(doc)
    assert state.expert_assessment.vote_category == "A"
    cautions = {item.id for item in state.evidence_dossier.processing_cautions}
    assert "PC-colour-foreground-ambiguity" in cautions
    assert "PC-colour-A-cap" not in cautions


def test_core_crossing_broad_nontangential_disc_residual_is_not_arc_evidence():
    doc = make_observation(structure_count=1)
    structure = doc.structures[0]
    structure.subtraction_core_overlap_fraction = 0.4
    structure.radial_fractional_width = 0.8
    structure.local_tangential_fraction = 0.0
    structure.tangential_misalignment_deg = 10.0
    structure.axis_ratio = 3.0
    structure.angular_extent_deg = 100.0
    state = run(doc)
    assert state.expert_assessment.vote_category != "A"
    assert any(
        item.id == "PC-foreground-disc-residual-risk"
        for item in state.evidence_dossier.processing_cautions
    )


def test_colour_consistency_supports_but_does_not_create_lens_morphology():
    doc = make_observation(structure_count=1, coherent=False)
    doc.structures[0].source_candidate_ids = ["C1"]
    doc.supplementary_diagnostics = {"multiband": {
        "status": "measured", "decontaminated_apertures": {"C1": {}},
        "relationship_audit": {"component_pairs": [], "candidate_vs_foreground": [
            {"candidate": "C1", "status": "distinct_from_foreground"}]}}}
    assert run(doc).expert_assessment.vote_category != "A"


def test_foreground_colour_difference_is_named_support_but_not_a_grade_creator():
    doc = make_observation(structure_count=1, coherent=False)
    doc.structures[0].source_candidate_ids = ["C1"]
    doc.supplementary_diagnostics = {"multiband": {
        "status": "measured", "decontaminated_apertures": {"C1": {}},
        "relationship_audit": {"component_pairs": [], "candidate_vs_foreground": [
            {"candidate": "C1", "status": "distinct_from_foreground"}]}}}
    state = run(doc)
    assert state.expert_assessment.assessment_questions["colour_evidence_state"] == "foreground_distinct_support"
    assert state.expert_assessment.vote_category != "A"


def test_low_snr_flux_shape_is_audited_without_casting_a_colour_vote():
    doc = make_observation()
    for index, structure in enumerate(doc.structures, 1):
        structure.source_candidate_ids = [f"C{index}"]
    doc.supplementary_diagnostics = {"multiband": {
        "status": "measured", "decontaminated_apertures": {"C1": {}, "C2": {}},
        "relationship_audit": {"component_pairs": [{
            "first": "C1", "second": "C2", "status": "unknown",
            "flux_scale_diagnostic": {
                "status": "measured_not_calibrated_for_grading",
                "same_sed_reduced_chi2": 0.4,
            },
        }], "candidate_vs_foreground": []}}}
    state = run(doc)
    assert state.expert_assessment.assessment_questions["colour_evidence_state"] == "low_snr_flux_shape_diagnostic_only"
    assert state.expert_assessment.vote_category == "A"


def test_unrelated_candidate_colour_conflict_cannot_veto_supported_fold_pair():
    doc = make_observation(structure_count=3)
    doc.scene_geometry.common_radius_cv = 0.9
    for structure, candidate, angle in zip(
        doc.structures, ("C1", "C2", "C3"), (10.0, 30.0, 210.0)
    ):
        structure.source_candidate_ids = [candidate]
        structure.position_angle_deg = angle
    doc.supplementary_diagnostics = {"multiband": {
        "status": "measured", "decontaminated_apertures": {"C1": {}, "C2": {}, "C3": {}},
        "relationship_audit": {
            "component_pairs": [
                {"first": "C1", "second": "C2", "status": "consistent_with_achromatic_scaling"},
                {"first": "C1", "second": "C3", "status": "strongly_inconsistent"},
                {"first": "C2", "second": "C3", "status": "strongly_inconsistent"},
            ],
            "candidate_vs_foreground": [],
        },
    }}
    state = run(doc)
    assert state.expert_assessment.vote_category == "A"
    assert state.expert_assessment.assessment_questions["colour_evidence_state"] == "supports_common_source"
    decision = next(
        item["content"] for item in state.reasoning_trace
        if item["action"] == "decision_conditions"
    )
    assert decision["colour_configuration_pair_count"] == 1
    assert decision["colour_configuration_pairs"][0]["configuration_ids"] == ["fold_pair_1"]
