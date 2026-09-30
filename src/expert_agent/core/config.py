from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VisionConfig:
    band: str = "VIS"
    ring_inner_px: float = 4.0
    ring_outer_px: float = 25.0
    center_search_half_size: int = 40
    target_center_xy: tuple[float, float] | None = None
    center_max_offset_px: float = 10.0
    center_ambiguity_margin_px: float = 2.0
    center_blend_radius_px: float = 50.0
    center_competing_size_ratio: float = 1.25
    center_competing_flux_ratio: float = 2.0
    target_candidate_max: int = 4
    target_candidate_min_area_px: float = 8.0
    target_morphology_angle_bins: int = 36
    # Foreground/source separation is a continuous elliptical boundary fitted
    # from adjacent radial profiles that contain a central-to-outer-light
    # minimum.  The old isolated-sector soft weights were removed because they
    # did not define a physical boundary and almost never activated.
    source_boundary_angle_bins: int = 72
    source_boundary_min_contrast_sigma: float = 1.0
    source_boundary_min_angular_support_bins: int = 3
    source_boundary_radius_tolerance_px: float = 2.5
    source_boundary_sky_start_scale: float = 1.5
    source_detect_sigma: float = 1.5
    snr_thresholds: tuple[float, ...] = (2.0, 2.58, 3.34, 4.31, 5.57, 7.19, 9.29, 12.0)
    snr_min_area: int = 3
    snr_min_survival: int = 3
    adaptive_residual_snr: bool = True
    adaptive_snr_scale_min: float = 0.60
    adaptive_snr_scale_max: float = 2.50
    adaptive_snr_threshold_floor: float = 1.00
    sep_nthresh: tuple[int, ...] = (16, 32, 64)
    sep_deblend_cont: tuple[float, ...] = (0.01, 0.00215, 0.00046, 0.0001)
    sep_detect_sigma: float = 2.0
    sep_min_area: int = 5
    sep_min_survival: int = 8
    cluster_distance_px: float = 3.0
    residual_structure_snr_low: float = 2.0
    # Geometry-guided path search. These settings define proposal geometry and
    # reuse the existing residual_structure_snr_low for pixel recovery; they
    # are not lens classification thresholds.
    path_min_half_width_px: float = 1.5
    path_angle_bins: int = 72
    path_center_offset_px: float = 2.0
    path_center_steps: int = 5
    path_axis_ratios: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7)
    path_position_angles_deg: tuple[float, ...] = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)
    path_max_candidates: int = 5
    # Deprecated input alias retained for external v0.5 configuration scripts.
    arc_snr_low: float | None = None
    near_core_radius_px: float = 6.5
    # Multiband colour measurement policy.  These are provisional measurement
    # quality boundaries, not Euclid-Q1 grade cuts.  Keep them in provenance so
    # labelled A/B/C samples can be used to calibrate them later.
    colour_min_band_snr: float = 3.0
    colour_min_common_bands: int = 3
    # A second, explicitly non-decisive tier keeps useful weak-arc colour
    # information without granting noisy measurements promotion/veto power.
    colour_suggestive_min_band_snr: float = 2.5
    colour_suggestive_min_common_bands: int = 2
    colour_min_local_controls: int = 4
    colour_systematic_floor_mag: float = 0.10
    colour_consistent_max_departure_mag: float = 0.20
    colour_consistent_max_reduced_chi2: float = 2.0
    colour_inconsistent_min_departure_mag: float = 0.35
    colour_inconsistent_min_sigma: float = 3.0
    colour_candidate_overlap_max: float = 0.20
    colour_foreground_core_overlap_max: float = 0.15

    @property
    def structure_snr_low(self) -> float:
        return (
            float(self.arc_snr_low)
            if self.arc_snr_low is not None
            else float(self.residual_structure_snr_low)
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReasoningConfig:
    observation_min_finite_fraction: float = 0.99
    observation_min_reliability: float = 0.20
    evidence_low_specificity_threshold: float = 0.35
    common_radius_cv_max: float = 0.30
    tangential_misalignment_max_deg: float = 25.0
    unified_system_min_coverage_deg: float = 60.0
    clear_system_min_coverage_deg: float = 90.0
    # Configuration-recognition bands.  These are interpretation aids rather
    # than universal physical cuts: no single value assigns a grade.
    tangential_compatible_max_deg: float = 35.0
    radial_alignment_min_deg: float = 65.0
    morphology_min_axis_ratio: float = 1.5
    short_arc_min_extent_deg: float = 20.0
    robust_arc_min_extent_deg: float = 30.0
    fold_pair_max_separation_deg: float = 60.0
    cusp_group_max_span_deg: float = 100.0
    compact_image_max_extent_deg: float = 25.0
    # Conservative physical-image grouping for PSF-scale fragments.  This
    # never changes or fills a pixel mask; it only records a grouping
    # hypothesis used by system-level configuration checks.  Complete-link
    # grouping prevents chains of nearby fragments from swallowing a region.
    physical_group_max_separation_psf: float = 0.75
    physical_group_max_polar_separation_deg: float = 15.0
    # A cross-type quad occupies four well-separated sectors around the
    # foreground centre.  Fold/cusp systems intentionally use other branches.
    cross_quad_min_gap_deg: float = 35.0
    cross_quad_max_gap_deg: float = 145.0
    # Multiple images in an elliptical potential or under external shear need
    # not lie at one circular radius. This broad envelope rejects unrelated
    # near/far mixtures without making equal radius a configuration gate.
    configuration_max_radius_ratio: float = 3.0
    # Provisional, auditable A-boundary audit.  These limits do not alter
    # candidate discovery or the official vote.  They ask whether an already
    # detected configuration is globally specific enough for an A vote.  A
    # factor-of-two radial envelope deliberately permits elliptical critical
    # curves and external shear; it is not an equal-radius/circular-ring rule.
    a_shadow_long_arc_min_extent_deg: float = 120.0
    a_shadow_configuration_max_radius_ratio: float = 2.0
    a_shadow_cusp_group_max_span_deg: float = 80.0
    a_shadow_multi_min_total_extent_deg: float = 45.0
    a_shadow_double_min_separation_deg: float = 120.0
    a_shadow_distributed_min_span_deg: float = 120.0
    a_shadow_distributed_min_pair_separation_deg: float = 30.0
    # Identity audit for neighbouring masks.  A close pair can be two merging
    # images or two fragments of one arc; the agent records that ambiguity
    # instead of turning segmentation count into physical image count.
    identity_ambiguous_max_angular_gap_deg: float = 15.0
    identity_ambiguous_max_radius_ratio: float = 1.5
    identity_separated_min_angular_gap_deg: float = 30.0
    # Input-image observability descriptors.  They classify measurement
    # quality for audit and model development; none is a grade threshold.
    observability_min_control_coverage: float = 0.50
    observability_visible_min_contrast_snr: float = 3.0
    observability_visible_min_positive_fraction: float = 0.50
    observability_broad_min_width_psf: float = 1.75
    independent_signal_survival_min: float = 0.50
    independent_signal_robust_min: float = 0.75
    tidal_asymmetry_proxy_high: float = 0.65
    curved_min_coverage_deg: float = 90.0
    curved_radial_width_max: float = 0.5
    curved_local_alignment_min: float = 0.67
    # A broad residual crossing the fitted light centre and lacking local
    # tangential alignment is compatible with an edge-on foreground disc or
    # subtraction mismatch.  All three conditions are required; no single
    # morphology number is allowed to veto a lensed feature.
    foreground_disc_core_overlap_min: float = 0.25
    foreground_disc_radial_width_min: float = 0.55
    foreground_disc_local_tangential_max: float = 0.34
    # Candidate-inclusive grading policy. Comparison occupancy is deliberately
    # not a deletion threshold because both models use the same exposure.
    candidate_grade_min_pixels: int = 5
    candidate_grade_min_snr: float = 4.0
    candidate_grade_min_extent_deg: float = 8.0
    candidate_grade_min_axis_ratio: float = 1.4
    candidate_grade_tangential_max_deg: float = 35.0
    candidate_grade_clear_snr: float = 7.0
    candidate_grade_clear_extent_deg: float = 50.0
    candidate_grade_clear_axis_ratio: float = 1.5
    candidate_grade_clear_tangential_max_deg: float = 25.0
    candidate_grade_ringlike_extent_deg: float = 90.0
    candidate_grade_companion_extent_deg: float = 15.0
    candidate_grade_min_azimuth_separation_deg: float = 30.0
    candidate_grade_max_radius_ratio: float = 3.0


@dataclass(frozen=True)
class AppConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    output_dir: Path = Path("outputs")
