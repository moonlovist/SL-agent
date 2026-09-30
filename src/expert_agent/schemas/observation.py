from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(StrictModel):
    pipeline: str = "expert_agent.vision"
    pipeline_version: str = "0.29.0"
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    parameters: dict[str, Any] = Field(default_factory=dict)


class DataQuality(StrictModel):
    usable: bool
    supplied_rms: bool
    finite_fraction: float = Field(ge=0.0, le=1.0)
    background_median: float
    background_rms: float = Field(ge=0.0)
    model_method: str
    model_success: bool
    residual_median: float
    residual_robust_rms: float = Field(ge=0.0)
    flags: list[str] = Field(default_factory=list)


MeasurementValue = float | int | bool | str | list[float] | list[int] | None


class Measurement(StrictModel):
    id: str
    kind: str
    value: MeasurementValue
    unit: str | None = None
    reliability: float = Field(ge=0.0, le=1.0)
    method: str
    description: str
    structure_id: str | None = None
    correlation_group: str
    flags: list[str] = Field(default_factory=list)


class StructureObservation(StrictModel):
    id: str
    centroid_xy: tuple[float, float]
    radius_px: float = Field(ge=0.0)
    position_angle_deg: float = Field(ge=0.0, lt=360.0)
    centroid_ra_deg: float | None = None
    centroid_dec_deg: float | None = None
    radius_arcsec: float | None = Field(default=None, ge=0.0)
    sky_position_angle_deg: float | None = Field(default=None, ge=0.0, lt=360.0)
    pixel_count: int = Field(ge=1)
    integrated_flux: float
    integrated_snr: float
    median_radius_px: float = Field(ge=0.0)
    angular_extent_deg: float = Field(ge=0.0, le=360.0)
    axis_ratio: float | None = Field(default=None, ge=1.0)
    tangential_misalignment_deg: float | None = Field(default=None, ge=0.0, le=90.0)
    threshold_survival_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    sep_survival_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    inner_connection_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    connects_to_inner_region: bool | None = None
    near_core: bool
    subtraction_core_overlap_fraction: float | None = Field(default=None, ge=0, le=1)
    comparison_integrated_snr: float | None = None
    occupied_angle_bins_5deg: list[int] | None = None
    foreground_comparison_survival_fraction: float | None = Field(default=None, ge=0, le=1)
    angular_occupied_deg: float | None = Field(default=None, ge=0, le=360)
    radial_fractional_width: float | None = Field(default=None, ge=0)
    local_tangential_fraction: float | None = Field(default=None, ge=0, le=1)
    local_shape_count: int = Field(default=0, ge=0)
    comparison_nearby_survival_fraction: float | None = Field(default=None, ge=0, le=1)
    comparison_negative_fraction: float | None = Field(default=None, ge=0, le=1)
    # Visibility in the input VIS image after subtracting a label-blind radial
    # control estimated at the same foreground-centric radius.  These are
    # measurements, not proof that the structure is lensed or behind the
    # foreground galaxy.
    input_radial_contrast_snr: float | None = None
    input_radial_positive_fraction: float | None = Field(default=None, ge=0, le=1)
    input_radial_control_coverage: float | None = Field(default=None, ge=0, le=1)
    # Observed radial thickness of the exact candidate mask.  PSF-normalized
    # width is omitted when no valid PSF measurement exists.
    radial_width_px: float | None = Field(default=None, ge=0)
    radial_width_psf: float | None = Field(default=None, ge=0)
    source_candidate_ids: list[str] = Field(default_factory=list)
    candidate_audit_ids: list[str] = Field(default_factory=list)
    measurement_ids: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)

    @field_validator("centroid_xy")
    @classmethod
    def finite_centroid(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not all(float(v) == float(v) for v in value):
            raise ValueError("centroid must be finite")
        return value


class ForegroundGalaxyObservation(StrictModel):
    """Measured properties of the central light distribution, not a lens label."""

    detected: bool
    center_selection: dict[str, Any] = Field(default_factory=dict)
    center_xy: tuple[float, float]
    peak_value: float
    integrated_flux: float | None = None
    ellipticity: float | None = Field(default=None, ge=0.0, le=1.0)
    position_angle_deg: float | None = Field(default=None, ge=0.0, lt=180.0)
    axis_ratio: float | None = Field(default=None, ge=1.0)
    reliability: float = Field(ge=0.0, le=1.0)
    method: str
    measurement_ids: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class SceneGeometryObservation(StrictModel):
    structure_count: int = Field(ge=0)
    common_radius_cv: float | None = Field(default=None, ge=0.0)
    total_angular_coverage_deg: float = Field(ge=0.0, le=360.0)
    mean_tangential_misalignment_deg: float | None = Field(
        default=None, ge=0.0, le=90.0
    )
    tangential_alignment_fraction: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    measurement_ids: list[str] = Field(default_factory=list)


class AlternativeIndicators(StrictModel):
    """Observation-only proxies useful for testing non-lens explanations."""

    inner_connected_structure_count: int = Field(ge=0)
    multiple_core_count: int | None = Field(default=None, ge=0)
    tidal_asymmetry: float | None = Field(default=None, ge=0.0, le=1.0)
    core_residual_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    bad_pixel_fraction: float = Field(ge=0.0, le=1.0)
    psf_information_available: bool = False
    measurement_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AstrometryObservation(StrictModel):
    """WCS-derived sky mapping; never a target identity or lens decision."""

    wcs_available: bool
    wcs_valid: bool
    frame: str | None = None
    projection: str | None = None
    pixel_scale_arcsec: float | None = Field(default=None, gt=0.0)
    center_ra_deg: float | None = None
    center_dec_deg: float | None = None
    north_vector_xy: tuple[float, float] | None = None
    east_vector_xy: tuple[float, float] | None = None
    display_origin: Literal["lower", "upper"]
    orientation_description: str
    flags: list[str] = Field(default_factory=list)


class EuclidQ1InputObservation(StrictModel):
    """Deprecated legacy field retained only for reading older observations."""

    profile: str
    compatible: bool
    expected_shape: tuple[int, int] = (300, 300)
    available_hdus: list[str] = Field(default_factory=list)
    missing_hdus: list[str] = Field(default_factory=list)
    available_bands: list[str] = Field(default_factory=list)
    psf_bands: list[str] = Field(default_factory=list)
    rms_bands: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class ForegroundModelComparisonObservation(StrictModel):
    """Audit-only comparison; it does not replace the detection model."""

    primary_method: str
    comparison_method: str
    comparison_success: bool
    psf_applied: bool
    used_for_detection: bool = False
    primary_residual_robust_rms: float = Field(ge=0.0)
    comparison_residual_robust_rms: float | None = Field(default=None, ge=0.0)
    measured_structure_pixel_survival_fraction: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    residual_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)
    flags: list[str] = Field(default_factory=list)


class RingCandidateObservation(StrictModel):
    """A model-driven annular residual measurement, not a lens decision."""

    id: str
    center_xy: tuple[float, float]
    semi_major_radius_px: float = Field(gt=0.0)
    axis_ratio: float = Field(gt=0.0, le=1.0)
    position_angle_deg: float = Field(ge=0.0, lt=180.0)
    width_px: float = Field(gt=0.0)
    above_local_background_fraction: float = Field(ge=0.0, le=1.0)
    longest_above_local_background_deg: float = Field(ge=0.0, le=360.0)
    median_on_ring_snr: float
    on_minus_off_ring_snr: float
    proposal_sources: list[str] = Field(default_factory=list)
    measurement_ids: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class RecoveredPathSegmentObservation(StrictModel):
    """Pixels newly recovered inside a geometric search path."""

    id: str
    centroid_xy: tuple[float, float]
    pixel_count: int = Field(gt=0)
    integrated_snr: float
    angular_extent_deg: float = Field(ge=0.0, le=360.0)


class GeometricPathObservation(StrictModel):
    """A search proposal; the path itself is not an observed ring."""

    id: str
    mode: Literal["arc_seeded", "center_fallback"]
    seed_structure_ids: list[str] = Field(default_factory=list)
    linked_structure_ids: list[str] = Field(default_factory=list)
    center_xy: tuple[float, float]
    semi_major_radius_px: float = Field(gt=0.0)
    axis_ratio: float = Field(gt=0.0, le=1.0)
    position_angle_deg: float = Field(ge=0.0, lt=180.0)
    half_width_px: float = Field(gt=0.0)
    fit_scatter_px: float | None = Field(default=None, ge=0.0)
    median_on_path_snr: float
    on_minus_off_path_snr: float
    above_local_background_fraction: float = Field(ge=0.0, le=1.0)
    longest_above_local_background_deg: float = Field(ge=0.0, le=360.0)
    recovered_segments: list[RecoveredPathSegmentObservation] = Field(default_factory=list)
    seed_candidate_ids: list[str] = Field(default_factory=list)
    linked_candidate_ids: list[str] = Field(default_factory=list)
    recovered_candidate_ids: list[str] = Field(default_factory=list)
    measurement_ids: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class ModelCandidateObservation(StrictModel):
    """A SEP detection after geometry-chain processing, not a lens label."""

    id: str
    origin: Literal["sep_seed", "path_recovery"]
    status: Literal[
        "sep_only",
        "sep_geometry_consistent",
        "geometry_recovered",
        # v0.10 compatibility values retained for loading older observations.
        "geometry_linked",
        "unexplained_residual",
    ]
    structural_support_level: Literal[
        "raw_sep_only", "sep_plus_geometry", "route_recovered"
    ] | None = None
    parent_sep_component_id: str | None = None
    pixel_partition: Literal[
        "route_supported", "off_route_remainder", "route_recovery"
    ] | None = None
    review_priority: Literal["priority_keep", "downgraded_trace"] | None = None
    support_sources: list[str] = Field(default_factory=list)
    raw_detection_ids: list[str] = Field(default_factory=list)
    path_ids: list[str] = Field(default_factory=list)
    geometry_consistent_path_ids: list[str] = Field(default_factory=list)
    candidate_system_ids: list[str] = Field(default_factory=list)
    path_overlap_fractions: dict[str, float] = Field(default_factory=dict)
    path_median_deviation_px: dict[str, float] = Field(default_factory=dict)
    centroid_xy: tuple[float, float]
    radius_px: float = Field(ge=0.0)
    position_angle_deg: float = Field(ge=0.0, lt=360.0)
    pixel_count: int = Field(gt=0)
    integrated_snr: float
    angular_extent_deg: float = Field(ge=0.0, le=360.0)
    snr_threshold_survival_count: int | None = Field(default=None, ge=0)
    snr_threshold_total_count: int | None = Field(default=None, ge=0)
    snr_threshold_survival_fraction: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    sep_survival_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    foreground_comparison_survival_fraction: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    near_core: bool
    subtraction_core_overlap_fraction: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    measurement_ids: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class CandidateSystemObservation(StrictModel):
    """Candidate components associated through one geometric search route."""

    id: str
    path_id: str
    member_candidate_ids: list[str]
    sep_candidate_ids: list[str] = Field(default_factory=list)
    recovered_candidate_ids: list[str] = Field(default_factory=list)
    component_count: int = Field(gt=0)
    union_pixel_count: int = Field(gt=0)
    radius_range_px: tuple[float, float]
    angular_extent_deg: float = Field(ge=0.0, le=360.0)
    support_sources: list[str] = Field(default_factory=list)
    audit_flags: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class CandidateStructureAuditObservation(StrictModel):
    """Transparent bridge from visual candidates to measured structures."""

    id: str
    candidate_id: str
    verdict: Literal["promoted", "hold_for_review", "trace_only"]
    structure_id: str | None = None
    audit_scope: Literal["signal_reality"] = "signal_reality"
    handoff_class: Literal[
        "primary_candidate",
        "recovery_candidate",
        "foreground_contaminated",
        "trace_candidate",
    ] = "trace_candidate"
    discovery_context: Literal[
        "global_sep", "global_sep_plus_geometry", "route_recovery"
    ] | None = None
    geometry_used_for_verdict: bool = False
    geometry_used_for_handoff: bool = False
    sections: dict[str, dict[str, bool | None]] = Field(default_factory=dict)
    checks: dict[str, bool | None] = Field(default_factory=dict)
    supporting_facts: list[str] = Field(default_factory=list)
    caution_flags: list[str] = Field(default_factory=list)
    rationale: str


class CandidateHandoffEntry(StrictModel):
    """Full pixel measurements for review, separate from grading eligibility."""

    candidate_id: str
    formal_structure_id: str | None = None
    legacy_verdict: str | None = None
    legacy_handoff_class: str | None = None
    status: Literal["formal_reference", "measured_for_review", "unmeasurable"]
    physical_origin: Literal["unresolved"] = "unresolved"
    mask_key: str
    mask_sha256: str | None = None
    measurement: StructureObservation | None = None
    # Diagonal input-RMS propagation only; foreground-fit uncertainty and
    # pixel covariance are not included and this is not a calibrated error.
    aperture_noise_diagonal: float | None = Field(default=None, gt=0)
    comparison_aperture_flux: float | None = None
    comparison_detection: Literal["detected", "below_threshold", "unavailable"]
    comparison_threshold_sigma: float = Field(gt=0)
    uncertainty_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_measurement_state(self):
        if (self.status == "unmeasurable") != (self.measurement is None):
            raise ValueError("measurement availability must match review status")
        if self.status == "formal_reference" and self.formal_structure_id is None:
            raise ValueError("formal reference requires an existing structure ID")
        if self.measurement is not None:
            if self.measurement.source_candidate_ids != [self.candidate_id]:
                raise ValueError("review measurement must reference its source candidate")
            fraction = self.measurement.foreground_comparison_survival_fraction
            if self.comparison_detection == "unavailable" and fraction is not None:
                raise ValueError("unavailable comparison cannot carry a survival fraction")
            if self.comparison_detection == "below_threshold" and fraction != 0:
                raise ValueError("below-threshold comparison requires zero threshold occupancy")
            if self.comparison_detection == "detected" and (fraction is None or fraction <= 0):
                raise ValueError("comparison detection requires positive threshold occupancy")
        return self


class CandidateHandoffReview(StrictModel):
    policy: Literal["complete_review_v1"] = "complete_review_v1"
    grading_policy: Literal["integrated_candidate_evidence_v1"] = "integrated_candidate_evidence_v1"
    source: Literal["live_pipeline", "saved_pixel_artifacts"]
    entries: list[CandidateHandoffEntry] = Field(default_factory=list)

    @field_validator("grading_policy", mode="before")
    @classmethod
    def upgrade_legacy_grading_policy(cls, value):
        if value in {"baseline_unchanged", "provisional_candidate_inclusive_v1"}:
            return "integrated_candidate_evidence_v1"
        return value


class ObservationDocument(StrictModel):
    # Versioned measurements, never class labels or externally supplied votes.
    arc_chains: list[StructureObservation] = Field(default_factory=list)
    supplementary_diagnostics: dict = Field(default_factory=dict)
    schema_version: Literal["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "1.10", "1.11", "1.12", "1.13", "1.14", "1.15", "1.16", "1.17", "1.18"] = "1.18"
    object_id: str
    input_path: str
    input_format: Literal["fits", "png"]
    band: str
    image_shape: tuple[int, int]
    nucleus_xy: tuple[float, float]
    data_quality: DataQuality
    foreground_galaxy: ForegroundGalaxyObservation | None = None
    scene_geometry: SceneGeometryObservation | None = None
    alternative_indicators: AlternativeIndicators | None = None
    astrometry: AstrometryObservation | None = None
    euclid_q1_input: EuclidQ1InputObservation | None = None  # legacy; no longer emitted
    foreground_model_comparison: ForegroundModelComparisonObservation | None = None
    ring_candidates: list[RingCandidateObservation] = Field(default_factory=list)
    geometric_paths: list[GeometricPathObservation] = Field(default_factory=list)
    model_candidates: list[ModelCandidateObservation] = Field(default_factory=list)
    candidate_systems: list[CandidateSystemObservation] = Field(default_factory=list)
    candidate_structure_audits: list[CandidateStructureAuditObservation] = Field(default_factory=list)
    candidate_handoff_review: CandidateHandoffReview | None = None
    structures: list[StructureObservation] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    provenance: Provenance

    @model_serializer(mode="wrap")
    def omit_inactive_trial_for_legacy_readers(self, handler):
        result = handler(self)
        if self.candidate_handoff_review is None:
            result.pop("candidate_handoff_review", None)
        return result

    def measurement_map(self) -> dict[str, Measurement]:
        return {item.id: item for item in self.measurements}
