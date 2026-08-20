from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(StrictModel):
    pipeline: str = "expert_agent.vision"
    pipeline_version: str = "0.9.0"
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
    measurement_ids: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class ObservationDocument(StrictModel):
    schema_version: Literal["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7"] = "1.7"
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
    structures: list[StructureObservation] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    provenance: Provenance

    def measurement_map(self) -> dict[str, Measurement]:
        return {item.id: item for item in self.measurements}
