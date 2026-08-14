from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(StrictModel):
    pipeline: str = "expert_agent.vision"
    pipeline_version: str = "0.4.0"
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


class ObservationDocument(StrictModel):
    schema_version: Literal["1.0", "1.1"] = "1.1"
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
    structures: list[StructureObservation] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    provenance: Provenance

    def measurement_map(self) -> dict[str, Measurement]:
        return {item.id: item for item in self.measurements}
