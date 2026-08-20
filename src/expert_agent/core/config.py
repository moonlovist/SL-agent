from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VisionConfig:
    band: str = "VIS"
    ring_inner_px: float = 4.0
    ring_outer_px: float = 25.0
    center_search_half_size: int = 40
    source_detect_sigma: float = 1.5
    snr_thresholds: tuple[float, ...] = (2.0, 2.58, 3.34, 4.31, 5.57, 7.19, 9.29, 12.0)
    snr_min_area: int = 3
    snr_min_survival: int = 3
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
    decision_min_score: float = 0.12
    decision_min_margin: float = 0.06
    high_confidence_score: float = 0.72
    medium_confidence_score: float = 0.48
    observation_min_finite_fraction: float = 0.99
    observation_min_reliability: float = 0.20
    evidence_low_specificity_threshold: float = 0.35
    decision_downgrade_completeness: float = 0.60


@dataclass(frozen=True)
class AppConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    output_dir: Path = Path("outputs")
