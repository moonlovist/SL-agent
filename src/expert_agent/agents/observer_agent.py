from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class ObserverAgent(BaseAgent):
    """Normalize machine measurements without reading pixels or classifying."""

    def __init__(self):
        super().__init__("ObserverAgent")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("整理视觉测量事实")
        self.load_skill("observation_skill")
        document = state.require_observation()
        state.object_id = document.object_id
        state.image_path = document.input_path
        state.observations = [
            {
                "object_id": document.object_id,
                "data_usable": document.data_quality.usable,
                "band": document.band,
                "nucleus_xy": list(document.nucleus_xy),
                "foreground_galaxy": (
                    document.foreground_galaxy.model_dump(mode="json")
                    if document.foreground_galaxy
                    else None
                ),
                "scene_geometry": (
                    document.scene_geometry.model_dump(mode="json")
                    if document.scene_geometry
                    else None
                ),
                "alternative_indicators": (
                    document.alternative_indicators.model_dump(mode="json")
                    if document.alternative_indicators
                    else None
                ),
                "structure_count": len(document.structures),
                "geometric_path_count": len(document.geometric_paths),
                "guided_recovered_segment_count": sum(
                    len(item.recovered_segments) for item in document.geometric_paths
                ),
                "geometric_paths": [
                    {
                        "id": item.id,
                        "mode": item.mode,
                        "seed_structure_ids": item.seed_structure_ids,
                        "linked_structure_ids": item.linked_structure_ids,
                        "center_xy": list(item.center_xy),
                        "semi_major_radius_px": item.semi_major_radius_px,
                        "axis_ratio": item.axis_ratio,
                        "half_width_px": item.half_width_px,
                        "fit_scatter_px": item.fit_scatter_px,
                        "above_local_background_fraction": item.above_local_background_fraction,
                        "longest_above_local_background_deg": item.longest_above_local_background_deg,
                        "median_on_path_snr": item.median_on_path_snr,
                        "on_minus_off_path_snr": item.on_minus_off_path_snr,
                        "recovered_segments": [
                            recovered.model_dump(mode="json")
                            for recovered in item.recovered_segments
                        ],
                        "measurement_ids": item.measurement_ids,
                    }
                    for item in document.geometric_paths
                ],
                "structures": [
                    {
                        "id": item.id,
                        "radius_px": item.radius_px,
                        "position_angle_deg": item.position_angle_deg,
                        "integrated_snr": item.integrated_snr,
                        "angular_extent_deg": item.angular_extent_deg,
                        "axis_ratio": item.axis_ratio,
                        "tangential_misalignment_deg": item.tangential_misalignment_deg,
                        "threshold_survival_fraction": item.threshold_survival_fraction,
                        "sep_survival_fraction": item.sep_survival_fraction,
                        "inner_connection_fraction": item.inner_connection_fraction,
                        "connects_to_inner_region": item.connects_to_inner_region,
                        "near_core": item.near_core,
                        "measurement_ids": item.measurement_ids,
                    }
                    for item in document.structures
                ],
                "quality_flags": document.data_quality.flags,
                "limitations": document.limitations,
            }
        ]
        self.record_reasoning(
            state,
            "observation_normalization",
            {
                "structure_count": len(document.structures),
                "geometric_path_count": len(document.geometric_paths),
                "guided_recovered_segment_count": sum(
                    len(item.recovered_segments) for item in document.geometric_paths
                ),
                "measurement_count": len(document.measurements),
                "classification_performed": False,
            },
        )
        return state
