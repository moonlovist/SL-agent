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
                "structure_count": len(document.structures),
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
                "measurement_count": len(document.measurements),
                "classification_performed": False,
            },
        )
        return state
