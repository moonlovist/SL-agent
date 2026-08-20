from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState
from ..schemas.reasoning import EvidenceQuestion


class EvidencePlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("EvidencePlannerAgent")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("规划区分性证据")
        self.load_skill("evidence_planning_skill")
        document = state.require_observation()
        available = {item.kind for item in document.measurements}
        candidates = [
            EvidenceQuestion(
                id="Q0",
                question="Is a measurable central light concentration available as the common geometric reference?",
                target_hypotheses=["H1", "H4"],
                measurement_kinds=[
                    "foreground_detected",
                    "foreground_peak_value",
                    "foreground_ellipticity",
                ],
                priority="high",
                reason="A central reference is required to interpret radii and tangential geometry, without assuming that the light source is itself a lens.",
            ),
            EvidenceQuestion(
                id="Q1",
                question="Do residual structures share a common radius around the central source?",
                target_hypotheses=["H1", "H2"],
                measurement_kinds=["common_radius_cv", "radius", "structure_count"],
                priority="high",
                reason="A common lens center and radius is more characteristic of lens geometry than an arbitrary internal feature.",
            ),
            EvidenceQuestion(
                id="Q2",
                question="Are the structures extended and tangentially aligned?",
                target_hypotheses=["H1", "H2"],
                measurement_kinds=["tangential_misalignment", "angular_extent", "axis_ratio"],
                priority="high",
                reason="Tangential elongation is a discriminating geometric signature, while radial or compact structure favors alternatives.",
            ),
            EvidenceQuestion(
                id="Q3",
                question="Are the structures significant and stable across detection settings?",
                target_hypotheses=["H1", "H4"],
                measurement_kinds=[
                    "integrated_snr",
                    "threshold_survival_fraction",
                    "sep_survival_fraction",
                    "galaxy_model_success",
                ],
                priority="high",
                reason="Repeatability weakens a pure-noise explanation, but shared preprocessing keeps these measurements correlated.",
            ),
            EvidenceQuestion(
                id="Q4",
                question="Are multiple cores or tidal structures present?",
                target_hypotheses=["H3"],
                measurement_kinds=["multiple_core_count", "tidal_asymmetry"],
                priority="medium",
                reason="These measurements discriminate an interaction from lensing.",
            ),
            EvidenceQuestion(
                id="Q6",
                question="Are candidate residual structures connected to the inner galaxy light distribution?",
                target_hypotheses=["H1", "H2", "H3"],
                measurement_kinds=["inner_connection_fraction"],
                priority="high",
                reason="Connectivity to the inner galaxy can favor intrinsic or interaction morphology over a detached background-source interpretation.",
            ),
            EvidenceQuestion(
                id="Q7",
                question="Could PSF behavior, non-finite pixels, or concentrated core residuals explain the structures?",
                target_hypotheses=["H1", "H4"],
                measurement_kinds=[
                    "psf_information_available",
                    "bad_pixel_fraction",
                    "core_residual_fraction",
                ],
                priority="high",
                reason="These checks distinguish astrophysical residual structure from image or subtraction artifacts.",
            ),
            EvidenceQuestion(
                id="Q5",
                question="Do candidate images have consistent colors across bands, and are the color measurements reliable at this signal level?",
                target_hypotheses=["H1", "H2", "H3"],
                measurement_kinds=["color_consistency", "color_measurement_uncertainty", "band_count"],
                priority="medium",
                reason="Comparable colors can link multiple images, but color inferred at low S/N is not reliable and the current single-band pipeline may not provide this test.",
            ),
            EvidenceQuestion(
                id="Q8",
                question="If several compact candidate images are present, do they form a plausible common-centre configuration rather than lying on one side?",
                target_hypotheses=["H1", "H2", "H3"],
                measurement_kinds=["multi_image_group_count", "opposite_side_configuration", "candidate_image_radius_consistency"],
                priority="medium",
                reason="A collection of bright structures is not automatically a double or quad lens system; their arrangement about one centre must be measured.",
            ),
            EvidenceQuestion(
                id="Q9",
                question="Do the candidate images share a compatible morphology or point-source character, without separate galaxy-like shapes or independent bright cores?",
                target_hypotheses=["H1", "H2", "H3", "H4"],
                measurement_kinds=["candidate_image_morphology_consistency", "point_source_likeness", "independent_bright_core_count"],
                priority="medium",
                reason="Different intrinsic galaxy morphologies or independent bright cores can imitate a multiple-image configuration.",
            ),
        ]
        state.evidence_plan = sorted(
            candidates,
            key=lambda item: (
                0 if any(kind in available for kind in item.measurement_kinds) else 1,
                {"high": 0, "medium": 1, "low": 2}[item.priority],
            ),
        )
        self.record_reasoning(
            state,
            "evidence_planning",
            {
                "question_ids": [item.id for item in state.evidence_plan],
                "available_measurement_kinds": sorted(available),
            },
        )
        return state
