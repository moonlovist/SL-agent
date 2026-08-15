import unittest

from expert_agent.schemas.observation import (
    DataQuality,
    ObservationDocument,
    Provenance,
)
from expert_agent.vision.summary_cn import render_observation_summary_cn


class ChineseObservationSummaryTests(unittest.TestCase):
    def test_summary_preserves_visual_only_boundary(self):
        document = ObservationDocument(
            object_id="demo",
            input_path="demo.fits",
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
                model_method="robust_radial_fallback",
                model_success=False,
                residual_median=0.0,
                residual_robust_rms=1.0,
                flags=["ellipse_model_failed:RuntimeError"],
            ),
            provenance=Provenance(parameters={}),
        )
        summary = render_observation_summary_cn(document)
        self.assertIn("固定模板", summary)
        self.assertIn("备用处理结果", summary)
        self.assertIn("不输出‘透镜/非透镜’", summary)
        self.assertNotIn("强透镜候选", summary)


if __name__ == "__main__":
    unittest.main()
