import unittest
from types import SimpleNamespace

import numpy as np

from expert_agent.core.config import VisionConfig
from expert_agent.vision.detectors import circular_span, threshold_masks


class GeometryTests(unittest.TestCase):
    def test_circular_span_handles_zero_crossing(self):
        self.assertAlmostEqual(circular_span(np.array([350.0, 355.0, 2.0, 10.0])), 20.0)

    def test_threshold_diagnostic_masks_match_scientific_annulus(self):
        residual_snr = np.zeros((11, 11), dtype=float)
        residual_snr[5, 8] = 5.0
        residual_snr[5, 9] = 5.0
        source_mask = np.zeros((11, 11), dtype=bool)
        source_mask[5, 9] = True
        prep = SimpleNamespace(
            image=SimpleNamespace(data=np.zeros((11, 11))),
            residual_snr=residual_snr,
            source_mask=source_mask,
            nucleus_yx=(5, 5),
        )
        config = VisionConfig(
            ring_inner_px=1.0,
            ring_outer_px=4.5,
            snr_thresholds=(2.0, 6.0),
        )
        masks, ring = threshold_masks(prep, config)
        self.assertTrue(ring[5, 8])
        self.assertTrue(masks[0][5, 8])
        self.assertFalse(masks[0][5, 9])
        self.assertFalse(masks[1].any())


if __name__ == "__main__":
    unittest.main()
