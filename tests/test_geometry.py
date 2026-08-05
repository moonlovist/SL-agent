import unittest

import numpy as np

from expert_agent.vision.detectors import circular_span


class GeometryTests(unittest.TestCase):
    def test_circular_span_handles_zero_crossing(self):
        self.assertAlmostEqual(circular_span(np.array([350.0, 355.0, 2.0, 10.0])), 20.0)


if __name__ == "__main__":
    unittest.main()
