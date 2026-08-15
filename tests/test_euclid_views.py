import unittest

import numpy as np

from expert_agent.vision.diagnostics import (
    arcsinh_stretch,
    midtone_transfer,
    normalize_euclid_display,
    solve_midtone_parameter,
)


class EuclidExpertViewTests(unittest.TestCase):
    def test_stretches_are_bounded_and_monotonic(self):
        raw = np.linspace(-2.0, 10.0, 10000).reshape(100, 100)
        normalized = normalize_euclid_display(raw)
        arcsinh = arcsinh_stretch(normalized)
        m = solve_midtone_parameter(normalized)
        mtf = midtone_transfer(normalized, m)
        for values in (normalized, arcsinh, mtf):
            self.assertGreaterEqual(float(values.min()), 0.0)
            self.assertLessEqual(float(values.max()), 1.0)
            self.assertTrue(np.all(np.diff(values.ravel()) >= -1e-12))

    def test_mtf_targets_central_mean(self):
        y, x = np.indices((300, 300))
        normalized = normalize_euclid_display(
            np.exp(-np.hypot(x - 150, y - 150) / 25.0)
        )
        m = solve_midtone_parameter(normalized, target_mean=0.2)
        shown = midtone_transfer(normalized, m)
        self.assertAlmostEqual(float(shown[100:200, 100:200].mean()), 0.2, places=3)


if __name__ == "__main__":
    unittest.main()
