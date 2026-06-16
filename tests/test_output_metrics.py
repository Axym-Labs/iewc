import unittest

import torch

from iewc.output_metrics import (
    euclidean_squared_distance,
    wasserstein_1d_cdf_squared_distance,
    wasserstein_1d_cdf_dual_quadratic_form,
)


class OutputMetricTests(unittest.TestCase):
    def test_wasserstein_cdf_metric_is_lower_for_adjacent_mass_shift(self):
        old = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        adjacent = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        far = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

        adjacent_euclidean = euclidean_squared_distance(old, adjacent)
        adjacent_wasserstein = wasserstein_1d_cdf_squared_distance(old, adjacent)
        far_wasserstein = wasserstein_1d_cdf_squared_distance(old, far)

        self.assertLess(adjacent_wasserstein.item(), adjacent_euclidean.item())
        self.assertGreater(far_wasserstein.item(), adjacent_wasserstein.item())

    def test_wasserstein_dual_form_ignores_simplex_null_direction(self):
        constant_covector = torch.ones(4)
        self.assertLess(
            wasserstein_1d_cdf_dual_quadratic_form(constant_covector).item(),
            1e-8,
        )


if __name__ == "__main__":
    unittest.main()
