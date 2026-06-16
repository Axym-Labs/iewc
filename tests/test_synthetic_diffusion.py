import unittest

from iewc.synthetic_diffusion import (
    SyntheticDiffusionConfig,
    run_synthetic_diffusion_suite,
)


class SyntheticDiffusionRunTests(unittest.TestCase):
    def test_suite_returns_diffusion_metrics_and_tau_values(self):
        config = SyntheticDiffusionConfig(
            seed=0,
            n_train_per_task=8,
            n_test_per_task=8,
            train_steps=1,
            batch_size=4,
            tau_values=(1e-3, 1e-2),
        )

        results = run_synthetic_diffusion_suite(
            config=config, methods=("sequential", "ef", "ief_diag")
        )
        by_method = {item["method"]: item for item in results}

        self.assertEqual(set(by_method), {"sequential", "ef", "ief_diag"})
        for method_result in results:
            self.assertIn("task_a_denoise_mse_after_task_a", method_result)
            self.assertIn("task_a_denoise_mse_after_task_b", method_result)
            self.assertIn("task_b_denoise_mse_after_task_b", method_result)
            self.assertIn("task_a_denoise_mse_increase", method_result)

        tau_results = by_method["ief_diag"]["tau_results"]
        self.assertEqual([item["tau"] for item in tau_results], [1e-3, 1e-2])
        self.assertIn("old_task_loss_scale_mean", by_method["ief_diag"])


if __name__ == "__main__":
    unittest.main()
