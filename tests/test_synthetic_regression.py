import unittest

from iewc.synthetic_regression import (
    SyntheticRegressionConfig,
    run_synthetic_regression_suite,
)


class SyntheticRegressionRunTests(unittest.TestCase):
    def test_suite_returns_regression_metrics_and_tau_values(self):
        config = SyntheticRegressionConfig(
            seed=0,
            n_train=64,
            n_test=64,
            n_tasks=3,
            train_epochs=3,
            hidden_size=16,
            batch_size=32,
            ewc_lambda=1.0,
            tau_values=(0.001, 0.1),
        )

        results = run_synthetic_regression_suite(
            config=config,
            methods=("sequential", "ef", "ief_diag"),
        )

        method_names = {result["method"] for result in results}
        self.assertEqual(method_names, {"sequential", "ef", "ief_diag"})
        ief_result = next(result for result in results if result["method"] == "ief_diag")
        self.assertEqual(len(ief_result["tau_results"]), 2)
        for result in results:
            self.assertIn("task_a_mse_after_task_a", result)
            self.assertIn("task_a_mse_after_task_b", result)
            self.assertIn("task_b_mse_after_task_b", result)
            self.assertIn("task_a_mse_increase", result)
            self.assertEqual(result["n_tasks"], 3)
            self.assertEqual(len(result["mse_after_learning"]), 3)
            self.assertEqual(len(result["final_task_mses"]), 3)
            self.assertIn("final_avg_mse", result)
            self.assertIn("avg_forgetting_mse", result)


if __name__ == "__main__":
    unittest.main()
