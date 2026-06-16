import unittest

from iewc.synthetic_classification import (
    SyntheticRunConfig,
    run_synthetic_classification_suite,
)


class SyntheticClassificationRunTests(unittest.TestCase):
    def test_suite_returns_metrics_for_requested_methods_and_tau_values(self):
        config = SyntheticRunConfig(
            seed=0,
            n_train=64,
            n_test=64,
            train_epochs=2,
            hidden_size=16,
            batch_size=32,
            ewc_lambda=1.0,
            tau_values=(0.001, 0.1),
        )

        results = run_synthetic_classification_suite(
            config=config,
            methods=("sequential", "ef", "ewc_dr", "ief_diag"),
        )

        method_names = {result["method"] for result in results}
        self.assertEqual(method_names, {"sequential", "ef", "ewc_dr", "ief_diag"})
        ief_result = next(result for result in results if result["method"] == "ief_diag")
        self.assertEqual(len(ief_result["tau_results"]), 2)
        for result in results:
            self.assertIn("task_a_accuracy_after_task_a", result)
            self.assertIn("task_a_accuracy_after_task_b", result)
            self.assertIn("task_b_accuracy_after_task_b", result)
            self.assertIn("task_a_forgetting", result)

    def test_naive_alias_is_reported_as_sequential(self):
        config = SyntheticRunConfig(
            seed=0,
            n_train=32,
            n_test=32,
            train_epochs=1,
            hidden_size=8,
            batch_size=16,
        )

        results = run_synthetic_classification_suite(
            config=config, methods=("naive",)
        )

        self.assertEqual(results[0]["method"], "sequential")


if __name__ == "__main__":
    unittest.main()
