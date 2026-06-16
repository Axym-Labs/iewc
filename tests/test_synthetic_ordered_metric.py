import unittest

from iewc.synthetic_ordered_metric import (
    SyntheticOrderedMetricConfig,
    run_ordered_metric_suite,
)


class SyntheticOrderedMetricRunTests(unittest.TestCase):
    def test_suite_compares_euclidean_and_wasserstein_iewc_metrics(self):
        config = SyntheticOrderedMetricConfig(
            seed=0,
            n_train=32,
            n_test=32,
            train_epochs=2,
            batch_size=16,
            tau=1e-3,
            ewc_lambda=2.0,
            wasserstein_ewc_lambda=20.0,
        )

        results = run_ordered_metric_suite(config=config)
        by_method = {item["method"]: item for item in results}

        self.assertEqual(
            set(by_method), {"sequential", "ief_euclidean", "ief_wasserstein"}
        )
        for item in results:
            self.assertIn("task_a_accuracy_after_task_b", item)
            self.assertIn("task_b_accuracy_after_task_b", item)
            self.assertIn("old_output_euclidean_drift", item)
            self.assertIn("old_output_wasserstein_drift", item)

        self.assertEqual(by_method["ief_wasserstein"]["output_metric"], "wasserstein_1d_cdf")
        self.assertEqual(by_method["ief_euclidean"]["output_metric"], "euclidean")
        self.assertEqual(by_method["ief_euclidean"]["ewc_lambda"], 2.0)
        self.assertEqual(by_method["ief_wasserstein"]["ewc_lambda"], 20.0)


if __name__ == "__main__":
    unittest.main()
