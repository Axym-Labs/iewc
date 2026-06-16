import unittest

from iewc.split_mnist_benchmark import SplitMNISTConfig, run_split_mnist_suite


class SplitMNISTBenchmarkTests(unittest.TestCase):
    def test_suite_returns_real_benchmark_metrics_and_tau_values(self):
        config = SplitMNISTConfig(
            benchmark_name="split_mnist",
            seed=0,
            n_experiences=2,
            train_epochs=1,
            hidden_size=32,
            train_mb_size=16,
            eval_mb_size=32,
            max_train_per_experience=16,
            max_test_per_experience=32,
            tau_values=(1e-3, 1e-2),
        )

        results = run_split_mnist_suite(
            config=config, methods=("sequential", "ef", "ief_diag")
        )
        by_method = {item["method"]: item for item in results}

        self.assertEqual(set(by_method), {"sequential", "ef", "ief_diag"})
        for item in results:
            self.assertIn("final_average_accuracy", item)
            self.assertIn("average_forgetting", item)
            self.assertIn("accuracy_matrix", item)
            self.assertIn("wall_seconds_total", item)
            self.assertIn("estimated_flops", item)
            self.assertGreater(item["estimated_flops"]["total"], 0)

        self.assertEqual(
            by_method["sequential"]["estimated_flops"]["importance_model"], 0
        )
        self.assertEqual(
            by_method["sequential"]["estimated_flops"]["penalty_overhead"], 0
        )

        self.assertEqual(
            [item["tau"] for item in by_method["ief_diag"]["tau_results"]],
            [1e-3, 1e-2],
        )
        self.assertIn("last_loss_scale_mean", by_method["ief_diag"])

    def test_suite_can_run_permuted_mnist_with_importance_sample_budget(self):
        config = SplitMNISTConfig(
            benchmark_name="permuted_mnist",
            seed=0,
            n_experiences=2,
            train_epochs=1,
            hidden_size=32,
            train_mb_size=16,
            eval_mb_size=32,
            max_train_per_experience=16,
            max_test_per_experience=32,
            max_importance_samples=8,
            tau_values=(1e-3,),
        )

        results = run_split_mnist_suite(config=config, methods=("ef", "ief_diag"))

        self.assertEqual({item["method"] for item in results}, {"ef", "ief_diag"})
        for item in results:
            self.assertEqual(item["benchmark_name"], "permuted_mnist")
            self.assertEqual(item["max_importance_samples"], 8)

    def test_suite_can_run_low_rank_iewc_with_rank_sweep(self):
        config = SplitMNISTConfig(
            benchmark_name="permuted_mnist",
            seed=0,
            n_experiences=2,
            train_epochs=1,
            hidden_size=32,
            train_mb_size=16,
            eval_mb_size=32,
            max_train_per_experience=16,
            max_test_per_experience=32,
            max_importance_samples=8,
            tau_values=(1e-3,),
            rank_values=(1, 2),
        )

        results = run_split_mnist_suite(config=config, methods=("ief_low_rank",))

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["method"], "ief_low_rank")
        self.assertIn(result["rank"], {1, 2})
        self.assertEqual([item["rank"] for item in result["rank_results"]], [1, 2])
        self.assertIn("last_explained_variance_ratio", result)

    def test_suite_can_run_low_rank_plus_diagonal_iewc(self):
        config = SplitMNISTConfig(
            benchmark_name="permuted_mnist",
            seed=0,
            n_experiences=2,
            train_epochs=1,
            hidden_size=32,
            train_mb_size=16,
            eval_mb_size=32,
            max_train_per_experience=16,
            max_test_per_experience=32,
            max_importance_samples=8,
            tau_values=(1e-3,),
            rank_values=(1,),
        )

        results = run_split_mnist_suite(config=config, methods=("ief_low_rank_diag",))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["method"], "ief_low_rank_diag")
        self.assertIn("last_residual_diagonal_mass", results[0])
        self.assertGreaterEqual(results[0]["last_residual_diagonal_mass"], 0.0)


if __name__ == "__main__":
    unittest.main()
