import unittest

from iewc.synthetic_segmentation import (
    SyntheticSegmentationConfig,
    run_synthetic_segmentation_suite,
)


class SyntheticSegmentationRunTests(unittest.TestCase):
    def test_suite_returns_segmentation_metrics_and_tau_values(self):
        config = SyntheticSegmentationConfig(
            seed=0,
            n_train=16,
            n_test=16,
            train_epochs=1,
            batch_size=8,
            tau_values=(1e-3, 1e-2),
        )

        results = run_synthetic_segmentation_suite(
            config=config, methods=("sequential", "ef", "ief_diag")
        )
        by_method = {item["method"]: item for item in results}

        self.assertEqual(set(by_method), {"sequential", "ef", "ief_diag"})
        for method_result in results:
            self.assertIn("task_a_iou_after_task_a", method_result)
            self.assertIn("task_a_iou_after_task_b", method_result)
            self.assertIn("task_b_iou_after_task_b", method_result)
            self.assertIn("task_a_iou_drop", method_result)

        tau_results = by_method["ief_diag"]["tau_results"]
        self.assertEqual([item["tau"] for item in tau_results], [1e-3, 1e-2])
        self.assertIn("old_task_loss_scale_mean", by_method["ief_diag"])


if __name__ == "__main__":
    unittest.main()
