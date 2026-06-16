import argparse
import json
import math
from pathlib import Path

from iewc.synthetic_diffusion import SyntheticDiffusionConfig, run_one as run_diffusion
from iewc.synthetic_regression import SyntheticRegressionConfig, run_one as run_regression
from iewc.synthetic_segmentation import (
    SyntheticSegmentationConfig,
    run_one as run_segmentation,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "empirical-evidence" / "artifacts"


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def std(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def metric_for(task_name: str, result: dict) -> float:
    if task_name == "regression":
        return float(result["task_a_mse_increase"])
    if task_name == "diffusion":
        return float(result["task_a_denoise_mse_increase"])
    if task_name == "segmentation":
        return float(result["task_a_iou_drop"])
    raise ValueError(task_name)


def new_metric_for(task_name: str, result: dict) -> float:
    if task_name == "regression":
        return float(result["task_b_mse_after_task_b"])
    if task_name == "diffusion":
        return float(result["task_b_denoise_mse_after_task_b"])
    if task_name == "segmentation":
        return float(result["task_b_iou_after_task_b"])
    raise ValueError(task_name)


def config_and_runner(task_name: str, seed: int, ewc_lambda: float):
    if task_name == "regression":
        return (
            SyntheticRegressionConfig(
                seed=seed,
                n_train=1024,
                n_test=2048,
                train_epochs=300,
                hidden_size=96,
                batch_size=128,
                learning_rate=0.01,
                ewc_lambda=ewc_lambda,
                tau_values=(1e-3, 1e-2, 1e-1),
            ),
            run_regression,
        )
    if task_name == "diffusion":
        return (
            SyntheticDiffusionConfig(
                seed=seed,
                n_train_per_task=192,
                n_test_per_task=192,
                train_steps=240,
                batch_size=24,
                learning_rate=1e-3,
                ewc_lambda=ewc_lambda,
                tau_values=(1e-3, 1e-2, 1e-1),
            ),
            run_diffusion,
        )
    if task_name == "segmentation":
        return (
            SyntheticSegmentationConfig(
                seed=seed,
                n_train=192,
                n_test=256,
                train_epochs=120,
                hidden_channels=24,
                batch_size=32,
                learning_rate=0.01,
                ewc_lambda=ewc_lambda,
                tau_values=(1e-12, 2e-12, 5e-12, 1e-11),
                task_b_foreground_value=0.0,
                task_b_background_value=1.0,
            ),
            run_segmentation,
        )
    raise ValueError(f"Unknown task: {task_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["regression", "diffusion", "segmentation"],
        choices=["regression", "diffusion", "segmentation"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--regression-lambda",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
    )
    parser.add_argument(
        "--diffusion-lambda",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 10.0, 25.0, 75.0, 200.0],
    )
    parser.add_argument(
        "--segmentation-lambda",
        type=float,
        nargs="+",
        default=[10.0, 30.0, 100.0, 200.0, 500.0, 1000.0],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS / "nonclassification-ef-lambda-sweep.json",
    )
    args = parser.parse_args()

    grids = {
        "regression": args.regression_lambda,
        "diffusion": args.diffusion_lambda,
        "segmentation": args.segmentation_lambda,
    }
    payload = {"experiment": "nonclassification_ef_lambda_sweep", "tasks": {}}
    for task_name in args.tasks:
        records = []
        for ewc_lambda in grids[task_name]:
            for seed in args.seeds:
                config, runner = config_and_runner(task_name, seed, ewc_lambda)
                result = runner(config=config, method="ef")
                records.append(
                    {
                        "task": task_name,
                        "seed": seed,
                        "ewc_lambda": ewc_lambda,
                        "old_task_metric": metric_for(task_name, result),
                        "new_task_metric": new_metric_for(task_name, result),
                        "result": result,
                    }
                )
                print(
                    json.dumps(
                        {
                            "task": task_name,
                            "seed": seed,
                            "ewc_lambda": ewc_lambda,
                            "old_task_metric": records[-1]["old_task_metric"],
                            "new_task_metric": records[-1]["new_task_metric"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        summary = []
        for ewc_lambda in grids[task_name]:
            items = [r for r in records if r["ewc_lambda"] == ewc_lambda]
            summary.append(
                {
                    "ewc_lambda": ewc_lambda,
                    "n": len(items),
                    "old_task_metric_mean": mean(r["old_task_metric"] for r in items),
                    "old_task_metric_std": std(r["old_task_metric"] for r in items),
                    "new_task_metric_mean": mean(r["new_task_metric"] for r in items),
                    "new_task_metric_std": std(r["new_task_metric"] for r in items),
                }
            )
        best = min(summary, key=lambda item: item["old_task_metric_mean"])
        payload["tasks"][task_name] = {
            "lambda_grid": grids[task_name],
            "records": records,
            "summary": summary,
            "best_by_old_task_metric": best,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
