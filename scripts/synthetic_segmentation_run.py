import argparse
import json
from pathlib import Path

from iewc.synthetic_segmentation import (
    SyntheticSegmentationConfig,
    run_synthetic_segmentation_suite,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a synthetic binary-segmentation continual-learning suite."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=96)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--train-epochs", type=int, default=50)
    parser.add_argument("--hidden-channels", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--ewc-lambda", type=float, default=2.0)
    parser.add_argument("--tau", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--task-b-foreground-value", type=float, default=-1.0)
    parser.add_argument("--task-b-background-value", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/empirical-evidence/artifacts/synthetic-segmentation-seed0.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = SyntheticSegmentationConfig(
        seed=args.seed,
        n_train=args.n_train,
        n_test=args.n_test,
        train_epochs=args.train_epochs,
        hidden_channels=args.hidden_channels,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ewc_lambda=args.ewc_lambda,
        tau_values=tuple(args.tau),
        task_b_foreground_value=args.task_b_foreground_value,
        task_b_background_value=args.task_b_background_value,
    )
    results = run_synthetic_segmentation_suite(config=config)
    payload = {
        "experiment": "synthetic_binary_segmentation",
        "config": {
            "seed": config.seed,
            "image_size": config.image_size,
            "n_train": config.n_train,
            "n_test": config.n_test,
            "train_epochs": config.train_epochs,
            "hidden_channels": config.hidden_channels,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "ewc_lambda": config.ewc_lambda,
            "tau_values": list(config.tau_values),
            "task_b_foreground_value": config.task_b_foreground_value,
            "task_b_background_value": config.task_b_background_value,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
