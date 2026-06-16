import argparse
import json
from pathlib import Path

from iewc.synthetic_ordered_metric import (
    SyntheticOrderedMetricConfig,
    run_ordered_metric_suite,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an ordered-output metric comparison for IEWC."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--n-classes", type=int, default=5)
    parser.add_argument("--train-epochs", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--ewc-lambda", type=float, default=1.0)
    parser.add_argument("--euclidean-ewc-lambda", type=float, default=None)
    parser.add_argument("--wasserstein-ewc-lambda", type=float, default=None)
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/empirical-evidence/artifacts/synthetic-ordered-metric-seed0.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = SyntheticOrderedMetricConfig(
        seed=args.seed,
        n_train=args.n_train,
        n_test=args.n_test,
        n_classes=args.n_classes,
        train_epochs=args.train_epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ewc_lambda=args.ewc_lambda,
        euclidean_ewc_lambda=args.euclidean_ewc_lambda,
        wasserstein_ewc_lambda=args.wasserstein_ewc_lambda,
        tau=args.tau,
    )
    results = run_ordered_metric_suite(config=config)
    payload = {
        "experiment": "synthetic_ordered_output_metric_comparison",
        "config": {
            "seed": config.seed,
            "n_train": config.n_train,
            "n_test": config.n_test,
            "n_classes": config.n_classes,
            "train_epochs": config.train_epochs,
            "hidden_size": config.hidden_size,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "ewc_lambda": config.ewc_lambda,
            "euclidean_ewc_lambda": config.euclidean_ewc_lambda,
            "wasserstein_ewc_lambda": config.wasserstein_ewc_lambda,
            "tau": config.tau,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
