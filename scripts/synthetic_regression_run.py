import argparse
import json
from pathlib import Path

from iewc.synthetic_regression import (
    SyntheticRegressionConfig,
    run_synthetic_regression_suite,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--n-tasks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--ewc-lambda", type=float, default=1.0)
    parser.add_argument("--tau", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["sequential", "ef", "ief_diag"],
        choices=["sequential", "ef", "iewc", "ief_diag"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = SyntheticRegressionConfig(
        seed=args.seed,
        n_train=args.n_train,
        n_test=args.n_test,
        n_tasks=args.n_tasks,
        train_epochs=args.epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ewc_lambda=args.ewc_lambda,
        tau_values=tuple(args.tau),
        device=args.device,
    )
    methods = tuple("ief_diag" if method == "iewc" else method for method in args.methods)
    results = run_synthetic_regression_suite(config=config, methods=methods)
    report = {
        "experiment": "synthetic_phase_shift_regression",
        "config": {
            "seed": config.seed,
            "n_train": config.n_train,
            "n_test": config.n_test,
            "n_tasks": config.n_tasks,
            "train_epochs": config.train_epochs,
            "hidden_size": config.hidden_size,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "ewc_lambda": config.ewc_lambda,
            "tau_values": list(config.tau_values),
            "methods": list(methods),
            "device": config.device,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
