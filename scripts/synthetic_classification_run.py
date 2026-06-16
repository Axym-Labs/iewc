import argparse
import json
from pathlib import Path

from iewc.synthetic_classification import (
    SyntheticRunConfig,
    run_synthetic_classification_suite,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--ewc-lambda", type=float, default=10.0)
    parser.add_argument("--tau", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = SyntheticRunConfig(
        seed=args.seed,
        n_train=args.n_train,
        n_test=args.n_test,
        train_epochs=args.epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ewc_lambda=args.ewc_lambda,
        tau_values=tuple(args.tau),
    )
    results = run_synthetic_classification_suite(config=config)
    report = {
        "experiment": "synthetic_domain_shift_classification",
        "config": {
            "seed": config.seed,
            "n_train": config.n_train,
            "n_test": config.n_test,
            "train_epochs": config.train_epochs,
            "hidden_size": config.hidden_size,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "ewc_lambda": config.ewc_lambda,
            "tau_values": list(config.tau_values),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
