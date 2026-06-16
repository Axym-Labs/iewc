import argparse
import json
from pathlib import Path

from iewc.split_mnist_benchmark import SplitMNISTConfig, run_split_mnist_suite


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a full SplitMNIST benchmark with EF/EWC-DR/IEWC variants."
    )
    allowed_methods = {
        "sequential",
        "naive",
        "ef",
        "ewc_dr",
        "ief_diag",
        "ef_low_rank",
        "ief_low_rank",
        "ef_low_rank_diag",
        "ief_low_rank_diag",
        "ef_diag_low_rank",
        "ief_diag_low_rank",
        "ef_corr_low_rank",
        "ief_corr_low_rank",
    }
    parser.add_argument(
        "--benchmark-name",
        choices=["split_mnist", "permuted_mnist"],
        default="split_mnist",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-experiences", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=400)
    parser.add_argument("--train-mb-size", type=int, default=128)
    parser.add_argument("--eval-mb-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--ewc-lambda", type=float, default=100.0)
    parser.add_argument("--tau", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--rank", type=int, nargs="+", default=[10])
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-per-experience", type=int, default=None)
    parser.add_argument("--max-test-per-experience", type=int, default=None)
    parser.add_argument("--max-importance-samples", type=int, default=None)
    parser.add_argument("--importance-sample-seed", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["sequential", "ef", "ewc_dr", "ief_diag"],
        metavar="METHOD",
        help=(
            "Methods: sequential, ef, ewc_dr, ief_diag, ef_low_rank, "
            "ief_low_rank, ef_low_rank_diag, ief_low_rank_diag, "
            "ef_diag_low_rank, ief_diag_low_rank, ef_corr_low_rank, "
            "ief_corr_low_rank."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    invalid_methods = [method for method in args.methods if method not in allowed_methods]
    if invalid_methods:
        parser.error(f"unknown method(s): {', '.join(invalid_methods)}")
    args.methods = [
        "sequential" if method == "naive" else method for method in args.methods
    ]
    return args


def main():
    args = parse_args()
    config = SplitMNISTConfig(
        benchmark_name=args.benchmark_name,
        seed=args.seed,
        n_experiences=args.n_experiences,
        train_epochs=args.epochs,
        hidden_size=args.hidden_size,
        train_mb_size=args.train_mb_size,
        eval_mb_size=args.eval_mb_size,
        learning_rate=args.learning_rate,
        ewc_lambda=args.ewc_lambda,
        tau_values=tuple(args.tau),
        rank_values=tuple(args.rank),
        dataset_root=args.dataset_root,
        device=args.device,
        max_train_per_experience=args.max_train_per_experience,
        max_test_per_experience=args.max_test_per_experience,
        max_importance_samples=args.max_importance_samples,
        importance_sample_seed=args.importance_sample_seed,
    )
    results = run_split_mnist_suite(config=config, methods=tuple(args.methods))
    report = {
        "experiment": "mnist_full_data_benchmark",
        "scale": (
            "full"
            if config.max_train_per_experience is None
            and config.max_test_per_experience is None
            else "capped"
        ),
        "importance_approximation": (
            "low_rank_plus_diagonal"
            if any(
                    "low_rank_diag" in method
                    or "diag_low_rank" in method
                    or "corr_low_rank" in method
                for method in args.methods
            )
            else "low_rank"
            if any("low_rank" in method for method in args.methods)
            else "diagonal"
        ),
        "config": {
            "seed": config.seed,
            "benchmark_name": config.benchmark_name,
            "n_experiences": config.n_experiences,
            "train_epochs": config.train_epochs,
            "hidden_size": config.hidden_size,
            "train_mb_size": config.train_mb_size,
            "eval_mb_size": config.eval_mb_size,
            "learning_rate": config.learning_rate,
            "ewc_lambda": config.ewc_lambda,
            "tau_values": list(config.tau_values),
            "rank_values": list(config.rank_values),
            "dataset_root": config.dataset_root,
            "device": config.device,
            "max_train_per_experience": config.max_train_per_experience,
            "max_test_per_experience": config.max_test_per_experience,
            "max_importance_samples": config.max_importance_samples,
            "importance_sample_seed": config.importance_sample_seed,
            "methods": list(args.methods),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
