import argparse
import json
from pathlib import Path

from iewc.synthetic_diffusion import (
    SyntheticDiffusionConfig,
    run_synthetic_diffusion_suite,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a class-conditional diffusion continual-learning suite."
    )
    parser.add_argument("--dataset", choices=["synthetic", "mnist"], default="synthetic")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train-per-task", type=int, default=96)
    parser.add_argument("--n-test-per-task", type=int, default=96)
    parser.add_argument("--max-importance-samples", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ewc-lambda", type=float, default=25.0)
    parser.add_argument("--num-train-timesteps", type=int, default=50)
    parser.add_argument("--block-channels", default="16,16")
    parser.add_argument("--layers-per-block", type=int, default=1)
    parser.add_argument("--beta-schedule", default="linear")
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--tau", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-metric",
        choices=["euclidean", "sliced_wasserstein"],
        default="euclidean",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["sequential", "ef", "ief_diag"],
        choices=["sequential", "ef", "iewc", "ief_diag"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/empirical-evidence/artifacts/synthetic-diffusion-seed0.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    block_out_channels = tuple(
        int(item.strip()) for item in args.block_channels.split(",") if item.strip()
    )
    if len(block_out_channels) < 2:
        raise SystemExit("--block-channels must contain at least two widths")
    config = SyntheticDiffusionConfig(
        seed=args.seed,
        image_size=args.image_size if args.image_size is not None else (32 if args.dataset == "mnist" else 16),
        n_train_per_task=args.n_train_per_task,
        n_test_per_task=args.n_test_per_task,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ewc_lambda=args.ewc_lambda,
        num_train_timesteps=args.num_train_timesteps,
        tau_values=tuple(args.tau),
        output_metric=args.output_metric,
        device=args.device,
        block_out_channels=block_out_channels,
        layers_per_block=args.layers_per_block,
        beta_schedule=args.beta_schedule,
        dataset=args.dataset,
        data_root=args.data_root,
        max_importance_samples=args.max_importance_samples,
        progress_interval=args.progress_interval,
    )
    methods = tuple("ief_diag" if method == "iewc" else method for method in args.methods)
    results = run_synthetic_diffusion_suite(config=config, methods=methods)
    payload = {
        "experiment": f"{config.dataset}_conditional_diffusion",
        "config": {
            "seed": config.seed,
            "dataset": config.dataset,
            "data_root": config.data_root,
            "image_size": config.image_size,
            "n_train_per_task": config.n_train_per_task,
            "n_test_per_task": config.n_test_per_task,
            "max_importance_samples": config.max_importance_samples,
            "train_steps": config.train_steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "ewc_lambda": config.ewc_lambda,
            "num_train_timesteps": config.num_train_timesteps,
            "block_out_channels": list(config.block_out_channels),
            "layers_per_block": config.layers_per_block,
            "beta_schedule": config.beta_schedule,
            "tau_values": list(config.tau_values),
            "output_metric": config.output_metric,
            "methods": list(methods),
            "device": config.device,
            "progress_interval": config.progress_interval,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)
    for result in results:
        print(
            f"{result['method']}: old_after_new={result['task_a_denoise_mse_after_task_b']:.4f} "
            f"forgetting={result['task_a_denoise_mse_increase']:.4f} "
            f"new={result['task_b_denoise_mse_after_task_b']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
