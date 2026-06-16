import argparse
import json
from pathlib import Path

import torch

from iewc.synthetic_diffusion import (
    SyntheticDiffusionConfig,
    TinyConditionalDiffusion,
    _limit_importance_dataset,
    _resolve_device,
    compute_diffusion_importance,
    denoising_mse,
    make_diffusion_stream,
    old_output_drift,
    train_task,
)


def parse_block_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(channels) < 2:
        raise argparse.ArgumentTypeError("expected at least two comma-separated widths")
    return channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep lambda values for MNIST DDPM continual-learning runs."
    )
    parser.add_argument("--method", choices=["ef", "ief_diag"], required=True)
    parser.add_argument("--output-metric", choices=["euclidean", "sliced_wasserstein"], default="euclidean")
    parser.add_argument("--lambdas", type=float, nargs="+", required=True)
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default="/home/davwis/main/data/mnist")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--n-train-per-task", type=int, default=4096)
    parser.add_argument("--n-test-per-task", type=int, default=512)
    parser.add_argument("--max-importance-samples", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--block-channels", type=parse_block_channels, default=(64, 128, 128))
    parser.add_argument("--layers-per-block", type=int, default=2)
    parser.add_argument("--beta-schedule", default="squaredcos_cap_v2")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--progress-interval", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_model(config: SyntheticDiffusionConfig, device: torch.device) -> TinyConditionalDiffusion:
    return TinyConditionalDiffusion(
        image_size=config.image_size,
        num_train_timesteps=config.num_train_timesteps,
        block_out_channels=config.block_out_channels,
        layers_per_block=config.layers_per_block,
        beta_schedule=config.beta_schedule,
    ).to(device)


def main() -> None:
    args = parse_args()
    config = SyntheticDiffusionConfig(
        seed=args.seed,
        image_size=args.image_size,
        n_train_per_task=args.n_train_per_task,
        n_test_per_task=args.n_test_per_task,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ewc_lambda=0.0,
        num_train_timesteps=args.num_train_timesteps,
        tau_values=(args.tau,),
        output_metric=args.output_metric,
        device=args.device,
        block_out_channels=args.block_channels,
        layers_per_block=args.layers_per_block,
        beta_schedule=args.beta_schedule,
        dataset="mnist",
        data_root=args.data_root,
        max_importance_samples=args.max_importance_samples,
        progress_interval=args.progress_interval,
    )
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    task_a_train, task_b_train, task_a_test, task_b_test = make_diffusion_stream(config)
    model = make_model(config, device)
    train_task(
        model=model,
        dataset=task_a_train,
        steps=config.train_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed + 100,
        progress_prefix=f"{args.method}:old",
        progress_interval=config.progress_interval,
    )
    task_a_mse_after_task_a = denoising_mse(
        model, task_a_test, batch_size=config.batch_size, seed=config.seed + 200
    )
    old_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    saved_params = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }
    importance_dataset = _limit_importance_dataset(
        task_a_train, config.max_importance_samples, seed=config.seed + 250
    )
    importances, loss_scales, ef_traces, stored_traces = compute_diffusion_importance(
        model=model,
        dataset=importance_dataset,
        kind=args.method,
        tau=args.tau,
        seed=config.seed + 300,
        output_metric=config.output_metric,
    )

    results = []
    for lamb in args.lambdas:
        new_model = make_model(config, device)
        new_model.load_state_dict(old_state)
        train_task(
            model=new_model,
            dataset=task_b_train,
            steps=config.train_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            ewc_lambda=lamb,
            saved_params=saved_params,
            importances=importances,
            seed=config.seed + 400,
            progress_prefix=f"{args.method}:new:lambda={lamb:g}",
            progress_interval=config.progress_interval,
        )
        task_a_mse_after_task_b = denoising_mse(
            new_model, task_a_test, batch_size=config.batch_size, seed=config.seed + 500
        )
        task_b_mse_after_task_b = denoising_mse(
            new_model, task_b_test, batch_size=config.batch_size, seed=config.seed + 600
        )
        old_model = make_model(config, device)
        old_model.load_state_dict(old_state)
        old_euclidean_drift, old_wasserstein_drift = old_output_drift(
            old_model,
            new_model,
            task_a_test,
            batch_size=config.batch_size,
            seed=config.seed + 700,
        )
        result = {
            "method": args.method,
            "ewc_lambda": lamb,
            "tau": args.tau if args.method == "ief_diag" else None,
            "output_metric": config.output_metric,
            "task_a_denoise_mse_after_task_a": task_a_mse_after_task_a,
            "task_a_denoise_mse_after_task_b": task_a_mse_after_task_b,
            "task_b_denoise_mse_after_task_b": task_b_mse_after_task_b,
            "task_a_denoise_mse_increase": task_a_mse_after_task_b
            - task_a_mse_after_task_a,
            "old_output_euclidean_drift": old_euclidean_drift,
            "old_output_sliced_wasserstein_drift": old_wasserstein_drift,
            "old_task_loss_scale_mean": float(loss_scales.mean().item()),
            "old_task_loss_scale_median": float(loss_scales.median().item()),
            "old_task_loss_scales": [float(value) for value in loss_scales.tolist()],
            "old_task_ef_summand_traces": [float(value) for value in ef_traces.tolist()],
            "old_task_stored_summand_traces": [
                float(value) for value in stored_traces.tolist()
            ],
        }
        print(
            f"{args.method} lambda={lamb:g}: "
            f"old_after_new={task_a_mse_after_task_b:.4f} "
            f"forgetting={result['task_a_denoise_mse_increase']:.4f} "
            f"new={task_b_mse_after_task_b:.4f} "
            f"old_sw_drift={old_wasserstein_drift:.4f}",
            flush=True,
        )
        results.append(result)

    payload = {
        "experiment": "mnist_conditional_diffusion_lambda_sweep",
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
            "lambdas": list(args.lambdas),
            "num_train_timesteps": config.num_train_timesteps,
            "block_out_channels": list(config.block_out_channels),
            "layers_per_block": config.layers_per_block,
            "beta_schedule": config.beta_schedule,
            "tau": args.tau,
            "output_metric": config.output_metric,
            "method": args.method,
            "device": config.device,
            "progress_interval": config.progress_interval,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
