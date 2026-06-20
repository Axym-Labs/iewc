import argparse
import json
from pathlib import Path

from iewc.empirical2_forecasting import ForecastingConfig, run_forecasting_cl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["m4", "ett", "long_horizon"], default="m4")
    parser.add_argument("--data-root", default="/home/davwis/main/data/m4/tsf")
    parser.add_argument("--frequencies", nargs="+", default=["hourly", "weekly", "daily"])
    parser.add_argument("--method", choices=["sequential", "ef", "iewc", "iewc_gss"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--max-series-per-task", type=int, default=64)
    parser.add_argument("--windows-per-series", type=int, default=4)
    parser.add_argument("--eval-windows-per-series", type=int, default=1)
    parser.add_argument("--epochs-per-task", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ewc-lambda", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=1e-2)
    parser.add_argument("--importance-samples", type=int, default=128)
    parser.add_argument("--model-type", choices=["encoder", "patchtst"], default="encoder")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--patch-length", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--normalization", choices=["series", "context"], default="series")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root
    if args.dataset == "ett" and data_root == "/home/davwis/main/data/m4/tsf":
        data_root = "/home/davwis/main/data/ett"
    if args.dataset == "long_horizon" and data_root == "/home/davwis/main/data/m4/tsf":
        data_root = "/home/davwis/main/data/long_horizon"
    config = ForecastingConfig(
        dataset=args.dataset,
        data_root=data_root,
        frequencies=tuple(args.frequencies),
        seed=args.seed,
        context_length=args.context_length,
        horizon=args.horizon,
        max_series_per_task=args.max_series_per_task,
        windows_per_series=args.windows_per_series,
        eval_windows_per_series=args.eval_windows_per_series,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ewc_lambda=args.ewc_lambda,
        tau=args.tau,
        importance_samples=args.importance_samples,
        model_type=args.model_type,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        dropout=args.dropout,
        num_workers=args.num_workers,
        device=args.device,
        normalization=args.normalization,
    )
    result = run_forecasting_cl(config, args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
