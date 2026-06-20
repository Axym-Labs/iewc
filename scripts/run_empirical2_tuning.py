import argparse
import gc
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from iewc.empirical2_forecasting import ForecastingConfig, run_forecasting_cl
from iewc.empirical2_nlp import NLPCLConfig, run_nlp_cl
from iewc.empirical2_trace import TraceCLConfig, run_trace_cl
from iewc.empirical2_vision import VisionCLConfig, run_vision_cl


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "empirical-2" / "artifacts"

METHODS = {
    "vision": ["ef", "ewc_dr", "iewc", "iewc_gss", "iewc_fromp"],
    "nlp": ["ef", "ewc_dr", "iewc", "iewc_gss", "iewc_fromp"],
    "forecasting": ["ef", "iewc", "iewc_gss"],
    "trace": ["ef", "ewc_dr", "iewc", "iewc_gss"],
}


def lambda_slug(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:g}".replace("-", "m").replace(".", "p")


def metric(result: dict[str, Any], group: str) -> float:
    if group in {"vision", "nlp"}:
        return float(result["final_avg_accuracy"])
    if group == "trace":
        return float(result["final_avg_score"])
    return float(result["final_avg_mse"])


def is_better(candidate: float, incumbent: float | None, group: str) -> bool:
    if incumbent is None:
        return True
    if group in {"vision", "nlp", "trace"}:
        return candidate > incumbent
    return candidate < incumbent


def maybe_clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_once(
    *,
    group: str,
    config,
    method: str,
    seed: int,
    lamb: float | None,
    tag: str,
    output_dir: Path,
    force: bool,
) -> dict[str, Any]:
    suffix = f"{group}-{tag}-{method}-seed{seed}"
    if lamb is not None:
        suffix += f"-lam{lambda_slug(lamb)}"
    path = output_dir / f"{suffix}.json"
    if path.exists() and not force:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        return data

    run_config = replace(config, seed=seed)
    if lamb is not None:
        run_config = replace(run_config, ewc_lambda=float(lamb))
    if group == "vision":
        result = run_vision_cl(run_config, method)
    elif group == "nlp":
        result = run_nlp_cl(run_config, method)
    elif group == "forecasting":
        result = run_forecasting_cl(run_config, method)
    elif group == "trace":
        result = run_trace_cl(run_config, method)
    else:
        raise ValueError(f"Unknown group: {group}")
    write_json(path, result)
    result["_path"] = str(path)
    maybe_clear_cuda()
    return result


def best_record(records: list[dict[str, Any]], group: str) -> dict[str, Any]:
    best = records[0]
    best_value = metric(best["result"], group)
    for record in records[1:]:
        value = metric(record["result"], group)
        if is_better(value, best_value, group):
            best = record
            best_value = value
    return best


def tune_method(
    *,
    group: str,
    config,
    method: str,
    seed: int,
    initial_lambdas: list[float],
    edge_factor: float,
    max_edge_extensions: int,
    tag: str,
    output_dir: Path,
    force: bool,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    tried = set()

    def run_lamb(lamb: float) -> None:
        if lamb in tried:
            return
        tried.add(lamb)
        result = run_once(
            group=group,
            config=config,
            method=method,
            seed=seed,
            lamb=lamb,
            tag=tag,
            output_dir=output_dir,
            force=force,
        )
        records.append({"lambda": lamb, "metric": metric(result, group), "result": result})

    for lamb in sorted(initial_lambdas):
        run_lamb(float(lamb))

    extensions = 0
    while extensions < max_edge_extensions:
        ordered = sorted(record["lambda"] for record in records)
        best = best_record(records, group)
        best_lamb = float(best["lambda"])
        if best_lamb == ordered[0]:
            next_lamb = best_lamb / edge_factor
        elif best_lamb == ordered[-1]:
            next_lamb = best_lamb * edge_factor
        else:
            break
        if not math.isfinite(next_lamb) or next_lamb <= 0:
            break
        run_lamb(float(next_lamb))
        extensions += 1

    selected = best_record(records, group)
    return {
        "method": method,
        "records": [
            {
                "lambda": record["lambda"],
                "metric": record["metric"],
                "path": record["result"].get("_path", ""),
            }
            for record in sorted(records, key=lambda item: item["lambda"])
        ],
        "selected_lambda": selected["lambda"],
        "selected_metric": selected["metric"],
        "edge_extensions": extensions,
    }


def make_config(args):
    if args.group == "vision":
        return VisionCLConfig(
            dataset=args.dataset,
            data_root=args.data_root,
            model_name=args.model_name,
            pretrained=args.pretrained,
            adaptation=args.adaptation,
            seed=args.seed,
            n_tasks=args.n_tasks,
            classes_per_task=args.classes_per_task,
            image_size=args.image_size,
            train_samples_per_class=args.train_samples_per_class,
            test_samples_per_class=args.test_samples_per_class,
            epochs_per_task=args.epochs_per_task,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            ewc_lambda=args.ewc_lambda,
            tau=args.tau,
            importance_samples=args.importance_samples,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            num_workers=args.num_workers,
            device=args.device,
            download=args.download,
            evaluation=args.evaluation,
        )
    if args.group == "nlp":
        return NLPCLConfig(
            model_name=args.model_name,
            tasks=tuple(args.tasks),
            seed=args.seed,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            epochs_per_task=args.epochs_per_task,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            ewc_lambda=args.ewc_lambda,
            tau=args.tau,
            importance_samples=args.importance_samples,
            max_length=args.max_length,
            adaptation=args.adaptation,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            device=args.device,
            synthetic=args.synthetic,
        )
    if args.group == "forecasting":
        data_root = args.data_root
        if data_root == "/home/davwis/main/data":
            data_root = {
                "ett": "/home/davwis/main/data/ett",
                "long_horizon": "/home/davwis/main/data/long_horizon",
                "m4": "/home/davwis/main/data/m4/tsf",
            }[args.forecast_dataset]
        return ForecastingConfig(
            dataset=args.forecast_dataset,
            data_root=data_root,
            frequencies=tuple(args.tasks),
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
    if args.group == "trace":
        data_root = args.data_root
        if data_root == "/home/davwis/main/data":
            data_root = f"/home/davwis/main/data/trace/TRACE-Benchmark/LLM-CL-Benchmark_{args.trace_size}"
        return TraceCLConfig(
            data_root=data_root,
            model_name=args.model_name,
            tasks=tuple(args.tasks),
            seed=args.seed,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            epochs_per_task=args.epochs_per_task,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            ewc_lambda=args.ewc_lambda,
            tau=args.tau,
            importance_samples=args.importance_samples,
            max_prompt_length=args.max_prompt_length,
            max_answer_length=args.max_answer_length,
            generation_max_new_tokens=args.generation_max_new_tokens,
            answer_mode=args.answer_mode,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            dtype=args.dtype,
            device=args.device,
        )
    raise ValueError(f"Unknown group: {args.group}")


def parse_method_grids(items: list[str]) -> dict[str, list[float]]:
    grids: dict[str, list[float]] = {}
    for item in items:
        if ":" not in item:
            raise ValueError("--method-grid entries must look like method:1,3,10")
        method, raw_values = item.split(":", 1)
        values = [float(value) for value in raw_values.split(",") if value]
        if not values:
            raise ValueError(f"Empty lambda grid for method {method}")
        grids[method] = values
    return grids


def centered_lambda_grid(center: float, factor: float) -> list[float]:
    if not math.isfinite(center) or center <= 0:
        raise ValueError("--lambda-center and --method-center values must be positive finite numbers")
    if not math.isfinite(factor) or factor <= 1:
        raise ValueError("--edge-factor must be > 1 when constructing centered lambda grids")
    return sorted(
        {
            float(center) / (factor**2),
            float(center) / factor,
            float(center),
            float(center) * factor,
            float(center) * (factor**2),
        }
    )


def parse_method_centers(items: list[str]) -> dict[str, float]:
    centers: dict[str, float] = {}
    for item in items:
        if ":" not in item:
            raise ValueError("--method-center entries must look like method:100")
        method, raw_value = item.split(":", 1)
        centers[method] = float(raw_value)
    return centers


def initial_lambdas_for_method(
    *,
    method: str,
    method_grids: dict[str, list[float]],
    method_centers: dict[str, float],
    lambda_center: float | None,
    default_lambdas: list[float],
    edge_factor: float,
) -> list[float]:
    if method in method_grids:
        return method_grids[method]
    if method in method_centers:
        return centered_lambda_grid(method_centers[method], edge_factor)
    if lambda_center is not None:
        return centered_lambda_grid(lambda_center, edge_factor)
    return default_lambdas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["vision", "nlp", "forecasting", "trace"], required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    parser.add_argument(
        "--method-grid",
        nargs="*",
        default=[],
        help="Optional method-specific grids like ef:1000,3000,9000 iewc:300,1000,3000.",
    )
    parser.add_argument(
        "--lambda-center",
        type=float,
        default=None,
        help="Optional center for an automatic five-point lambda grid: c/f^2,c/f,c,c*f,c*f^2.",
    )
    parser.add_argument(
        "--method-center",
        nargs="*",
        default=[],
        help="Optional method-specific automatic grid centers like ef:1000 iewc:300.",
    )
    parser.add_argument("--edge-factor", type=float, default=3.0)
    parser.add_argument("--max-edge-extensions", type=int, default=3)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--dataset", choices=["cifar100", "tiny_imagenet", "imagenet_r"], default="imagenet_r")
    parser.add_argument("--forecast-dataset", choices=["m4", "ett", "long_horizon"], default="ett")
    parser.add_argument("--data-root", default="/home/davwis/main/data")
    parser.add_argument("--model-name", default="vit_base_patch16_224")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--adaptation", choices=["full", "lora"], default="lora")
    parser.add_argument("--tasks", nargs="+", default=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    parser.add_argument("--synthetic", action="store_true")

    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--classes-per-task", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-samples-per-class", type=int, default=0)
    parser.add_argument("--test-samples-per-class", type=int, default=0)
    parser.add_argument("--evaluation", choices=["task_aware", "class_incremental"], default="class_incremental")

    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--trace-size", type=int, default=500)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-answer-length", type=int, default=32)
    parser.add_argument("--generation-max-new-tokens", type=int, default=24)
    parser.add_argument("--answer-mode", choices=["full", "choice"], default="full")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")

    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--max-series-per-task", type=int, default=64)
    parser.add_argument("--windows-per-series", type=int, default=512)
    parser.add_argument("--eval-windows-per-series", type=int, default=128)
    parser.add_argument("--model-type", choices=["encoder", "patchtst"], default="encoder")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--patch-length", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--normalization", choices=["series", "context", "task"], default="series")

    parser.add_argument("--epochs-per-task", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ewc-lambda", type=float, default=100.0)
    parser.add_argument("--tau", type=float, default=1e-2)
    parser.add_argument("--importance-samples", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    config = make_config(args)
    methods = args.methods if args.methods is not None else METHODS[args.group]
    method_grids = parse_method_grids(args.method_grid)
    method_centers = parse_method_centers(args.method_center)
    output_dir = args.output_dir
    summary_path = output_dir / f"{args.group}-{args.tag}-tuning-summary.json"

    sequential = run_once(
        group=args.group,
        config=config,
        method="sequential",
        seed=args.seed,
        lamb=None,
        tag=args.tag,
        output_dir=output_dir,
        force=args.force,
    )
    selections = []
    for method in methods:
        if method == "sequential":
            continue
        selections.append(
            tune_method(
                group=args.group,
                config=config,
                method=method,
                seed=args.seed,
                initial_lambdas=initial_lambdas_for_method(
                    method=method,
                    method_grids=method_grids,
                    method_centers=method_centers,
                    lambda_center=args.lambda_center,
                    default_lambdas=args.lambdas,
                    edge_factor=args.edge_factor,
                ),
                edge_factor=args.edge_factor,
                max_edge_extensions=args.max_edge_extensions,
                tag=args.tag,
                output_dir=output_dir,
                force=args.force,
            )
        )

    final_records = []
    for final_seed in args.final_seeds:
        final_records.append(
            {
                "method": "sequential",
                "seed": final_seed,
                "lambda": None,
                "result": run_once(
                    group=args.group,
                    config=config,
                    method="sequential",
                    seed=final_seed,
                    lamb=None,
                    tag=args.tag,
                    output_dir=output_dir,
                    force=args.force,
                ).get("_path", ""),
            }
        )
        for selection in selections:
            final_records.append(
                {
                    "method": selection["method"],
                    "seed": final_seed,
                    "lambda": selection["selected_lambda"],
                    "result": run_once(
                        group=args.group,
                        config=config,
                        method=selection["method"],
                        seed=final_seed,
                        lamb=float(selection["selected_lambda"]),
                        tag=args.tag,
                        output_dir=output_dir,
                        force=args.force,
                    ).get("_path", ""),
                }
            )

    summary = {
        "experiment": "empirical2_tuning",
        "group": args.group,
        "tag": args.tag,
        "config": asdict(config),
        "objective": "max_final_avg_accuracy"
        if args.group in {"vision", "nlp"}
        else ("max_final_avg_score" if args.group == "trace" else "min_final_avg_mse"),
        "initial_lambdas": args.lambdas,
        "method_grids": method_grids,
        "lambda_center": args.lambda_center,
        "method_centers": method_centers,
        "centered_grid_rule": "center/edge_factor^2, center/edge_factor, center, center*edge_factor, center*edge_factor^2",
        "edge_factor": args.edge_factor,
        "max_edge_extensions": args.max_edge_extensions,
        "tune_seed": args.seed,
        "sequential": {
            "metric": metric(sequential, args.group),
            "path": sequential.get("_path", ""),
        },
        "selections": selections,
        "final_records": final_records,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
