import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from iewc.synthetic_regression import (
    RegressionMLP,
    SyntheticRegressionConfig,
    make_dataset,
    make_regression_shift,
)


@dataclass(frozen=True)
class WeightedImportance:
    importances: dict[str, torch.Tensor]
    loss_scales: torch.Tensor
    row_traces: torch.Tensor
    weights: torch.Tensor
    trace_scale: float
    diag_leverage: torch.Tensor | None
    diag_consensus: torch.Tensor | None
    diag_shape_deviation: torch.Tensor | None


def mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x.to(device))
        return float(nn.functional.mse_loss(pred, y.to(device)).item())


def compute_penalty(model: nn.Module, saved_params, importances):
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name not in saved_params:
            continue
        penalty = penalty + (importances[name] * (param - saved_params[name]).pow(2)).sum()
    return penalty


def train_task(
    *,
    model: nn.Module,
    dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    criterion: nn.Module,
    device: torch.device,
    ewc_lambda: float = 0.0,
    saved_params=None,
    importances=None,
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if importances is not None and saved_params is not None:
                loss = loss + ewc_lambda * compute_penalty(model, saved_params, importances)
            loss.backward()
            optimizer.step()
    return optimizer


def _renormalize_weights(weights: torch.Tensor) -> torch.Tensor:
    weights = torch.nan_to_num(weights.float(), nan=0.0, posinf=0.0, neginf=0.0)
    weights = weights.clamp_min(0.0)
    mean_weight = weights.mean()
    if float(mean_weight.item()) <= 0.0:
        return torch.ones_like(weights)
    return weights / mean_weight


def _parse_percent_power(mode: str, prefix: str) -> float:
    return float(mode.removeprefix(prefix).replace("p", ".")) / 100.0


def _diag_leverage_scores(
    contributions: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    names = list(contributions[0])
    mean_diag = {
        name: torch.zeros_like(contributions[0][name])
        for name in names
    }
    for sample in contributions:
        for name in names:
            mean_diag[name] += sample[name]
    sample_count = float(len(contributions))
    mean_diag = {name: value / sample_count for name, value in mean_diag.items()}

    positive_parts = [
        value.detach().flatten()[value.detach().flatten() > 0]
        for value in mean_diag.values()
    ]
    positive_parts = [value for value in positive_parts if value.numel() > 0]
    if positive_parts:
        positive_values = torch.cat(positive_parts)
        ridge = positive_values.median().clamp_min(torch.finfo(positive_values.dtype).eps) * 1e-4
    else:
        ridge = torch.tensor(torch.finfo(next(iter(mean_diag.values())).dtype).eps)

    scores = []
    for sample in contributions:
        score = torch.zeros((), device=next(iter(sample.values())).device)
        for name in names:
            score = score + (sample[name] / (mean_diag[name] + ridge.to(mean_diag[name].device))).sum()
        scores.append(score.detach().cpu())
    return torch.stack(scores).float()


def _diag_consensus_scores(
    contributions: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    names = list(contributions[0])
    mean_diag = {
        name: torch.zeros_like(contributions[0][name])
        for name in names
    }
    for sample in contributions:
        for name in names:
            mean_diag[name] += sample[name]
    sample_count = float(len(contributions))
    mean_diag = {name: value / sample_count for name, value in mean_diag.items()}
    mean_norm_sq = torch.zeros((), device=next(iter(mean_diag.values())).device)
    for value in mean_diag.values():
        mean_norm_sq = mean_norm_sq + value.pow(2).sum()
    mean_norm = mean_norm_sq.sqrt()

    eps = torch.finfo(mean_norm.dtype).eps
    scores = []
    for sample in contributions:
        dot = torch.zeros((), device=mean_norm.device)
        sample_norm_sq = torch.zeros((), device=mean_norm.device)
        for name in names:
            dot = dot + (sample[name] * mean_diag[name]).sum()
            sample_norm_sq = sample_norm_sq + sample[name].pow(2).sum()
        score = dot / (sample_norm_sq.sqrt() * mean_norm + eps)
        scores.append(score.detach().cpu())
    return torch.stack(scores).float().clamp_min(0.0)


def _diag_shape_deviation_scores(
    contributions: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    names = list(contributions[0])
    mean_diag = {
        name: torch.zeros_like(contributions[0][name])
        for name in names
    }
    for sample in contributions:
        for name in names:
            mean_diag[name] += sample[name]
    sample_count = float(len(contributions))
    mean_diag = {name: value / sample_count for name, value in mean_diag.items()}

    positive_parts = [
        value.detach().flatten()[value.detach().flatten() > 0]
        for value in mean_diag.values()
    ]
    positive_parts = [value for value in positive_parts if value.numel() > 0]
    if positive_parts:
        positive_values = torch.cat(positive_parts)
        ridge = positive_values.median().clamp_min(torch.finfo(positive_values.dtype).eps) * 1e-4
    else:
        ridge = torch.tensor(torch.finfo(next(iter(mean_diag.values())).dtype).eps)

    scores = []
    for sample in contributions:
        score = torch.zeros((), device=next(iter(sample.values())).device)
        for name in names:
            denom = mean_diag[name] + ridge.to(mean_diag[name].device)
            score = score + ((sample[name] - mean_diag[name]) / denom).pow(2).sum()
        scores.append(score.sqrt().detach().cpu())
    return torch.stack(scores).float()


def make_weights(
    mode: str,
    row_traces: torch.Tensor,
    loss_scales: torch.Tensor,
    tau: float,
    diag_leverage: torch.Tensor | None = None,
    diag_consensus: torch.Tensor | None = None,
    diag_shape_deviation: torch.Tensor | None = None,
) -> torch.Tensor:
    eps = torch.finfo(row_traces.dtype).eps
    if mode == "uniform":
        return torch.ones_like(row_traces)

    if mode == "outer_row_fro_diag":
        return torch.ones_like(row_traces)

    if mode.startswith("outer_row_fro_diag_scale_p"):
        power = _parse_percent_power(mode, "outer_row_fro_diag_scale_p")
        center = torch.quantile(row_traces, 0.5).clamp_min(eps)
        weights = (row_traces / center).clamp_min(eps).pow(power).clamp(0.1, 10.0)
        return _renormalize_weights(weights)

    if mode.startswith("row_clip_q"):
        quantile = float(mode.removeprefix("row_clip_q")) / 100.0
        cap = torch.quantile(row_traces, quantile)
        return _renormalize_weights(torch.minimum(torch.ones_like(row_traces), cap / (row_traces + eps)))

    if mode.startswith("row_trim_q"):
        quantile = float(mode.removeprefix("row_trim_q")) / 100.0
        cap = torch.quantile(row_traces, quantile)
        return _renormalize_weights((row_traces <= cap).float())

    if mode == "row_equalize_sqrt":
        center = torch.quantile(row_traces, 0.5)
        weights = torch.sqrt(center / (row_traces + eps)).clamp(0.25, 4.0)
        return _renormalize_weights(weights)

    if mode == "row_equalize_linear":
        center = torch.quantile(row_traces, 0.5)
        weights = (center / (row_traces + eps)).clamp(0.1, 10.0)
        return _renormalize_weights(weights)

    if mode.startswith("loss_clip_q"):
        quantile = float(mode.removeprefix("loss_clip_q")) / 100.0
        cap = torch.quantile(loss_scales, quantile)
        return _renormalize_weights(torch.minimum(torch.ones_like(loss_scales), cap / (loss_scales + eps)))

    if mode == "loss_equalize_sqrt":
        center = torch.quantile(loss_scales, 0.5)
        weights = torch.sqrt(center / (loss_scales + eps)).clamp(0.25, 4.0)
        return _renormalize_weights(weights)

    if mode == "row_and_loss_clip_q90":
        row_cap = torch.quantile(row_traces, 0.9)
        loss_cap = torch.quantile(loss_scales, 0.9)
        weights = torch.minimum(torch.ones_like(row_traces), row_cap / (row_traces + eps))
        weights = weights * torch.minimum(torch.ones_like(loss_scales), loss_cap / (loss_scales + eps))
        return _renormalize_weights(weights)

    if mode.startswith("outer_product_scale_raw_p"):
        power = _parse_percent_power(mode, "outer_product_scale_raw_p")
        center = torch.quantile(row_traces, 0.5).clamp_min(eps)
        weights = (row_traces / center).clamp_min(eps).pow(power)
        return _renormalize_weights(weights)

    if mode.startswith("outer_product_scale_p"):
        power = _parse_percent_power(mode, "outer_product_scale_p")
        center = torch.quantile(row_traces, 0.5).clamp_min(eps)
        weights = (row_traces / center).clamp_min(eps).pow(power).clamp(0.1, 10.0)
        return _renormalize_weights(weights)

    if mode.startswith("diag_leverage_boost_p"):
        if diag_leverage is None:
            raise ValueError(f"{mode} requires diagonal leverage scores")
        power = _parse_percent_power(mode, "diag_leverage_boost_p")
        center = torch.quantile(diag_leverage, 0.5).clamp_min(eps)
        weights = (diag_leverage / center).clamp_min(eps).pow(power).clamp(0.25, 4.0)
        return _renormalize_weights(weights)

    if mode.startswith("diag_leverage_temper_p"):
        if diag_leverage is None:
            raise ValueError(f"{mode} requires diagonal leverage scores")
        power = _parse_percent_power(mode, "diag_leverage_temper_p")
        center = torch.quantile(diag_leverage, 0.5).clamp_min(eps)
        weights = (center / diag_leverage.clamp_min(eps)).pow(power).clamp(0.25, 4.0)
        return _renormalize_weights(weights)

    if mode.startswith("diag_leverage_clip_q"):
        if diag_leverage is None:
            raise ValueError(f"{mode} requires diagonal leverage scores")
        quantile = float(mode.removeprefix("diag_leverage_clip_q")) / 100.0
        cap = torch.quantile(diag_leverage, quantile)
        return _renormalize_weights(torch.minimum(torch.ones_like(diag_leverage), cap / (diag_leverage + eps)))

    if mode.startswith("diag_consensus_boost_p"):
        if diag_consensus is None:
            raise ValueError(f"{mode} requires diagonal consensus scores")
        power = _parse_percent_power(mode, "diag_consensus_boost_p")
        center = torch.quantile(diag_consensus, 0.5).clamp_min(eps)
        weights = (diag_consensus / center).clamp_min(eps).pow(power).clamp(0.25, 4.0)
        return _renormalize_weights(weights)

    if mode.startswith("diag_consensus_temper_p"):
        if diag_consensus is None:
            raise ValueError(f"{mode} requires diagonal consensus scores")
        power = _parse_percent_power(mode, "diag_consensus_temper_p")
        center = torch.quantile(diag_consensus, 0.5).clamp_min(eps)
        weights = (center / diag_consensus.clamp_min(eps)).pow(power).clamp(0.25, 4.0)
        return _renormalize_weights(weights)

    if mode.startswith("diag_shape_clip_q"):
        if diag_shape_deviation is None:
            raise ValueError(f"{mode} requires diagonal shape-deviation scores")
        quantile = float(mode.removeprefix("diag_shape_clip_q")) / 100.0
        cap = torch.quantile(diag_shape_deviation, quantile)
        return _renormalize_weights(
            torch.minimum(torch.ones_like(diag_shape_deviation), cap / (diag_shape_deviation + eps))
        )

    if mode.startswith("diag_shape_temper_p"):
        if diag_shape_deviation is None:
            raise ValueError(f"{mode} requires diagonal shape-deviation scores")
        power = _parse_percent_power(mode, "diag_shape_temper_p")
        center = torch.quantile(diag_shape_deviation, 0.5).clamp_min(eps)
        weights = (center / diag_shape_deviation.clamp_min(eps)).pow(power).clamp(0.25, 4.0)
        return _renormalize_weights(weights)

    if mode.startswith("coef_temper_p"):
        power = _parse_percent_power(mode, "coef_temper_p")
        weights = (loss_scales + float(tau)).clamp_min(eps).pow(power)
        return _renormalize_weights(weights)

    if mode.startswith("coef_sharpen_p"):
        power = _parse_percent_power(mode, "coef_sharpen_p")
        weights = (loss_scales + float(tau)).clamp_min(eps).pow(-power)
        return _renormalize_weights(weights)

    raise ValueError(f"Unknown weighting mode: {mode}")


def compute_diag_importance(
    *,
    model: nn.Module,
    dataset,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
    kind: str,
    tau: float,
    weighting: str,
    match_trace: bool,
) -> WeightedImportance:
    if kind not in {"ef", "ief_diag"}:
        raise ValueError(f"Unknown kind: {kind}")
    model.eval()
    contributions: list[dict[str, torch.Tensor]] = []
    loss_scales = []
    row_traces = []
    use_outer_row_fro_diag = weighting.startswith("outer_row_fro_diag")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for x_batch, y_batch, _ in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        for idx in range(x_batch.shape[0]):
            model.zero_grad(set_to_none=True)
            x = x_batch[idx : idx + 1]
            y = y_batch[idx : idx + 1]
            out = model(x)
            loss = criterion(out, y)
            if kind == "ief_diag":
                output_grad = torch.autograd.grad(loss, out, retain_graph=True)[0]
                loss_scale = output_grad.detach().pow(2).sum()
                denom = loss_scale + tau
            else:
                loss_scale = torch.ones((), device=device)
                denom = torch.ones((), device=device)
            if not torch.isfinite(loss_scale):
                raise FloatingPointError("Non-finite loss scale in weighting search")
            loss.backward()
            standard_contrib = {}
            row_trace = torch.zeros((), device=device)
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                value = param.grad.detach().pow(2) / denom
                standard_contrib[name] = value
                row_trace = row_trace + value.sum()
            if use_outer_row_fro_diag:
                row_norm = row_trace.clamp_min(torch.finfo(row_trace.dtype).eps).sqrt()
                sample_contrib = {
                    name: value.clamp_min(0.0).sqrt() * row_norm
                    for name, value in standard_contrib.items()
                }
            else:
                sample_contrib = standard_contrib
            contributions.append(sample_contrib)
            loss_scales.append(loss_scale.detach().cpu())
            row_traces.append(row_trace.detach().cpu())
    if not contributions:
        raise ValueError("Cannot compute importance from an empty dataset")

    loss_scales_t = torch.stack(loss_scales).float()
    row_traces_t = torch.stack(row_traces).float()
    diag_leverage = (
        _diag_leverage_scores(contributions)
        if weighting.startswith("diag_leverage_")
        else None
    )
    diag_consensus = (
        _diag_consensus_scores(contributions)
        if weighting.startswith("diag_consensus_")
        else None
    )
    diag_shape_deviation = (
        _diag_shape_deviation_scores(contributions)
        if weighting.startswith("diag_shape_")
        else None
    )
    weights = make_weights(
        weighting,
        row_traces_t,
        loss_scales_t,
        tau,
        diag_leverage,
        diag_consensus,
        diag_shape_deviation,
    )
    names = list(contributions[0])
    importances = {
        name: torch.zeros_like(contributions[0][name], device=device)
        for name in names
    }
    if use_outer_row_fro_diag:
        unweighted_trace = float(row_traces_t.mean().item())
    else:
        unweighted_trace = 0.0
        for sample in contributions:
            unweighted_trace += float(sum(value.sum().item() for value in sample.values()))
        unweighted_trace = unweighted_trace / float(len(contributions))
    for weight, sample in zip(weights, contributions):
        weight_device = weight.to(device)
        for name in names:
            importances[name] += sample[name] * weight_device
    sample_count = float(len(contributions))
    importances = {name: value / sample_count for name, value in importances.items()}
    weighted_trace = float(sum(value.sum().item() for value in importances.values()))
    trace_scale = 1.0
    if match_trace and weighted_trace > 0.0:
        trace_scale = unweighted_trace / weighted_trace
        importances = {name: value * trace_scale for name, value in importances.items()}
    model.train()
    return WeightedImportance(
        importances=importances,
        loss_scales=loss_scales_t,
        row_traces=row_traces_t,
        weights=weights,
        trace_scale=trace_scale,
        diag_leverage=diag_leverage,
        diag_consensus=diag_consensus,
        diag_shape_deviation=diag_shape_deviation,
    )


def run_one(
    *,
    config: SyntheticRegressionConfig,
    method: str,
    tau: float,
    weighting: str,
    device: torch.device,
    match_trace: bool,
) -> dict:
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    task_a_train, task_b_train, task_a_test, task_b_test = make_regression_shift(
        config.seed, config.n_train, config.n_test
    )
    train_a = make_dataset(*task_a_train)
    train_b = make_dataset(*task_b_train)
    criterion = nn.MSELoss()
    model = RegressionMLP(config.hidden_size).to(device)
    train_task(
        model=model,
        dataset=train_a,
        epochs=config.train_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        criterion=criterion,
        device=device,
    )
    task_a_mse_after_task_a = mse(model, *task_a_test, device)

    saved_params = None
    importances = None
    importance_stats = {}
    if method != "sequential":
        importance = compute_diag_importance(
            model=model,
            dataset=train_a,
            criterion=criterion,
            batch_size=config.batch_size,
            device=device,
            kind=method,
            tau=tau,
            weighting=weighting if method == "ief_diag" else "uniform",
            match_trace=match_trace,
        )
        saved_params = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        importances = importance.importances
        weights = importance.weights
        importance_stats = {
            "loss_scale_mean": float(importance.loss_scales.mean().item()),
            "loss_scale_median": float(importance.loss_scales.median().item()),
            "row_trace_mean": float(importance.row_traces.mean().item()),
            "row_trace_median": float(importance.row_traces.median().item()),
            "weight_min": float(weights.min().item()),
            "weight_median": float(weights.median().item()),
            "weight_max": float(weights.max().item()),
            "weight_ess": float((weights.sum().pow(2) / weights.pow(2).sum()).item()),
            "trace_scale": float(importance.trace_scale),
        }
        if importance.diag_leverage is not None:
            importance_stats.update(
                {
                    "diag_leverage_mean": float(importance.diag_leverage.mean().item()),
                    "diag_leverage_median": float(importance.diag_leverage.median().item()),
                    "diag_leverage_max": float(importance.diag_leverage.max().item()),
                }
            )
        if importance.diag_consensus is not None:
            importance_stats.update(
                {
                    "diag_consensus_mean": float(importance.diag_consensus.mean().item()),
                    "diag_consensus_median": float(importance.diag_consensus.median().item()),
                    "diag_consensus_min": float(importance.diag_consensus.min().item()),
                }
            )
        if importance.diag_shape_deviation is not None:
            importance_stats.update(
                {
                    "diag_shape_deviation_mean": float(importance.diag_shape_deviation.mean().item()),
                    "diag_shape_deviation_median": float(importance.diag_shape_deviation.median().item()),
                    "diag_shape_deviation_max": float(importance.diag_shape_deviation.max().item()),
                }
            )

    train_task(
        model=model,
        dataset=train_b,
        epochs=config.train_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        criterion=criterion,
        device=device,
        ewc_lambda=config.ewc_lambda,
        saved_params=saved_params,
        importances=importances,
    )

    task_a_mse_after_task_b = mse(model, *task_a_test, device)
    task_b_mse_after_task_b = mse(model, *task_b_test, device)
    output = {
        "seed": config.seed,
        "method": method,
        "tau": tau if method == "ief_diag" else None,
        "weighting": weighting if method == "ief_diag" else "uniform",
        "task_a_mse_after_task_a": task_a_mse_after_task_a,
        "task_a_mse_after_task_b": task_a_mse_after_task_b,
        "task_b_mse_after_task_b": task_b_mse_after_task_b,
        "task_a_mse_increase": task_a_mse_after_task_b - task_a_mse_after_task_a,
    }
    output.update(importance_stats)
    return output


def summarize(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["method"], record["weighting"]), []).append(record)
    summary = []
    for (method, weighting), items in sorted(grouped.items()):
        old = [item["task_a_mse_increase"] for item in items]
        new = [item["task_b_mse_after_task_b"] for item in items]
        summary.append(
            {
                "method": method,
                "weighting": weighting,
                "n": len(items),
                "old_mse_increase_mean": sum(old) / len(old),
                "old_mse_increase_std": math.sqrt(
                    sum((value - sum(old) / len(old)) ** 2 for value in old)
                    / max(1, len(old) - 1)
                ),
                "new_mse_mean": sum(new) / len(new),
                "new_mse_std": math.sqrt(
                    sum((value - sum(new) / len(new)) ** 2 for value in new)
                    / max(1, len(new) - 1)
                ),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--ewc-lambda", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument(
        "--weightings",
        nargs="+",
        default=[
            "uniform",
            "row_clip_q90",
            "row_clip_q75",
            "row_trim_q95",
            "row_equalize_sqrt",
            "row_equalize_linear",
            "loss_clip_q90",
            "loss_equalize_sqrt",
            "row_and_loss_clip_q90",
            "outer_product_scale_p25",
            "outer_product_scale_p50",
            "outer_product_scale_p100",
            "outer_product_scale_raw_p100",
            "outer_row_fro_diag",
            "outer_row_fro_diag_scale_p50",
            "outer_row_fro_diag_scale_p100",
            "diag_leverage_boost_p25",
            "diag_leverage_boost_p50",
            "diag_leverage_temper_p25",
            "diag_leverage_temper_p50",
            "diag_leverage_clip_q90",
            "diag_leverage_clip_q95",
            "diag_consensus_boost_p25",
            "diag_consensus_boost_p50",
            "diag_consensus_temper_p25",
            "diag_shape_clip_q90",
            "diag_shape_clip_q95",
            "diag_shape_temper_p25",
            "diag_shape_temper_p50",
            "coef_temper_p25",
            "coef_temper_p50",
            "coef_sharpen_p25",
            "coef_sharpen_p50",
        ],
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-match-trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    records = []
    for seed in args.seeds:
        config = SyntheticRegressionConfig(
            seed=seed,
            n_train=args.n_train,
            n_test=args.n_test,
            train_epochs=args.epochs,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            ewc_lambda=args.ewc_lambda,
            tau_values=(args.tau,),
        )
        records.append(
            run_one(
                config=config,
                method="sequential",
                tau=args.tau,
                weighting="uniform",
                device=device,
                match_trace=not args.no_match_trace,
            )
        )
        records.append(
            run_one(
                config=config,
                method="ef",
                tau=args.tau,
                weighting="uniform",
                device=device,
                match_trace=not args.no_match_trace,
            )
        )
        for weighting in args.weightings:
            records.append(
                run_one(
                    config=config,
                    method="ief_diag",
                    tau=args.tau,
                    weighting=weighting,
                    device=device,
                    match_trace=not args.no_match_trace,
                )
            )

    payload = {
        "experiment": "synthetic_regression_iewc_weighting_search",
        "config": {
            "seeds": args.seeds,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "ewc_lambda": args.ewc_lambda,
            "tau": args.tau,
            "match_trace": not args.no_match_trace,
            "device": str(device),
            "weightings": args.weightings,
        },
        "records": records,
        "summary": summarize(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
