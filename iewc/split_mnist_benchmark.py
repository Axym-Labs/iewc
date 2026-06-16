from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import time
import warnings

import torch
from avalanche.benchmarks import tensors_benchmark
from avalanche.benchmarks.classic import PermutedMNIST, SplitMNIST
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive
from torch import nn
from torch.utils.data import DataLoader

from .avalanche import IEWCPlugin
from .low_rank import LowRankIEWCPlugin


@dataclass(frozen=True)
class SplitMNISTConfig:
    benchmark_name: str = "split_mnist"
    seed: int = 0
    n_experiences: int = 5
    train_epochs: int = 10
    hidden_size: int = 400
    train_mb_size: int = 128
    eval_mb_size: int = 256
    learning_rate: float = 0.1
    ewc_lambda: float = 100.0
    tau_values: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    rank_values: tuple[int, ...] = (10,)
    dataset_root: str = "data"
    device: str = "auto"
    max_train_per_experience: int | None = None
    max_test_per_experience: int | None = None
    max_importance_samples: int | None = None
    importance_sample_seed: int = 0


class SplitMNISTMLP(nn.Module):
    def __init__(self, hidden_size: int, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _device_from_config(config: SplitMNISTConfig):
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def _maybe_cap_dataset(dataset, max_items: int | None):
    if max_items is None or len(dataset) <= max_items:
        return dataset
    return dataset.subset(range(max_items))


def _experience_to_tensors(dataset, max_items: int | None):
    count = len(dataset) if max_items is None else min(len(dataset), max_items)
    xs = []
    ys = []
    for idx in range(count):
        item = dataset[idx]
        xs.append(item[0])
        ys.append(int(item[1]))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def make_split_mnist(config: SplitMNISTConfig):
    if config.benchmark_name == "split_mnist":
        benchmark = SplitMNIST(
            n_experiences=config.n_experiences,
            seed=config.seed,
            dataset_root=Path(config.dataset_root),
        )
    elif config.benchmark_name == "permuted_mnist":
        benchmark = PermutedMNIST(
            n_experiences=config.n_experiences,
            seed=config.seed,
            dataset_root=Path(config.dataset_root),
        )
    else:
        raise ValueError(f"Unknown benchmark_name: {config.benchmark_name}")
    if (
        config.max_train_per_experience is None
        and config.max_test_per_experience is None
    ):
        return benchmark

    train_tensors = []
    test_tensors = []
    for experience in benchmark.train_stream:
        train_tensors.append(
            _experience_to_tensors(
                experience.dataset, config.max_train_per_experience
            )
        )
    for experience in benchmark.test_stream:
        test_tensors.append(
            _experience_to_tensors(
                experience.dataset, config.max_test_per_experience
            )
        )
    return tensors_benchmark(
        train_tensors=train_tensors,
        test_tensors=test_tensors,
        task_labels=[0] * config.n_experiences,
        complete_test_set_only=False,
    )


@torch.no_grad()
def experience_accuracy(model: nn.Module, dataset, *, batch_size: int, device: torch.device):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    if total == 0:
        raise ValueError("Cannot evaluate an empty SplitMNIST experience")
    return correct / float(total)


def evaluate_seen(model, benchmark, *, upto_exp: int, batch_size: int, device: torch.device):
    accuracies = []
    for exp_id, experience in enumerate(benchmark.test_stream):
        if exp_id > upto_exp:
            break
        accuracies.append(
            experience_accuracy(
                model,
                experience.dataset,
                batch_size=batch_size,
                device=device,
            )
        )
    return accuracies


def summarize_accuracy_matrix(accuracy_matrix: list[list[float]]):
    final_row = accuracy_matrix[-1]
    final_average_accuracy = sum(final_row) / float(len(final_row))
    forgetting_values = []
    for exp_id in range(len(final_row) - 1):
        best_before_final = max(row[exp_id] for row in accuracy_matrix[exp_id:])
        forgetting_values.append(best_before_final - final_row[exp_id])
    average_forgetting = (
        sum(forgetting_values) / float(len(forgetting_values))
        if forgetting_values
        else 0.0
    )
    return final_average_accuracy, average_forgetting, forgetting_values


def _forward_flops_per_sample(hidden_size: int) -> int:
    return int(
        2
        * (
            28 * 28 * hidden_size
            + hidden_size * hidden_size
            + hidden_size * 10
        )
    )


def estimate_flops(
    *,
    config: SplitMNISTConfig,
    benchmark,
    method: str,
    rank: int | None,
) -> dict[str, float]:
    forward = _forward_flops_per_sample(config.hidden_size)
    train_samples = [len(exp.dataset) for exp in benchmark.train_stream]
    test_samples = [len(exp.dataset) for exp in benchmark.test_stream]
    train_model = float(3 * forward * config.train_epochs * sum(train_samples))
    eval_model = float(
        forward
        * sum(sum(test_samples[: exp_id + 1]) for exp_id in range(len(train_samples)))
    )

    is_importance_method = method not in {"sequential", "naive"}
    importance_counts = []
    if is_importance_method:
        for count in train_samples:
            if config.max_importance_samples is None:
                importance_counts.append(count)
            else:
                importance_counts.append(min(count, config.max_importance_samples))
    importance_model = float(3 * forward * sum(importance_counts))

    parameter_count = (
        28 * 28 * config.hidden_size
        + config.hidden_size
        + config.hidden_size * config.hidden_size
        + config.hidden_size
        + config.hidden_size * 10
        + 10
    )
    minibatches = sum(
        ((count + config.train_mb_size - 1) // config.train_mb_size)
        * config.train_epochs
        for count in train_samples
    )
    rank_value = 0 if rank is None else int(rank)
    low_rank = "low_rank" in method
    hybrid = (
        "low_rank_diag" in method
        or "diag_low_rank" in method
        or "corr_low_rank" in method
    )
    penalty_overhead = 0.0
    sketch_linear_algebra = 0.0
    if low_rank:
        previous_states_by_experience = [max(0, exp_id) for exp_id in range(len(train_samples))]
        penalty_overhead = float(
            sum(previous_states_by_experience)
            * minibatches
            * max(1, rank_value)
            * parameter_count
            * 2
        )
        if hybrid:
            penalty_overhead += float(
                sum(previous_states_by_experience) * minibatches * parameter_count
            )
        for n_samples in importance_counts:
            sketch_linear_algebra += float(
                n_samples * n_samples * parameter_count
                + n_samples * n_samples * n_samples
                + max(1, rank_value) * n_samples * parameter_count
            )

    total = train_model + eval_model + importance_model + sketch_linear_algebra + penalty_overhead
    return {
        "forward_per_sample": float(forward),
        "train_model": train_model,
        "eval_model": eval_model,
        "importance_model": importance_model,
        "sketch_linear_algebra": sketch_linear_algebra,
        "penalty_overhead": penalty_overhead,
        "total": float(total),
    }


def run_one(
    *,
    config: SplitMNISTConfig,
    method: str,
    tau: float | None = None,
    rank: int | None = None,
) -> dict:
    run_start = time.perf_counter()
    torch.manual_seed(config.seed)
    device = _device_from_config(config)
    benchmark = make_split_mnist(config)
    model = SplitMNISTMLP(config.hidden_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    plugins = []
    canonical_method = "sequential" if method == "naive" else method
    if canonical_method in {"ef", "ewc_dr", "ief_diag"}:
        plugins.append(
            IEWCPlugin(
                ewc_lambda=config.ewc_lambda,
                importance_kind=canonical_method,
                tau=0.0 if tau is None else tau,
                max_importance_samples=config.max_importance_samples,
            )
        )
    elif canonical_method in {
        "ef_low_rank",
        "ief_low_rank",
        "ef_low_rank_diag",
        "ief_low_rank_diag",
        "ef_diag_low_rank",
        "ief_diag_low_rank",
        "ef_corr_low_rank",
        "ief_corr_low_rank",
    }:
        plugins.append(
            LowRankIEWCPlugin(
                ewc_lambda=config.ewc_lambda,
                importance_kind=canonical_method,
                tau=0.0 if tau is None else tau,
                rank=config.rank_values[0] if rank is None else rank,
                max_importance_samples=config.max_importance_samples,
                importance_sample_seed=config.importance_sample_seed,
            )
        )
    elif canonical_method != "sequential":
        raise ValueError(f"Unknown method: {method}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No loggers specified, metrics will not be logged",
            category=UserWarning,
        )
        evaluator = EvaluationPlugin()

    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=config.train_mb_size,
        train_epochs=config.train_epochs,
        eval_mb_size=config.eval_mb_size,
        device=device,
        plugins=plugins,
        evaluator=evaluator,
    )

    accuracy_matrix = []
    train_wall_seconds_by_experience = []
    eval_wall_seconds_by_experience = []
    for exp_id, experience in enumerate(benchmark.train_stream):
        train_start = time.perf_counter()
        strategy.train(experience, num_workers=0)
        train_wall_seconds_by_experience.append(time.perf_counter() - train_start)
        eval_start = time.perf_counter()
        accuracy_matrix.append(
            evaluate_seen(
                model,
                benchmark,
                upto_exp=exp_id,
                batch_size=config.eval_mb_size,
                device=device,
            )
        )
        eval_wall_seconds_by_experience.append(time.perf_counter() - eval_start)

    final_average_accuracy, average_forgetting, forgetting_values = (
        summarize_accuracy_matrix(accuracy_matrix)
    )
    result = {
        "method": canonical_method,
        "benchmark_name": config.benchmark_name,
        "tau": tau,
        "rank": rank,
        "device": str(device),
        "max_importance_samples": config.max_importance_samples,
        "final_average_accuracy": final_average_accuracy,
        "average_forgetting": average_forgetting,
        "forgetting_values": forgetting_values,
        "accuracy_matrix": accuracy_matrix,
        "train_wall_seconds": sum(train_wall_seconds_by_experience),
        "eval_wall_seconds": sum(eval_wall_seconds_by_experience),
        "wall_seconds_total": time.perf_counter() - run_start,
        "train_wall_seconds_by_experience": train_wall_seconds_by_experience,
        "eval_wall_seconds_by_experience": eval_wall_seconds_by_experience,
        "estimated_flops": estimate_flops(
            config=config,
            benchmark=benchmark,
            method=canonical_method,
            rank=rank,
        ),
    }
    if plugins and plugins[0].last_importance_result is not None:
        loss_scales = plugins[0].last_importance_result.loss_scales
        result["last_loss_scale_mean"] = float(loss_scales.mean().item())
        result["last_loss_scale_median"] = float(loss_scales.median().item())
        if hasattr(plugins[0].last_importance_result, "explained_variance_ratio"):
            result["last_explained_variance_ratio"] = float(
                plugins[0].last_importance_result.explained_variance_ratio
            )
            result["last_actual_rank"] = int(plugins[0].last_importance_result.rank)
        if (
            hasattr(plugins[0].last_importance_result, "residual_diagonal_mass")
            and plugins[0].last_importance_result.residual_diagonal is not None
        ):
            result["last_residual_diagonal_mass"] = float(
                plugins[0].last_importance_result.residual_diagonal_mass
            )
    return result


def run_split_mnist_suite(
    *,
    config: SplitMNISTConfig,
    methods: Sequence[str] = ("sequential", "ef", "ewc_dr", "ief_diag"),
) -> list[dict]:
    results = []
    for raw_method in methods:
        method = "sequential" if raw_method == "naive" else raw_method
        if method == "ief_diag":
            tau_results = [
                run_one(config=config, method=method, tau=tau)
                for tau in config.tau_values
            ]
            best = max(
                tau_results,
                key=lambda item: (
                    item["final_average_accuracy"],
                    -item["average_forgetting"],
                ),
            )
            combined = dict(best)
            combined["tau_results"] = tau_results
            results.append(combined)
        elif method in {
            "ief_low_rank",
            "ief_low_rank_diag",
            "ief_diag_low_rank",
            "ief_corr_low_rank",
        }:
            rank_results = []
            for rank in config.rank_values:
                for tau in config.tau_values:
                    rank_results.append(
                        run_one(config=config, method=method, tau=tau, rank=rank)
                    )
            best = max(
                rank_results,
                key=lambda item: (
                    item["final_average_accuracy"],
                    -item["average_forgetting"],
                ),
            )
            combined = dict(best)
            combined["rank_results"] = rank_results
            results.append(combined)
        elif method in {
            "ef_low_rank",
            "ef_low_rank_diag",
            "ef_diag_low_rank",
            "ef_corr_low_rank",
        }:
            rank_results = [
                run_one(config=config, method=method, rank=rank)
                for rank in config.rank_values
            ]
            best = max(
                rank_results,
                key=lambda item: (
                    item["final_average_accuracy"],
                    -item["average_forgetting"],
                ),
            )
            combined = dict(best)
            combined["rank_results"] = rank_results
            results.append(combined)
        else:
            results.append(run_one(config=config, method=method))
    return results
