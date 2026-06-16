from dataclasses import dataclass
import math
from typing import Sequence
import warnings

import torch
from avalanche.benchmarks import tensors_benchmark
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive
from torch import nn

from .avalanche import IEWCPlugin


SEQUENTIAL_ALIASES = {"sequential", "naive"}


@dataclass(frozen=True)
class SyntheticRunConfig:
    seed: int = 0
    n_train: int = 256
    n_test: int = 512
    train_epochs: int = 20
    hidden_size: int = 32
    batch_size: int = 64
    learning_rate: float = 0.05
    ewc_lambda: float = 10.0
    tau_values: tuple[float, ...] = (1e-3, 1e-2, 1e-1)


class SmallMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x):
        return self.net(x)


def make_domain_shift(seed: int, n_train: int, n_test: int):
    generator = torch.Generator().manual_seed(seed)

    def sample(n_samples: int, angle: float):
        x = torch.randn(n_samples, 2, generator=generator)
        direction = torch.tensor([math.cos(angle), math.sin(angle)])
        y = ((x @ direction) > 0).long()
        return x, y

    task_a_train = sample(n_train, 0.0)
    task_b_train = sample(n_train, 1.2)
    task_a_test = sample(n_test, 0.0)
    task_b_test = sample(n_test, 1.2)
    return task_a_train, task_b_train, task_a_test, task_b_test


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    pred = model(x).argmax(dim=1)
    return float((pred == y).float().mean().item())


def run_one(
    *,
    config: SyntheticRunConfig,
    method: str,
    tau: float | None = None,
) -> dict:
    canonical_method = "sequential" if method in SEQUENTIAL_ALIASES else method
    torch.manual_seed(config.seed)
    task_a_train, task_b_train, task_a_test, task_b_test = make_domain_shift(
        config.seed, config.n_train, config.n_test
    )
    benchmark = tensors_benchmark(
        train_tensors=[task_a_train, task_b_train],
        test_tensors=[task_a_test, task_b_test],
        task_labels=[0, 0],
        complete_test_set_only=False,
    )

    model = SmallMLP(config.hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    plugins = []
    if canonical_method != "sequential":
        plugins.append(
            IEWCPlugin(
                ewc_lambda=config.ewc_lambda,
                importance_kind=canonical_method,
                tau=0.0 if tau is None else tau,
            )
        )

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
        train_mb_size=config.batch_size,
        train_epochs=config.train_epochs,
        eval_mb_size=config.batch_size,
        device=torch.device("cpu"),
        plugins=plugins,
        evaluator=evaluator,
    )

    task_a_acc_after_task_a = None
    for exp_idx, experience in enumerate(benchmark.train_stream):
        strategy.train(experience, num_workers=0)
        if exp_idx == 0:
            task_a_acc_after_task_a = accuracy(model, *task_a_test)

    task_a_acc_after_task_b = accuracy(model, *task_a_test)
    task_b_acc_after_task_b = accuracy(model, *task_b_test)
    result = {
        "method": canonical_method,
        "tau": tau,
        "task_a_accuracy_after_task_a": task_a_acc_after_task_a,
        "task_a_accuracy_after_task_b": task_a_acc_after_task_b,
        "task_b_accuracy_after_task_b": task_b_acc_after_task_b,
        "task_a_forgetting": task_a_acc_after_task_a - task_a_acc_after_task_b,
    }
    if plugins and plugins[0].last_importance_result is not None:
        loss_scales = plugins[0].last_importance_result.loss_scales
        result["last_loss_scale_mean"] = float(loss_scales.mean().item())
        result["last_loss_scale_median"] = float(loss_scales.median().item())
    return result


def run_synthetic_classification_suite(
    *,
    config: SyntheticRunConfig,
    methods: Sequence[str] = ("sequential", "ef", "ewc_dr", "ief_diag"),
) -> list[dict]:
    results = []
    for method in methods:
        if method == "ief_diag":
            tau_results = [
                run_one(config=config, method=method, tau=tau)
                for tau in config.tau_values
            ]
            best = min(
                tau_results,
                key=lambda item: (
                    item["task_a_forgetting"],
                    -item["task_b_accuracy_after_task_b"],
                ),
            )
            combined = dict(best)
            combined["tau_results"] = tau_results
            results.append(combined)
        else:
            results.append(run_one(config=config, method=method))
    return results
