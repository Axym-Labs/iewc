from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .importance import ImportanceEstimator, OutputMetricKind
from .output_metrics import (
    euclidean_squared_distance,
    wasserstein_1d_cdf_squared_distance,
)


MethodName = Literal["sequential", "naive", "ief_euclidean", "ief_wasserstein"]
SEQUENTIAL_ALIASES = {"sequential", "naive"}


@dataclass(frozen=True)
class SyntheticOrderedMetricConfig:
    seed: int = 0
    n_train: int = 256
    n_test: int = 512
    n_classes: int = 5
    train_epochs: int = 60
    hidden_size: int = 32
    batch_size: int = 64
    learning_rate: float = 0.03
    ewc_lambda: float = 1.0
    euclidean_ewc_lambda: float | None = None
    wasserstein_ewc_lambda: float | None = None
    tau: float = 1e-3


class OrdinalMLP(nn.Module):
    def __init__(self, hidden_size: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _labels_from_x(x: torch.Tensor, n_classes: int, shift: int):
    scaled = ((x.squeeze(-1) + 1.0) / 2.0 * n_classes).floor().long()
    scaled = scaled.clamp(0, n_classes - 1)
    return (scaled + shift).clamp(0, n_classes - 1)


def _make_task(
    *,
    n_samples: int,
    n_classes: int,
    shift: int,
    generator: torch.Generator,
):
    x = torch.empty(n_samples, 1).uniform_(-1.0, 1.0, generator=generator)
    x = x + 0.02 * torch.randn(x.shape, generator=generator)
    x = x.clamp(-1.0, 1.0)
    y = _labels_from_x(x, n_classes=n_classes, shift=shift)
    task = torch.zeros(n_samples, dtype=torch.long)
    return TensorDataset(x, y, task)


def make_ordered_shift(config: SyntheticOrderedMetricConfig):
    generator = torch.Generator().manual_seed(config.seed)
    task_a_train = _make_task(
        n_samples=config.n_train,
        n_classes=config.n_classes,
        shift=0,
        generator=generator,
    )
    task_b_train = _make_task(
        n_samples=config.n_train,
        n_classes=config.n_classes,
        shift=1,
        generator=generator,
    )
    task_a_test = _make_task(
        n_samples=config.n_test,
        n_classes=config.n_classes,
        shift=0,
        generator=generator,
    )
    task_b_test = _make_task(
        n_samples=config.n_test,
        n_classes=config.n_classes,
        shift=1,
        generator=generator,
    )
    return task_a_train, task_b_train, task_a_test, task_b_test


def compute_penalty(model: nn.Module, saved_params, importances):
    penalty = torch.zeros(())
    for name, param in model.named_parameters():
        if name not in saved_params:
            continue
        delta = param - saved_params[name]
        penalty = penalty + (importances[name].data * delta.pow(2)).sum()
    return penalty


def train_task(
    *,
    model: nn.Module,
    dataset: TensorDataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    criterion: nn.Module,
    ewc_lambda: float = 0.0,
    saved_params=None,
    importances=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for x, y, _ in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if importances is not None and saved_params is not None:
                loss = loss + ewc_lambda * compute_penalty(
                    model, saved_params, importances
                )
            loss.backward()
            optimizer.step()
    return optimizer


@torch.no_grad()
def accuracy(model: nn.Module, dataset: TensorDataset, batch_size: int) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    for x, y, _ in loader:
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return correct / float(total)


@torch.no_grad()
def probabilities(model: nn.Module, dataset: TensorDataset, batch_size: int):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = []
    for x, _, _ in loader:
        outputs.append(torch.softmax(model(x), dim=-1))
    return torch.cat(outputs, dim=0)


def run_one(
    *,
    config: SyntheticOrderedMetricConfig,
    method: MethodName,
) -> dict:
    canonical_method = "sequential" if method in SEQUENTIAL_ALIASES else method
    torch.manual_seed(config.seed)
    task_a_train, task_b_train, task_a_test, task_b_test = make_ordered_shift(config)
    criterion = nn.CrossEntropyLoss()
    model = OrdinalMLP(config.hidden_size, config.n_classes)
    optimizer = train_task(
        model=model,
        dataset=task_a_train,
        epochs=config.train_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        criterion=criterion,
    )
    teacher_old_probs = probabilities(
        model, task_a_test, batch_size=config.batch_size
    )
    task_a_accuracy_after_task_a = accuracy(
        model, task_a_test, batch_size=config.batch_size
    )

    output_metric: OutputMetricKind | None = None
    importances = None
    saved_params = None
    loss_scale_mean = None
    loss_scale_median = None
    if canonical_method != "sequential":
        output_metric = (
            "wasserstein_1d_cdf"
            if canonical_method == "ief_wasserstein"
            else "euclidean"
        )
        ewc_lambda = (
            config.wasserstein_ewc_lambda
            if canonical_method == "ief_wasserstein"
            else config.euclidean_ewc_lambda
        )
        if ewc_lambda is None:
            ewc_lambda = config.ewc_lambda
        result = ImportanceEstimator(
            kind="ief_diag",
            tau=config.tau,
            output_metric=output_metric,
        ).compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=task_a_train,
            device=torch.device("cpu"),
            batch_size=config.batch_size,
        )
        importances = result.importances
        saved_params = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        loss_scale_mean = float(result.loss_scales.mean().item())
        loss_scale_median = float(result.loss_scales.median().item())

    train_task(
        model=model,
        dataset=task_b_train,
        epochs=config.train_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        criterion=criterion,
        ewc_lambda=0.0 if canonical_method == "sequential" else float(ewc_lambda),
        saved_params=saved_params,
        importances=importances,
    )

    old_probs_after_b = probabilities(model, task_a_test, batch_size=config.batch_size)
    euclidean_drift = euclidean_squared_distance(
        teacher_old_probs, old_probs_after_b
    ).mean()
    wasserstein_drift = wasserstein_1d_cdf_squared_distance(
        teacher_old_probs, old_probs_after_b
    ).mean()

    output = {
        "method": canonical_method,
        "output_metric": output_metric,
        "ewc_lambda": None if canonical_method == "sequential" else float(ewc_lambda),
        "tau": config.tau if canonical_method != "sequential" else None,
        "task_a_accuracy_after_task_a": task_a_accuracy_after_task_a,
        "task_a_accuracy_after_task_b": accuracy(
            model, task_a_test, batch_size=config.batch_size
        ),
        "task_b_accuracy_after_task_b": accuracy(
            model, task_b_test, batch_size=config.batch_size
        ),
        "old_output_euclidean_drift": float(euclidean_drift.item()),
        "old_output_wasserstein_drift": float(wasserstein_drift.item()),
    }
    output["task_a_forgetting"] = (
        output["task_a_accuracy_after_task_a"]
        - output["task_a_accuracy_after_task_b"]
    )
    if loss_scale_mean is not None:
        output["old_task_loss_scale_mean"] = loss_scale_mean
        output["old_task_loss_scale_median"] = loss_scale_median
    return output


def run_ordered_metric_suite(
    *, config: SyntheticOrderedMetricConfig
) -> list[dict]:
    return [
        run_one(config=config, method="sequential"),
        run_one(config=config, method="ief_euclidean"),
        run_one(config=config, method="ief_wasserstein"),
    ]
