from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .importance import ImportanceEstimator


SEQUENTIAL_ALIASES = {"sequential", "naive"}


@dataclass(frozen=True)
class SyntheticRegressionConfig:
    seed: int = 0
    n_train: int = 256
    n_test: int = 512
    n_tasks: int = 2
    train_epochs: int = 100
    hidden_size: int = 32
    batch_size: int = 64
    learning_rate: float = 0.01
    ewc_lambda: float = 1.0
    tau_values: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    device: str = "cpu"


class RegressionMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_regression_shift(seed: int, n_train: int, n_test: int):
    train_tasks, test_tasks = make_regression_stream(
        seed=seed,
        n_train=n_train,
        n_test=n_test,
        n_tasks=2,
    )
    return train_tasks[0], train_tasks[1], test_tasks[0], test_tasks[1]


def make_regression_stream(seed: int, n_train: int, n_test: int, n_tasks: int):
    if n_tasks < 2:
        raise ValueError("n_tasks must be at least 2")
    generator = torch.Generator().manual_seed(seed)

    def sample(n_samples: int, phase: float, amplitude: float):
        x = torch.empty(n_samples, 1).uniform_(-2.0, 2.0, generator=generator)
        y = amplitude * torch.sin(2.5 * x + phase)
        return x, y

    specs = []
    for task_id in range(n_tasks):
        phase = float(task_id)
        amplitude = 1.0 + 0.15 * math.sin(1.7 * task_id)
        specs.append((phase, amplitude))
    train_tasks = [sample(n_train, phase=phase, amplitude=amp) for phase, amp in specs]
    test_tasks = [sample(n_test, phase=phase, amplitude=amp) for phase, amp in specs]
    return train_tasks, test_tasks


def make_dataset(x: torch.Tensor, y: torch.Tensor):
    task = torch.zeros(len(x), dtype=torch.long)
    return TensorDataset(x, y, task)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(device)


@torch.no_grad()
def mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    pred = model(x.to(device))
    return float(nn.functional.mse_loss(pred, y.to(device)).item())


def compute_penalty(model: nn.Module, saved_params, importances):
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name not in saved_params:
            continue
        delta = param - saved_params[name]
        penalty = penalty + (importances[name].data * delta.pow(2)).sum()
    return penalty


def compute_penalty_states(model: nn.Module, penalty_states):
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for saved_params, importances in penalty_states:
        penalty = penalty + compute_penalty(model, saved_params, importances)
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
    penalty_states=None,
    device: torch.device | None = None,
):
    if device is None:
        device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if penalty_states:
                loss = loss + ewc_lambda * compute_penalty_states(
                    model, penalty_states
                )
            if importances is not None and saved_params is not None:
                loss = loss + ewc_lambda * compute_penalty(
                    model, saved_params, importances
                )
            loss.backward()
            optimizer.step()
    return optimizer


def run_one(
    *,
    config: SyntheticRegressionConfig,
    method: str,
    tau: float | None = None,
) -> dict:
    canonical_method = "sequential" if method in SEQUENTIAL_ALIASES else method
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    train_tasks_raw, test_tasks_raw = make_regression_stream(
        config.seed, config.n_train, config.n_test, config.n_tasks
    )
    train_tasks = [make_dataset(*task) for task in train_tasks_raw]
    criterion = nn.MSELoss()
    model = RegressionMLP(config.hidden_size).to(device)
    penalty_states = []
    mse_after_learning = []
    loss_scale_values = []
    ef_summand_trace_values = []
    stored_summand_trace_values = []
    optimizer = None
    for task_id, train_task_dataset in enumerate(train_tasks):
        optimizer = train_task(
            model=model,
            dataset=train_task_dataset,
            epochs=config.train_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            criterion=criterion,
            ewc_lambda=config.ewc_lambda,
            penalty_states=penalty_states,
            device=device,
        )
        mse_after_learning.append(mse(model, *test_tasks_raw[task_id], device))
        if canonical_method != "sequential" and task_id < config.n_tasks - 1:
            result = ImportanceEstimator(
                kind=canonical_method, tau=0.0 if tau is None else tau
            ).compute(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                dataset=train_task_dataset,
                device=device,
                batch_size=config.batch_size,
            )
            saved_params = {
                name: param.detach().clone() for name, param in model.named_parameters()
            }
            penalty_states.append((saved_params, result.importances))
            loss_scale_values.extend(float(value) for value in result.loss_scales.tolist())
            ef_summand_trace_values.extend(
                float(value) for value in result.ef_summand_traces.tolist()
            )
            stored_summand_trace_values.extend(
                float(value) for value in result.stored_summand_traces.tolist()
            )

    final_mses = [mse(model, *task, device) for task in test_tasks_raw]
    old_final_mses = final_mses[:-1]
    old_forgetting = [
        final_mses[idx] - mse_after_learning[idx] for idx in range(config.n_tasks - 1)
    ]
    task_a_mse_after_task_a = mse_after_learning[0]
    task_a_mse_after_task_b = final_mses[0]
    task_b_mse_after_task_b = final_mses[-1]
    output = {
        "method": canonical_method,
        "tau": tau,
        "n_tasks": config.n_tasks,
        "mse_after_learning": mse_after_learning,
        "final_task_mses": final_mses,
        "final_avg_mse": sum(final_mses) / len(final_mses),
        "old_avg_mse_after_final": sum(old_final_mses) / len(old_final_mses),
        "new_task_mse_after_final": final_mses[-1],
        "avg_forgetting_mse": sum(old_forgetting) / len(old_forgetting),
        "task_a_mse_after_task_a": task_a_mse_after_task_a,
        "task_a_mse_after_task_b": task_a_mse_after_task_b,
        "task_b_mse_after_task_b": task_b_mse_after_task_b,
        "task_a_mse_increase": task_a_mse_after_task_b - task_a_mse_after_task_a,
    }
    if loss_scale_values:
        scales = torch.tensor(loss_scale_values)
        output["old_task_loss_scale_mean"] = float(scales.mean().item())
        output["old_task_loss_scale_median"] = float(scales.median().item())
        output["old_task_loss_scales"] = loss_scale_values
        output["old_task_ef_summand_traces"] = ef_summand_trace_values
        output["old_task_stored_summand_traces"] = stored_summand_trace_values
    return output


def run_synthetic_regression_suite(
    *,
    config: SyntheticRegressionConfig,
    methods: Sequence[str] = ("sequential", "ef", "ief_diag"),
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
                    item["avg_forgetting_mse"],
                    item["new_task_mse_after_final"],
                ),
            )
            combined = dict(best)
            combined["tau_results"] = tau_results
            results.append(combined)
        else:
            results.append(run_one(config=config, method=method))
    return results
