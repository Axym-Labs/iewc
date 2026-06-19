from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .diagonal_regularization import (
    DiagonalImportance,
    compute_diagonal_importance,
    diagonal_ewc_penalties,
)


ForecastMethod = Literal["sequential", "ef", "iewc"]


@dataclass(frozen=True)
class ForecastingConfig:
    data_root: str = "/home/davwis/main/data/m4/tsf"
    frequencies: tuple[str, ...] = ("hourly", "weekly", "daily")
    seed: int = 0
    context_length: int = 48
    horizon: int = 12
    max_series_per_task: int = 64
    windows_per_series: int = 4
    epochs_per_task: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    ewc_lambda: float = 10.0
    tau: float = 1e-2
    importance_samples: int = 128
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0
    num_workers: int = 0
    device: str = "cuda"


class ForecastWindowDataset(Dataset):
    def __init__(
        self,
        series: list[torch.Tensor],
        *,
        context_length: int,
        horizon: int,
        windows_per_series: int,
        seed: int,
        train: bool,
    ):
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        generator = torch.Generator().manual_seed(seed)
        needed = context_length + horizon
        for values in series:
            if values.numel() < needed + 1:
                continue
            if train:
                max_start = values.numel() - needed - horizon
                if max_start <= 0:
                    max_start = values.numel() - needed
                starts = torch.randint(
                    low=0,
                    high=max(1, max_start),
                    size=(windows_per_series,),
                    generator=generator,
                ).tolist()
            else:
                starts = [values.numel() - needed]
            for start in starts:
                context = values[start : start + context_length].float()
                target = values[start + context_length : start + context_length + horizon].float()
                loc = context.mean()
                scale = context.std().clamp_min(1e-4)
                self.samples.append(((context - loc) / scale, (target - loc) / scale, loc, scale))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        *,
        context_length: int,
        horizon: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, context_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(context.unsqueeze(-1)) + self.pos_embed
        encoded = self.encoder(x)
        return self.head(encoded[:, -1])


def _read_tsf(path: Path, *, max_series: int) -> list[torch.Tensor]:
    series = []
    in_data = False
    for line in path.read_text(encoding="latin-1").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower() == "@data":
            in_data = True
            continue
        if not in_data or line.startswith("#") or line.startswith("@"):
            continue
        values_text = line.rsplit(":", 1)[-1]
        values = [float(value) for value in values_text.split(",") if value != "?"]
        if values:
            series.append(torch.tensor(values, dtype=torch.float32))
        if 0 < max_series <= len(series):
            break
    return series


def _make_tasks(config: ForecastingConfig):
    tasks = []
    root = Path(config.data_root)
    for task_id, frequency in enumerate(config.frequencies):
        path = root / f"m4_{frequency}_dataset.tsf"
        series = _read_tsf(path, max_series=config.max_series_per_task)
        train = ForecastWindowDataset(
            series,
            context_length=config.context_length,
            horizon=config.horizon,
            windows_per_series=config.windows_per_series,
            seed=config.seed + task_id * 13,
            train=True,
        )
        test = ForecastWindowDataset(
            series,
            context_length=config.context_length,
            horizon=config.horizon,
            windows_per_series=1,
            seed=config.seed,
            train=False,
        )
        tasks.append((frequency, train, test))
    return tasks


def _loader(dataset: Dataset, *, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)


def _loss_output_fn(device: torch.device):
    def fn(model: nn.Module, batch, reverse_output: bool):
        del reverse_output
        context, target = batch[0].to(device), batch[1].to(device)
        output = model(context)
        return nn.functional.mse_loss(output, target), output

    return fn


def train_one_task(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    ewc_lambda: float,
    importances: list[DiagonalImportance],
) -> float:
    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        for context, target, _, _ in loader:
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(context)
            loss = nn.functional.mse_loss(output, target)
            if importances:
                loss = loss + float(ewc_lambda) * diagonal_ewc_penalties(model, importances, device)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
    return last_loss


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    se = 0.0
    ae = 0.0
    count = 0
    rollouts = []
    for context, target, loc, scale in loader:
        context = context.to(device)
        target = target.to(device)
        output = model(context)
        se += float((output - target).pow(2).sum().detach().cpu())
        ae += float((output - target).abs().sum().detach().cpu())
        count += int(target.numel())
        if len(rollouts) < 3:
            pred_raw = output.detach().cpu()[0] * scale[0] + loc[0]
            target_raw = target.detach().cpu()[0] * scale[0] + loc[0]
            context_raw = context.detach().cpu()[0] * scale[0] + loc[0]
            rollouts.append(
                {
                    "context": [float(v) for v in context_raw],
                    "target": [float(v) for v in target_raw],
                    "prediction": [float(v) for v in pred_raw],
                }
            )
    return {"mse": se / max(1, count), "mae": ae / max(1, count), "rollouts": rollouts}


def run_forecasting_cl(config: ForecastingConfig, method: ForecastMethod) -> dict:
    torch.manual_seed(config.seed)
    device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    tasks = _make_tasks(config)
    model = TransformerForecaster(
        context_length=config.context_length,
        horizon=config.horizon,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    importances: list[DiagonalImportance] = []
    mse_matrix: list[list[float]] = []
    mae_matrix: list[list[float]] = []
    train_losses = []
    rollout_records = {}
    importance_summaries = []

    for task_id, (frequency, train_dataset, _) in enumerate(tasks):
        train_loss = train_one_task(
            model=model,
            loader=_loader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                workers=config.num_workers,
            ),
            optimizer=optimizer,
            device=device,
            epochs=config.epochs_per_task,
            ewc_lambda=config.ewc_lambda,
            importances=importances,
        )
        train_losses.append(train_loss)

        mse_row = []
        mae_row = []
        for eval_frequency, _, test_dataset in tasks:
            metrics = evaluate(
                model,
                _loader(
                    test_dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    workers=config.num_workers,
                ),
                device,
            )
            mse_row.append(metrics["mse"])
            mae_row.append(metrics["mae"])
            if task_id == len(tasks) - 1:
                rollout_records[eval_frequency] = metrics["rollouts"]
        mse_matrix.append(mse_row)
        mae_matrix.append(mae_row)

        if method != "sequential" and task_id < len(tasks) - 1:
            importance = compute_diagonal_importance(
                model=model,
                dataloader=_loader(
                    train_dataset,
                    batch_size=1,
                    shuffle=False,
                    workers=config.num_workers,
                ),
                loss_output_fn=_loss_output_fn(device),
                device=device,
                kind="ef" if method == "ef" else "iewc",
                tau=config.tau,
                max_samples=config.importance_samples,
            )
            importances.append(importance)
            importance_summaries.append(
                {
                    "task_id": task_id,
                    "frequency": frequency,
                    "sample_count": importance.sample_count,
                    "mean_loss_scale": float(importance.loss_scales.float().mean().item()),
                    "mean_summand_trace": float(importance.stored_summand_traces.float().mean().item()),
                }
            )

    final_mses = mse_matrix[-1]
    forgetting = []
    for task_id in range(len(tasks) - 1):
        best = min(row[task_id] for row in mse_matrix)
        forgetting.append(final_mses[task_id] - best)
    return {
        "experiment": "empirical2_m4_forecasting_cl",
        "config": asdict(config),
        "method": method,
        "frequencies": list(config.frequencies),
        "n_parameters": int(sum(param.numel() for param in model.parameters())),
        "mse_matrix": mse_matrix,
        "mae_matrix": mae_matrix,
        "final_task_mses": final_mses,
        "final_avg_mse": float(sum(final_mses) / len(final_mses)),
        "avg_forgetting_mse": float(sum(forgetting) / len(forgetting)) if forgetting else 0.0,
        "train_losses": train_losses,
        "importance_summaries": importance_summaries,
        "rollouts": rollout_records,
    }
