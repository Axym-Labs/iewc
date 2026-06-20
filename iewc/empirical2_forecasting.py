from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import csv
import math
import urllib.request
import zipfile

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .diagonal_regularization import (
    DiagonalImportance,
    compute_diagonal_importance,
    diagonal_ewc_penalties,
)


ForecastDataset = Literal["m4", "ett", "long_horizon"]
ForecastMethod = Literal["sequential", "ef", "iewc", "iewc_gss"]
ForecastNormalization = Literal["series", "context", "task"]
ForecastModel = Literal["encoder", "patchtst"]


@dataclass(frozen=True)
class ForecastingConfig:
    dataset: ForecastDataset = "m4"
    data_root: str = "/home/davwis/main/data/m4/tsf"
    frequencies: tuple[str, ...] = ("hourly", "weekly", "daily")
    seed: int = 0
    context_length: int = 48
    horizon: int = 12
    max_series_per_task: int = 64
    windows_per_series: int = 4
    eval_windows_per_series: int = 1
    epochs_per_task: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    ewc_lambda: float = 10.0
    tau: float = 1e-2
    importance_samples: int = 128
    model_type: ForecastModel = "encoder"
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    patch_length: int = 16
    patch_stride: int = 8
    dropout: float = 0.0
    num_workers: int = 0
    device: str = "cuda"
    normalization: ForecastNormalization = "series"


class ForecastWindowDataset(Dataset):
    def __init__(
        self,
        series: list[torch.Tensor],
        *,
        context_length: int,
        horizon: int,
        windows_per_series: int,
        eval_windows_per_series: int,
        seed: int,
        train: bool,
        normalization: ForecastNormalization,
        task_loc: torch.Tensor | None = None,
        task_scale: torch.Tensor | None = None,
    ):
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        generator = torch.Generator().manual_seed(seed)
        needed = context_length + horizon
        if normalization == "task" and (task_loc is None or task_scale is None):
            values = torch.cat([item.float().reshape(-1) for item in series])
            task_loc = values.mean()
            task_scale = values.std().clamp_min(1e-3)
        for values in series:
            if values.numel() < needed + 1:
                continue
            series_loc = values.mean()
            series_scale = values.std().clamp_min(1e-3)
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
                max_start = values.numel() - needed
                if eval_windows_per_series <= 1:
                    starts = [max_start]
                else:
                    start_min = max(0, max_start // 2)
                    starts = (
                        torch.linspace(
                            start_min,
                            max_start,
                            steps=min(eval_windows_per_series, max_start - start_min + 1),
                        )
                        .round()
                        .long()
                        .unique()
                        .tolist()
                    )
            for start in starts:
                context = values[start : start + context_length].float()
                target = values[start + context_length : start + context_length + horizon].float()
                if normalization == "context":
                    loc = context.mean()
                    scale = context.std().clamp_min(1e-3)
                elif normalization == "task":
                    loc = task_loc
                    scale = task_scale
                else:
                    loc = series_loc
                    scale = series_scale
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


class PatchTSTForecaster(nn.Module):
    """Channel-independent PatchTST-style forecaster for univariate windows."""

    def __init__(
        self,
        *,
        context_length: int,
        horizon: int,
        patch_length: int,
        patch_stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        if patch_length > context_length:
            raise ValueError("patch_length must be <= context_length")
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        n_patches = 1 + (context_length - patch_length) // patch_stride
        self.patch_proj = nn.Linear(patch_length, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(n_patches * d_model),
            nn.Linear(n_patches * d_model, horizon),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        patches = context.unfold(dimension=1, size=self.patch_length, step=self.patch_stride)
        x = self.patch_proj(patches) + self.pos_embed[:, : patches.shape[1]]
        encoded = self.encoder(x)
        return self.head(encoded)


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


def _download_ett(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.csv"
    if path.exists():
        return path
    url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
    urllib.request.urlretrieve(url, path)
    return path


def _read_ett(path: Path) -> torch.Tensor:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        values = [float(row["OT"]) for row in reader if row.get("OT") not in {None, ""}]
    return torch.tensor(values, dtype=torch.float32)


def _download_long_horizon(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / "datasets.zip"
    if path.exists() and path.stat().st_size > 300_000_000:
        return path
    url = "https://nhits-experiments.s3.amazonaws.com/datasets.zip"
    urllib.request.urlretrieve(url, path)
    return path


def _read_long_horizon_group(root: Path, group: str, *, max_series: int) -> list[torch.Tensor]:
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    group_key = group.lower()
    cache_path = cache / f"{group_key}_m.pt"
    if cache_path.exists():
        series = torch.load(cache_path, weights_only=True)
    else:
        import pandas as pd

        zip_path = _download_long_horizon(root)
        candidates = {
            "ecl": "ECL/M/df_y.csv",
            "electricity": "ECL/M/df_y.csv",
            "traffic": "traffic/M/df_y.csv",
            "weather": "weather/M/df_y.csv",
            "exchange": "Exchange/M/df_y.csv",
            "ili": "ili/M/df_y.csv",
            "etth1": "ETTh1/df_y.csv",
            "etth2": "ETTh2/df_y.csv",
            "ettm1": "ETTm1/M/df_y.csv",
            "ettm2": "ETTm2/M/df_y.csv",
        }
        if group_key not in candidates:
            raise ValueError(f"Unknown long-horizon group: {group}")
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(candidates[group_key]) as handle:
                frame = pd.read_csv(handle, usecols=["unique_id", "y"])
        series = [
            torch.tensor(values["y"].to_numpy(dtype="float32"), dtype=torch.float32)
            for _, values in frame.groupby("unique_id", sort=False)
        ]
        torch.save(series, cache_path)
    if max_series > 0:
        return series[:max_series]
    return series


def _make_m4_tasks(config: ForecastingConfig):
    tasks = []
    root = Path(config.data_root)
    for task_id, frequency in enumerate(config.frequencies):
        path = root / f"m4_{frequency}_dataset.tsf"
        series = _read_tsf(path, max_series=config.max_series_per_task)
        task_values = torch.cat([values.float().reshape(-1) for values in series])
        task_loc = task_values.mean()
        task_scale = task_values.std().clamp_min(1e-3)
        train = ForecastWindowDataset(
            series,
            context_length=config.context_length,
            horizon=config.horizon,
            windows_per_series=config.windows_per_series,
            eval_windows_per_series=config.eval_windows_per_series,
            seed=config.seed + task_id * 13,
            train=True,
            normalization=config.normalization,
            task_loc=task_loc,
            task_scale=task_scale,
        )
        test = ForecastWindowDataset(
            series,
            context_length=config.context_length,
            horizon=config.horizon,
            windows_per_series=1,
            eval_windows_per_series=config.eval_windows_per_series,
            seed=config.seed,
            train=False,
            normalization=config.normalization,
            task_loc=task_loc,
            task_scale=task_scale,
        )
        tasks.append((frequency, train, test))
    return tasks


def _make_ett_tasks(config: ForecastingConfig):
    tasks = []
    root = Path(config.data_root)
    for task_id, name in enumerate(config.frequencies):
        values = _read_ett(_download_ett(root, name))
        train_end = int(values.numel() * 0.7)
        test_start = int(values.numel() * 0.8)
        task_loc = values[:train_end].mean()
        task_scale = values[:train_end].std().clamp_min(1e-3)
        train = ForecastWindowDataset(
            [values[:train_end]],
            context_length=config.context_length,
            horizon=config.horizon,
            windows_per_series=config.windows_per_series,
            eval_windows_per_series=config.eval_windows_per_series,
            seed=config.seed + task_id * 13,
            train=True,
            normalization=config.normalization,
            task_loc=task_loc,
            task_scale=task_scale,
        )
        test = ForecastWindowDataset(
            [values[test_start:]],
            context_length=config.context_length,
            horizon=config.horizon,
            windows_per_series=1,
            eval_windows_per_series=config.eval_windows_per_series,
            seed=config.seed,
            train=False,
            normalization=config.normalization,
            task_loc=task_loc,
            task_scale=task_scale,
        )
        tasks.append((name, train, test))
    return tasks


def _make_long_horizon_tasks(config: ForecastingConfig):
    tasks = []
    root = Path(config.data_root)
    for task_id, name in enumerate(config.frequencies):
        series = _read_long_horizon_group(root, name, max_series=config.max_series_per_task)
        train_series = []
        test_series = []
        for values in series:
            train_end = int(values.numel() * 0.7)
            test_start = int(values.numel() * 0.8)
            train_series.append(values[:train_end])
            test_series.append(values[test_start:])
        task_values = torch.cat([values.float().reshape(-1) for values in train_series])
        task_loc = task_values.mean()
        task_scale = task_values.std().clamp_min(1e-3)
        train = ForecastWindowDataset(
            train_series,
            context_length=config.context_length,
            horizon=config.horizon,
            windows_per_series=config.windows_per_series,
            eval_windows_per_series=config.eval_windows_per_series,
            seed=config.seed + task_id * 13,
            train=True,
            normalization=config.normalization,
            task_loc=task_loc,
            task_scale=task_scale,
        )
        test = ForecastWindowDataset(
            test_series,
            context_length=config.context_length,
            horizon=config.horizon,
            windows_per_series=1,
            eval_windows_per_series=config.eval_windows_per_series,
            seed=config.seed,
            train=False,
            normalization=config.normalization,
            task_loc=task_loc,
            task_scale=task_scale,
        )
        tasks.append((name, train, test))
    return tasks


def _make_tasks(config: ForecastingConfig):
    if config.dataset == "m4":
        return _make_m4_tasks(config)
    if config.dataset == "ett":
        return _make_ett_tasks(config)
    if config.dataset == "long_horizon":
        return _make_long_horizon_tasks(config)
    raise ValueError(f"Unknown forecasting dataset: {config.dataset}")


def _make_model(config: ForecastingConfig) -> nn.Module:
    if config.model_type == "encoder":
        return TransformerForecaster(
            context_length=config.context_length,
            horizon=config.horizon,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
    if config.model_type == "patchtst":
        return PatchTSTForecaster(
            context_length=config.context_length,
            horizon=config.horizon,
            patch_length=config.patch_length,
            patch_stride=config.patch_stride,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
    raise ValueError(f"Unknown forecasting model_type: {config.model_type}")


def _loader(dataset: Dataset, *, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)


def _loss_output_fn(device: torch.device):
    def fn(model: nn.Module, batch, reverse_output: bool):
        del reverse_output
        context, target = batch[0].to(device), batch[1].to(device)
        output = model(context)
        return nn.functional.mse_loss(output, target), output

    return fn


def _importance_kind_and_weight(method: ForecastMethod) -> tuple[str | None, str]:
    if method == "sequential":
        return None, "uniform"
    if method == "ef":
        return "ef", "uniform"
    if method == "iewc":
        return "iewc", "uniform"
    if method == "iewc_gss":
        return "iewc", "gss_residual"
    raise ValueError(f"Unknown forecasting method: {method}")


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
    model = _make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    importances: list[DiagonalImportance] = []
    importance_kind, sample_weighting = _importance_kind_and_weight(method)
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

        if importance_kind is not None and task_id < len(tasks) - 1:
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
                kind=importance_kind,
                tau=config.tau,
                sample_weighting=sample_weighting,
                max_samples=config.importance_samples,
            )
            importances.append(importance)
            importance_summaries.append(
                {
                    "task_id": task_id,
                    "frequency": frequency,
                    "sample_count": importance.sample_count,
                    "mean_loss_scale": float(importance.loss_scales.float().mean().item()),
                    "mean_sample_weight": float(importance.sample_weights.float().mean().item()),
                    "max_sample_weight": float(importance.sample_weights.float().max().item()),
                    "mean_summand_trace": float(importance.stored_summand_traces.float().mean().item()),
                }
            )

    final_mses = mse_matrix[-1]
    forgetting = []
    for task_id in range(len(tasks) - 1):
        best = min(row[task_id] for row in mse_matrix)
        forgetting.append(final_mses[task_id] - best)
    return {
        "experiment": "empirical2_forecasting_cl",
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
