from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .importance import ImportanceEstimator


SEQUENTIAL_ALIASES = {"sequential", "naive"}


@dataclass(frozen=True)
class SyntheticSegmentationConfig:
    seed: int = 0
    image_size: int = 16
    n_train: int = 96
    n_test: int = 128
    train_epochs: int = 50
    hidden_channels: int = 12
    batch_size: int = 16
    learning_rate: float = 0.01
    ewc_lambda: float = 2.0
    tau_values: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    task_b_foreground_value: float = -1.0
    task_b_background_value: float = 1.0


class SmallSegNet(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def _shape_mask(kind: int, image_size: int, generator: torch.Generator):
    mask = torch.zeros(image_size, image_size, dtype=torch.long)
    center = int(torch.randint(4, image_size - 4, (1,), generator=generator).item())
    width = 2
    if kind == 0:
        mask[:, center - width : center + width] = 1
    elif kind == 1:
        mask[center - width : center + width, :] = 1
    elif kind == 2:
        offset = int(torch.randint(-2, 3, (1,), generator=generator).item())
        for row in range(image_size):
            col = row + offset
            for delta in (-1, 0, 1):
                c = col + delta
                if 0 <= c < image_size:
                    mask[row, c] = 1
    elif kind == 3:
        lo = max(1, center - 3)
        hi = min(image_size - 1, center + 3)
        mask[lo:hi, lo:hi] = 1
        mask[lo + 2 : hi - 2, lo + 2 : hi - 2] = 0
    else:
        raise ValueError(f"Unknown segmentation shape kind: {kind}")
    return mask


def _make_task(
    *,
    kinds: tuple[int, int],
    n_samples: int,
    image_size: int,
    foreground_value: float,
    background_value: float,
    generator: torch.Generator,
):
    images = []
    masks = []
    for idx in range(n_samples):
        mask = _shape_mask(kinds[idx % len(kinds)], image_size, generator)
        image = torch.where(
            mask.bool(),
            torch.full_like(mask, foreground_value, dtype=torch.float32),
            torch.full_like(mask, background_value, dtype=torch.float32),
        )
        noise = 0.25 * torch.randn(
            image.shape, generator=generator, dtype=image.dtype
        )
        images.append((image + noise).clamp(-1.5, 1.5).unsqueeze(0))
        masks.append(mask)

    task_labels = torch.zeros(n_samples, dtype=torch.long)
    return TensorDataset(torch.stack(images), torch.stack(masks), task_labels)


def make_segmentation_stream(config: SyntheticSegmentationConfig):
    generator = torch.Generator().manual_seed(config.seed)
    task_a_train = _make_task(
        kinds=(0, 1),
        n_samples=config.n_train,
        image_size=config.image_size,
        foreground_value=1.0,
        background_value=-1.0,
        generator=generator,
    )
    task_b_train = _make_task(
        kinds=(2, 3),
        n_samples=config.n_train,
        image_size=config.image_size,
        foreground_value=config.task_b_foreground_value,
        background_value=config.task_b_background_value,
        generator=generator,
    )
    task_a_test = _make_task(
        kinds=(0, 1),
        n_samples=config.n_test,
        image_size=config.image_size,
        foreground_value=1.0,
        background_value=-1.0,
        generator=generator,
    )
    task_b_test = _make_task(
        kinds=(2, 3),
        n_samples=config.n_test,
        image_size=config.image_size,
        foreground_value=config.task_b_foreground_value,
        background_value=config.task_b_background_value,
        generator=generator,
    )
    return task_a_train, task_b_train, task_a_test, task_b_test


@torch.no_grad()
def foreground_iou(model: nn.Module, dataset: TensorDataset, batch_size: int) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    intersections = 0.0
    unions = 0.0
    for images, masks, _ in loader:
        pred = model(images).argmax(dim=1)
        pred_fg = pred == 1
        true_fg = masks == 1
        intersections += float((pred_fg & true_fg).sum().item())
        unions += float((pred_fg | true_fg).sum().item())
    if unions == 0:
        raise ValueError("Cannot compute foreground IoU without foreground pixels")
    return intersections / unions


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
        for images, masks, _ in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            if importances is not None and saved_params is not None:
                loss = loss + ewc_lambda * compute_penalty(
                    model, saved_params, importances
                )
            loss.backward()
            optimizer.step()
    return optimizer


def run_one(
    *,
    config: SyntheticSegmentationConfig,
    method: str,
    tau: float | None = None,
) -> dict:
    canonical_method = "sequential" if method in SEQUENTIAL_ALIASES else method
    torch.manual_seed(config.seed)
    task_a_train, task_b_train, task_a_test, task_b_test = make_segmentation_stream(
        config
    )
    criterion = nn.CrossEntropyLoss()
    model = SmallSegNet(config.hidden_channels)
    optimizer = train_task(
        model=model,
        dataset=task_a_train,
        epochs=config.train_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        criterion=criterion,
    )
    task_a_iou_after_task_a = foreground_iou(
        model, task_a_test, batch_size=config.batch_size
    )

    importances = None
    saved_params = None
    loss_scale_mean = None
    loss_scale_median = None
    loss_scale_values = None
    ef_summand_trace_values = None
    stored_summand_trace_values = None
    if canonical_method != "sequential":
        result = ImportanceEstimator(
            kind=canonical_method, tau=0.0 if tau is None else tau
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
        loss_scale_values = [float(value) for value in result.loss_scales.tolist()]
        ef_summand_trace_values = [
            float(value) for value in result.ef_summand_traces.tolist()
        ]
        stored_summand_trace_values = [
            float(value) for value in result.stored_summand_traces.tolist()
        ]

    train_task(
        model=model,
        dataset=task_b_train,
        epochs=config.train_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        criterion=criterion,
        ewc_lambda=config.ewc_lambda,
        saved_params=saved_params,
        importances=importances,
    )

    task_a_iou_after_task_b = foreground_iou(
        model, task_a_test, batch_size=config.batch_size
    )
    task_b_iou_after_task_b = foreground_iou(
        model, task_b_test, batch_size=config.batch_size
    )
    output = {
        "method": canonical_method,
        "tau": tau,
        "task_a_iou_after_task_a": task_a_iou_after_task_a,
        "task_a_iou_after_task_b": task_a_iou_after_task_b,
        "task_b_iou_after_task_b": task_b_iou_after_task_b,
        "task_a_iou_drop": task_a_iou_after_task_a - task_a_iou_after_task_b,
    }
    if loss_scale_mean is not None:
        output["old_task_loss_scale_mean"] = loss_scale_mean
        output["old_task_loss_scale_median"] = loss_scale_median
        output["old_task_loss_scales"] = loss_scale_values
        output["old_task_ef_summand_traces"] = ef_summand_trace_values
        output["old_task_stored_summand_traces"] = stored_summand_trace_values
    return output


def run_synthetic_segmentation_suite(
    *,
    config: SyntheticSegmentationConfig,
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
                    item["task_a_iou_drop"],
                    -item["task_b_iou_after_task_b"],
                ),
            )
            combined = dict(best)
            combined["tau_results"] = tau_results
            results.append(combined)
        else:
            results.append(run_one(config=config, method=method))
    return results
