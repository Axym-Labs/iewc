from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Sequence

import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from .output_metrics import wasserstein_1d_cdf_dual_quadratic_form


SEQUENTIAL_ALIASES = {"sequential", "naive"}


@dataclass(frozen=True)
class SyntheticDiffusionConfig:
    seed: int = 0
    image_size: int = 16
    n_train_per_task: int = 96
    n_test_per_task: int = 96
    train_steps: int = 80
    batch_size: int = 16
    learning_rate: float = 1e-3
    ewc_lambda: float = 25.0
    num_train_timesteps: int = 50
    tau_values: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    output_metric: str = "euclidean"
    device: str = "cpu"
    block_out_channels: tuple[int, ...] = (16, 16)
    layers_per_block: int = 1
    beta_schedule: str = "linear"
    dataset: str = "synthetic"
    data_root: str = "data"
    max_importance_samples: int | None = None
    progress_interval: int = 0


class TinyConditionalDiffusion(nn.Module):
    def __init__(
        self,
        image_size: int,
        num_train_timesteps: int,
        block_out_channels: tuple[int, ...] = (16, 16),
        layers_per_block: int = 1,
        beta_schedule: str = "linear",
    ):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=1,
            out_channels=1,
            block_out_channels=block_out_channels,
            down_block_types=tuple("DownBlock2D" for _ in block_out_channels),
            up_block_types=tuple("UpBlock2D" for _ in block_out_channels),
            layers_per_block=layers_per_block,
            norm_num_groups=8,
            num_class_embeds=4,
        )
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
        )

    def forward(self, noisy_images, timesteps, labels):
        return self.unet(noisy_images, timesteps, labels).sample


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(device)


def _class_pattern(label: int, image_size: int) -> torch.Tensor:
    image = torch.zeros(1, image_size, image_size)
    lo = image_size // 4
    hi = 3 * image_size // 4
    if label == 0:
        image[:, :, image_size // 2 - 1 : image_size // 2 + 1] = 1.0
    elif label == 1:
        image[:, image_size // 2 - 1 : image_size // 2 + 1, :] = 1.0
    elif label == 2:
        idx = torch.arange(lo, hi)
        image[:, idx, idx] = 1.0
        image[:, idx, idx + 1] = 1.0
    elif label == 3:
        image[:, lo:hi, lo:hi] = 1.0
        image[:, lo + 2 : hi - 2, lo + 2 : hi - 2] = 0.0
    else:
        raise ValueError(f"Unsupported synthetic class label: {label}")
    return image * 2.0 - 1.0


def _make_task(
    *,
    labels: tuple[int, int],
    n_samples: int,
    image_size: int,
    generator: torch.Generator,
) -> TensorDataset:
    images = []
    ys = []
    for idx in range(n_samples):
        label = labels[idx % len(labels)]
        image = _class_pattern(label, image_size)
        image = image + 0.08 * torch.randn(
            image.shape, generator=generator, dtype=image.dtype
        )
        images.append(image.clamp(-1.0, 1.0))
        ys.append(label)
    return TensorDataset(torch.stack(images), torch.tensor(ys, dtype=torch.long))


def _make_mnist_task(
    *,
    labels: tuple[int, int],
    n_samples: int,
    image_size: int,
    root: str,
    train: bool,
) -> TensorDataset:
    transform_steps = []
    if image_size != 28:
        transform_steps.append(transforms.Resize((image_size, image_size)))
    transform_steps += [
        transforms.ToTensor(),
        transforms.Lambda(lambda image: image * 2.0 - 1.0),
    ]
    dataset = datasets.MNIST(
        root=str(Path(root)),
        train=train,
        download=False,
        transform=transforms.Compose(transform_steps),
    )
    per_label = n_samples // len(labels)
    remainder = n_samples % len(labels)
    budgets = {label: per_label + (idx < remainder) for idx, label in enumerate(labels)}
    counts = {label: 0 for label in labels}
    images = []
    ys = []
    label_set = set(labels)
    for image, label in dataset:
        if label not in label_set or counts[label] >= budgets[label]:
            continue
        images.append(image)
        ys.append(label)
        counts[label] += 1
        if all(counts[label] >= budgets[label] for label in labels):
            break
    if len(images) != n_samples:
        raise RuntimeError(
            f"MNIST root {root} did not provide {n_samples} samples for labels {labels}"
        )
    return TensorDataset(torch.stack(images), torch.tensor(ys, dtype=torch.long))


def make_diffusion_stream(config: SyntheticDiffusionConfig):
    if config.dataset == "mnist":
        task_a_train = _make_mnist_task(
            labels=(0, 1),
            n_samples=config.n_train_per_task,
            image_size=config.image_size,
            root=config.data_root,
            train=True,
        )
        task_b_train = _make_mnist_task(
            labels=(2, 3),
            n_samples=config.n_train_per_task,
            image_size=config.image_size,
            root=config.data_root,
            train=True,
        )
        task_a_test = _make_mnist_task(
            labels=(0, 1),
            n_samples=config.n_test_per_task,
            image_size=config.image_size,
            root=config.data_root,
            train=False,
        )
        task_b_test = _make_mnist_task(
            labels=(2, 3),
            n_samples=config.n_test_per_task,
            image_size=config.image_size,
            root=config.data_root,
            train=False,
        )
        return task_a_train, task_b_train, task_a_test, task_b_test
    if config.dataset != "synthetic":
        raise ValueError(f"Unsupported diffusion dataset: {config.dataset}")
    generator = torch.Generator().manual_seed(config.seed)
    task_a_train = _make_task(
        labels=(0, 1),
        n_samples=config.n_train_per_task,
        image_size=config.image_size,
        generator=generator,
    )
    task_b_train = _make_task(
        labels=(2, 3),
        n_samples=config.n_train_per_task,
        image_size=config.image_size,
        generator=generator,
    )
    task_a_test = _make_task(
        labels=(0, 1),
        n_samples=config.n_test_per_task,
        image_size=config.image_size,
        generator=generator,
    )
    task_b_test = _make_task(
        labels=(2, 3),
        n_samples=config.n_test_per_task,
        image_size=config.image_size,
        generator=generator,
    )
    return task_a_train, task_b_train, task_a_test, task_b_test


def _limit_importance_dataset(
    dataset: TensorDataset,
    max_samples: int | None,
    *,
    seed: int,
) -> TensorDataset:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    images, labels = dataset.tensors
    generator = torch.Generator().manual_seed(seed)
    unique_labels = sorted(int(label) for label in labels.unique().tolist())
    per_label = max_samples // len(unique_labels)
    remainder = max_samples % len(unique_labels)
    selected = []
    for label_position, label in enumerate(unique_labels):
        budget = per_label + (label_position < remainder)
        label_indices = (labels == label).nonzero(as_tuple=False).flatten()
        order = torch.randperm(label_indices.numel(), generator=generator)
        selected.append(label_indices[order[:budget]])
    indices = torch.cat(selected)
    indices = indices[torch.randperm(indices.numel(), generator=generator)]
    return TensorDataset(images[indices], labels[indices])


def _diffusion_loss(
    model: TinyConditionalDiffusion,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
):
    device = images.device
    noise = torch.randn(
        images.shape, generator=generator, device=device, dtype=images.dtype
    )
    timesteps = torch.randint(
        0,
        model.scheduler.config.num_train_timesteps,
        (images.shape[0],),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    noisy_images = model.scheduler.add_noise(images, noise, timesteps)
    pred_noise = model(noisy_images, timesteps, labels)
    return nn.functional.mse_loss(pred_noise, noise), pred_noise, noise


def _sliced_wasserstein_dual_loss_scale(output_grad: torch.Tensor) -> torch.Tensor:
    images = output_grad.reshape(-1, output_grad.shape[-2], output_grad.shape[-1])
    values = []
    for image in images:
        for row in image:
            values.append(wasserstein_1d_cdf_dual_quadratic_form(row))
        for col in image.transpose(0, 1):
            values.append(wasserstein_1d_cdf_dual_quadratic_form(col))
    return torch.stack(values).mean()


def _output_loss_scale(output_grad: torch.Tensor, output_metric: str) -> torch.Tensor:
    if output_metric == "euclidean":
        return output_grad.detach().pow(2).sum()
    if output_metric == "sliced_wasserstein":
        return _sliced_wasserstein_dual_loss_scale(output_grad.detach())
    raise ValueError(f"Unsupported diffusion output metric: {output_metric}")


def _sliced_wasserstein_image_distance(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    delta = (old - new).reshape(-1, old.shape[-2], old.shape[-1])
    width = float(old.shape[-1] - 1)
    height = float(old.shape[-2] - 1)
    row_cdf = torch.cumsum(delta, dim=-1)[..., :-1]
    col_cdf = torch.cumsum(delta, dim=-2)[:, :-1, :]
    row_dist = row_cdf.pow(2).sum(dim=(-1, -2)) / width
    col_dist = col_cdf.pow(2).sum(dim=(-1, -2)) / height
    return 0.5 * (row_dist + col_dist)


@torch.no_grad()
def denoising_mse(
    model: TinyConditionalDiffusion,
    dataset: TensorDataset,
    *,
    batch_size: int,
    seed: int,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        loss, _, _ = _diffusion_loss(model, images, labels, generator=generator)
        total_loss += float(loss.item()) * images.shape[0]
        total_count += images.shape[0]
    if total_count == 0:
        raise ValueError("Cannot evaluate an empty diffusion dataset")
    return total_loss / float(total_count)


@torch.no_grad()
def old_output_drift(
    old_model: TinyConditionalDiffusion,
    new_model: TinyConditionalDiffusion,
    dataset: TensorDataset,
    *,
    batch_size: int,
    seed: int,
) -> tuple[float, float]:
    old_model.eval()
    new_model.eval()
    device = next(old_model.parameters()).device
    new_model.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    euclidean_total = 0.0
    wasserstein_total = 0.0
    total_count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        noise = torch.randn(
            images.shape, generator=generator, device=images.device, dtype=images.dtype
        )
        timesteps = torch.randint(
            0,
            old_model.scheduler.config.num_train_timesteps,
            (images.shape[0],),
            generator=generator,
            device=images.device,
            dtype=torch.long,
        )
        noisy_images = old_model.scheduler.add_noise(images, noise, timesteps)
        old_pred = old_model(noisy_images, timesteps, labels)
        new_pred = new_model(noisy_images, timesteps, labels)
        euclidean = (old_pred - new_pred).pow(2).mean(dim=(1, 2, 3))
        wasserstein = _sliced_wasserstein_image_distance(old_pred, new_pred)
        euclidean_total += float(euclidean.sum().item())
        wasserstein_total += float(wasserstein.sum().item())
        total_count += images.shape[0]
    if total_count == 0:
        raise ValueError("Cannot evaluate output drift on an empty diffusion dataset")
    return euclidean_total / total_count, wasserstein_total / total_count


@torch.no_grad()
def sample_images(
    model: TinyConditionalDiffusion,
    labels: torch.Tensor,
    *,
    seed: int,
    num_inference_steps: int | None = None,
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    labels = labels.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    sample_size = model.unet.config.sample_size
    if isinstance(sample_size, (tuple, list)):
        height, width = int(sample_size[0]), int(sample_size[1])
    else:
        height = width = int(sample_size)
    images = torch.randn(
        (labels.shape[0], model.unet.config.in_channels, height, width),
        generator=generator,
        device=device,
    )
    steps = num_inference_steps or model.scheduler.config.num_train_timesteps
    try:
        model.scheduler.set_timesteps(steps, device=device)
    except TypeError:
        model.scheduler.set_timesteps(steps)
    for timestep in model.scheduler.timesteps:
        timestep_batch = torch.full(
            (labels.shape[0],),
            int(timestep.item()),
            device=device,
            dtype=torch.long,
        )
        pred_noise = model(images, timestep_batch, labels)
        try:
            images = model.scheduler.step(
                pred_noise, timestep, images, generator=generator
            ).prev_sample
        except TypeError:
            images = model.scheduler.step(pred_noise, timestep, images).prev_sample
    return images.clamp(-1.0, 1.0).detach().cpu()


def compute_penalty(model: nn.Module, saved_params, importances):
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name not in saved_params:
            continue
        delta = param - saved_params[name]
        penalty = penalty + (importances[name] * delta.pow(2)).sum()
    return penalty


def train_task(
    *,
    model: TinyConditionalDiffusion,
    dataset: TensorDataset,
    steps: int,
    batch_size: int,
    learning_rate: float,
    ewc_lambda: float = 0.0,
    saved_params=None,
    importances=None,
    seed: int,
    progress_prefix: str = "",
    progress_interval: int = 0,
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    loss_window = []
    for step in range(1, steps + 1):
        images, labels = next(loader)
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss, _, _ = _diffusion_loss(model, images, labels, generator=generator)
        if importances is not None and saved_params is not None:
            loss = loss + ewc_lambda * compute_penalty(
                model, saved_params, importances
            )
        loss.backward()
        optimizer.step()
        if progress_interval:
            loss_window.append(float(loss.item()))
            if step % progress_interval == 0:
                window = loss_window[-progress_interval:]
                label = progress_prefix or "train"
                print(
                    f"{label} step {step}/{steps} loss={sum(window) / len(window):.4f}",
                    flush=True,
                )
    return optimizer


def compute_diffusion_importance(
    *,
    model: TinyConditionalDiffusion,
    dataset: TensorDataset,
    kind: str,
    tau: float,
    seed: int,
    output_metric: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if kind not in {"ef", "ief_diag"}:
        raise ValueError(f"Unsupported diffusion importance kind: {kind}")

    model.eval()
    importances = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    loss_scales = []
    ef_summand_traces = []
    stored_summand_traces = []
    sample_count = 0
    for image, label in dataset:
        image = image.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        loss, pred_noise, target_noise = _diffusion_loss(
            model, image, label, generator=generator
        )

        if kind == "ief_diag":
            output_grad = torch.autograd.grad(loss, pred_noise, retain_graph=True)[0]
            loss_scale = _output_loss_scale(output_grad, output_metric)
            denom = loss_scale + tau
        else:
            loss_scale = torch.ones(())
            denom = torch.ones(())

        loss.backward()
        grad_squares = {}
        ef_trace = torch.zeros((), device=device)
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_sq = param.grad.detach().pow(2)
                grad_squares[name] = grad_sq
                ef_trace = ef_trace + grad_sq.sum()
        stored_by_name = {
            name: grad_sq / denom for name, grad_sq in grad_squares.items()
        }
        stored_trace = torch.zeros((), device=device)
        for name, stored in stored_by_name.items():
            importances[name] += stored
            stored_trace = stored_trace + stored.sum()
        loss_scales.append(loss_scale.detach())
        ef_summand_traces.append(ef_trace.detach())
        stored_summand_traces.append(stored_trace.detach())
        sample_count += 1

    if sample_count == 0:
        raise ValueError("Cannot compute diffusion importances from an empty dataset")
    for name in importances:
        importances[name] /= float(sample_count)
    stored_summand_trace_tensor = torch.stack(stored_summand_traces)
    model.zero_grad(set_to_none=True)
    return (
        importances,
        torch.stack(loss_scales),
        torch.stack(ef_summand_traces),
        stored_summand_trace_tensor,
    )


def run_one(
    *,
    config: SyntheticDiffusionConfig,
    method: str,
    tau: float | None = None,
) -> dict:
    canonical_method = "sequential" if method in SEQUENTIAL_ALIASES else method
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    task_a_train, task_b_train, task_a_test, task_b_test = make_diffusion_stream(
        config
    )
    model = TinyConditionalDiffusion(
        image_size=config.image_size,
        num_train_timesteps=config.num_train_timesteps,
        block_out_channels=config.block_out_channels,
        layers_per_block=config.layers_per_block,
        beta_schedule=config.beta_schedule,
    ).to(device)
    train_task(
        model=model,
        dataset=task_a_train,
        steps=config.train_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed + 100,
        progress_prefix=f"{canonical_method}:old",
        progress_interval=config.progress_interval,
    )
    task_a_mse_after_task_a = denoising_mse(
        model, task_a_test, batch_size=config.batch_size, seed=config.seed + 200
    )
    old_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    importances = None
    saved_params = None
    loss_scale_mean = None
    loss_scale_median = None
    loss_scale_values = None
    if canonical_method != "sequential":
        importance_dataset = _limit_importance_dataset(
            task_a_train,
            config.max_importance_samples,
            seed=config.seed + 250,
        )
        (
            importances,
            loss_scales,
            ef_summand_traces,
            stored_summand_traces,
        ) = compute_diffusion_importance(
            model=model,
            dataset=importance_dataset,
            kind=canonical_method,
            tau=0.0 if tau is None else tau,
            seed=config.seed + 300,
            output_metric=config.output_metric,
        )
        saved_params = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        loss_scale_mean = float(loss_scales.mean().item())
        loss_scale_median = float(loss_scales.median().item())
        loss_scale_values = [float(value) for value in loss_scales.tolist()]
        ef_summand_trace_values = [
            float(value) for value in ef_summand_traces.tolist()
        ]
        stored_summand_trace_values = [
            float(value) for value in stored_summand_traces.tolist()
        ]

    train_task(
        model=model,
        dataset=task_b_train,
        steps=config.train_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        ewc_lambda=config.ewc_lambda,
        saved_params=saved_params,
        importances=importances,
        seed=config.seed + 400,
        progress_prefix=f"{canonical_method}:new",
        progress_interval=config.progress_interval,
    )

    task_a_mse_after_task_b = denoising_mse(
        model, task_a_test, batch_size=config.batch_size, seed=config.seed + 500
    )
    task_b_mse_after_task_b = denoising_mse(
        model, task_b_test, batch_size=config.batch_size, seed=config.seed + 600
    )
    old_model = TinyConditionalDiffusion(
        image_size=config.image_size,
        num_train_timesteps=config.num_train_timesteps,
        block_out_channels=config.block_out_channels,
        layers_per_block=config.layers_per_block,
        beta_schedule=config.beta_schedule,
    ).to(device)
    old_model.load_state_dict(old_state)
    old_output_euclidean_drift, old_output_wasserstein_drift = old_output_drift(
        old_model,
        model,
        task_a_test,
        batch_size=config.batch_size,
        seed=config.seed + 700,
    )
    output = {
        "method": canonical_method,
        "tau": tau,
        "output_metric": config.output_metric,
        "task_a_denoise_mse_after_task_a": task_a_mse_after_task_a,
        "task_a_denoise_mse_after_task_b": task_a_mse_after_task_b,
        "task_b_denoise_mse_after_task_b": task_b_mse_after_task_b,
        "task_a_denoise_mse_increase": (
            task_a_mse_after_task_b - task_a_mse_after_task_a
        ),
        "old_output_euclidean_drift": old_output_euclidean_drift,
        "old_output_sliced_wasserstein_drift": old_output_wasserstein_drift,
    }
    if loss_scale_mean is not None:
        output["old_task_loss_scale_mean"] = loss_scale_mean
        output["old_task_loss_scale_median"] = loss_scale_median
        output["old_task_loss_scales"] = loss_scale_values
        output["old_task_ef_summand_traces"] = ef_summand_trace_values
        output["old_task_stored_summand_traces"] = stored_summand_trace_values
    return output


def run_synthetic_diffusion_suite(
    *,
    config: SyntheticDiffusionConfig,
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
                    item["task_a_denoise_mse_increase"],
                    item["task_b_denoise_mse_after_task_b"],
                ),
            )
            combined = dict(best)
            combined["tau_results"] = tau_results
            results.append(combined)
        else:
            results.append(run_one(config=config, method=method))
    return results
