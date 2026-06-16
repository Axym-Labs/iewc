import argparse
import copy
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from iewc.synthetic_diffusion import (
    SyntheticDiffusionConfig,
    TinyConditionalDiffusion,
    _diffusion_loss,
    compute_penalty,
    compute_diffusion_importance,
    make_diffusion_stream,
    sample_images,
)


def parse_block_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(channels) < 2:
        raise argparse.ArgumentTypeError("expected at least two comma-separated widths")
    return channels


def train_continual_model(
    *,
    config: SyntheticDiffusionConfig,
    method: str,
    task_a_train,
    task_b_train,
    importance_dataset,
    ewc_lambda: float,
    tau: float,
    device: torch.device,
    ema_decay: float,
    num_workers: int,
    progress_interval: int,
):
    if method not in {"sequential", "ef", "ief_diag"}:
        raise ValueError(f"Unsupported sample-grid method: {method}")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = TinyConditionalDiffusion(
        image_size=config.image_size,
        num_train_timesteps=config.num_train_timesteps,
        block_out_channels=config.block_out_channels,
        layers_per_block=config.layers_per_block,
        beta_schedule=config.beta_schedule,
    ).to(device)
    ema_model = copy.deepcopy(model).eval() if ema_decay > 0.0 else None
    if ema_model is not None:
        for param in ema_model.parameters():
            param.requires_grad_(False)
    train_task_for_samples(
        model=model,
        ema_model=ema_model,
        dataset=task_a_train,
        steps=config.train_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed + 100,
        ema_decay=ema_decay,
        num_workers=num_workers,
        progress_prefix=f"{method}:old",
        progress_interval=progress_interval,
    )
    importances = None
    saved_params = None
    if method != "sequential":
        importances, _, _, _ = compute_diffusion_importance(
            model=model,
            dataset=importance_dataset,
            kind=method,
            tau=tau,
            seed=config.seed + 300,
            output_metric="euclidean",
        )
        saved_params = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
    train_task_for_samples(
        model=model,
        ema_model=ema_model,
        dataset=task_b_train,
        steps=config.train_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        ewc_lambda=ewc_lambda,
        saved_params=saved_params,
        importances=importances,
        seed=config.seed + 400,
        ema_decay=ema_decay,
        num_workers=num_workers,
        progress_prefix=f"{method}:new",
        progress_interval=progress_interval,
    )
    return ema_model if ema_model is not None else model


def train_task_for_samples(
    *,
    model: TinyConditionalDiffusion,
    ema_model: TinyConditionalDiffusion | None,
    dataset: TensorDataset,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    ema_decay: float,
    num_workers: int,
    progress_prefix: str,
    progress_interval: int,
    ewc_lambda: float = 0.0,
    saved_params=None,
    importances=None,
):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    iterator = iter(loader)
    generator = torch.Generator(device=device).manual_seed(seed)
    loss_window = []
    for step in range(1, steps + 1):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = _diffusion_loss(model, images, labels, generator=generator)
        if importances is not None and saved_params is not None:
            loss = loss + ewc_lambda * compute_penalty(model, saved_params, importances)
        loss.backward()
        optimizer.step()
        if ema_model is not None:
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.mul_(ema_decay).add_(
                        param.detach(), alpha=1.0 - ema_decay
                    )
                for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
                    ema_buffer.copy_(buffer)
        loss_window.append(float(loss.item()))
        if progress_interval and step % progress_interval == 0:
            window = loss_window[-progress_interval:]
            print(
                f"{progress_prefix} step {step}/{steps} loss={sum(window) / len(window):.4f}",
                flush=True,
            )


def mnist_task_dataset(
    *,
    root: Path,
    labels: tuple[int, int],
    n_samples: int,
    image_size: int,
) -> TensorDataset:
    transform_steps = []
    if image_size != 28:
        transform_steps.append(transforms.Resize((image_size, image_size)))
    transform_steps += [
        transforms.ToTensor(),
        transforms.Lambda(lambda image: image * 2.0 - 1.0),
    ]
    transform = transforms.Compose(transform_steps)
    dataset = datasets.MNIST(root=str(root), train=True, download=False, transform=transform)
    per_label = n_samples // len(labels)
    remainder = n_samples % len(labels)
    label_budgets = {
        label: per_label + (idx < remainder) for idx, label in enumerate(labels)
    }
    counts = {label: 0 for label in labels}
    images = []
    ys = []
    label_set = set(labels)
    for image, label in dataset:
        if label not in label_set or counts[label] >= label_budgets[label]:
            continue
        images.append(image)
        ys.append(label)
        counts[label] += 1
        if all(counts[label] >= label_budgets[label] for label in labels):
            break
    if len(images) != n_samples:
        raise RuntimeError(
            f"MNIST root {root} did not provide {n_samples} samples for labels {labels}"
        )
    return TensorDataset(torch.stack(images), torch.tensor(ys, dtype=torch.long))


def make_mnist_stream(
    *,
    root: Path,
    n_train_per_task: int,
    n_importance_samples: int,
    image_size: int,
):
    task_a_train = mnist_task_dataset(
        root=root, labels=(0, 1), n_samples=n_train_per_task, image_size=image_size
    )
    task_b_train = mnist_task_dataset(
        root=root, labels=(2, 3), n_samples=n_train_per_task, image_size=image_size
    )
    task_a_importance = mnist_task_dataset(
        root=root,
        labels=(0, 1),
        n_samples=n_importance_samples,
        image_size=image_size,
    )
    return task_a_train, task_b_train, task_a_importance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "synthetic"], default="mnist")
    parser.add_argument(
        "--mnist-root",
        type=Path,
        default=Path("/home/davwis/main/data/mnist"),
    )
    parser.add_argument("--mnist-image-size", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train-per-task", type=int, default=192)
    parser.add_argument("--importance-samples", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--block-channels", type=parse_block_channels, default=(16, 16))
    parser.add_argument("--layers-per-block", type=int, default=1)
    parser.add_argument("--num-train-timesteps", type=int, default=50)
    parser.add_argument("--beta-schedule", default="linear")
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--ef-lambda", type=float, default=200.0)
    parser.add_argument("--iewc-lambda", type=float, default=25.0)
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/empirical-evidence/artifacts/paper-plots/diffusion_generated_samples.png"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the diffusion sample grid")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = SyntheticDiffusionConfig(
        seed=args.seed,
        image_size=args.mnist_image_size if args.dataset == "mnist" else 16,
        n_train_per_task=args.n_train_per_task,
        n_test_per_task=args.n_train_per_task,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ewc_lambda=args.iewc_lambda,
        num_train_timesteps=args.num_train_timesteps,
        tau_values=(args.tau,),
        block_out_channels=args.block_channels,
        layers_per_block=args.layers_per_block,
        beta_schedule=args.beta_schedule,
    )
    if args.dataset == "mnist":
        task_a_train, task_b_train, task_a_importance = make_mnist_stream(
            root=args.mnist_root,
            n_train_per_task=args.n_train_per_task,
            n_importance_samples=args.importance_samples,
            image_size=config.image_size,
        )
    else:
        task_a_train, task_b_train, _, _ = make_diffusion_stream(config)
        task_a_importance = task_a_train

    models = {
        "Sequential": train_continual_model(
            config=config,
            method="sequential",
            task_a_train=task_a_train,
            task_b_train=task_b_train,
            importance_dataset=task_a_importance,
            ewc_lambda=0.0,
            tau=args.tau,
            device=device,
            ema_decay=args.ema_decay,
            num_workers=args.num_workers,
            progress_interval=args.progress_interval,
        ),
        "EF-EWC": train_continual_model(
            config=replace(config, ewc_lambda=args.ef_lambda),
            method="ef",
            task_a_train=task_a_train,
            task_b_train=task_b_train,
            importance_dataset=task_a_importance,
            ewc_lambda=args.ef_lambda,
            tau=args.tau,
            device=device,
            ema_decay=args.ema_decay,
            num_workers=args.num_workers,
            progress_interval=args.progress_interval,
        ),
        "IEWC": train_continual_model(
            config=replace(config, ewc_lambda=args.iewc_lambda),
            method="ief_diag",
            task_a_train=task_a_train,
            task_b_train=task_b_train,
            importance_dataset=task_a_importance,
            ewc_lambda=args.iewc_lambda,
            tau=args.tau,
            device=device,
            ema_decay=args.ema_decay,
            num_workers=args.num_workers,
            progress_interval=args.progress_interval,
        ),
    }
    distribution_labels = {
        "Old distribution": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "New distribution": torch.tensor([2, 3, 2, 3], dtype=torch.long),
    }
    image_blocks = {}
    for method_name, model in models.items():
        for row_idx, (distribution, labels) in enumerate(distribution_labels.items()):
            image_blocks[(method_name, distribution)] = sample_images(
                model,
                labels,
                seed=config.seed + 900 + row_idx,
                num_inference_steps=args.num_inference_steps,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.titlesize": 8,
        }
    )
    methods = ("Sequential", "EF-EWC", "IEWC")
    distributions = ("Old distribution", "New distribution")
    fig, axes = plt.subplots(
        len(methods),
        len(distributions) * 4,
        figsize=(5.55, 2.55),
        squeeze=False,
    )
    for row_idx, method_name in enumerate(methods):
        for distribution_idx, distribution in enumerate(distributions):
            block = image_blocks[(method_name, distribution)]
            for sample_idx, image in enumerate(block):
                ax = axes[row_idx, distribution_idx * 4 + sample_idx]
                ax.imshow(image.squeeze(0), cmap="gray", vmin=-1, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.35)
    fig.subplots_adjust(
        left=0.145,
        right=0.995,
        bottom=0.035,
        top=0.88,
        wspace=0.035,
        hspace=0.10,
    )
    fig.text(0.36, 0.94, "Old distribution", ha="center", va="center", fontsize=7.5)
    fig.text(0.78, 0.94, "New distribution", ha="center", va="center", fontsize=7.5)
    fig.text(0.015, 0.725, "Sequential", ha="left", va="center", fontsize=7.5)
    fig.text(0.015, 0.455, "EF-EWC", ha="left", va="center", fontsize=7.5)
    fig.text(0.015, 0.185, "IEWC", ha="left", va="center", fontsize=7.5)
    fig.add_artist(
        Line2D(
            [0.57, 0.57],
            [0.035, 0.88],
            transform=fig.transFigure,
            color="black",
            linewidth=0.35,
            alpha=0.45,
        )
    )
    fig.savefig(args.output, dpi=300)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
