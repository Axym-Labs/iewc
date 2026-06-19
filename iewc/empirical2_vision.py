from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import math
import tarfile
import urllib.request

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
from PIL import Image
import yaml

from .diagonal_regularization import (
    DiagonalImportance,
    compute_diagonal_importance,
    diagonal_ewc_penalties,
)


VisionDataset = Literal["cifar100", "tiny_imagenet", "imagenet_r"]
VisionAdaptation = Literal["full", "lora"]
VisionEvaluation = Literal["task_aware", "class_incremental"]
VisionMethod = Literal[
    "sequential",
    "ef",
    "ewc_dr",
    "iewc",
    "iewc_gss",
    "iewc_fromp",
]


@dataclass(frozen=True)
class VisionCLConfig:
    dataset: VisionDataset = "cifar100"
    data_root: str = "/home/davwis/main/data"
    model_name: str = "vit_tiny_patch16_224"
    pretrained: bool = False
    adaptation: VisionAdaptation = "full"
    seed: int = 0
    n_tasks: int = 2
    classes_per_task: int = 5
    image_size: int = 224
    train_samples_per_class: int = 8
    test_samples_per_class: int = 8
    epochs_per_task: int = 1
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    ewc_lambda: float = 100.0
    tau: float = 1e-2
    importance_samples: int = 64
    lora_rank: int = 4
    lora_alpha: float = 8.0
    num_workers: int = 2
    device: str = "cuda"
    download: bool = False
    evaluation: VisionEvaluation = "task_aware"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        for param in self.base.parameters():
            param.requires_grad = False
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = (x @ self.lora_a.t()) @ self.lora_b.t()
        return self.base(x) + update * self.scaling


class ImagePathLabelDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path], targets: list[int], transform=None):
        self.paths = paths
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[idx])


def _set_child(module: nn.Module, name: str, child: nn.Module) -> None:
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], child)


def add_lora_to_vit(model: nn.Module, *, rank: int, alpha: float) -> None:
    target_names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith("head") or name.endswith("fc"):
            continue
        if any(token in name for token in ("qkv", "proj", "fc1", "fc2")):
            target_names.append(name)
    for name in target_names:
        parent = model
        for part in name.split(".")[:-1]:
            parent = getattr(parent, part)
        child = getattr(parent, name.split(".")[-1])
        _set_child(model, name, LoRALinear(child, rank=rank, alpha=alpha))


def configure_trainable_parameters(
    model: nn.Module,
    *,
    adaptation: VisionAdaptation,
    lora_rank: int,
    lora_alpha: float,
) -> None:
    if adaptation == "full":
        for param in model.parameters():
            param.requires_grad = True
        return
    if adaptation != "lora":
        raise ValueError(f"Unknown adaptation: {adaptation}")
    for param in model.parameters():
        param.requires_grad = False
    add_lora_to_vit(model, rank=lora_rank, alpha=lora_alpha)
    for name, param in model.named_parameters():
        if "lora_" in name or name.startswith("head.") or ".head." in name:
            param.requires_grad = True


def _vision_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return train_transform, test_transform


def _download_imagenet_r(root: Path) -> None:
    target = root / "imagenet-r"
    if target.exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "imagenet-r.tar"
    urllib.request.urlretrieve("https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar", archive)
    with tarfile.open(archive) as handle:
        handle.extractall(root)
    archive.unlink()


def _imagenet_r_split(data_root: Path, split: str, transform) -> ImagePathLabelDataset:
    yaml_path = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "mammoth"
        / "datasets"
        / "imagenet_r_utils"
        / f"imagenet-r_{split}.yaml"
    )
    config = yaml.safe_load(yaml_path.read_text())
    paths = []
    for raw_path in config["data"]:
        path = Path(raw_path)
        if path.parts and path.parts[0] == "data":
            path = data_root.joinpath(*path.parts[1:])
        elif not path.is_absolute():
            path = data_root / path
        paths.append(path)
    return ImagePathLabelDataset(paths, [int(value) for value in config["targets"]], transform=transform)


def _load_base_datasets(config: VisionCLConfig):
    data_root = Path(config.data_root)
    train_transform, test_transform = _vision_transforms(config.image_size)
    if config.dataset == "cifar100":
        root = data_root / "cifar-100"
        train = datasets.CIFAR100(root=root, train=True, transform=train_transform, download=config.download)
        test = datasets.CIFAR100(root=root, train=False, transform=test_transform, download=config.download)
        return train, test, 100
    if config.dataset == "tiny_imagenet":
        root = data_root / "tiny-imagenet-200"
        train = datasets.ImageFolder(root / "train", transform=train_transform)
        test_root = root / "val"
        test = datasets.ImageFolder(test_root, transform=test_transform) if test_root.exists() else train
        return train, test, len(train.classes)
    if config.dataset == "imagenet_r":
        root = data_root
        if config.download:
            _download_imagenet_r(root)
        train = _imagenet_r_split(root, "train", train_transform)
        test = _imagenet_r_split(root, "test", test_transform)
        return train, test, 200
    raise ValueError(f"Unknown vision dataset: {config.dataset}")


def _targets(dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return [int(value) for value in dataset.targets]
    if hasattr(dataset, "samples"):
        return [int(sample[1]) for sample in dataset.samples]
    raise ValueError("Cannot read dataset targets")


def _class_indices(dataset, classes: set[int], max_per_class: int, *, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    by_class: dict[int, list[int]] = {klass: [] for klass in classes}
    for idx, target in enumerate(_targets(dataset)):
        if target in by_class:
            by_class[target].append(idx)
    selected = []
    for klass in sorted(by_class):
        indices = by_class[klass]
        if max_per_class > 0 and len(indices) > max_per_class:
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in order[:max_per_class]]
        selected.extend(indices)
    return selected


def _task_subsets(config: VisionCLConfig, train, test) -> list[tuple[list[int], Subset, Subset]]:
    total_classes = min(config.n_tasks * config.classes_per_task, len(set(_targets(train))))
    class_order = list(range(total_classes))
    tasks = []
    for task_id in range(config.n_tasks):
        classes = class_order[
            task_id * config.classes_per_task : (task_id + 1) * config.classes_per_task
        ]
        class_set = set(classes)
        train_indices = _class_indices(
            train,
            class_set,
            config.train_samples_per_class,
            seed=config.seed + 17 * task_id,
        )
        test_indices = _class_indices(
            test,
            class_set,
            config.test_samples_per_class,
            seed=config.seed + 101 + 17 * task_id,
        )
        tasks.append((classes, Subset(train, train_indices), Subset(test, test_indices)))
    return tasks


def _loader(dataset, *, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def _single_loader(dataset, *, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)


def _loss_output_fn(device: torch.device):
    def fn(model: nn.Module, batch, reverse_output: bool):
        x, y = batch[0].to(device), batch[1].to(device)
        output = model(x)
        loss_input = -output if reverse_output else output
        return nn.functional.cross_entropy(loss_input, y), output

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
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            if importances:
                loss = loss + float(ewc_lambda) * diagonal_ewc_penalties(model, importances, device)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
    return last_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    allowed_classes: list[int] | None = None,
) -> float:
    model.eval()
    mask = None
    correct = 0
    total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x)
        if allowed_classes is not None and mask is None:
            allowed = torch.as_tensor(allowed_classes, device=device, dtype=torch.long)
            mask = torch.full((logits.shape[-1],), float("-inf"), device=device)
            mask[allowed] = 0.0
        if mask is not None:
            logits = logits + mask
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / float(max(1, total))


def _importance_kind_and_weight(method: VisionMethod) -> tuple[str | None, str]:
    if method == "sequential":
        return None, "uniform"
    if method == "ef":
        return "ef", "uniform"
    if method == "ewc_dr":
        return "ewc_dr", "uniform"
    if method == "iewc":
        return "iewc", "uniform"
    if method == "iewc_gss":
        return "iewc", "gss_residual"
    if method == "iewc_fromp":
        return "iewc", "fromp_trace"
    raise ValueError(f"Unknown method: {method}")


def run_vision_cl(config: VisionCLConfig, method: VisionMethod) -> dict:
    torch.manual_seed(config.seed)
    device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    train_dataset, test_dataset, n_classes = _load_base_datasets(config)
    tasks = _task_subsets(config, train_dataset, test_dataset)
    model = timm.create_model(
        config.model_name,
        pretrained=config.pretrained,
        num_classes=n_classes,
        img_size=config.image_size,
    )
    configure_trainable_parameters(
        model,
        adaptation=config.adaptation,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    importance_kind, sample_weighting = _importance_kind_and_weight(method)
    importances: list[DiagonalImportance] = []
    accuracy_matrix: list[list[float]] = []
    train_losses = []
    importance_summaries = []

    for task_id, (_, train_subset, _) in enumerate(tasks):
        train_loss = train_one_task(
            model=model,
            loader=_loader(
                train_subset,
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
        row = []
        for eval_classes, _, test_subset in tasks:
            row.append(
                evaluate(
                    model,
                    _loader(
                        test_subset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        workers=config.num_workers,
                    ),
                    device,
                    allowed_classes=eval_classes if config.evaluation == "task_aware" else None,
                )
            )
        accuracy_matrix.append(row)

        if importance_kind is not None and task_id < len(tasks) - 1:
            importance = compute_diagonal_importance(
                model=model,
                dataloader=_single_loader(train_subset, workers=config.num_workers),
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
                    "sample_count": importance.sample_count,
                    "mean_loss_scale": float(importance.loss_scales.float().mean().item()),
                    "mean_sample_weight": float(importance.sample_weights.float().mean().item()),
                    "max_sample_weight": float(importance.sample_weights.float().max().item()),
                    "mean_summand_trace": float(importance.stored_summand_traces.float().mean().item()),
                }
            )

    final_accs = accuracy_matrix[-1]
    forgetting = []
    for task_id in range(len(tasks) - 1):
        best = max(row[task_id] for row in accuracy_matrix[:])
        forgetting.append(best - final_accs[task_id])
    return {
        "experiment": "empirical2_vision_cl",
        "config": asdict(config),
        "method": method,
        "n_trainable_parameters": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
        "n_total_parameters": int(sum(param.numel() for param in model.parameters())),
        "accuracy_matrix": accuracy_matrix,
        "final_task_accuracies": final_accs,
        "final_avg_accuracy": float(sum(final_accs) / len(final_accs)),
        "avg_forgetting": float(sum(forgetting) / len(forgetting)) if forgetting else 0.0,
        "train_losses": train_losses,
        "importance_summaries": importance_summaries,
    }
