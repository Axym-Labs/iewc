import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import OxfordIIITPet, VOCSegmentation
from torchvision.transforms import functional as TF

from .importance import ImportanceEstimator


CAT_CLASSES = {
    "abyssinian",
    "bengal",
    "birman",
    "bombay",
    "british_shorthair",
    "egyptian_mau",
    "maine_coon",
    "persian",
    "ragdoll",
    "russian_blue",
    "siamese",
    "sphynx",
}
SEQUENTIAL_ALIASES = {"sequential", "naive"}
VOC_CLASS_TO_INDEX = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20,
}
VOC_ANIMAL_CLASSES = ("bird", "cat", "cow", "dog", "horse", "sheep")
VOC_VEHICLE_CLASSES = (
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorbike",
    "train",
)


@dataclass(frozen=True)
class PetSegmentationConfig:
    seed: int = 0
    root: str = "/home/davwis/main/data/oxford-iiit-pet"
    image_size: int = 96
    n_train_per_task: int = 600
    n_test_per_task: int = 400
    train_epochs: int = 20
    hidden_channels: int = 24
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    ewc_lambda: float = 10.0
    importance_samples: int = 256
    tau_values: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    device: str = "auto"


@dataclass(frozen=True)
class VOCSegmentationConfig:
    seed: int = 0
    root: str = "/home/davwis/main/data/voc"
    image_size: int = 96
    n_train_per_task: int = 400
    n_test_per_task: int = 300
    train_epochs: int = 15
    hidden_channels: int = 24
    batch_size: int = 24
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    ewc_lambda: float = 10.0
    importance_samples: int = 192
    tau_values: tuple[float, ...] = (1e-2,)
    min_foreground_fraction: float = 0.01
    foreground_ce_weight: float = 2.0
    device: str = "auto"


class PetBinarySegmentation(Dataset):
    def __init__(
        self,
        *,
        base: OxfordIIITPet,
        indices: Sequence[int],
        image_size: int,
    ):
        self.base = base
        self.indices = list(indices)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, mask = self.base[self.indices[item]]
        if not isinstance(image, Image.Image):
            raise TypeError("Expected PIL image from OxfordIIITPet")
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        raw_mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        target = torch.full_like(raw_mask, 255)
        target[raw_mask == 1] = 1
        target[raw_mask == 2] = 0
        task = torch.zeros((), dtype=torch.long)
        return image_tensor, target, task


class VOCBinaryClassSetSegmentation(Dataset):
    def __init__(
        self,
        *,
        base: VOCSegmentation,
        indices: Sequence[int],
        selected_classes: Sequence[int],
        image_size: int,
    ):
        self.base = base
        self.indices = list(indices)
        self.selected_classes = torch.as_tensor(list(selected_classes), dtype=torch.long)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, mask = self.base[self.indices[item]]
        if not isinstance(image, Image.Image):
            raise TypeError("Expected PIL image from VOCSegmentation")
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        raw_mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        target = torch.zeros_like(raw_mask)
        target[raw_mask == 255] = 255
        valid = raw_mask != 255
        selected = torch.isin(raw_mask, self.selected_classes)
        target[selected & valid] = 1
        task = torch.zeros((), dtype=torch.long)
        return image_tensor, target, task


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        h = int(hidden_channels)
        self.enc1 = ConvBlock(3, h)
        self.enc2 = ConvBlock(h, 2 * h)
        self.bottleneck = ConvBlock(2 * h, 4 * h)
        self.up2 = nn.ConvTranspose2d(4 * h, 2 * h, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(4 * h, 2 * h)
        self.up1 = nn.ConvTranspose2d(2 * h, h, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(2 * h, h)
        self.out = nn.Conv2d(h, 2, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2))
        b = self.bottleneck(F.max_pool2d(e2, kernel_size=2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(device)


def _class_groups(base: OxfordIIITPet):
    cats = []
    dogs = []
    for idx, label in enumerate(base._labels):
        class_name = base.classes[int(label)].lower().replace(" ", "_")
        if class_name in CAT_CLASSES:
            cats.append(idx)
        else:
            dogs.append(idx)
    return cats, dogs


def _take(indices: Sequence[int], count: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(indices), generator=generator).tolist()
    return [indices[idx] for idx in perm[: min(int(count), len(indices))]]


def make_pet_stream(config: PetSegmentationConfig):
    root = Path(config.root)
    train_base = OxfordIIITPet(
        root,
        split="trainval",
        target_types="segmentation",
        download=True,
    )
    test_base = OxfordIIITPet(
        root,
        split="test",
        target_types="segmentation",
        download=True,
    )
    train_cats, train_dogs = _class_groups(train_base)
    test_cats, test_dogs = _class_groups(test_base)
    train_a = PetBinarySegmentation(
        base=train_base,
        indices=_take(train_cats, config.n_train_per_task, config.seed + 10),
        image_size=config.image_size,
    )
    train_b = PetBinarySegmentation(
        base=train_base,
        indices=_take(train_dogs, config.n_train_per_task, config.seed + 20),
        image_size=config.image_size,
    )
    test_a = PetBinarySegmentation(
        base=test_base,
        indices=_take(test_cats, config.n_test_per_task, config.seed + 30),
        image_size=config.image_size,
    )
    test_b = PetBinarySegmentation(
        base=test_base,
        indices=_take(test_dogs, config.n_test_per_task, config.seed + 40),
        image_size=config.image_size,
    )
    return train_a, train_b, test_a, test_b


def _voc_class_indices(names: Sequence[str]) -> tuple[int, ...]:
    return tuple(VOC_CLASS_TO_INDEX[name] for name in names)


def _voc_indices_for_classes(
    base: VOCSegmentation,
    class_indices: Sequence[int],
    *,
    min_foreground_fraction: float,
    cache_path: Path | None = None,
) -> list[int]:
    if cache_path is not None and cache_path.exists():
        return [int(index) for index in json.loads(cache_path.read_text())]
    wanted = set(int(index) for index in class_indices)
    selected = []
    for index, mask_path in enumerate(base.masks):
        mask = np.array(Image.open(mask_path), dtype=np.int64)
        valid = mask != 255
        if not np.any(valid):
            continue
        fg = np.isin(mask, list(wanted))
        if float(fg.sum()) / float(valid.sum()) >= min_foreground_fraction:
            selected.append(index)
    if cache_path is not None:
        cache_path.write_text(json.dumps(selected), encoding="utf-8")
    return selected


def make_voc_class_set_stream(config: VOCSegmentationConfig):
    root = Path(config.root)
    train_base = VOCSegmentation(
        root,
        year="2012",
        image_set="train",
        download=True,
    )
    test_base = VOCSegmentation(
        root,
        year="2012",
        image_set="val",
        download=True,
    )
    animal_classes = _voc_class_indices(VOC_ANIMAL_CLASSES)
    vehicle_classes = _voc_class_indices(VOC_VEHICLE_CLASSES)
    cache_tag = str(config.min_foreground_fraction).replace(".", "p")
    cache_dir = root / ".iewc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_animals = _voc_indices_for_classes(
        train_base,
        animal_classes,
        min_foreground_fraction=config.min_foreground_fraction,
        cache_path=cache_dir / f"train_animals_minfg_{cache_tag}.json",
    )
    train_vehicles = _voc_indices_for_classes(
        train_base,
        vehicle_classes,
        min_foreground_fraction=config.min_foreground_fraction,
        cache_path=cache_dir / f"train_vehicles_minfg_{cache_tag}.json",
    )
    test_animals = _voc_indices_for_classes(
        test_base,
        animal_classes,
        min_foreground_fraction=config.min_foreground_fraction,
        cache_path=cache_dir / f"val_animals_minfg_{cache_tag}.json",
    )
    test_vehicles = _voc_indices_for_classes(
        test_base,
        vehicle_classes,
        min_foreground_fraction=config.min_foreground_fraction,
        cache_path=cache_dir / f"val_vehicles_minfg_{cache_tag}.json",
    )
    if not train_animals or not train_vehicles or not test_animals or not test_vehicles:
        raise ValueError("VOC class-set split produced an empty task")

    train_a = VOCBinaryClassSetSegmentation(
        base=train_base,
        indices=_take(train_animals, config.n_train_per_task, config.seed + 10),
        selected_classes=animal_classes,
        image_size=config.image_size,
    )
    train_b = VOCBinaryClassSetSegmentation(
        base=train_base,
        indices=_take(train_vehicles, config.n_train_per_task, config.seed + 20),
        selected_classes=vehicle_classes,
        image_size=config.image_size,
    )
    test_a = VOCBinaryClassSetSegmentation(
        base=test_base,
        indices=_take(test_animals, config.n_test_per_task, config.seed + 30),
        selected_classes=animal_classes,
        image_size=config.image_size,
    )
    test_b = VOCBinaryClassSetSegmentation(
        base=test_base,
        indices=_take(test_vehicles, config.n_test_per_task, config.seed + 40),
        selected_classes=vehicle_classes,
        image_size=config.image_size,
    )
    metadata = {
        "task_a": "VOC animals",
        "task_b": "VOC vehicles",
        "task_a_classes": list(VOC_ANIMAL_CLASSES),
        "task_b_classes": list(VOC_VEHICLE_CLASSES),
        "available_train_a": len(train_animals),
        "available_train_b": len(train_vehicles),
        "available_test_a": len(test_animals),
        "available_test_b": len(test_vehicles),
    }
    return train_a, train_b, test_a, test_b, metadata


@torch.no_grad()
def foreground_iou(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    intersection = 0.0
    union = 0.0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        pred = model(images).argmax(dim=1)
        valid = masks != 255
        pred_fg = (pred == 1) & valid
        true_fg = (masks == 1) & valid
        intersection += float((pred_fg & true_fg).sum().item())
        union += float((pred_fg | true_fg).sum().item())
    if union <= 0:
        raise ValueError("Cannot compute foreground IoU without foreground pixels")
    return intersection / union


def compute_penalty(model: nn.Module, saved_params, importances):
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name not in saved_params:
            continue
        delta = param - saved_params[name]
        penalty = penalty + (importances[name].data * delta.pow(2)).sum()
    return penalty


def train_task(
    *,
    model: nn.Module,
    dataset: Dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    criterion: nn.Module,
    device: torch.device,
    ewc_lambda: float = 0.0,
    saved_params=None,
    importances=None,
):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    for _ in range(epochs):
        model.train()
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            if importances is not None and saved_params is not None:
                loss = loss + ewc_lambda * compute_penalty(
                    model, saved_params, importances
                )
            loss.backward()
            optimizer.step()
    return optimizer


def _importance_subset(dataset: Dataset, sample_count: int, seed: int):
    if sample_count <= 0 or sample_count >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count]
    return Subset(dataset, indices.tolist())


def run_one(
    *,
    config: PetSegmentationConfig,
    method: str,
    tau: float | None = None,
) -> dict:
    canonical_method = "sequential" if method in SEQUENTIAL_ALIASES else method
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    train_a, train_b, test_a, test_b = make_pet_stream(config)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    model = SmallUNet(config.hidden_channels).to(device)
    optimizer = train_task(
        model=model,
        dataset=train_a,
        epochs=config.train_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        criterion=criterion,
        device=device,
    )
    task_a_iou_after_task_a = foreground_iou(
        model, test_a, batch_size=config.batch_size, device=device
    )

    importances = None
    saved_params = None
    loss_scale_mean = None
    loss_scale_median = None
    ef_summand_trace_values = None
    stored_summand_trace_values = None
    if canonical_method != "sequential":
        importance_dataset = _importance_subset(
            train_a, config.importance_samples, config.seed + 100
        )
        result = ImportanceEstimator(
            kind=canonical_method, tau=0.0 if tau is None else tau
        ).compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=importance_dataset,
            device=device,
            batch_size=config.batch_size,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )
        importances = result.importances
        saved_params = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        loss_scale_mean = float(result.loss_scales.mean().item())
        loss_scale_median = float(result.loss_scales.median().item())
        ef_summand_trace_values = [
            float(value) for value in result.ef_summand_traces.tolist()
        ]
        stored_summand_trace_values = [
            float(value) for value in result.stored_summand_traces.tolist()
        ]

    train_task(
        model=model,
        dataset=train_b,
        epochs=config.train_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        criterion=criterion,
        device=device,
        ewc_lambda=config.ewc_lambda,
        saved_params=saved_params,
        importances=importances,
    )

    task_a_iou_after_task_b = foreground_iou(
        model, test_a, batch_size=config.batch_size, device=device
    )
    task_b_iou_after_task_b = foreground_iou(
        model, test_b, batch_size=config.batch_size, device=device
    )
    output = {
        "method": canonical_method,
        "tau": tau,
        "task_a_iou_after_task_a": task_a_iou_after_task_a,
        "task_a_iou_after_task_b": task_a_iou_after_task_b,
        "task_b_iou_after_task_b": task_b_iou_after_task_b,
        "task_a_iou_forgetting": task_a_iou_after_task_a - task_a_iou_after_task_b,
    }
    if loss_scale_mean is not None:
        output["old_task_loss_scale_mean"] = loss_scale_mean
        output["old_task_loss_scale_median"] = loss_scale_median
        output["old_task_ef_summand_traces"] = ef_summand_trace_values
        output["old_task_stored_summand_traces"] = stored_summand_trace_values
    return output


def run_pet_segmentation_suite(
    *,
    config: PetSegmentationConfig,
    methods: Sequence[str] = ("sequential", "ef", "ief_diag"),
) -> list[dict]:
    results = []
    for method in methods:
        if method == "ief_diag":
            tau_results = [
                run_one(config=config, method=method, tau=tau)
                for tau in config.tau_values
            ]
            best = max(
                tau_results,
                key=lambda item: (
                    item["task_a_iou_after_task_b"],
                    item["task_b_iou_after_task_b"],
                ),
            )
            combined = dict(best)
            combined["tau_results"] = tau_results
            results.append(combined)
        else:
            results.append(run_one(config=config, method=method))
    return results


def run_voc_class_set_one(
    *,
    config: VOCSegmentationConfig,
    method: str,
    tau: float | None = None,
) -> dict:
    canonical_method = "sequential" if method in SEQUENTIAL_ALIASES else method
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    train_a, train_b, test_a, test_b, metadata = make_voc_class_set_stream(config)
    class_weight = torch.tensor(
        [1.0, float(config.foreground_ce_weight)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weight)
    model = SmallUNet(config.hidden_channels).to(device)
    optimizer = train_task(
        model=model,
        dataset=train_a,
        epochs=config.train_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        criterion=criterion,
        device=device,
    )
    task_a_iou_after_task_a = foreground_iou(
        model, test_a, batch_size=config.batch_size, device=device
    )

    importances = None
    saved_params = None
    loss_scale_mean = None
    loss_scale_median = None
    ef_summand_trace_values = None
    stored_summand_trace_values = None
    if canonical_method != "sequential":
        importance_dataset = _importance_subset(
            train_a, config.importance_samples, config.seed + 100
        )
        result = ImportanceEstimator(
            kind=canonical_method, tau=0.0 if tau is None else tau
        ).compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=importance_dataset,
            device=device,
            batch_size=config.batch_size,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )
        importances = result.importances
        saved_params = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        loss_scale_mean = float(result.loss_scales.mean().item())
        loss_scale_median = float(result.loss_scales.median().item())
        ef_summand_trace_values = [
            float(value) for value in result.ef_summand_traces.tolist()
        ]
        stored_summand_trace_values = [
            float(value) for value in result.stored_summand_traces.tolist()
        ]

    train_task(
        model=model,
        dataset=train_b,
        epochs=config.train_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        criterion=criterion,
        device=device,
        ewc_lambda=config.ewc_lambda,
        saved_params=saved_params,
        importances=importances,
    )

    task_a_iou_after_task_b = foreground_iou(
        model, test_a, batch_size=config.batch_size, device=device
    )
    task_b_iou_after_task_b = foreground_iou(
        model, test_b, batch_size=config.batch_size, device=device
    )
    output = {
        "method": canonical_method,
        "tau": tau,
        "task_a_iou_after_task_a": task_a_iou_after_task_a,
        "task_a_iou_after_task_b": task_a_iou_after_task_b,
        "task_b_iou_after_task_b": task_b_iou_after_task_b,
        "task_a_iou_forgetting": task_a_iou_after_task_a - task_a_iou_after_task_b,
        "metadata": metadata,
    }
    if loss_scale_mean is not None:
        output["old_task_loss_scale_mean"] = loss_scale_mean
        output["old_task_loss_scale_median"] = loss_scale_median
        output["old_task_ef_summand_traces"] = ef_summand_trace_values
        output["old_task_stored_summand_traces"] = stored_summand_trace_values
    return output


def run_voc_class_set_segmentation_suite(
    *,
    config: VOCSegmentationConfig,
    methods: Sequence[str] = ("sequential", "ef", "ief_diag"),
) -> list[dict]:
    results = []
    for method in methods:
        if method == "ief_diag":
            tau_results = [
                run_voc_class_set_one(config=config, method=method, tau=tau)
                for tau in config.tau_values
            ]
            best = max(
                tau_results,
                key=lambda item: (
                    item["task_a_iou_after_task_b"],
                    item["task_b_iou_after_task_b"],
                ),
            )
            combined = dict(best)
            combined["tau_results"] = tau_results
            results.append(combined)
        else:
            results.append(run_voc_class_set_one(config=config, method=method))
    return results
