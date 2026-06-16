import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Config:
    seed: int = 0
    n_train: int = 256
    n_test: int = 512
    epochs: int = 40
    hidden_size: int = 64
    batch_size: int = 64
    learning_rate: float = 0.05
    tau: float = 1e-3
    feature_weight: float = 1.0


class FeatureMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.input = nn.Linear(2, hidden_size)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        features = self.activation(self.input(x))
        logits = self.output(features)
        if return_features:
            return logits, features
        return logits


def make_domain_shift(
    *, seed: int, n_train: int, n_test: int, device: torch.device
) -> tuple[TensorDataset, TensorDataset, tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def sample(n_samples: int, angle: float):
        x = torch.randn(n_samples, 2, generator=generator)
        direction = torch.tensor([math.cos(angle), math.sin(angle)])
        y = ((x @ direction) > 0).long()
        return x.to(device), y.to(device)

    task_a_train = sample(n_train, 0.0)
    task_b_train = sample(n_train, 1.2)
    task_a_test = sample(n_test, 0.0)
    task = torch.zeros(n_train, dtype=torch.long, device=device)
    return (
        TensorDataset(task_a_train[0], task_a_train[1], task),
        TensorDataset(task_b_train[0], task_b_train[1], task.clone()),
        task_a_test,
    )


def train_task_a(
    model: FeatureMLP,
    dataset: TensorDataset,
    *,
    config: Config,
):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    for _ in range(config.epochs):
        for x, y, _ in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return optimizer


@torch.no_grad()
def save_features(model: FeatureMLP, dataset: TensorDataset) -> torch.Tensor:
    model.eval()
    features = []
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    for x, _, _ in loader:
        _, hidden = model(x, return_features=True)
        features.append(hidden.detach())
    return torch.cat(features, dim=0)


def zero_like_importances(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(param.detach())
        for name, param in model.named_parameters()
    }


def flatten_importances(importances: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in importances.values()])


def compute_iewc_importances(
    model: FeatureMLP,
    optimizer: torch.optim.Optimizer,
    dataset: TensorDataset,
    saved_features: torch.Tensor | None,
    *,
    config: Config,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    importances = zero_like_importances(model)
    loss_scales = []
    feature_loss_values = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for idx, (x, y, _) in enumerate(loader):
        optimizer.zero_grad()
        logits, hidden = model(x, return_features=True)
        loss = criterion(logits, y)
        if saved_features is not None:
            feature_target = saved_features[idx : idx + 1].to(hidden.device)
            feature_loss = nn.functional.mse_loss(hidden, feature_target)
            feature_loss_values.append(float(feature_loss.detach().cpu().item()))
            loss = loss + config.feature_weight * feature_loss
        output_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        loss_scale = output_grad.detach().pow(2).sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            importances[name] += param.grad.detach().pow(2) / (loss_scale + config.tau)
        loss_scales.append(loss_scale.detach().cpu())
    for name in importances:
        importances[name] /= float(len(dataset))
    mean_feature_loss = (
        float(torch.tensor(feature_loss_values).mean().item())
        if feature_loss_values
        else 0.0
    )
    return importances, torch.stack(loss_scales), mean_feature_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/empirical-evidence/artifacts/feature-preservation-ief-diagnostic.json"
        ),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this task arc diagnostic")
    device = torch.device("cuda")
    config = Config(
        seed=args.seed,
        epochs=args.epochs,
        feature_weight=args.feature_weight,
    )
    torch.manual_seed(config.seed)
    task_a_train, _, task_a_test = make_domain_shift(
        seed=config.seed,
        n_train=config.n_train,
        n_test=config.n_test,
        device=device,
    )
    model = FeatureMLP(config.hidden_size).to(device)
    optimizer = train_task_a(model, task_a_train, config=config)
    saved_features = save_features(model, task_a_train)

    base_importances, base_scales, _ = compute_iewc_importances(
        model,
        optimizer,
        task_a_train,
        None,
        config=config,
    )
    feature_importances, feature_scales, mean_feature_loss = compute_iewc_importances(
        model,
        optimizer,
        task_a_train,
        saved_features,
        config=config,
    )
    base_flat = flatten_importances(base_importances)
    feature_flat = flatten_importances(feature_importances)
    diff = (feature_flat - base_flat).abs()

    with torch.no_grad():
        task_a_accuracy = float(
            (model(task_a_test[0]).argmax(dim=1) == task_a_test[1])
            .float()
            .mean()
            .item()
        )
    payload = {
        "experiment": "feature_preservation_ief_diagnostic",
        "device": torch.cuda.get_device_name(device),
        "config": config.__dict__,
        "task_a_accuracy_after_task_a": task_a_accuracy,
        "mean_feature_anchor_loss_at_theta_a": mean_feature_loss,
        "base_importance_trace": float(base_flat.sum().item()),
        "feature_importance_trace": float(feature_flat.sum().item()),
        "importance_max_abs_difference": float(diff.max().item()),
        "importance_l1_difference": float(diff.sum().item()),
        "loss_scale_max_abs_difference": float(
            (feature_scales - base_scales).abs().max().item()
        ),
        "interpretation": (
            "Feature-anchor MSE is zero at theta_A, so its first derivative is zero; "
            "the IEF importance equals ordinary IEWC up to numerical precision."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
