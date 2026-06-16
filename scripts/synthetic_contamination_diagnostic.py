import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from iewc.importance import ImportanceEstimator


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def make_dataset(seed: int, n_samples: int, contamination: float):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, 2, generator=generator)
    true_y = (x[:, 0] + 0.75 * x[:, 1] < 0).long()
    train_y = true_y.clone()
    n_flip = int(round(n_samples * contamination))
    flip_indices = torch.randperm(n_samples, generator=generator)[:n_flip]
    train_y[flip_indices] = 1 - train_y[flip_indices]
    clean_mask = train_y == true_y
    task = torch.zeros(n_samples, dtype=torch.long)
    return TensorDataset(x, train_y, task), true_y, clean_mask


def train_old_task(model, dataset, epochs: int, lr: float):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y, _ in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return optimizer, criterion


def summarize(values: torch.Tensor):
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p90": float(values.quantile(0.9).item()),
        "max": float(values.max().item()),
    }


def summarize_dataset(model, dataset, true_y, clean_mask, optimizer, criterion, tau: float):
    ief_result = ImportanceEstimator(kind="ief_diag", tau=tau).compute(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataset=dataset,
        device=torch.device("cpu"),
        batch_size=64,
    )
    with torch.no_grad():
        x = dataset.tensors[0]
        train_y = dataset.tensors[1]
        logits = model(x)
        pred = logits.argmax(dim=1)
        train_acc = (pred == train_y).float().mean()
        true_acc = (pred == true_y).float().mean()

    loss_scales = ief_result.loss_scales
    clean_scales = loss_scales[clean_mask]
    contaminated_scales = loss_scales[~clean_mask]
    ratio = contaminated_scales.mean() / clean_scales.mean()
    return {
        "accuracy_on_corrupted_labels": float(train_acc.item()),
        "accuracy_on_true_labels": float(true_acc.item()),
        "clean_count": int(clean_mask.sum().item()),
        "contaminated_count": int((~clean_mask).sum().item()),
        "loss_scale_clean": summarize(clean_scales),
        "loss_scale_contaminated": summarize(contaminated_scales),
        "contaminated_to_clean_mean_loss_scale_ratio": float(ratio.item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--n-heldout", type=int, default=512)
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset, true_y, clean_mask = make_dataset(
        args.seed, args.n_samples, args.contamination
    )
    model = MLP()
    optimizer, criterion = train_old_task(model, dataset, args.epochs, args.lr)

    train_summary = summarize_dataset(
        model, dataset, true_y, clean_mask, optimizer, criterion, args.tau
    )
    heldout_dataset, heldout_true_y, heldout_clean_mask = make_dataset(
        args.seed + 1000, args.n_heldout, args.contamination
    )
    heldout_summary = summarize_dataset(
        model,
        heldout_dataset,
        heldout_true_y,
        heldout_clean_mask,
        optimizer,
        criterion,
        args.tau,
    )

    report = {
        "hypothesis": (
            "EF implicit sample weights r^T r are larger on contaminated "
            "old-task samples than on clean samples after old-task fitting."
        ),
        "seed": args.seed,
        "n_samples": args.n_samples,
        "n_heldout": args.n_heldout,
        "contamination": args.contamination,
        "epochs": args.epochs,
        "lr": args.lr,
        "tau": args.tau,
        "train": train_summary,
        "heldout": heldout_summary,
        "train_accuracy_on_corrupted_labels": train_summary[
            "accuracy_on_corrupted_labels"
        ],
        "accuracy_on_true_labels": train_summary["accuracy_on_true_labels"],
        "clean_count": train_summary["clean_count"],
        "contaminated_count": train_summary["contaminated_count"],
        "loss_scale_clean": train_summary["loss_scale_clean"],
        "loss_scale_contaminated": train_summary["loss_scale_contaminated"],
        "contaminated_to_clean_mean_loss_scale_ratio": train_summary[
            "contaminated_to_clean_mean_loss_scale_ratio"
        ],
        "interpretation": (
            "supported"
            if train_summary["contaminated_to_clean_mean_loss_scale_ratio"] > 2.0
            and heldout_summary["contaminated_to_clean_mean_loss_scale_ratio"] > 2.0
            else "inconclusive_or_rejected_under_this_configuration"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
