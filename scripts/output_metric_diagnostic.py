import argparse
import json
from pathlib import Path

import torch

from iewc.output_metrics import (
    euclidean_squared_distance,
    wasserstein_1d_cdf_squared_distance,
)


def make_shifted_distributions(
    *,
    n_samples: int,
    n_bins: int,
    shift: int,
    sigma: float,
    seed: int,
):
    generator = torch.Generator().manual_seed(seed)
    grid = torch.arange(n_bins, dtype=torch.float32)
    low = max(0, -shift)
    high = min(n_bins - 1, n_bins - 1 - shift)
    centers = torch.randint(low, high + 1, (n_samples,), generator=generator)
    old = []
    new = []
    for center in centers:
        old_logits = -0.5 * ((grid - float(center)) / sigma).pow(2)
        new_logits = -0.5 * ((grid - float(center + shift)) / sigma).pow(2)
        old.append(torch.softmax(old_logits, dim=0))
        new.append(torch.softmax(new_logits, dim=0))
    return torch.stack(old), torch.stack(new)


def summarize(old: torch.Tensor, new: torch.Tensor):
    euclidean = euclidean_squared_distance(old, new)
    wasserstein = wasserstein_1d_cdf_squared_distance(old, new)
    return {
        "euclidean_mean": float(euclidean.mean().item()),
        "euclidean_std": float(euclidean.std(unbiased=False).item()),
        "wasserstein_1d_cdf_mean": float(wasserstein.mean().item()),
        "wasserstein_1d_cdf_std": float(wasserstein.std(unbiased=False).item()),
        "wasserstein_to_euclidean_ratio": float(
            wasserstein.mean().item() / euclidean.mean().item()
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Euclidean and 1D Wasserstein-style output metrics."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--n-bins", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--local-shift", type=int, default=1)
    parser.add_argument("--far-shift", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/empirical-evidence/artifacts/output-metric-diagnostic.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    old_local, new_local = make_shifted_distributions(
        n_samples=args.n_samples,
        n_bins=args.n_bins,
        shift=args.local_shift,
        sigma=args.sigma,
        seed=args.seed,
    )
    old_far, new_far = make_shifted_distributions(
        n_samples=args.n_samples,
        n_bins=args.n_bins,
        shift=args.far_shift,
        sigma=args.sigma,
        seed=args.seed + 1,
    )
    payload = {
        "experiment": "ordered_output_metric_diagnostic",
        "config": {
            "seed": args.seed,
            "n_samples": args.n_samples,
            "n_bins": args.n_bins,
            "sigma": args.sigma,
            "local_shift": args.local_shift,
            "far_shift": args.far_shift,
        },
        "local_shift": summarize(old_local, new_local),
        "far_shift": summarize(old_far, new_far),
        "interpretation": (
            "The 1D Wasserstein-style CDF metric gives lower loss than "
            "Euclidean for adjacent ordered-output shifts, while increasing "
            "for longer transport distances."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
