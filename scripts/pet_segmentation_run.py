import argparse
import json
from pathlib import Path

from iewc.real_segmentation import (
    PetSegmentationConfig,
    run_pet_segmentation_suite,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Oxford-IIIT Pet foreground-segmentation CL experiments."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--root",
        type=str,
        default="/home/davwis/main/data/oxford-iiit-pet",
    )
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--n-train-per-task", type=int, default=600)
    parser.add_argument("--n-test-per-task", type=int, default=400)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ewc-lambda", type=float, default=10.0)
    parser.add_argument("--importance-samples", type=int, default=256)
    parser.add_argument("--tau", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["sequential", "ef", "ief_diag"],
        choices=["sequential", "ef", "ief_diag"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/empirical-evidence/artifacts/pet-segmentation-seed0.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = PetSegmentationConfig(
        seed=args.seed,
        root=args.root,
        image_size=args.image_size,
        n_train_per_task=args.n_train_per_task,
        n_test_per_task=args.n_test_per_task,
        train_epochs=args.train_epochs,
        hidden_channels=args.hidden_channels,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ewc_lambda=args.ewc_lambda,
        importance_samples=args.importance_samples,
        tau_values=tuple(args.tau),
        device=args.device,
    )
    results = run_pet_segmentation_suite(config=config, methods=tuple(args.methods))
    payload = {
        "experiment": "oxford_iiit_pet_foreground_segmentation",
        "config": {
            "seed": config.seed,
            "root": config.root,
            "image_size": config.image_size,
            "n_train_per_task": config.n_train_per_task,
            "n_test_per_task": config.n_test_per_task,
            "train_epochs": config.train_epochs,
            "hidden_channels": config.hidden_channels,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "ewc_lambda": config.ewc_lambda,
            "importance_samples": config.importance_samples,
            "tau_values": list(config.tau_values),
            "device": config.device,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
