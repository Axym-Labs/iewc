import argparse
import json
from pathlib import Path

from iewc.empirical2_vision import VisionCLConfig, run_vision_cl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar100", "tiny_imagenet", "imagenet_r"], default="cifar100")
    parser.add_argument("--data-root", default="/home/davwis/main/data")
    parser.add_argument("--model-name", default="vit_tiny_patch16_224")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--adaptation", choices=["full", "lora"], default="full")
    parser.add_argument("--method", choices=["sequential", "ef", "ewc_dr", "iewc", "iewc_gss", "iewc_fromp"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-tasks", type=int, default=2)
    parser.add_argument("--classes-per-task", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-samples-per-class", type=int, default=8)
    parser.add_argument("--test-samples-per-class", type=int, default=8)
    parser.add_argument("--epochs-per-task", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ewc-lambda", type=float, default=100.0)
    parser.add_argument("--tau", type=float, default=1e-2)
    parser.add_argument("--importance-samples", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--evaluation", choices=["task_aware", "class_incremental"], default="task_aware")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = VisionCLConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        model_name=args.model_name,
        pretrained=args.pretrained,
        adaptation=args.adaptation,
        seed=args.seed,
        n_tasks=args.n_tasks,
        classes_per_task=args.classes_per_task,
        image_size=args.image_size,
        train_samples_per_class=args.train_samples_per_class,
        test_samples_per_class=args.test_samples_per_class,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ewc_lambda=args.ewc_lambda,
        tau=args.tau,
        importance_samples=args.importance_samples,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        num_workers=args.num_workers,
        device=args.device,
        download=args.download,
        evaluation=args.evaluation,
    )
    result = run_vision_cl(config, args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
