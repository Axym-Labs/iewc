import argparse
import json
from pathlib import Path

from iewc.empirical2_nlp import NLPCLConfig, run_nlp_cl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--tasks", nargs="+", default=["sst2", "mrpc", "rte"])
    parser.add_argument(
        "--method",
        choices=["sequential", "ef", "ewc_dr", "iewc", "iewc_gss", "iewc_fromp"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=128)
    parser.add_argument("--epochs-per-task", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ewc-lambda", type=float, default=100.0)
    parser.add_argument("--tau", type=float, default=1e-2)
    parser.add_argument("--importance-samples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--adaptation", choices=["full", "lora"], default="lora")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = NLPCLConfig(
        model_name=args.model_name,
        tasks=tuple(args.tasks),
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ewc_lambda=args.ewc_lambda,
        tau=args.tau,
        importance_samples=args.importance_samples,
        max_length=args.max_length,
        adaptation=args.adaptation,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=args.device,
        synthetic=args.synthetic,
    )
    result = run_nlp_cl(config, args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
