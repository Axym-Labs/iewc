from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    DataCollatorWithPadding,
)

from .diagonal_regularization import (
    DiagonalImportance,
    compute_diagonal_importance,
    diagonal_ewc_penalties,
)


NLPMethod = Literal[
    "sequential",
    "ef",
    "ewc_dr",
    "iewc",
    "iewc_gss",
    "iewc_fromp",
]
NLPAdaptation = Literal["full", "lora"]


@dataclass(frozen=True)
class NLPCLConfig:
    model_name: str = "roberta-base"
    tasks: tuple[str, ...] = ("sst2", "mrpc", "rte")
    seed: int = 0
    max_train_samples: int = 128
    max_eval_samples: int = 128
    epochs_per_task: int = 1
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    ewc_lambda: float = 100.0
    tau: float = 1e-2
    importance_samples: int = 64
    max_length: int = 128
    adaptation: NLPAdaptation = "lora"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    device: str = "cuda"
    synthetic: bool = False


class SyntheticTextDataset(Dataset):
    def __init__(self, *, task_id: int, n_samples: int, length: int, vocab_size: int, seed: int):
        generator = torch.Generator().manual_seed(seed + task_id * 1009)
        self.input_ids = torch.randint(5, vocab_size, (n_samples, length), generator=generator)
        signal = self.input_ids[:, : max(1, length // 4)].float().mean(dim=1)
        threshold = signal.median() + float(task_id) * 0.15
        self.labels = (signal > threshold).long()
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


class DictDataset(Dataset):
    def __init__(self, items: list[dict[str, torch.Tensor]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


GLUE_FIELDS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
}


def _make_model_and_tokenizer(config: NLPCLConfig):
    if config.synthetic:
        model_config = BertConfig(
            vocab_size=30522,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            num_labels=2,
        )
        return BertForSequenceClassification(model_config), None
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    return model, tokenizer


def _configure_lora(model: nn.Module, config: NLPCLConfig) -> None:
    if config.adaptation == "full":
        for param in model.parameters():
            param.requires_grad = True
        return
    if config.adaptation != "lora":
        raise ValueError(f"Unknown NLP adaptation: {config.adaptation}")
    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = ["query", "value"]
    if "t5" in config.model_name.lower():
        target_modules = ["q", "v"]
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        if any(token in name for token in ("classifier", "classification_head", "score")):
            param.requires_grad = True
    return model


def _glue_dataset(task: str, tokenizer, *, split: str, limit: int, max_length: int):
    from datasets import load_dataset

    if task not in GLUE_FIELDS:
        raise ValueError(f"Unsupported GLUE task: {task}")
    try:
        raw = load_dataset("glue", task, split=split)
    except Exception as bare_error:
        try:
            raw = load_dataset("nyu-mll/glue", task, split=split)
        except Exception:
            raise bare_error
    if limit > 0:
        raw = raw.select(range(min(limit, len(raw))))
    field_a, field_b = GLUE_FIELDS[task]
    items = []
    for example in raw:
        if field_b is None:
            encoded = tokenizer(
                example[field_a],
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
        else:
            encoded = tokenizer(
                example[field_a],
                example[field_b],
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
        encoded["labels"] = int(example["label"])
        items.append({key: torch.tensor(value) for key, value in encoded.items()})
    return DictDataset(items)


def _make_tasks(config: NLPCLConfig, tokenizer):
    if config.synthetic:
        tasks = []
        for task_id, name in enumerate(config.tasks):
            tasks.append(
                (
                    name,
                    SyntheticTextDataset(
                        task_id=task_id,
                        n_samples=config.max_train_samples,
                        length=min(config.max_length, 32),
                        vocab_size=30522,
                        seed=config.seed,
                    ),
                    SyntheticTextDataset(
                        task_id=task_id,
                        n_samples=config.max_eval_samples,
                        length=min(config.max_length, 32),
                        vocab_size=30522,
                        seed=config.seed + 999,
                    ),
                )
            )
        return tasks
    return [
        (
            task,
            _glue_dataset(task, tokenizer, split="train", limit=config.max_train_samples, max_length=config.max_length),
            _glue_dataset(
                task,
                tokenizer,
                split="validation",
                limit=config.max_eval_samples,
                max_length=config.max_length,
            ),
        )
        for task in config.tasks
    ]


def _collator(tokenizer):
    if tokenizer is not None:
        return DataCollatorWithPadding(tokenizer)

    def collate(batch):
        return {
            key: torch.stack([item[key] for item in batch])
            for key in batch[0]
        }

    return collate


def _loader(dataset: Dataset, *, batch_size: int, shuffle: bool, collate_fn) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def _loss_output_fn(device: torch.device):
    def fn(model: nn.Module, batch, reverse_output: bool):
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")
        output = model(**batch).logits
        loss_input = -output if reverse_output else output
        return nn.functional.cross_entropy(loss_input, labels), output

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
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = nn.functional.cross_entropy(logits, labels)
            if importances:
                loss = loss + float(ewc_lambda) * diagonal_ewc_penalties(model, importances, device)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
    return last_loss


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")
        pred = model(**batch).logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return correct / float(max(1, total))


def _importance_kind_and_weight(method: NLPMethod) -> tuple[str | None, str]:
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
    raise ValueError(f"Unknown NLP method: {method}")


def run_nlp_cl(config: NLPCLConfig, method: NLPMethod) -> dict:
    torch.manual_seed(config.seed)
    device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model, tokenizer = _make_model_and_tokenizer(config)
    maybe_model = _configure_lora(model, config)
    if maybe_model is not None:
        model = maybe_model
    model.to(device)
    tasks = _make_tasks(config, tokenizer)
    collator = _collator(tokenizer)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    importances: list[DiagonalImportance] = []
    importance_kind, sample_weighting = _importance_kind_and_weight(method)
    accuracy_matrix = []
    train_losses = []
    importance_summaries = []

    for task_id, (task_name, train_dataset, _) in enumerate(tasks):
        train_loss = train_one_task(
            model=model,
            loader=_loader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=collator,
            ),
            optimizer=optimizer,
            device=device,
            epochs=config.epochs_per_task,
            ewc_lambda=config.ewc_lambda,
            importances=importances,
        )
        train_losses.append(train_loss)
        row = []
        for _, _, eval_dataset in tasks:
            row.append(
                evaluate(
                    model,
                    _loader(
                        eval_dataset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        collate_fn=collator,
                    ),
                    device,
                )
            )
        accuracy_matrix.append(row)
        if importance_kind is not None and task_id < len(tasks) - 1:
            importance = compute_diagonal_importance(
                model=model,
                dataloader=_loader(
                    train_dataset,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=collator,
                ),
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
                    "task_name": task_name,
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
        best = max(row[task_id] for row in accuracy_matrix)
        forgetting.append(best - final_accs[task_id])
    return {
        "experiment": "empirical2_nlp_cl",
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
