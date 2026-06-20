from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .diagonal_regularization import (
    DiagonalImportance,
    compute_diagonal_importance,
    diagonal_ewc_penalties,
    trainable_named_parameters,
)


TraceMethod = Literal["sequential", "ef", "ewc_dr", "iewc", "iewc_gss"]
TraceDType = Literal["auto", "float32", "bfloat16", "float16"]
TraceAnswerMode = Literal["full", "choice"]


ACCURACY_TASKS = {"C-STANCE", "FOMC", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds"}
CHOICE_TASKS = {
    "C-STANCE": "ABC",
    "FOMC": "ABC",
    "ScienceQA": "ABCDE",
}


@dataclass(frozen=True)
class TraceCLConfig:
    data_root: str = "/home/davwis/main/data/trace/TRACE-Benchmark/LLM-CL-Benchmark_500"
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tasks: tuple[str, ...] = ("C-STANCE", "FOMC", "ScienceQA")
    seed: int = 0
    max_train_samples: int = 128
    max_eval_samples: int = 64
    epochs_per_task: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    ewc_lambda: float = 100.0
    tau: float = 1e-2
    importance_samples: int = 16
    max_prompt_length: int = 256
    max_answer_length: int = 32
    generation_max_new_tokens: int = 24
    answer_mode: TraceAnswerMode = "full"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    dtype: TraceDType = "auto"
    device: str = "cuda"


class TracePromptDataset(Dataset):
    def __init__(
        self,
        path: Path,
        *,
        task_name: str,
        answer_mode: TraceAnswerMode,
        limit: int,
        seed: int,
        shuffle: bool,
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        if shuffle:
            generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(data), generator=generator).tolist()
            data = [data[idx] for idx in order]
        if limit > 0:
            data = data[: min(limit, len(data))]
        self.items = [
            {
                "prompt": str(item["prompt"]),
                "answer": _training_answer(task_name, str(item["answer"]), answer_mode),
                "full_answer": str(item["answer"]),
            }
            for item in data
            if item.get("prompt") is not None and item.get("answer") is not None
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.items[idx]


class TraceCollator:
    def __init__(self, tokenizer, *, max_prompt_length: int, max_answer_length: int):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_answer_length = max_answer_length

    def _encode_pair(self, prompt: str, answer: str) -> tuple[list[int], list[int], int]:
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length,
        )["input_ids"]
        answer_ids = self.tokenizer(
            answer,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_answer_length,
        )["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [self.tokenizer.eos_token_id]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return input_ids, labels, len(prompt_ids)

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, torch.Tensor | list[str]]:
        encoded = [self._encode_pair(item["prompt"], item["answer"]) for item in batch]
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(input_ids) for input_ids, _, _ in encoded)
        input_rows = []
        label_rows = []
        masks = []
        prompt_lens = []
        for input_ids, labels, prompt_len in encoded:
            pad_len = max_len - len(input_ids)
            input_rows.append(input_ids + [pad_id] * pad_len)
            label_rows.append(labels + [-100] * pad_len)
            masks.append([1] * len(input_ids) + [0] * pad_len)
            prompt_lens.append(prompt_len)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "prompt_lengths": torch.tensor(prompt_lens, dtype=torch.long),
            "prompts": [item["prompt"] for item in batch],
            "answers": [item["answer"] for item in batch],
        }


def _torch_dtype(dtype: TraceDType, device: torch.device):
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def _make_model_and_tokenizer(config: TraceCLConfig, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=_torch_dtype(config.dtype, device),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def _make_tasks(config: TraceCLConfig):
    root = Path(config.data_root)
    tasks = []
    for task_id, task_name in enumerate(config.tasks):
        task_root = root / task_name
        if not task_root.exists():
            raise FileNotFoundError(f"TRACE task directory not found: {task_root}")
        tasks.append(
            (
                task_name,
                TracePromptDataset(
                    task_root / "train.json",
                    task_name=task_name,
                    answer_mode=config.answer_mode,
                    limit=config.max_train_samples,
                    seed=config.seed + task_id * 1009,
                    shuffle=True,
                ),
                TracePromptDataset(
                    task_root / "eval.json",
                    task_name=task_name,
                    answer_mode=config.answer_mode,
                    limit=config.max_eval_samples,
                    seed=config.seed + task_id * 1009 + 17,
                    shuffle=False,
                ),
            )
        )
    return tasks


def _loader(dataset: Dataset, *, batch_size: int, shuffle: bool, collate_fn) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def _tensor_batch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor) and key != "prompt_lengths"
    }


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, *, reverse_output: bool = False) -> torch.Tensor:
    if reverse_output:
        logits = -logits
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return nn.functional.cross_entropy(
        shifted_logits.float().view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
    )


def _supervised_logits_and_labels(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    mask = shifted_labels != -100
    if not bool(mask.any()):
        raise ValueError("TRACE batch has no supervised answer tokens")
    return shifted_logits[mask].contiguous(), shifted_labels[mask].contiguous()


def _loss_output_fn(device: torch.device):
    def fn(model: nn.Module, batch, reverse_output: bool):
        tensors = _tensor_batch(batch, device)
        labels = tensors.pop("labels")
        logits = model(**tensors).logits
        supervised_logits, supervised_labels = _supervised_logits_and_labels(logits, labels)
        if reverse_output:
            supervised_logits = -supervised_logits
        loss = nn.functional.cross_entropy(supervised_logits.float(), supervised_labels)
        return loss, supervised_logits

    return fn


def _importance_to_training_device(
    importance: DiagonalImportance,
    *,
    model: nn.Module,
    device: torch.device,
) -> DiagonalImportance:
    param_dtypes = {name: param.dtype for name, param in trainable_named_parameters(model)}
    return DiagonalImportance(
        centers={
            name: value.to(device=device, dtype=param_dtypes.get(name, value.dtype))
            for name, value in importance.centers.items()
        },
        importances={
            name: value.to(device=device, dtype=param_dtypes.get(name, value.dtype))
            for name, value in importance.importances.items()
        },
        loss_scales=importance.loss_scales,
        sample_weights=importance.sample_weights,
        stored_summand_traces=importance.stored_summand_traces,
        sample_count=importance.sample_count,
    )


def _importance_kind_and_weight(method: TraceMethod) -> tuple[str | None, str]:
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
    raise ValueError(f"Unknown TRACE method: {method}")


def train_one_task(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    gradient_accumulation_steps: int,
    ewc_lambda: float,
    importances: list[DiagonalImportance],
) -> float:
    model.train()
    last_loss = 0.0
    accumulation = max(1, int(gradient_accumulation_steps))
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            tensors = _tensor_batch(batch, device)
            labels = tensors.pop("labels")
            logits = model(**tensors).logits
            loss = _causal_lm_loss(logits, labels)
            if importances:
                loss = loss + float(ewc_lambda) * diagonal_ewc_penalties(model, importances, device)
            (loss / accumulation).backward()
            if step % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            last_loss = float(loss.detach().cpu())
        if len(loader) % accumulation != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return last_loss


def _letter_answer(text: str, letters: str) -> str:
    match = re.search(rf"([{letters}])", text.strip().upper())
    if match:
        return match.group(1)
    stripped = text.strip().upper()
    return stripped[:1]


def _numeric_answer(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _score_prediction(task_name: str, prediction: str, target: str) -> float:
    if task_name == "C-STANCE":
        labels = {"支持": "A", "反对": "B", "中立": "C"}
        predicted = next((value for key, value in labels.items() if key in prediction), None)
        if predicted is None:
            predicted = _letter_answer(prediction, "ABC")
        return float(predicted == _letter_answer(target, "ABC"))
    if task_name == "FOMC":
        lower = prediction.lower()
        labels = {"dovish": "A", "hawkish": "B", "neutral": "C"}
        predicted = next((value for key, value in labels.items() if key in lower), None)
        if predicted is None:
            predicted = _letter_answer(prediction, "ABC")
        return float(predicted == _letter_answer(target, "ABC"))
    if task_name == "ScienceQA":
        return float(_letter_answer(prediction, "ABCDE") == _letter_answer(target, "ABCDE"))
    if task_name in {"NumGLUE-cm", "NumGLUE-ds"}:
        pred_num = _numeric_answer(prediction)
        target_num = _numeric_answer(target)
        if pred_num is None or target_num is None:
            return 0.0
        return float(math.isclose(pred_num, target_num, rel_tol=1e-3, abs_tol=1e-3))
    return float(prediction.strip() == target.strip())


def _training_answer(task_name: str, answer: str, answer_mode: TraceAnswerMode) -> str:
    if answer_mode == "full":
        return answer
    if answer_mode != "choice":
        raise ValueError(f"Unknown TRACE answer mode: {answer_mode}")
    if task_name in CHOICE_TASKS:
        return _target_letter(task_name, answer)
    return answer


def _target_letter(task_name: str, target: str) -> str:
    if task_name == "C-STANCE":
        return _letter_answer(target, "ABC")
    if task_name == "FOMC":
        return _letter_answer(target, "ABC")
    if task_name == "ScienceQA":
        return _letter_answer(target, "ABCDE")
    raise ValueError(f"Task {task_name} is not a multiple-choice TRACE task")


def _choice_token_ids(tokenizer, letters: str) -> dict[str, int]:
    token_ids = {}
    for letter in letters:
        encoded = tokenizer(letter, add_special_tokens=False)["input_ids"]
        if not encoded:
            raise ValueError(f"Could not tokenize TRACE choice letter: {letter}")
        token_ids[letter] = int(encoded[0])
    return token_ids


@torch.no_grad()
def evaluate(
    model: nn.Module,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    *,
    task_name: str,
    max_new_tokens: int,
) -> dict:
    model.eval()
    nll_total = 0.0
    nll_tokens = 0
    scores = []
    examples = []
    choice_ids = _choice_token_ids(tokenizer, CHOICE_TASKS[task_name]) if task_name in CHOICE_TASKS else None
    for batch in loader:
        tensors = _tensor_batch(batch, device)
        labels = tensors.pop("labels")
        logits = model(**tensors).logits
        loss = _causal_lm_loss(logits, labels)
        answer_tokens = int((labels[:, 1:] != -100).sum().item())
        nll_total += float(loss.detach().cpu()) * max(1, answer_tokens)
        nll_tokens += max(1, answer_tokens)

        prompts = batch["prompts"]
        if choice_ids is not None:
            prompt_lengths = batch["prompt_lengths"].to(device)
            next_token_logits = logits[
                torch.arange(logits.shape[0], device=device),
                (prompt_lengths - 1).clamp_min(0),
            ]
            choice_letters = list(choice_ids)
            choice_token_ids = torch.tensor([choice_ids[letter] for letter in choice_letters], device=device)
            choice_scores = next_token_logits.index_select(dim=-1, index=choice_token_ids)
            predictions = [
                choice_letters[int(index)]
                for index in choice_scores.argmax(dim=-1).detach().cpu().tolist()
            ]
        else:
            tokenizer.padding_side = "left"
            encoded = tokenizer(
                prompts,
                add_special_tokens=False,
                truncation=True,
                max_length=loader.collate_fn.max_prompt_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            predictions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            tokenizer.padding_side = "right"
        for prompt, prediction, target in zip(prompts, predictions, batch["answers"]):
            if choice_ids is not None:
                score = float(str(prediction) == _target_letter(task_name, target))
            else:
                score = _score_prediction(task_name, prediction, target)
            scores.append(score)
            if len(examples) < 3:
                examples.append(
                    {
                        "prompt": prompt[:500],
                        "target": target,
                        "prediction": prediction.strip(),
                        "score": score,
                    }
                )
    return {
        "score": float(sum(scores) / max(1, len(scores))),
        "nll": float(nll_total / max(1, nll_tokens)),
        "examples": examples,
    }


def run_trace_cl(config: TraceCLConfig, method: TraceMethod) -> dict:
    torch.manual_seed(config.seed)
    device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model, tokenizer = _make_model_and_tokenizer(config, device)
    model.to(device)
    tasks = _make_tasks(config)
    collator = TraceCollator(
        tokenizer,
        max_prompt_length=config.max_prompt_length,
        max_answer_length=config.max_answer_length,
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    importances: list[DiagonalImportance] = []
    importance_kind, sample_weighting = _importance_kind_and_weight(method)
    score_matrix = []
    nll_matrix = []
    train_losses = []
    examples_after_final = {}
    importance_summaries = []

    for task_id, (task_name, train_dataset, _) in enumerate(tasks):
        train_loss = train_one_task(
            model=model,
            loader=_loader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collator),
            optimizer=optimizer,
            device=device,
            epochs=config.epochs_per_task,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            ewc_lambda=config.ewc_lambda,
            importances=importances,
        )
        train_losses.append(train_loss)

        score_row = []
        nll_row = []
        for eval_name, _, eval_dataset in tasks:
            metrics = evaluate(
                model,
                tokenizer,
                _loader(eval_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collator),
                device,
                task_name=eval_name,
                max_new_tokens=config.generation_max_new_tokens,
            )
            score_row.append(metrics["score"])
            nll_row.append(metrics["nll"])
            if task_id == len(tasks) - 1:
                examples_after_final[eval_name] = metrics["examples"]
        score_matrix.append(score_row)
        nll_matrix.append(nll_row)

        if importance_kind is not None and task_id < len(tasks) - 1:
            importance = compute_diagonal_importance(
                model=model,
                dataloader=_loader(train_dataset, batch_size=1, shuffle=False, collate_fn=collator),
                loss_output_fn=_loss_output_fn(device),
                device=device,
                kind=importance_kind,
                tau=config.tau,
                sample_weighting=sample_weighting,
                max_samples=config.importance_samples,
            )
            importance = _importance_to_training_device(importance, model=model, device=device)
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

    final_scores = score_matrix[-1]
    final_nlls = nll_matrix[-1]
    forgetting = []
    for task_id in range(len(tasks) - 1):
        best = max(row[task_id] for row in score_matrix)
        forgetting.append(best - final_scores[task_id])

    return {
        "experiment": "empirical2_trace_cl",
        "config": asdict(config),
        "method": method,
        "tasks": list(config.tasks),
        "n_trainable_parameters": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
        "n_total_parameters": int(sum(param.numel() for param in model.parameters())),
        "score_matrix": score_matrix,
        "nll_matrix": nll_matrix,
        "final_task_scores": final_scores,
        "final_task_nlls": final_nlls,
        "final_avg_score": float(sum(final_scores) / len(final_scores)),
        "final_avg_nll": float(sum(final_nlls) / len(final_nlls)),
        "avg_forgetting_score": float(sum(forgetting) / len(forgetting)) if forgetting else 0.0,
        "train_losses": train_losses,
        "importance_summaries": importance_summaries,
        "examples_after_final": examples_after_final,
        "accuracy_tasks": sorted(ACCURACY_TASKS),
    }
