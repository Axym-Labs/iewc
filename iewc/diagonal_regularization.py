from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import torch
from torch import Tensor, nn


DiagonalKind = Literal["ef", "ewc_dr", "iewc"]
SampleWeighting = Literal["uniform", "gss_residual", "fromp_trace"]

LossOutputFn = Callable[[nn.Module, object, bool], tuple[Tensor, Tensor]]


@dataclass
class DiagonalImportance:
    centers: dict[str, Tensor]
    importances: dict[str, Tensor]
    loss_scales: Tensor
    sample_weights: Tensor
    stored_summand_traces: Tensor
    sample_count: int


def trainable_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def _flatten_grads(params: Iterable[tuple[str, nn.Parameter]]) -> Tensor:
    rows = []
    for _, param in params:
        if param.grad is None:
            rows.append(torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1))
        else:
            rows.append(param.grad.detach().reshape(-1))
    if not rows:
        raise ValueError("No trainable parameters found")
    return torch.cat(rows)


def _squared_grad_dict(params: Iterable[tuple[str, nn.Parameter]], denom: Tensor) -> dict[str, Tensor]:
    result = {}
    for name, param in params:
        if param.grad is None:
            result[name] = torch.zeros_like(param.detach())
        else:
            result[name] = param.grad.detach().pow(2) / denom.detach()
    return result


def _softmax_hessian_trace(output: Tensor) -> Tensor:
    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError("fromp_trace weighting expects single-sample classification logits")
    probabilities = output.detach().softmax(dim=-1)
    return (1.0 - probabilities.pow(2).sum(dim=-1)).squeeze(0).clamp_min(0.0)


def _gss_residual_weights(
    rows: list[Tensor],
    *,
    max_basis: int = 64,
    eps: float = 1e-12,
    device: torch.device | None = None,
) -> Tensor:
    work_device = device if device is not None else rows[0].device
    matrix = torch.stack([row.detach().float() for row in rows]).to(work_device)
    basis: Tensor | None = None
    weights = []
    for row in matrix:
        residual = row
        if basis is not None:
            coeffs = torch.mv(basis, residual)
            residual = residual - torch.mv(basis.t(), coeffs)
        weight = residual.pow(2).sum()
        weights.append(weight.detach().cpu())
        if weight.item() > eps and (basis is None or basis.shape[0] < max_basis):
            vector = (residual / torch.sqrt(weight + eps)).detach().unsqueeze(0)
            basis = vector if basis is None else torch.cat([basis, vector], dim=0)
    return torch.stack(weights)


def _normalize_weights(weights: Tensor) -> Tensor:
    mean = weights.mean()
    if not torch.isfinite(mean) or mean.item() <= 0.0:
        return torch.ones_like(weights)
    return weights / mean


def compute_diagonal_importance(
    *,
    model: nn.Module,
    dataloader,
    loss_output_fn: LossOutputFn,
    device: torch.device,
    kind: DiagonalKind,
    tau: float = 1e-2,
    sample_weighting: SampleWeighting = "uniform",
    max_samples: int | None = None,
    normalize_sample_weights: bool = True,
    gss_max_basis: int = 64,
) -> DiagonalImportance:
    """Compute a trainable-parameter diagonal EF/EWC-DR/IEWC importance.

    The caller supplies a single-sample-compatible `loss_output_fn`. It receives
    `reverse_output=True` for EWC-DR so classification code can apply the same
    criterion to negated logits.
    """

    if kind not in {"ef", "ewc_dr", "iewc"}:
        raise ValueError(f"Unknown diagonal importance kind: {kind}")
    if sample_weighting not in {"uniform", "gss_residual", "fromp_trace"}:
        raise ValueError(f"Unknown sample weighting: {sample_weighting}")
    if tau < 0:
        raise ValueError("tau must be non-negative")

    was_training = model.training
    model.eval()
    model.to(device)
    params = trainable_named_parameters(model)
    if not params:
        raise ValueError("Cannot compute importance without trainable parameters")

    centers = {name: param.detach().clone() for name, param in params}
    sample_summands: list[dict[str, Tensor]] = []
    normalized_rows: list[Tensor] = []
    loss_scales = []
    immediate_weights = []
    stored_traces = []

    for sample_index, batch in enumerate(dataloader):
        if max_samples is not None and sample_index >= max_samples:
            break
        model.zero_grad(set_to_none=True)
        loss, output = loss_output_fn(model, batch, kind == "ewc_dr")
        if kind == "iewc":
            output_grad = torch.autograd.grad(loss, output, retain_graph=True)[0]
            loss_scale = output_grad.detach().pow(2).sum()
            denom = loss_scale + float(tau)
        else:
            loss_scale = torch.ones((), device=device)
            denom = torch.ones((), device=device)
        loss.backward()

        summand = _squared_grad_dict(params, denom)
        flat_grad = _flatten_grads(params) / torch.sqrt(denom.detach())
        sample_summands.append({name: value.detach().cpu() for name, value in summand.items()})
        normalized_rows.append(flat_grad.detach().float().cpu())
        loss_scales.append(loss_scale.detach().cpu())
        if sample_weighting == "fromp_trace":
            immediate_weights.append(_softmax_hessian_trace(output).detach().cpu())
        else:
            immediate_weights.append(torch.ones((), dtype=torch.float32))

    if not sample_summands:
        raise ValueError("Cannot compute importances from an empty dataloader")

    if sample_weighting == "gss_residual":
        sample_weights = _gss_residual_weights(normalized_rows, max_basis=gss_max_basis, device=device)
    else:
        sample_weights = torch.stack([weight.float() for weight in immediate_weights])
    if normalize_sample_weights:
        sample_weights = _normalize_weights(sample_weights)

    importances = {
        name: torch.zeros_like(value, dtype=value.dtype)
        for name, value in sample_summands[0].items()
    }
    for summand, weight in zip(sample_summands, sample_weights):
        stored_trace = torch.zeros((), dtype=torch.float32)
        for name, value in summand.items():
            weighted = value * weight
            importances[name] += weighted
            stored_trace = stored_trace + weighted.float().sum()
        stored_traces.append(stored_trace)

    count = float(len(sample_summands))
    importances = {name: value / count for name, value in importances.items()}
    model.zero_grad(set_to_none=True)
    model.train(was_training)

    return DiagonalImportance(
        centers={name: value.detach().cpu() for name, value in centers.items()},
        importances=importances,
        loss_scales=torch.stack(loss_scales),
        sample_weights=sample_weights,
        stored_summand_traces=torch.stack(stored_traces),
        sample_count=len(sample_summands),
    )


def diagonal_ewc_penalty(model: nn.Module, importance: DiagonalImportance, device: torch.device) -> Tensor:
    penalty = torch.zeros((), device=device)
    for name, param in trainable_named_parameters(model):
        if name not in importance.importances:
            continue
        center = importance.centers[name].to(device=device, dtype=param.dtype)
        diagonal = importance.importances[name].to(device=device, dtype=param.dtype)
        penalty = penalty + (diagonal * (param - center).pow(2)).sum()
    return penalty


def diagonal_ewc_penalties(
    model: nn.Module,
    importances: Iterable[DiagonalImportance],
    device: torch.device,
) -> Tensor:
    total = torch.zeros((), device=device)
    for importance in importances:
        total = total + diagonal_ewc_penalty(model, importance, device)
    return total
