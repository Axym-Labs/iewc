from dataclasses import dataclass
from typing import Dict, Literal

import torch
from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import ParamData, zerolike_params_dict
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from .output_metrics import wasserstein_1d_cdf_dual_quadratic_form
from .config import IEWCConfig


ImportanceKind = Literal["ef", "ewc_dr", "ief_diag"]
OutputMetricKind = Literal["euclidean", "wasserstein_1d_cdf"]


@dataclass
class ImportanceResult:
    importances: Dict[str, ParamData]
    loss_scales: Tensor
    ef_summand_traces: Tensor
    stored_summand_traces: Tensor
    sample_count: int


class ImportanceEstimator:
    """Compute diagonal EWC-style importance estimates with IEWC instrumentation."""

    def __init__(
        self,
        *,
        kind: ImportanceKind,
        tau: float | None = None,
        output_metric: OutputMetricKind | None = None,
        config: IEWCConfig | None = None,
    ):
        if config is not None:
            config.validate()
            if tau is not None and float(tau) != config.tau:
                raise ValueError("tau conflicts with IEWCConfig.tau")
            if output_metric is not None and output_metric != config.output_metric:
                raise ValueError("output_metric conflicts with IEWCConfig.geometry")
            tau = config.tau
            output_metric = config.output_metric
        if tau is None:
            tau = 0.0
        if output_metric is None:
            output_metric = "euclidean"
        if kind not in {"ef", "ewc_dr", "ief_diag"}:
            raise ValueError(f"Unknown importance kind: {kind}")
        if tau < 0:
            raise ValueError("tau must be non-negative")
        if output_metric not in {"euclidean", "wasserstein_1d_cdf"}:
            raise ValueError(f"Unknown output metric: {output_metric}")
        self.kind = kind
        self.tau = float(tau)
        self.output_metric = output_metric

    def _loss_scale(self, output_grad: Tensor) -> Tensor:
        if self.output_metric == "euclidean":
            return output_grad.detach().pow(2).sum()
        if self.output_metric == "wasserstein_1d_cdf":
            if output_grad.shape[0] != 1:
                raise ValueError("Expected a single-sample output gradient")
            return wasserstein_1d_cdf_dual_quadratic_form(output_grad.detach())
        raise ValueError(f"Unknown output metric: {self.output_metric}")

    def compute(
        self,
        *,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        dataset: Dataset,
        device: torch.device,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> ImportanceResult:
        was_training = model.training
        model.eval()
        model.to(device)

        importances = zerolike_params_dict(model)
        loss_scales = []
        ef_summand_traces = []
        stored_summand_traces = []
        sample_count = 0
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

        for batch in dataloader:
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            task_batch = batch[-1].to(device) if len(batch) > 2 else None
            for idx in range(x_batch.shape[0]):
                x = x_batch[idx : idx + 1]
                y = y_batch[idx : idx + 1]
                task_labels = (
                    task_batch[idx : idx + 1] if task_batch is not None else None
                )
                optimizer.zero_grad()
                out = avalanche_forward(model, x, task_labels)
                loss_input = -out if self.kind == "ewc_dr" else out
                loss = criterion(loss_input, y)

                if self.kind == "ief_diag":
                    output_grad = torch.autograd.grad(loss, out, retain_graph=True)[0]
                    loss_scale = self._loss_scale(output_grad)
                    denom = loss_scale + self.tau
                    loss.backward()
                else:
                    loss_scale = torch.ones((), device=device)
                    denom = torch.ones((), device=device)
                    loss.backward()
                grad_squares = {}
                ef_trace = torch.zeros((), device=device)
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    grad_sq = param.grad.detach().pow(2)
                    grad_squares[name] = grad_sq
                    ef_trace = ef_trace + grad_sq.sum()

                stored_by_name = {
                    name: grad_sq / denom for name, grad_sq in grad_squares.items()
                }

                stored_trace = torch.zeros((), device=device)
                for name, stored in stored_by_name.items():
                    importances[name].data += stored
                    stored_trace = stored_trace + stored.sum()

                loss_scales.append(loss_scale.detach().cpu())
                ef_summand_traces.append(ef_trace.detach().cpu())
                stored_summand_traces.append(stored_trace.detach().cpu())
                sample_count += 1

        if sample_count == 0:
            raise ValueError("Cannot compute importances from an empty dataset")

        for importance in importances.values():
            importance.data /= float(sample_count)

        stored_summand_trace_tensor = torch.stack(stored_summand_traces)

        optimizer.zero_grad()
        model.train(was_training)

        return ImportanceResult(
            importances=importances,
            loss_scales=torch.stack(loss_scales),
            ef_summand_traces=torch.stack(ef_summand_traces),
            stored_summand_traces=stored_summand_trace_tensor,
            sample_count=sample_count,
        )
