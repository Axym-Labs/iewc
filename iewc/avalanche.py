from typing import Optional

import torch
from avalanche.training.plugins import EWCPlugin
from avalanche.training.utils import ParamData
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset

from .importance import (
    ImportanceEstimator,
    ImportanceKind,
    ImportanceResult,
    OutputMetricKind,
)


class IEWCPlugin(EWCPlugin):
    """Avalanche EWC plugin with selectable EF, EWC-DR, and IEWC importances."""

    def __init__(
        self,
        ewc_lambda: float,
        *,
        importance_kind: ImportanceKind = "ief_diag",
        tau: float = 0.0,
        output_metric: OutputMetricKind = "euclidean",
        max_importance_samples: int | None = None,
        importance_num_workers: int | None = None,
        mode: str = "separate",
        decay_factor=None,
        keep_importance_data: bool = False,
    ):
        super().__init__(
            ewc_lambda=ewc_lambda,
            mode=mode,
            decay_factor=decay_factor,
            keep_importance_data=keep_importance_data,
        )
        self.importance_kind = importance_kind
        self.tau = float(tau)
        self.output_metric = output_metric
        self.max_importance_samples = max_importance_samples
        self.importance_num_workers = importance_num_workers
        self.last_importance_result: Optional[ImportanceResult] = None

    def compute_importances(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        dataset: Dataset,
        device: torch.device,
        batch_size: int,
        num_workers: int = 0,
    ) -> dict[str, ParamData]:
        estimator = ImportanceEstimator(
            kind=self.importance_kind,
            tau=self.tau,
            output_metric=self.output_metric,
        )
        if (
            self.max_importance_samples is not None
            and len(dataset) > self.max_importance_samples
        ):
            dataset = dataset.subset(range(self.max_importance_samples))
        effective_num_workers = (
            num_workers
            if self.importance_num_workers is None
            else self.importance_num_workers
        )
        result = estimator.compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
            num_workers=effective_num_workers,
            pin_memory=device.type == "cuda",
        )
        self.last_importance_result = result
        return result.importances
