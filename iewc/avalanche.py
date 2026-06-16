from typing import Optional

import torch
from avalanche.training.plugins import EWCPlugin
from avalanche.training.utils import ParamData
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset

from .config import IEWCConfig
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
        ewc_lambda: float | None = None,
        *,
        config: IEWCConfig | None = None,
        importance_kind: ImportanceKind = "ief_diag",
        tau: float | None = None,
        output_metric: OutputMetricKind | None = None,
        max_importance_samples: int | None = None,
        importance_num_workers: int | None = None,
        mode: str = "separate",
        decay_factor=None,
        keep_importance_data: bool = False,
    ):
        if config is not None:
            config.validate()
            if ewc_lambda is not None and float(ewc_lambda) != config.lambda_:
                raise ValueError("ewc_lambda conflicts with IEWCConfig.lambda_")
            if tau is not None and float(tau) != config.tau:
                raise ValueError("tau conflicts with IEWCConfig.tau")
            if output_metric is not None and output_metric != config.output_metric:
                raise ValueError("output_metric conflicts with IEWCConfig.geometry")
            ewc_lambda = config.lambda_
            tau = config.tau
            output_metric = config.output_metric
        if ewc_lambda is None:
            raise ValueError("ewc_lambda is required unless config is provided")
        if tau is None:
            tau = 0.0
        if output_metric is None:
            output_metric = "euclidean"
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
            config=IEWCConfig(
                lambda_=self.ewc_lambda,
                tau=self.tau,
                geometry=self.output_metric,
                sample_weighting="uniform",
            ),
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
