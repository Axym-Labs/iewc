from dataclasses import dataclass
from typing import Literal, Optional

import torch
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins import SupervisedPlugin
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset

from .config import IEWCConfig
from .output_metrics import wasserstein_1d_cdf_dual_quadratic_form


LowRankImportanceKind = Literal[
    "ef_low_rank",
    "ief_low_rank",
    "ef_low_rank_diag",
    "ief_low_rank_diag",
    "ef_centered_low_rank_diag",
    "ief_centered_low_rank_diag",
    "ef_diag_low_rank",
    "ief_diag_low_rank",
    "ef_corr_low_rank",
    "ief_corr_low_rank",
]
LowRankOutputMetricKind = Literal["euclidean", "wasserstein_1d_cdf"]


@dataclass
class LowRankImportanceResult:
    eigenvalues: Tensor
    eigenvectors: Tensor
    residual_diagonal: Optional[Tensor]
    loss_scales: Tensor
    sample_count: int
    total_variance: float

    @property
    def rank(self) -> int:
        return int(self.eigenvalues.numel())

    @property
    def explained_variance_ratio(self) -> float:
        if self.total_variance <= 0.0:
            return 0.0
        return float(self.eigenvalues.sum().item() / self.total_variance)

    @property
    def residual_diagonal_mass(self) -> float:
        if self.residual_diagonal is None:
            return 0.0
        return float(self.residual_diagonal.sum().item())


@dataclass
class _PenaltyState:
    result: LowRankImportanceResult
    saved_parameters: Tensor


def _flatten_model_parameters(model: nn.Module) -> Tensor:
    return torch.cat([param.reshape(-1) for param in model.parameters()])


def _flatten_model_gradients(model: nn.Module) -> Tensor:
    pieces = []
    for param in model.parameters():
        if param.grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
        else:
            pieces.append(param.grad.detach().reshape(-1))
    return torch.cat(pieces)


def _covariance_eigenpairs(
    gradient_matrix: Tensor,
    *,
    rank: int,
    min_eigenvalue: float,
) -> tuple[Tensor, Tensor]:
    sample_count = int(gradient_matrix.shape[0])
    gram = gradient_matrix @ gradient_matrix.T / float(sample_count)
    eigvals, eigvecs_left = torch.linalg.eigh(gram)
    if not torch.isfinite(eigvals).all():
        raise FloatingPointError(
            "Non-finite eigenvalues while computing low-rank importances"
        )
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs_left = eigvecs_left[:, order]

    positive = eigvals > min_eigenvalue
    keep = min(rank, int(positive.sum().item()))
    eigvals = eigvals[:keep].clamp_min(0.0)
    eigvecs_left = eigvecs_left[:, :keep]
    if keep == 0:
        eigenvectors = gradient_matrix.new_zeros((0, gradient_matrix.shape[1]))
    else:
        normalizers = torch.sqrt(eigvals * float(sample_count)).unsqueeze(0)
        eigenvectors = (eigvecs_left.T @ gradient_matrix) / normalizers.T
        eigenvectors = torch.nn.functional.normalize(eigenvectors, dim=1)
    return eigvals, eigenvectors


def _signed_residual_eigenpairs(
    gradient_matrix: Tensor,
    *,
    full_diagonal: Tensor,
    rank: int,
    min_eigenvalue: float,
) -> tuple[Tensor, Tensor]:
    """Approximate signed eigenpairs of X^T X / n - diag(X^T X / n).

    The zero-diagonal residual is generally indefinite. After truncation, the
    diagonal-plus-signed-low-rank surrogate is not automatically PSD, so the
    residual correction is shrunk only when the diagonal-whitened low-rank part
    would create negative curvature.
    """
    dim = int(gradient_matrix.shape[1])
    if dim == 0:
        return gradient_matrix.new_zeros((0,)), gradient_matrix.new_zeros((0, 0))

    basis_size = min(dim, max(rank + 8, 2 * rank))
    generator = torch.Generator(device=gradient_matrix.device).manual_seed(0)
    q = torch.randn(
        dim,
        basis_size,
        dtype=gradient_matrix.dtype,
        device=gradient_matrix.device,
        generator=generator,
    )

    def residual_mv(v: Tensor) -> Tensor:
        return (
            gradient_matrix.T @ (gradient_matrix @ v) / float(gradient_matrix.shape[0])
            - full_diagonal.unsqueeze(1) * v
        )

    for _ in range(3):
        q, _ = torch.linalg.qr(residual_mv(q), mode="reduced")
    projected = q.T @ residual_mv(q)
    projected = (projected + projected.T) / 2
    eigvals, eigvecs_small = torch.linalg.eigh(projected)
    if not torch.isfinite(eigvals).all():
        raise FloatingPointError(
            "Non-finite residual eigenvalues while computing diag-low-rank importances"
        )
    order = torch.argsort(eigvals.abs(), descending=True)
    eigvals = eigvals[order]
    eigvecs_small = eigvecs_small[:, order]
    large_enough = eigvals.abs() > min_eigenvalue
    keep = min(rank, int(large_enough.sum().item()))
    eigvals = eigvals[:keep]
    if keep == 0:
        return eigvals, gradient_matrix.new_zeros((0, dim))
    eigenvectors = (q @ eigvecs_small[:, :keep]).T
    eigenvectors = torch.nn.functional.normalize(eigenvectors, dim=1)
    safe_diagonal = full_diagonal.clamp_min(1e-12)
    whitened = eigenvectors.T / torch.sqrt(safe_diagonal).unsqueeze(1)
    whitened = whitened * torch.sqrt(eigvals.abs()).unsqueeze(0)
    signs = torch.diag(torch.sign(eigvals))
    small = whitened.T @ whitened @ signs
    min_whitened_eig = float(torch.linalg.eigvals(small).real.min().item())
    if keep < dim and min_whitened_eig < -0.99:
        eigvals = eigvals * (0.99 / -min_whitened_eig)
    return eigvals, eigenvectors


def _correlation_residual_eigenpairs(
    gradient_matrix: Tensor,
    *,
    full_diagonal: Tensor,
    rank: int,
    min_eigenvalue: float,
) -> tuple[Tensor, Tensor]:
    safe_diagonal = full_diagonal.clamp_min(1e-12)
    whitened_rows = gradient_matrix / torch.sqrt(safe_diagonal).unsqueeze(0)
    eigvals, whitened_eigenvectors = _signed_residual_eigenpairs(
        whitened_rows,
        full_diagonal=torch.ones_like(full_diagonal),
        rank=rank,
        min_eigenvalue=min_eigenvalue,
    )
    eigenvectors = whitened_eigenvectors * torch.sqrt(safe_diagonal).unsqueeze(0)
    return eigvals, eigenvectors


def _centered_covariance_eigenpairs(
    gradient_matrix: Tensor,
    *,
    rank: int,
    min_eigenvalue: float,
) -> tuple[Tensor, Tensor]:
    centered = gradient_matrix - gradient_matrix.mean(dim=0, keepdim=True)
    return _covariance_eigenpairs(
        centered,
        rank=rank,
        min_eigenvalue=min_eigenvalue,
    )


class LowRankImportanceEstimator:
    """Compute low-rank EF/IEWC factors from normalized per-sample gradients."""

    def __init__(
        self,
        *,
        kind: LowRankImportanceKind,
        rank: int,
        tau: float | None = None,
        output_metric: LowRankOutputMetricKind | None = None,
        config: IEWCConfig | None = None,
        min_eigenvalue: float = 1e-12,
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
        if kind not in {
            "ef_low_rank",
            "ief_low_rank",
            "ef_low_rank_diag",
            "ief_low_rank_diag",
            "ef_centered_low_rank_diag",
            "ief_centered_low_rank_diag",
            "ef_diag_low_rank",
            "ief_diag_low_rank",
            "ef_corr_low_rank",
            "ief_corr_low_rank",
        }:
            raise ValueError(f"Unknown low-rank importance kind: {kind}")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if tau < 0:
            raise ValueError("tau must be non-negative")
        if output_metric not in {"euclidean", "wasserstein_1d_cdf"}:
            raise ValueError(f"Unknown output metric: {output_metric}")
        self.kind = kind
        self.rank = int(rank)
        self.tau = float(tau)
        self.output_metric = output_metric
        self.min_eigenvalue = float(min_eigenvalue)

    @property
    def _uses_ief_normalization(self) -> bool:
        return self.kind in {
            "ief_low_rank",
            "ief_low_rank_diag",
            "ief_centered_low_rank_diag",
            "ief_diag_low_rank",
            "ief_corr_low_rank",
        }

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
    ) -> LowRankImportanceResult:
        was_training = model.training
        model.eval()
        model.to(device)

        rows = []
        loss_scales = []
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
                loss = criterion(out, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite loss while computing low-rank importances"
                    )

                if self._uses_ief_normalization:
                    output_grad = torch.autograd.grad(loss, out, retain_graph=True)[0]
                    loss_scale = self._loss_scale(output_grad)
                    denom = loss_scale + self.tau
                else:
                    loss_scale = torch.ones((), device=device)
                    denom = torch.ones((), device=device)

                loss.backward()
                row = _flatten_model_gradients(model) / torch.sqrt(denom)
                if not torch.isfinite(loss_scale):
                    raise FloatingPointError(
                        "Non-finite loss scale while computing low-rank importances"
                    )
                if not torch.isfinite(row).all():
                    raise FloatingPointError(
                        "Non-finite gradient row while computing low-rank importances"
                    )
                rows.append(row.detach().float().cpu())
                loss_scales.append(loss_scale.detach().float().cpu())

        if not rows:
            raise ValueError("Cannot compute low-rank importances from an empty dataset")

        gradient_matrix = torch.stack(rows)
        if not torch.isfinite(gradient_matrix).all():
            raise FloatingPointError(
                "Non-finite gradient sketch while computing low-rank importances"
            )
        sample_count = int(gradient_matrix.shape[0])
        full_diagonal = gradient_matrix.pow(2).mean(dim=0)
        if self.kind in {"ef_diag_low_rank", "ief_diag_low_rank"}:
            eigvals, eigenvectors = _signed_residual_eigenpairs(
                gradient_matrix,
                full_diagonal=full_diagonal,
                rank=self.rank,
                min_eigenvalue=self.min_eigenvalue,
            )
            residual_diagonal = full_diagonal
        elif self.kind in {
            "ef_centered_low_rank_diag",
            "ief_centered_low_rank_diag",
        }:
            eigvals, eigenvectors = _centered_covariance_eigenpairs(
                gradient_matrix,
                rank=self.rank,
                min_eigenvalue=self.min_eigenvalue,
            )
            if eigvals.numel() == 0:
                low_rank_diagonal = torch.zeros_like(full_diagonal)
            else:
                low_rank_diagonal = (
                    eigvals.unsqueeze(1) * eigenvectors.pow(2)
                ).sum(dim=0)
            residual_diagonal = (full_diagonal - low_rank_diagonal).clamp_min(0.0)
        elif self.kind in {"ef_corr_low_rank", "ief_corr_low_rank"}:
            eigvals, eigenvectors = _correlation_residual_eigenpairs(
                gradient_matrix,
                full_diagonal=full_diagonal,
                rank=self.rank,
                min_eigenvalue=self.min_eigenvalue,
            )
            residual_diagonal = full_diagonal
        else:
            eigvals, eigenvectors = _covariance_eigenpairs(
                gradient_matrix,
                rank=self.rank,
                min_eigenvalue=self.min_eigenvalue,
            )
            residual_diagonal = None
        if self.kind in {"ef_low_rank_diag", "ief_low_rank_diag"}:
            if eigvals.numel() == 0:
                low_rank_diagonal = torch.zeros_like(full_diagonal)
            else:
                low_rank_diagonal = (
                    eigvals.unsqueeze(1) * eigenvectors.pow(2)
                ).sum(dim=0)
            residual_diagonal = (full_diagonal - low_rank_diagonal).clamp_min(0.0)

        total_variance = float(gradient_matrix.pow(2).sum().item() / sample_count)
        optimizer.zero_grad()
        model.train(was_training)

        return LowRankImportanceResult(
            eigenvalues=eigvals.detach().cpu(),
            eigenvectors=eigenvectors.detach().cpu(),
            residual_diagonal=(
                residual_diagonal.detach().cpu()
                if residual_diagonal is not None
                else None
            ),
            loss_scales=torch.stack(loss_scales),
            sample_count=sample_count,
            total_variance=total_variance,
        )


class LowRankIEWCPlugin(SupervisedPlugin):
    """Avalanche plugin implementing a low-rank EF/IEWC quadratic penalty."""

    def __init__(
        self,
        ewc_lambda: float | None = None,
        *,
        config: IEWCConfig | None = None,
        importance_kind: LowRankImportanceKind = "ief_low_rank",
        rank: int = 20,
        tau: float | None = None,
        output_metric: LowRankOutputMetricKind | None = None,
        max_importance_samples: int | None = None,
        importance_sample_seed: int = 0,
        importance_num_workers: int | None = None,
        mode: Literal["separate", "online"] = "separate",
    ):
        super().__init__()
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
        if mode not in {"separate", "online"}:
            raise ValueError("mode must be 'separate' or 'online'")
        self.ewc_lambda = float(ewc_lambda)
        self.importance_kind = importance_kind
        self.rank = int(rank)
        self.tau = float(tau)
        self.output_metric = output_metric
        self.max_importance_samples = max_importance_samples
        self.importance_sample_seed = int(importance_sample_seed)
        self.importance_num_workers = importance_num_workers
        self.mode = mode
        self.states: list[_PenaltyState] = []
        self.last_importance_result: Optional[LowRankImportanceResult] = None

    def _importance_subset(self, dataset: Dataset, *, experience_index: int) -> Dataset:
        if (
            self.max_importance_samples is None
            or len(dataset) <= self.max_importance_samples
        ):
            return dataset
        generator = torch.Generator().manual_seed(
            self.importance_sample_seed + int(experience_index)
        )
        indices = torch.randperm(len(dataset), generator=generator)[
            : self.max_importance_samples
        ].tolist()
        if hasattr(dataset, "subset"):
            return dataset.subset(indices)
        return Subset(dataset, indices)

    def before_backward(self, strategy, **kwargs):
        if not self.states:
            return

        current = _flatten_model_parameters(strategy.model)
        penalty = torch.zeros((), device=strategy.device)
        for state in self.states:
            result = state.result
            saved = state.saved_parameters.to(strategy.device)
            delta = current - saved
            if result.rank > 0:
                eigenvectors = result.eigenvectors.to(strategy.device)
                eigenvalues = result.eigenvalues.to(strategy.device)
                projections = eigenvectors @ delta
                penalty = penalty + (eigenvalues * projections.pow(2)).sum()
            if result.residual_diagonal is not None:
                residual_diagonal = result.residual_diagonal.to(strategy.device)
                penalty = penalty + (residual_diagonal * delta.pow(2)).sum()

        strategy.loss = strategy.loss + self.ewc_lambda * penalty

    def after_training_exp(self, strategy, **kwargs):
        exp_counter = strategy.clock.train_exp_counter
        dataset = self._importance_subset(
            strategy.experience.dataset, experience_index=exp_counter
        )
        estimator = LowRankImportanceEstimator(
            kind=self.importance_kind,
            rank=self.rank,
            config=IEWCConfig(
                lambda_=self.ewc_lambda,
                tau=self.tau,
                geometry=self.output_metric,
                sample_weighting="uniform",
            ),
        )
        result = estimator.compute(
            model=strategy.model,
            criterion=strategy._criterion,
            optimizer=strategy.optimizer,
            dataset=dataset,
            device=strategy.device,
            batch_size=strategy.train_mb_size,
            num_workers=(
                kwargs.get("num_workers", 0)
                if self.importance_num_workers is None
                else self.importance_num_workers
            ),
            pin_memory=strategy.device.type == "cuda",
        )
        state = _PenaltyState(
            result=result,
            saved_parameters=_flatten_model_parameters(strategy.model).detach().cpu(),
        )
        if self.mode == "online":
            self.states = [state]
        else:
            self.states.append(state)
        self.last_importance_result = result
