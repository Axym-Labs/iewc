import torch


def _validate_same_shape(old: torch.Tensor, new: torch.Tensor):
    if old.shape != new.shape:
        raise ValueError(f"Shape mismatch: {old.shape} vs {new.shape}")
    if old.ndim < 1:
        raise ValueError("Expected at least one output dimension")


def euclidean_squared_distance(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    """Mean discretized squared Euclidean distance over the last dimension."""
    _validate_same_shape(old, new)
    return (old - new).pow(2).mean(dim=-1)


def wasserstein_1d_cdf_squared_distance(
    old: torch.Tensor, new: torch.Tensor
) -> torch.Tensor:
    """1D ordered-output Wasserstein-style CDF L2 distance.

    For probability vectors on an evenly spaced 1D grid, the cumulative
    difference encodes transport along the line. The final grid point is
    omitted because normalized distributions have zero terminal CDF difference.
    """
    _validate_same_shape(old, new)
    if old.shape[-1] < 2:
        raise ValueError("Wasserstein-style 1D distance needs at least 2 bins")
    cdf_delta = torch.cumsum(old - new, dim=-1)[..., :-1]
    dx = 1.0 / float(old.shape[-1] - 1)
    return cdf_delta.pow(2).sum(dim=-1) * dx


def wasserstein_1d_cdf_dual_quadratic_form(
    covector: torch.Tensor, *, ridge: float = 1e-8
) -> torch.Tensor:
    """Dual quadratic form r^T G^+ r for the 1D CDF L2 metric.

    The primal metric is ||C delta||^2 dx, where C maps a vector to its
    cumulative sums except the terminal CDF entry. On probability simplexes this
    metric is used on the zero-sum tangent space, so the full-space matrix is
    singular. A tiny ridge makes the solve numerically stable for finite
    precision while preserving the intended local geometry in small experiments.
    """
    if covector.ndim == 0:
        raise ValueError("Expected at least one output dimension")
    flat = covector.reshape(-1)
    n_bins = flat.numel()
    if n_bins < 2:
        raise ValueError("Wasserstein-style 1D metric needs at least 2 bins")
    flat = flat - flat.mean()

    cdf = torch.tril(
        torch.ones(
            n_bins - 1,
            n_bins,
            dtype=flat.dtype,
            device=flat.device,
        )
    )
    dx = 1.0 / float(n_bins - 1)
    metric = cdf.T @ cdf * dx
    metric = metric + torch.eye(n_bins, dtype=flat.dtype, device=flat.device) * ridge
    solved = torch.linalg.solve(metric, flat)
    return flat @ solved
