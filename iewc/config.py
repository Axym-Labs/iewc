from dataclasses import dataclass
from typing import Literal


OutputGeometry = Literal["euclidean", "wasserstein_1d_cdf"]
SampleWeighting = Literal["uniform", "gss_residual", "fromp_trace"]


@dataclass(frozen=True)
class IEWCConfig:
    """User-facing IEWC hyperparameter interface.

    This mirrors the paper-level choices (lambda, tau, geometry, h) while
    keeping names valid Python identifiers.
    """

    lambda_: float = 1.0
    tau: float = 1e-2
    geometry: OutputGeometry = "euclidean"
    sample_weighting: SampleWeighting = "uniform"

    def validate(self) -> None:
        if self.lambda_ < 0:
            raise ValueError("lambda_ must be non-negative")
        if self.tau < 0:
            raise ValueError("tau must be non-negative")
        if self.geometry not in {"euclidean", "wasserstein_1d_cdf"}:
            raise ValueError(f"Unknown IEWC geometry: {self.geometry}")
        if self.sample_weighting not in {"uniform", "gss_residual", "fromp_trace"}:
            raise ValueError(f"Unknown IEWC sample weighting: {self.sample_weighting}")

    @property
    def output_metric(self) -> OutputGeometry:
        return self.geometry
