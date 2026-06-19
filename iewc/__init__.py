from .avalanche import IEWCPlugin
from .config import IEWCConfig
from .diagonal_regularization import (
    DiagonalImportance,
    compute_diagonal_importance,
    diagonal_ewc_penalties,
    diagonal_ewc_penalty,
)
from .importance import ImportanceEstimator, ImportanceResult
from .low_rank import LowRankIEWCPlugin, LowRankImportanceEstimator

__all__ = [
    "DiagonalImportance",
    "IEWCConfig",
    "IEWCPlugin",
    "ImportanceEstimator",
    "ImportanceResult",
    "LowRankIEWCPlugin",
    "LowRankImportanceEstimator",
    "compute_diagonal_importance",
    "diagonal_ewc_penalties",
    "diagonal_ewc_penalty",
]
