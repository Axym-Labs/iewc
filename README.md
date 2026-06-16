# Improved Elastic Weight Consolidation

This repository contains the code used for the empirical IEWC experiments. The
maintained implementation exposes IEWC through `IEWCConfig`, which mirrors the
paper's four-tuple parametrization and can be passed directly to the Avalanche
plugin path.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

The experiment scripts expect PyTorch, Avalanche, and Diffusers. GPU execution
is recommended for the CIFAR-100 and DDPM runs.

## Basic Usage

```python
from iewc import IEWCConfig, IEWCPlugin

iewc = IEWCConfig(
    lambda_=10000.0,
    tau=1e-2,
    geometry="euclidean",
    sample_weighting="uniform",
)
plugin = IEWCPlugin(config=iewc)
```

The same configuration object is accepted by the maintained diagonal and
low-rank IEWC plugin paths. Existing keyword arguments such as `ewc_lambda`,
`tau`, and `output_metric` remain supported for compatibility.

## Empirical Artifacts

Paper-facing tables and figures are generated with:

```bash
python scripts/build_empirical_artifacts.py
```

The generated summary is written to
`docs/empirical-evidence/artifacts/paper-tables-and-plots.md`, and the selected
figures live under `docs/empirical-evidence/artifacts/paper-plots/`.

Classification experiments use a FACIL integration in `vendor/FACIL` when that
vendor checkout is available locally. The standalone regression, diffusion, and
segmentation scripts use the same IEWC configuration interface.

For paper provenance, commit `00f9d6d` records the pre-cleanup repository state
used for the reported empirical results.
