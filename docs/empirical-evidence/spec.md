# Empirical Evidence Task Arc

## Objective

Build empirical evidence for the IEWC paper in `workspace/iewc`, starting from a tested, working, modern EWC-capable project wherever possible.

## User Requirements

- Follow the global `AGENTS.md` instructions and local Codex skills.
- First find the best candidate project to base the empirical repo on.
- Prefer a base that works out of the box for EWC and requires only minimal changes.
- Cover multiple task types:
  - regression
  - classification
  - diffusion
  - a fourth task type if a credible candidate exists
- Verify EWC works before implementing IEWC variants.
- Implement or plan:
  - EF-EWC baseline
  - EWC-DR
  - IEWC / IEF-normalized importance
  - loss-scale computation for the IEF parameter
  - tau tuning
  - experiment runs
  - alternative output distance metric, preferably Wasserstein, with lower loss than Euclidean `G = I`
  - precise empirical result transfer into the paper
  - contamination model evidence
  - implicit loss-scale weight distribution measurements

## Acceptance Boundary For This Initial Stage

- Identify and justify the empirical base candidate using local inspection and current public sources.
- Establish the repo/task-arc structure for empirical work.
- Record a concrete experiment design with baselines, metrics, task types, and pivotal verification checks.
- If feasible in the current environment, verify an EWC example runs; if not, record the environment blocker and exact next verification command.

## Current Constraints

- `workspace/iewc` initially contained only the paper draft `N. IEWC Writing.md`; it was not a git repository and had no implementation.
- The system Python is 3.13.13 and initially had no `torch` or `avalanche` installed, so PyTorch/Avalanche verification may require a dedicated Python 3.10-3.12 environment.
- When performing runs, ignore baseline utilization from the other process. This work has higher priority for scheduling decisions, but the other process must not be stopped.
