import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from run_empirical_suite import ARTIFACTS, ROOT, facil_job


@dataclass(frozen=True)
class SpectrumJob:
    name: str
    command: list[str]
    log: Path


def noise_token(p: float) -> str:
    return str(p).replace(".", "p")


def build_job(method: str, seed: int, p: float) -> SpectrumJob:
    if method == "ef":
        job = facil_job(
            name=f"spectrum_cifar100_ef_diag_seed{seed}_e60_lam10000_p{noise_token(p)}",
            approach="iewc",
            seed=seed,
            stop_at_task=3,
            lamb=10000,
            importance_kind="ef_diag",
            fi_num_samples=512,
            results_subdir="facil-spectrum-prefix3",
        )
    elif method == "iewc":
        job = facil_job(
            name=f"spectrum_cifar100_iewc_diag_seed{seed}_e60_lam10000_tau1e-2_p{noise_token(p)}",
            approach="iewc",
            seed=seed,
            stop_at_task=3,
            lamb=10000,
            importance_kind="ief_diag",
            tau=0.01,
            fi_num_samples=512,
            match_ef_scale=True,
            results_subdir="facil-spectrum-prefix3",
        )
    else:
        raise ValueError(method)
    command = [
        *job.command,
        "--label-noise",
        str(p),
        "--label-noise-seed",
        "20260615",
    ]
    return SpectrumJob(
        name=job.name,
        command=command,
        log=ARTIFACTS / "run-logs" / f"{job.name}.log",
    )


def build_jobs() -> list[SpectrumJob]:
    return [
        build_job(method, seed, p)
        for p in (0.0, 0.1, 0.25)
        for method in ("ef", "iewc")
        for seed in (0, 1, 2)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs()
    if args.only:
        requested = set(args.only)
        jobs = [job for job in jobs if any(token in job.name for token in requested)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    (ARTIFACTS / "run-logs").mkdir(parents=True, exist_ok=True)
    for job in jobs:
        if args.skip_existing and any(
            path.is_dir() and job.name in path.name
            for root in (ARTIFACTS / "facil-spectrum-prefix3",)
            if root.exists()
            for path in root.iterdir()
        ):
            print(f"skipping existing {job.name}", flush=True)
            continue
        display = "CUDA_VISIBLE_DEVICES=0 " + " ".join(job.command)
        if args.dry_run:
            print(display)
            continue
        print(f"running {job.name}: log={job.log}", flush=True)
        with job.log.open("w") as log:
            completed = subprocess.run(
                job.command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        print(
            f"finished {job.name}: returncode={completed.returncode} log={job.log}",
            flush=True,
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
