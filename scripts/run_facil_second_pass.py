import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from run_empirical_suite import ARTIFACTS, ROOT, facil_job


@dataclass(frozen=True)
class SecondPassJob:
    name: str
    command: list[str]
    log: Path


def with_label_noise(job, *, p: float, seed: int) -> SecondPassJob:
    command = [
        *job.command,
        "--label-noise",
        str(p),
        "--label-noise-seed",
        "20260615",
    ]
    return SecondPassJob(
        name=job.name,
        command=command,
        log=ARTIFACTS / "run-logs" / f"{job.name}.log",
    )


def clean_ewcdr_job(seed: int) -> SecondPassJob:
    job = facil_job(
        name=f"secondpass_cifar100_ewcdr_diag_seed{seed}_e60_lam10000",
        approach="iewc",
        seed=seed,
        lamb=10000,
        importance_kind="ewc_dr_diag",
        fi_num_samples=512,
        results_subdir="facil-second-pass",
    )
    return SecondPassJob(job.name, job.command, job.log)


def contamination_job(method: str, seed: int, p: float) -> SecondPassJob:
    if method == "ef":
        job = facil_job(
            name=f"contam_cifar100_ef_diag_seed{seed}_e60_lam10000_p{str(p).replace('.', 'p')}",
            approach="iewc",
            seed=seed,
            lamb=10000,
            importance_kind="ef_diag",
            fi_num_samples=512,
            results_subdir="facil-contamination",
        )
    elif method == "ewcdr":
        job = facil_job(
            name=f"contam_cifar100_ewcdr_diag_seed{seed}_e60_lam10000_p{str(p).replace('.', 'p')}",
            approach="iewc",
            seed=seed,
            lamb=10000,
            importance_kind="ewc_dr_diag",
            fi_num_samples=512,
            results_subdir="facil-contamination",
        )
    elif method == "iewc":
        job = facil_job(
            name=f"contam_cifar100_iewc_diag_seed{seed}_e60_lam10000_tau1e-2_p{str(p).replace('.', 'p')}",
            approach="iewc",
            seed=seed,
            lamb=10000,
            importance_kind="ief_diag",
            tau=0.01,
            fi_num_samples=512,
            match_ef_scale=True,
            results_subdir="facil-contamination",
        )
    else:
        raise ValueError(method)
    return with_label_noise(job, p=p, seed=seed)


def build_jobs() -> list[SecondPassJob]:
    jobs = [clean_ewcdr_job(seed) for seed in (0, 1, 2)]
    for p in (0.0, 0.1, 0.25):
        for method in ("ef", "ewcdr", "iewc"):
            for seed in (0, 1, 2):
                jobs.append(contamination_job(method, seed, p))
    return jobs


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
            for root in (ARTIFACTS / "facil-second-pass", ARTIFACTS / "facil-contamination")
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
