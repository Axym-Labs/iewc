import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from run_empirical_suite import ARTIFACTS, ROOT, facil_job


@dataclass(frozen=True)
class MechanismJob:
    name: str
    command: list[str]
    log: Path


def weighting_job(mode: str, seed: int, lamb: int = 10000) -> MechanismJob:
    job = facil_job(
        name=(
            f"mechanism_prefix3_iewc_{mode}_seed{seed}"
            f"_e60_lam{lamb}_tau1e-2"
        ),
        approach="iewc",
        seed=seed,
        stop_at_task=3,
        lamb=lamb,
        importance_kind="ief_diag",
        tau=0.01,
        fi_num_samples=512,
        match_ef_scale=True,
        importance_weight_mode=mode,
        results_subdir="facil-mechanism-checks",
    )
    return MechanismJob(job.name, job.command, job.log)


def ewcd_r_job(seed: int, lamb: int) -> MechanismJob:
    job = facil_job(
        name=f"mechanism_prefix3_ewcdr_seed{seed}_e60_lam{lamb}",
        approach="iewc",
        seed=seed,
        stop_at_task=3,
        lamb=lamb,
        importance_kind="ewc_dr_diag",
        fi_num_samples=512,
        results_subdir="facil-mechanism-checks",
    )
    return MechanismJob(job.name, job.command, job.log)


def build_jobs() -> list[MechanismJob]:
    jobs: list[MechanismJob] = []
    for mode in ("gss_residual", "fromp_lambda"):
        for seed in (0, 1):
            jobs.append(weighting_job(mode, seed))
    for lamb in (100, 300, 1000, 3000, 10000):
        jobs.append(ewcd_r_job(seed=0, lamb=lamb))
    return jobs


def result_exists(job: MechanismJob) -> bool:
    root = ARTIFACTS / "facil-mechanism-checks"
    if not root.exists():
        return False
    return any(path.is_dir() and job.name in path.name for path in root.iterdir())


def run_parallel(jobs: list[MechanismJob], max_parallel: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    (ARTIFACTS / "run-logs").mkdir(parents=True, exist_ok=True)
    running: list[tuple[MechanismJob, subprocess.Popen, object]] = []
    pending = list(jobs)
    failures: list[tuple[str, int]] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            print(f"starting {job.name}: log={job.log}", flush=True)
            log_handle = job.log.open("w")
            process = subprocess.Popen(
                job.command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            running.append((job, process, log_handle))
        still_running = []
        for job, process, log_handle in running:
            returncode = process.poll()
            if returncode is None:
                still_running.append((job, process, log_handle))
                continue
            log_handle.close()
            print(f"finished {job.name}: returncode={returncode}", flush=True)
            if returncode != 0:
                failures.append((job.name, int(returncode)))
        running = still_running
        if running:
            time.sleep(15)
    if failures:
        for name, returncode in failures:
            print(f"failed {name}: returncode={returncode}", flush=True)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    jobs = build_jobs()
    if args.only:
        requested = set(args.only)
        jobs = [
            job for job in jobs if any(token in job.name for token in requested)
        ]
    if args.skip_existing:
        jobs = [job for job in jobs if not result_exists(job)]

    if args.dry_run:
        for job in jobs:
            print("CUDA_VISIBLE_DEVICES=0 " + " ".join(job.command))
        return

    if not jobs:
        print("No jobs to run.", flush=True)
        return
    run_parallel(jobs, max(1, int(args.max_parallel)))


if __name__ == "__main__":
    main()
