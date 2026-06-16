import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from run_empirical_suite import ARTIFACTS, ROOT, facil_job


OUT_ROOT = ARTIFACTS / "lambda-tradeoff-internal"


@dataclass(frozen=True)
class LambdaJob:
    name: str
    command: list[str]
    log: Path
    gpu: bool = True
    output: Path | None = None


def lambda_grid(midpoint: float) -> list[float]:
    return [midpoint * factor for factor in (1e-3, 1e-1, 1.0, 1e1, 1e3)]


def lambda_slug(value: float) -> str:
    text = f"{value:.8g}"
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def classification_jobs() -> list[LambdaJob]:
    jobs: list[LambdaJob] = []
    specs = [
        ("ef", "EF-EWC", 10000.0),
        ("ewcdr", "EWC-DR", 100.0),
        ("iewc", "IEWC", 10000.0),
    ]
    for short, _label, midpoint in specs:
        for lamb in lambda_grid(midpoint):
            slug = lambda_slug(lamb)
            name = f"lambda_tradeoff_prefix3_{short}_seed0_lam{slug}"
            if short == "ef":
                job = facil_job(
                    name=name,
                    approach="ewc",
                    seed=0,
                    stop_at_task=3,
                    lamb=lamb,
                    fi_num_samples=512,
                    results_subdir="lambda-tradeoff-internal/facil-classification",
                )
            elif short == "ewcdr":
                job = facil_job(
                    name=name,
                    approach="iewc",
                    seed=0,
                    stop_at_task=3,
                    lamb=lamb,
                    importance_kind="ewc_dr_diag",
                    fi_num_samples=512,
                    results_subdir="lambda-tradeoff-internal/facil-classification",
                )
            elif short == "iewc":
                job = facil_job(
                    name=name,
                    approach="iewc",
                    seed=0,
                    stop_at_task=3,
                    lamb=lamb,
                    importance_kind="ief_diag",
                    tau=0.01,
                    fi_num_samples=512,
                    match_ef_scale=True,
                    results_subdir="lambda-tradeoff-internal/facil-classification",
                )
            else:
                raise AssertionError(short)
            jobs.append(LambdaJob(job.name, job.command, job.log, gpu=True))
    return jobs


def regression_jobs() -> list[LambdaJob]:
    jobs: list[LambdaJob] = []
    out_dir = OUT_ROOT / "regression"
    for seed in range(5):
        output = out_dir / f"regression_sequential_seed{seed}.json"
        jobs.append(
            LambdaJob(
                name=f"lambda_tradeoff_regression_sequential_seed{seed}",
                command=[
                    str(ROOT / ".venv" / "bin" / "python"),
                    "scripts/synthetic_regression_run.py",
                    "--seed",
                    str(seed),
                    "--n-train",
                    "1024",
                    "--n-test",
                    "2048",
                    "--epochs",
                    "300",
                    "--hidden-size",
                    "96",
                    "--batch-size",
                    "128",
                    "--learning-rate",
                    "0.01",
                    "--ewc-lambda",
                    "1.0",
                    "--tau",
                    "1e-3",
                    "--device",
                    "cuda",
                    "--methods",
                    "sequential",
                    "--output",
                    str(output),
                ],
                log=ARTIFACTS
                / "run-logs"
                / f"lambda_tradeoff_regression_sequential_seed{seed}.log",
                output=output,
            )
        )
        for method, midpoint in (("ef", 30.0), ("iewc", 1.0)):
            for lamb in lambda_grid(midpoint):
                slug = lambda_slug(lamb)
                output = out_dir / f"regression_{method}_seed{seed}_lam{slug}.json"
                jobs.append(
                    LambdaJob(
                        name=f"lambda_tradeoff_regression_{method}_seed{seed}_lam{slug}",
                        command=[
                            str(ROOT / ".venv" / "bin" / "python"),
                            "scripts/synthetic_regression_run.py",
                            "--seed",
                            str(seed),
                            "--n-train",
                            "1024",
                            "--n-test",
                            "2048",
                            "--epochs",
                            "300",
                            "--hidden-size",
                            "96",
                            "--batch-size",
                            "128",
                            "--learning-rate",
                            "0.01",
                            "--ewc-lambda",
                            str(lamb),
                            "--tau",
                            "1e-3",
                            "--device",
                            "cuda",
                            "--methods",
                            "ief_diag" if method == "iewc" else method,
                            "--output",
                            str(output),
                        ],
                        log=ARTIFACTS
                        / "run-logs"
                        / f"lambda_tradeoff_regression_{method}_seed{seed}_lam{slug}.log",
                        output=output,
                    )
                )
    return jobs


def facil_result_exists(job: LambdaJob) -> bool:
    root = OUT_ROOT / "facil-classification"
    if not root.exists():
        return False
    return any(path.is_dir() and job.name in path.name for path in root.iterdir())


def result_exists(job: LambdaJob) -> bool:
    if job.output is not None:
        return job.output.exists()
    return facil_result_exists(job)


def run_jobs(jobs: list[LambdaJob], max_parallel: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    (ARTIFACTS / "run-logs").mkdir(parents=True, exist_ok=True)
    running: list[tuple[LambdaJob, subprocess.Popen, object]] = []
    pending = list(jobs)
    failures: list[tuple[str, int]] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            if job.output is not None:
                job.output.parent.mkdir(parents=True, exist_ok=True)
            print(f"starting {job.name}: log={job.log}", flush=True)
            handle = job.log.open("w")
            process = subprocess.Popen(
                job.command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            running.append((job, process, handle))
        still_running = []
        for job, process, handle in running:
            returncode = process.poll()
            if returncode is None:
                still_running.append((job, process, handle))
                continue
            handle.close()
            print(f"finished {job.name}: returncode={returncode}", flush=True)
            if returncode != 0:
                failures.append((job.name, int(returncode)))
        running = still_running
        if running:
            time.sleep(10)
    if failures:
        for name, returncode in failures:
            print(f"failed {name}: returncode={returncode}", flush=True)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["all", "classification", "regression"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    jobs: list[LambdaJob] = []
    if args.suite in {"all", "classification"}:
        jobs.extend(classification_jobs())
    if args.suite in {"all", "regression"}:
        jobs.extend(regression_jobs())
    if args.only:
        requested = set(args.only)
        jobs = [job for job in jobs if any(token in job.name for token in requested)]
    if args.skip_existing:
        jobs = [job for job in jobs if not result_exists(job)]

    if args.dry_run:
        for job in jobs:
            print("CUDA_VISIBLE_DEVICES=0 " + " ".join(job.command))
        return

    if not jobs:
        print("No jobs to run.", flush=True)
        return
    run_jobs(jobs, max(1, int(args.max_parallel)))


if __name__ == "__main__":
    main()
