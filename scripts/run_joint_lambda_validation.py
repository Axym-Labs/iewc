import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "empirical-evidence" / "artifacts"
OUT_ROOT = ARTIFACTS / "joint-lambda-validation"
RAW = OUT_ROOT / "raw"


@dataclass(frozen=True)
class Job:
    name: str
    command: list[str]
    log: Path
    output: Path


def lambda_slug(value: float) -> str:
    text = f"{value:.8g}"
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def regression_lambdas(method: str) -> list[float]:
    if method == "ef":
        return [0.3, 3.0, 30.0, 300.0, 3000.0]
    if method == "iewc":
        return [0.01, 0.1, 1.0, 10.0, 100.0]
    raise ValueError(method)


def diffusion_lambdas(method: str) -> list[float]:
    if method == "ef":
        return [2.0, 20.0, 200.0, 2000.0, 20000.0]
    if method == "iewc":
        return [0.25, 2.5, 25.0, 250.0, 2500.0]
    raise ValueError(method)


def regression_jobs() -> list[Job]:
    jobs: list[Job] = []
    out_dir = RAW / "regression"
    for seed in range(5):
        output = out_dir / f"regression_sequential_seed{seed}.json"
        jobs.append(
            Job(
                name=f"joint_lambda_regression_sequential_seed{seed}",
                command=[
                    str(ROOT / ".venv" / "bin" / "python"),
                    "scripts/synthetic_regression_run.py",
                    "--seed",
                    str(seed),
                    "--n-train",
                    "1024",
                    "--n-test",
                    "2048",
                    "--n-tasks",
                    "5",
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
                / f"joint_lambda_regression_sequential_seed{seed}.log",
                output=output,
            )
        )
        for method in ("ef", "iewc"):
            for lamb in regression_lambdas(method):
                slug = lambda_slug(lamb)
                output = out_dir / f"regression_{method}_seed{seed}_lam{slug}.json"
                jobs.append(
                    Job(
                        name=f"joint_lambda_regression_{method}_seed{seed}_lam{slug}",
                        command=[
                            str(ROOT / ".venv" / "bin" / "python"),
                            "scripts/synthetic_regression_run.py",
                            "--seed",
                            str(seed),
                            "--n-train",
                            "1024",
                            "--n-test",
                            "2048",
                            "--n-tasks",
                            "5",
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
                        / f"joint_lambda_regression_{method}_seed{seed}_lam{slug}.log",
                        output=output,
                    )
                )
    return jobs


def diffusion_jobs() -> list[Job]:
    jobs: list[Job] = []
    out_dir = RAW / "diffusion"
    for seed in range(5):
        output = out_dir / f"diffusion_sequential_seed{seed}.json"
        jobs.append(
            Job(
                name=f"joint_lambda_diffusion_sequential_seed{seed}",
                command=[
                    str(ROOT / ".venv" / "bin" / "python"),
                    "scripts/synthetic_diffusion_run.py",
                    "--seed",
                    str(seed),
                    "--n-train-per-task",
                    "192",
                    "--n-test-per-task",
                    "192",
                    "--train-steps",
                    "240",
                    "--batch-size",
                    "24",
                    "--learning-rate",
                    "0.001",
                    "--ewc-lambda",
                    "25",
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
                / f"joint_lambda_diffusion_sequential_seed{seed}.log",
                output=output,
            )
        )
        for method in ("ef", "iewc"):
            for lamb in diffusion_lambdas(method):
                slug = lambda_slug(lamb)
                output = out_dir / f"diffusion_{method}_seed{seed}_lam{slug}.json"
                jobs.append(
                    Job(
                        name=f"joint_lambda_diffusion_{method}_seed{seed}_lam{slug}",
                        command=[
                            str(ROOT / ".venv" / "bin" / "python"),
                            "scripts/synthetic_diffusion_run.py",
                            "--seed",
                            str(seed),
                            "--n-train-per-task",
                            "192",
                            "--n-test-per-task",
                            "192",
                            "--train-steps",
                            "240",
                            "--batch-size",
                            "24",
                            "--learning-rate",
                            "0.001",
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
                        / f"joint_lambda_diffusion_{method}_seed{seed}_lam{slug}.log",
                        output=output,
                    )
                )
    return jobs


def run_jobs(jobs: list[Job], max_parallel: int, *, skip_existing: bool) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    (ARTIFACTS / "run-logs").mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if not (skip_existing and job.output.exists())]
    running: list[tuple[Job, subprocess.Popen, object]] = []
    failures: list[tuple[str, int]] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
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
            time.sleep(5)
    if failures:
        for name, returncode in failures:
            print(f"failed {name}: returncode={returncode}", flush=True)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["all", "regression", "diffusion"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    jobs: list[Job] = []
    if args.suite in {"all", "regression"}:
        jobs.extend(regression_jobs())
    if args.suite in {"all", "diffusion"}:
        jobs.extend(diffusion_jobs())
    if args.only:
        needles = tuple(args.only)
        jobs = [job for job in jobs if any(needle in job.name for needle in needles)]
    if args.skip_existing:
        jobs = [job for job in jobs if not job.output.exists()]
    for job in jobs:
        print(" ".join(job.command))
    print(f"job_count={len(jobs)}")
    if args.dry_run:
        return
    run_jobs(jobs, args.max_parallel, skip_existing=False)


if __name__ == "__main__":
    main()
