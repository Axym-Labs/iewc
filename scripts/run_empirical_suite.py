import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "empirical-evidence" / "artifacts"


@dataclass(frozen=True)
class Job:
    name: str
    command: list[str]
    log: Path
    gpu: bool = False


def facil_job(
    *,
    name: str,
    approach: str,
    seed: int,
    network: str = "resnet32",
    stop_at_task: int = 0,
    nepochs: int = 60,
    lamb: float | None = None,
    importance_kind: str | None = None,
    tau: float | None = None,
    fi_num_samples: int | None = 512,
    match_ef_scale: bool = False,
    eval_on_train: bool = False,
    importance_weight_mode: str | None = None,
    results_subdir: str = "facil-final",
) -> Job:
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        "vendor/FACIL/src/main_incremental.py",
        "--approach",
        approach,
        "--datasets",
        "cifar100_icarl",
        "--network",
        network,
        "--seed",
        str(seed),
        "--num-tasks",
        "10",
        "--nepochs",
        str(nepochs),
        "--batch-size",
        "128",
        "--num-workers",
        "1",
        "--pin-memory",
        "true",
        "--lr",
        "0.05",
        "--lr-patience",
        "10",
        "--momentum",
        "0.9",
        "--weight-decay",
        "0.0002",
        "--log",
        "disk",
        "--results-path",
        str(ARTIFACTS / results_subdir),
        "--exp-name",
        name,
    ]
    if stop_at_task:
        cmd += ["--stop-at-task", str(stop_at_task)]
    if eval_on_train:
        cmd += ["--eval-on-train"]
    if lamb is not None:
        cmd += ["--lamb", str(lamb)]
    if importance_kind is not None:
        cmd += ["--importance-kind", importance_kind]
    if tau is not None:
        cmd += ["--tau", str(tau)]
    if fi_num_samples is not None:
        cmd += [
            "--fi-num-samples",
            str(fi_num_samples),
            "--fi-sampling-type",
            "max_pred",
        ]
        if approach == "iewc":
            cmd += ["--importance-sample-seed", "20260614"]
    if match_ef_scale:
        cmd += ["--match-ef-scale"]
    if importance_weight_mode is not None:
        cmd += ["--importance-weight-mode", importance_weight_mode]
    return Job(
        name=name,
        command=cmd,
        log=ARTIFACTS / "run-logs" / f"{name}.log",
        gpu=True,
    )


def python_job(name: str, args: list[str], *, gpu: bool = False) -> Job:
    return Job(
        name=name,
        command=[str(ROOT / ".venv" / "bin" / "python"), *args],
        log=ARTIFACTS / "run-logs" / f"{name}.log",
        gpu=gpu,
    )


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for seed in (0, 1, 2):
        jobs.append(
            facil_job(
                name=f"final_cifar100_ef_diag_seed{seed}_e60_lam10000",
                approach="ewc",
                seed=seed,
                lamb=10000,
                results_subdir="facil-final",
            )
        )
        jobs.append(
            facil_job(
                name=f"final_cifar100_iewc_diag_seed{seed}_e60_lam10000_tau1e-2",
                approach="iewc",
                seed=seed,
                lamb=10000,
                importance_kind="ief_diag",
                tau=0.01,
                fi_num_samples=512,
                match_ef_scale=True,
                results_subdir="facil-final",
            )
        )

    for seed in (0, 1, 2, 3, 4):
        jobs.append(
            python_job(
                f"final_synthetic_regression_seed{seed}",
                [
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
                    "1e-2",
                    "1e-1",
                    "--output",
                    str(ARTIFACTS / f"final-synthetic-regression-seed{seed}.json"),
                ],
            )
        )
        jobs.append(
            python_job(
                f"final_synthetic_diffusion_seed{seed}",
                [
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
                    "1e-3",
                    "--ewc-lambda",
                    "25",
                    "--tau",
                    "1e-3",
                    "1e-2",
                    "1e-1",
                    "--output",
                    str(ARTIFACTS / f"final-synthetic-diffusion-seed{seed}.json"),
                ],
            )
        )
        jobs.append(
            python_job(
                f"final_synthetic_segmentation_seed{seed}",
                [
                    "scripts/synthetic_segmentation_run.py",
                    "--seed",
                    str(seed),
                    "--n-train",
                    "192",
                    "--n-test",
                    "256",
                    "--train-epochs",
                    "120",
                    "--hidden-channels",
                    "24",
                    "--batch-size",
                    "32",
                    "--learning-rate",
                    "0.01",
                    "--ewc-lambda",
                    "200",
                    "--tau",
                    "1e-12",
                    "2e-12",
                    "5e-12",
                    "1e-11",
                    "--task-b-foreground-value",
                    "0.0",
                    "--task-b-background-value",
                    "1.0",
                    "--output",
                    str(ARTIFACTS / f"final-synthetic-segmentation-seed{seed}.json"),
                ],
            )
        )

    for seed in (0, 1, 2):
        jobs.append(
            python_job(
                f"final_permuted_mnist_sequential_seed{seed}",
                [
                    "scripts/split_mnist_benchmark_run.py",
                    "--benchmark-name",
                    "permuted_mnist",
                    "--seed",
                    str(seed),
                    "--epochs",
                    "10",
                    "--hidden-size",
                    "400",
                    "--train-mb-size",
                    "128",
                    "--eval-mb-size",
                    "256",
                    "--learning-rate",
                    "0.1",
                    "--methods",
                    "sequential",
                    "--dataset-root",
                    "data",
                    "--device",
                    "auto",
                    "--output",
                    str(ARTIFACTS / f"final-permuted-mnist-sequential-seed{seed}.json"),
                ],
                gpu=True,
            )
        )
        jobs.append(
            python_job(
                f"final_permuted_mnist_ef_low_rank_seed{seed}",
                [
                    "scripts/split_mnist_benchmark_run.py",
                    "--benchmark-name",
                    "permuted_mnist",
                    "--seed",
                    str(seed),
                    "--epochs",
                    "10",
                    "--hidden-size",
                    "400",
                    "--train-mb-size",
                    "128",
                    "--eval-mb-size",
                    "256",
                    "--learning-rate",
                    "0.1",
                    "--ewc-lambda",
                    "0.01",
                    "--rank",
                    "20",
                    "--methods",
                    "ef_low_rank",
                    "--max-importance-samples",
                    "2000",
                    "--importance-sample-seed",
                    "20260613",
                    "--dataset-root",
                    "data",
                    "--device",
                    "auto",
                    "--output",
                    str(ARTIFACTS / f"final-permuted-mnist-ef-low-rank-seed{seed}.json"),
                ],
                gpu=True,
            )
        )
        jobs.append(
            python_job(
                f"final_permuted_mnist_iewc_low_rank_seed{seed}",
                [
                    "scripts/split_mnist_benchmark_run.py",
                    "--benchmark-name",
                    "permuted_mnist",
                    "--seed",
                    str(seed),
                    "--epochs",
                    "10",
                    "--hidden-size",
                    "400",
                    "--train-mb-size",
                    "128",
                    "--eval-mb-size",
                    "256",
                    "--learning-rate",
                    "0.1",
                    "--ewc-lambda",
                    "0.003",
                    "--tau",
                    "1e-3",
                    "--rank",
                    "20",
                    "--methods",
                    "ief_low_rank",
                    "--max-importance-samples",
                    "2000",
                    "--importance-sample-seed",
                    "20260613",
                    "--dataset-root",
                    "data",
                    "--device",
                    "auto",
                    "--output",
                    str(ARTIFACTS / f"final-permuted-mnist-iewc-low-rank-seed{seed}.json"),
                ],
                gpu=True,
            )
        )

    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--gpu-sequential", action="store_true")
    parser.add_argument("--max-cpu-parallel", type=int, default=6)
    args = parser.parse_args()

    jobs = build_jobs()
    if args.only:
        requested = set(args.only)
        jobs = [job for job in jobs if any(token in job.name for token in requested)]
    if args.skip_gpu:
        jobs = [job for job in jobs if not job.gpu]

    def make_env(job: Job) -> dict[str, str]:
        env = os.environ.copy()
        if job.gpu:
            env["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("NUMEXPR_NUM_THREADS", "1")
        return env

    def reap_finished(
        active: list[tuple[Job, subprocess.Popen]], *, wait: bool = False
    ) -> int:
        failures = 0
        remaining = []
        for active_job, process in active:
            returncode = process.wait() if wait else process.poll()
            if returncode is None:
                remaining.append((active_job, process))
                continue
            print(
                f"finished {active_job.name}: returncode={returncode} "
                f"log={active_job.log}",
                flush=True,
            )
            if returncode != 0:
                failures += 1
        active[:] = remaining
        return failures

    (ARTIFACTS / "run-logs").mkdir(parents=True, exist_ok=True)
    active: list[tuple[Job, subprocess.Popen]] = []
    failures = 0
    for job in jobs:
        env_prefix = "CUDA_VISIBLE_DEVICES=0 " if job.gpu else ""
        display = env_prefix + " ".join(job.command)
        if args.dry_run:
            print(display)
            continue
        with job.log.open("w") as log:
            env = make_env(job)
            if args.gpu_sequential and job.gpu:
                print(f"running {job.name}: log={job.log}", flush=True)
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
            else:
                if not job.gpu:
                    while len(active) >= args.max_cpu_parallel:
                        failures += reap_finished(active)
                        if failures:
                            raise SystemExit(1)
                        if len(active) >= args.max_cpu_parallel:
                            time.sleep(2)
                process = subprocess.Popen(
                    job.command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                print(f"started {job.name}: pid={process.pid} log={job.log}")
                if not job.gpu:
                    active.append((job, process))

    while active:
        failures += reap_finished(active, wait=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
