import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "empirical-evidence" / "artifacts"
PLOTS = ARTIFACTS / "paper-plots"

SUMMAND_LAMBDAS = {
    ("Classification", "EF-EWC"): 10000.0,
    ("Classification", "IEWC"): 10000.0,
    ("Regression", "EF-EWC"): 30.0,
    ("Regression", "IEWC"): 1.0,
    ("Diffusion", "EF-EWC"): 10000.0,
    ("Diffusion", "IEWC"): 25.0,
    ("Segmentation", "EF-EWC"): 1.0,
    ("Segmentation", "IEWC"): 10.0,
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
    }
)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def std(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def stderr(values):
    values = list(values)
    return std(values) / math.sqrt(len(values)) if values else math.nan


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    value = float(value)
    if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def pm(mu, sd):
    return f"{fmt(mu)} $\\pm$ {fmt(sd)}"


def latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def read_matrix(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append([float(value) for value in line.split()])
    return rows


def final_nonzero_average(row: list[float]) -> float:
    values = [value for value in row if value != 0.0]
    return mean(values)


def final_nonzero_values(row: list[float]) -> list[float]:
    return [value for value in row if value != 0.0]


def parse_noise_probability(name: str) -> float:
    match = re.search(r"_p([0-9]+p[0-9]+)(?:_|$)", name)
    if not match:
        return 0.0
    return float(match.group(1).replace("p", "."))


def parse_seed(name: str) -> int:
    match = re.search(r"_seed([0-9]+)(?:_|$)", name)
    return int(match.group(1)) if match else -1


def method_from_name(name: str) -> str:
    lowered = name.lower()
    if "ewcdr" in lowered or "ewc_dr" in lowered:
        return "EWC-DR"
    if "finetuning" in lowered or "sequential" in lowered:
        return "Sequential"
    if "ef_diag" in lowered or "_ewc_" in lowered:
        return "EF-EWC"
    if "iewc" in lowered or "ief_diag" in lowered:
        return "IEWC"
    return "Other"


def collect_facil(root: Path) -> list[dict]:
    records = []
    if not root.exists():
        return records
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        results = run_dir / "results"
        selected = None
        for acc_path in reversed(sorted(results.glob("acc_taw-*.txt"))):
            acc = read_matrix(acc_path)
            if not acc:
                continue
            final_avg = final_nonzero_average(acc[-1])
            if math.isfinite(final_avg):
                stamp = acc_path.name.removeprefix("acc_taw-")
                forg_path = results / f"forg_taw-{stamp}"
                selected = (
                    acc,
                    forg_path if forg_path.exists() else latest_file(results, "forg_taw-*.txt"),
                    final_avg,
                )
                break
        if selected is None:
            continue
        acc, forg_path, final_avg = selected
        forg = read_matrix(forg_path) if forg_path is not None else []
        final_task_taw = final_nonzero_values(acc[-1])
        final_task_forgetting = forg[-1][: len(final_task_taw)] if forg else []
        records.append(
            {
                "name": run_dir.name,
                "method": method_from_name(run_dir.name),
                "noise": parse_noise_probability(run_dir.name),
                "final_avg_taw": final_avg,
                "final_avg_forgetting": final_nonzero_average(forg[-1])
                if forg
                else math.nan,
                "final_task_taw": final_task_taw,
                "final_task_forgetting": final_task_forgetting,
                "tasks": len(acc),
                "path": run_dir,
            }
        )
    return records


def collect_paper_classification_records() -> list[dict]:
    return collect_facil(ARTIFACTS / "facil-final") + collect_facil(
        ARTIFACTS / "facil-ewcdr-lam10"
    )


def collect_paper_contamination_records() -> list[dict]:
    legacy = [
        record
        for record in collect_facil(ARTIFACTS / "facil-contamination")
        if record["method"] != "EWC-DR"
    ]
    return legacy + collect_facil(ARTIFACTS / "facil-ewcdr-lam10-contamination")


def summarize_records(records: list[dict], value_key: str) -> dict:
    values = [record[value_key] for record in records if math.isfinite(record[value_key])]
    return {"n": len(values), "mean": mean(values), "std": std(values)}


def summarize_facil_by_method(records: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record)
    return {
        method: {
            "n": len(items),
            "final_avg_taw_mean": mean(item["final_avg_taw"] for item in items),
            "final_avg_taw_std": std(item["final_avg_taw"] for item in items),
            "final_avg_forgetting_mean": mean(
                item["final_avg_forgetting"] for item in items
            ),
            "final_avg_forgetting_std": std(
                item["final_avg_forgetting"] for item in items
            ),
        }
        for method, items in grouped.items()
    }


def collect_synthetic(pattern: str, old_key: str) -> list[dict]:
    records = []
    for path in sorted(ARTIFACTS.glob(pattern)):
        payload = json.loads(path.read_text())
        for result in payload.get("results", []):
            task_b = (
                result.get("task_b_accuracy_after_task_b")
                or result.get("task_b_mse_after_task_b")
                or result.get("task_b_iou_after_task_b")
                or result.get("task_b_denoise_mse_after_task_b")
                or math.nan
            )
            records.append(
                {
                    "file": path.name,
                    "method": result["method"],
                    "old": float(result[old_key]),
                    "new": float(task_b),
                    "result": result,
                }
            )
    return records


def method_label(method: str) -> str:
    return {
        "sequential": "Sequential",
        "ef": "EF-EWC",
        "ef_low_rank": "EF-EWC",
        "ief_diag": "IEWC",
        "ief_low_rank": "IEWC",
    }.get(method, method)


def summarize_synthetic(records: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[method_label(record["method"])].append(record)
    return {
        method: {
            "n": len(items),
            "old_mean": mean(item["old"] for item in items),
            "old_std": std(item["old"] for item in items),
            "new_mean": mean(item["new"] for item in items),
            "new_std": std(item["new"] for item in items),
        }
        for method, items in grouped.items()
    }


def collect_synthetic_results(pattern: str) -> list[dict]:
    records = []
    for path in sorted(ARTIFACTS.glob(pattern)):
        payload = json.loads(path.read_text())
        for result in payload.get("results", []):
            records.append(
                {
                    "file": path.name,
                    "method": method_label(result["method"]),
                    "result": result,
                }
            )
    return records


def summarize_metric(records: list[dict], value_fn) -> dict[str, dict]:
    grouped = defaultdict(list)
    for record in records:
        value = value_fn(record["result"])
        if value is None or not math.isfinite(float(value)):
            continue
        grouped[record["method"]].append(float(value))
    return {
        method: {
            "n": len(values),
            "mean": mean(values),
            "std": std(values),
        }
        for method, values in grouped.items()
    }


def best_ef_sweep_records(task: str) -> list[dict]:
    path = ARTIFACTS / "nonclassification-ef-lambda-sweep.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    task_payload = payload.get("tasks", {}).get(task, {})
    by_lamb = defaultdict(list)
    for record in task_payload.get("records", []):
        by_lamb[float(record["ewc_lambda"])].append(record)
    best_items = []
    best_old = math.inf
    for items in by_lamb.values():
        old_mean = mean(item["old_task_metric"] for item in items)
        if old_mean < best_old:
            best_old = old_mean
            best_items = items
    return [
        {
            "file": "nonclassification-ef-lambda-sweep.json",
            "method": "EF-EWC",
            "result": item["result"],
        }
        for item in best_items
    ]


def collect_ef_lambda_sweep() -> dict[str, dict]:
    path = ARTIFACTS / "nonclassification-ef-lambda-sweep.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    output = {}
    for task, task_payload in payload.get("tasks", {}).items():
        by_lamb = defaultdict(list)
        for record in task_payload.get("records", []):
            by_lamb[float(record["ewc_lambda"])].append(record)
        best_lamb = None
        best_items = []
        for lamb, items in by_lamb.items():
            old_mean = mean(item["old_task_metric"] for item in items)
            if best_lamb is None or old_mean < mean(
                item["old_task_metric"] for item in best_items
            ):
                best_lamb = lamb
                best_items = items
        if best_items:
            output[task] = {
                "lambda": best_lamb,
                "n": len(best_items),
                "old_mean": mean(item["old_task_metric"] for item in best_items),
                "old_std": std(item["old_task_metric"] for item in best_items),
                "new_mean": mean(item["new_task_metric"] for item in best_items),
                "new_std": std(item["new_task_metric"] for item in best_items),
            }
    return output


def collect_mnist_low_loss() -> dict[str, dict]:
    records = []
    files = sorted(ARTIFACTS.glob("final-permuted-mnist-*.json"))
    for path in files:
        payload = json.loads(path.read_text())
        for result in payload.get("results", []):
            records.append(
                {
                    "method": method_label(result["method"]),
                    "final_average_accuracy": float(result["final_average_accuracy"]),
                    "average_forgetting": float(result["average_forgetting"]),
                    "loss_scale_median": result.get("last_loss_scale_median"),
                }
            )
    grouped = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record)
    return {
        method: {
            "n": len(items),
            "final_average_accuracy_mean": mean(
                item["final_average_accuracy"] for item in items
            ),
            "final_average_accuracy_std": std(
                item["final_average_accuracy"] for item in items
            ),
            "average_forgetting_mean": mean(
                item["average_forgetting"] for item in items
            ),
            "average_forgetting_std": std(
                item["average_forgetting"] for item in items
            ),
            "loss_scale_median_mean": mean(
                item["loss_scale_median"]
                for item in items
                if item["loss_scale_median"] is not None
            ),
        }
        for method, items in grouped.items()
    }


def collect_facil_summand_norms() -> list[dict]:
    records = []
    for root in (
        ARTIFACTS / "facil-final",
        ARTIFACTS / "facil-second-pass",
        ARTIFACTS / "facil-prefix",
    ):
        for path in sorted(root.glob("**/importance_summand_scales.json")):
            if root.name == "facil-prefix" and "matchfacil_traces" not in path.parent.name:
                continue
            payload = json.loads(path.read_text())
            for task_record in payload:
                if "ief" not in str(task_record.get("importance_kind", "")).lower():
                    continue
                ef_traces = task_record.get("ef_summand_traces", [])
                stored_traces = task_record.get("stored_summand_traces", [])
                if not ef_traces or not stored_traces:
                    continue
                for ef_trace, stored_trace in zip(ef_traces, stored_traces):
                    ef_lambda = SUMMAND_LAMBDAS[("Classification", "EF-EWC")]
                    iewc_lambda = SUMMAND_LAMBDAS[("Classification", "IEWC")]
                    records.append(
                        {
                            "task": "Classification",
                            "method": "EF-EWC",
                            "trace": max(float(ef_trace), 1e-300),
                            "lambda": ef_lambda,
                            "effective_trace": max(float(ef_trace) * ef_lambda, 1e-300),
                        }
                    )
                    records.append(
                        {
                            "task": "Classification",
                            "method": "IEWC",
                            "trace": max(float(stored_trace), 1e-300),
                            "lambda": iewc_lambda,
                            "effective_trace": max(
                                float(stored_trace) * iewc_lambda, 1e-300
                            ),
                        }
                    )
    return records


def collect_synthetic_summand_norms() -> list[dict]:
    tasks = {
        "Regression": ("final-synthetic-regression-seed*.json", "ief_diag"),
        "Diffusion": ("final-mnist-diffusion-seed*.json", "ief_diag"),
    }
    records = []
    for task_name, (pattern, method_name) in tasks.items():
        for path in sorted(ARTIFACTS.glob(pattern)):
            payload = json.loads(path.read_text())
            for result in payload.get("results", []):
                if result.get("method") != method_name:
                    continue
                ef_traces = result.get("old_task_ef_summand_traces", [])
                stored_traces = result.get("old_task_stored_summand_traces", [])
                if not ef_traces or not stored_traces:
                    continue
                ef_lambda = SUMMAND_LAMBDAS[(task_name, "EF-EWC")]
                iewc_lambda = SUMMAND_LAMBDAS[(task_name, "IEWC")]
                for ef_trace, stored_trace in zip(ef_traces, stored_traces):
                    records.append(
                        {
                            "task": task_name,
                            "method": "EF-EWC",
                            "trace": max(float(ef_trace), 1e-300),
                            "lambda": ef_lambda,
                            "effective_trace": max(float(ef_trace) * ef_lambda, 1e-300),
                        }
                    )
                    records.append(
                        {
                            "task": task_name,
                            "method": "IEWC",
                            "trace": max(float(stored_trace), 1e-300),
                            "lambda": iewc_lambda,
                            "effective_trace": max(
                                float(stored_trace) * iewc_lambda, 1e-300
                            ),
                        }
                    )
    return records


def _single_result(path: Path, method: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    for result in payload.get("results", []):
        if result.get("method") == method:
            return result
    return None


def collect_segmentation_records() -> list[dict]:
    files = {
        "Sequential": [
            ("seed0", ARTIFACTS / "voc-segmentation-pilot-seed0-lam10.json", "sequential"),
            ("seed1", ARTIFACTS / "voc-segmentation-final-seed1-sequential.json", "sequential"),
            ("seed2", ARTIFACTS / "voc-segmentation-final-seed2-sequential.json", "sequential"),
        ],
        "EF-EWC": [
            ("seed0", ARTIFACTS / "voc-segmentation-pilot-seed0-lam1.json", "ef"),
            ("seed1", ARTIFACTS / "voc-segmentation-final-seed1-ef-lam1.json", "ef"),
            ("seed2", ARTIFACTS / "voc-segmentation-final-seed2-ef-lam1.json", "ef"),
        ],
        "IEWC": [
            ("seed0", ARTIFACTS / "voc-segmentation-pilot-seed0-lam10.json", "ief_diag"),
            ("seed1", ARTIFACTS / "voc-segmentation-final-seed1-iewc-lam10.json", "ief_diag"),
            ("seed2", ARTIFACTS / "voc-segmentation-final-seed2-iewc-lam10.json", "ief_diag"),
        ],
    }
    records = []
    for method, specs in files.items():
        for seed, path, raw_method in specs:
            result = _single_result(path, raw_method)
            if result is None:
                continue
            enriched = dict(result)
            enriched["old"] = float(result["task_a_iou_after_task_b"])
            enriched["new"] = float(result["task_b_iou_after_task_b"])
            enriched["forgetting"] = float(result["task_a_iou_forgetting"])
            records.append(
                {
                    "method": method,
                    "seed": seed,
                    "old": enriched["old"],
                    "new": enriched["new"],
                    "forgetting": enriched["forgetting"],
                    "result": enriched,
                }
            )
    return records


def collect_segmentation_summand_norms() -> list[dict]:
    records = []
    for path in [
        ARTIFACTS / "voc-segmentation-pilot-seed0-lam10.json",
        ARTIFACTS / "voc-segmentation-final-seed1-iewc-lam10.json",
        ARTIFACTS / "voc-segmentation-final-seed2-iewc-lam10.json",
    ]:
        result = _single_result(path, "ief_diag")
        if result is None:
            continue
        ef_traces = result.get("old_task_ef_summand_traces", [])
        stored_traces = result.get("old_task_stored_summand_traces", [])
        ef_lambda = SUMMAND_LAMBDAS[("Segmentation", "EF-EWC")]
        iewc_lambda = SUMMAND_LAMBDAS[("Segmentation", "IEWC")]
        for ef_trace, stored_trace in zip(ef_traces, stored_traces):
            records.append(
                {
                    "task": "Segmentation",
                    "method": "EF-EWC",
                    "trace": max(float(ef_trace), 1e-300),
                    "lambda": ef_lambda,
                    "effective_trace": max(float(ef_trace) * ef_lambda, 1e-300),
                }
            )
            records.append(
                {
                    "task": "Segmentation",
                    "method": "IEWC",
                    "trace": max(float(stored_trace), 1e-300),
                    "lambda": iewc_lambda,
                    "effective_trace": max(
                        float(stored_trace) * iewc_lambda, 1e-300
                    ),
                }
            )
    return records


def collect_diffusion_geometry() -> list[dict]:
    records = []
    patterns = {
        "Euclidean $G=I$": "mnist-diffusion-geometry-euclidean-seed*.json",
        "Sliced-Wasserstein $G$": "mnist-diffusion-geometry-wasserstein-lam*seed*.json",
    }
    for label, pattern in patterns.items():
        for path in sorted(ARTIFACTS.glob(pattern)):
            seed_match = re.search(r"seed([0-9]+)", path.name)
            seed = int(seed_match.group(1)) if seed_match else -1
            payload = json.loads(path.read_text())
            for result in payload.get("results", []):
                if result.get("method") != "ief_diag":
                    continue
                records.append(
                    {
                        "metric": label,
                        "seed": seed,
                        "old_wasserstein_drift": float(
                            result["old_output_sliced_wasserstein_drift"]
                        ),
                        "old_euclidean_drift": float(
                            result["old_output_euclidean_drift"]
                        ),
                        "old_denoise_mse_increase": float(
                            result["task_a_denoise_mse_increase"]
                        ),
                        "task_a_denoise_mse_after_task_b": float(
                            result["task_a_denoise_mse_after_task_b"]
                        ),
                        "task_b_denoise_mse": float(
                            result["task_b_denoise_mse_after_task_b"]
                        ),
                    }
                )
    return records


def summarize_diffusion_geometry(records: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["metric"]].append(record)
    return {
        metric: {
            "n": len(items),
            "old_wasserstein_drift_mean": mean(
                item["old_wasserstein_drift"] for item in items
            ),
            "old_wasserstein_drift_std": std(
                item["old_wasserstein_drift"] for item in items
            ),
            "old_euclidean_drift_mean": mean(
                item["old_euclidean_drift"] for item in items
            ),
            "old_denoise_mse_increase_mean": mean(
                item["old_denoise_mse_increase"] for item in items
            ),
            "task_b_denoise_mse_mean": mean(
                item["task_b_denoise_mse"] for item in items
            ),
        }
        for metric, items in grouped.items()
    }


def collect_contamination_table(clean_records: list[dict], contam_records: list[dict]):
    rows = []
    all_records = clean_records + contam_records
    methods = ["EF-EWC", "EWC-DR", "IEWC"]
    noises = sorted({0.0, *[record["noise"] for record in contam_records]})
    for method in methods:
        row = {"method": method}
        for noise in noises:
            if noise == 0.0:
                source = [
                    record
                    for record in clean_records
                    if record["method"] == method and record["noise"] == 0.0
                ]
            else:
                source = [
                    record
                    for record in all_records
                    if record["method"] == method
                    and abs(record["noise"] - noise) < 1e-12
                ]
            row[noise] = summarize_records(source, "final_avg_taw")
        rows.append(row)
    return rows, noises


def collect_spectra() -> list[dict]:
    records = []
    root = ARTIFACTS / "facil-spectrum-prefix3"
    if not root.exists():
        return records
    for stats_path in sorted(root.glob("*/importance_stats.json")):
        method = method_from_name(stats_path.parent.name)
        noise = parse_noise_probability(stats_path.parent.name)
        seed = parse_seed(stats_path.parent.name)
        payload = json.loads(stats_path.read_text())
        for task_stats in payload:
            spectrum = task_stats.get("top_diagonal_eigenvalues")
            if not spectrum:
                continue
            indices = task_stats.get("top_diagonal_indices")
            if indices and len(indices) != len(spectrum):
                indices = None
            records.append(
                {
                    "method": method,
                    "noise": noise,
                    "seed": seed,
                    "task": int(task_stats["task"]),
                    "values": [float(value) for value in spectrum],
                    "indices": [int(index) for index in indices] if indices else None,
                }
            )
    return records


def plot_summand_norm_ecdf(records: list[dict], output_dir: Path) -> None:
    if not records:
        return
    colors = {
        "EF-EWC": "#8a3f3f",
        "IEWC": "#2f7f75",
    }
    task_order = [
        task
        for task in ("Classification", "Regression", "Diffusion", "Segmentation")
        if any(item["task"] == task for item in records)
    ]
    if not task_order:
        return
    fig, axes = plt.subplots(
        1,
        len(task_order),
        figsize=(2.75 * len(task_order), 3.1),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    for ax, task in zip(axes, task_order):
        for method in ("EF-EWC", "IEWC"):
            values = sorted(
                item["effective_trace"]
                for item in records
                if item["task"] == task and item["method"] == method
            )
            if not values:
                continue
            # Subsample only for rendering density; ECDF coordinates remain exact.
            step = max(1, len(values) // 2000)
            xs = values[::step]
            ys = [(idx * step + 1) / len(values) for idx in range(len(xs))]
            if xs[-1] != values[-1]:
                xs.append(values[-1])
                ys.append(1.0)
            ax.plot(
                xs,
                ys,
                color=colors[method],
                linewidth=1.6,
                label=method,
            )
        ax.set_xscale("log")
        ax.grid(True, which="major", alpha=0.25)
        ax.text(
            0.04,
            0.94,
            task,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )
    axes[0].set_ylabel("Empirical CDF")
    axes[-1].legend(loc="lower right", frameon=False)
    fig.supxlabel(r"Effective diagonal summand trace ($\lambda \times$ trace)", y=0.02)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"summand_norm_ecdf.{suffix}", dpi=240)
    plt.close(fig)


def plot_diffusion_geometry(records: list[dict], output_dir: Path) -> None:
    if not records:
        return
    by_seed = defaultdict(dict)
    for record in records:
        by_seed[record["seed"]][record["metric"]] = record
    colors = {"Euclidean $G=I$": "#555555", "Sliced-Wasserstein $G$": "#2f7f75"}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharex=True)
    for seed, items in sorted(by_seed.items()):
        if len(items) == 2:
            ordered = [items["Euclidean $G=I$"], items["Sliced-Wasserstein $G$"]]
            axes[0].annotate(
                "",
                xy=(
                    ordered[1]["old_wasserstein_drift"],
                    ordered[1]["old_denoise_mse_increase"],
                ),
                xytext=(
                    ordered[0]["old_wasserstein_drift"],
                    ordered[0]["old_denoise_mse_increase"],
                ),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#aaaaaa",
                    "lw": 0.8,
                    "alpha": 0.7,
                },
            )
            axes[1].annotate(
                "",
                xy=(
                    ordered[1]["old_wasserstein_drift"],
                    ordered[1]["task_b_denoise_mse"],
                ),
                xytext=(
                    ordered[0]["old_wasserstein_drift"],
                    ordered[0]["task_b_denoise_mse"],
                ),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#aaaaaa",
                    "lw": 0.8,
                    "alpha": 0.7,
                },
            )
        for metric, item in items.items():
            axes[0].scatter(
                item["old_wasserstein_drift"],
                item["old_denoise_mse_increase"],
                color=colors[metric],
                s=28,
                label=metric if seed == min(by_seed) else None,
                zorder=3,
            )
            axes[1].scatter(
                item["old_wasserstein_drift"],
                item["task_b_denoise_mse"],
                color=colors[metric],
                s=28,
                zorder=3,
            )
    axes[0].set_ylabel("Old denoising MSE increase")
    axes[1].set_ylabel("New-distribution denoising MSE")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper left", frameon=False)
    fig.supxlabel("Old-output sliced-Wasserstein drift", y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"diffusion_wasserstein_geometry.{suffix}", dpi=240)
    plt.close(fig)


def plot_spectra(records: list[dict], output_dir: Path) -> None:
    if not records:
        return
    grouped = defaultdict(list)
    for record in records:
        if record["method"] not in {"EF-EWC", "IEWC"}:
            continue
        grouped[(record["method"], record["noise"])].append(record["values"])
    if not grouped:
        return
    colors = {"EF-EWC": "#8a3f3f", "IEWC": "#2f7f75"}
    plt.figure(figsize=(7.2, 4.0))
    for (method, noise), spectra in sorted(grouped.items(), key=lambda item: item[0]):
        max_len = max(len(values) for values in spectra)
        averaged = []
        for idx in range(max_len):
            vals = [values[idx] for values in spectra if idx < len(values)]
            averaged.append(mean(vals))
        xs = list(range(1, len(averaged) + 1))
        marker_positions = xs[:: max(1, len(xs) // 12)]
        marker_values = averaged[:: max(1, len(xs) // 12)]
        plt.plot(
            xs,
            averaged,
            color=colors[method],
            linewidth=1.5,
            linestyle="-" if noise == 0 else "--" if noise < 0.2 else ":",
            label=f"{method}, p={noise:g}",
        )
        plt.scatter(
            marker_positions,
            marker_values,
            color=colors[method],
            s=8,
            alpha=0.65,
        )
    plt.yscale("log")
    plt.xlabel("Sorted diagonal-entry rank")
    plt.ylabel("Mean top diagonal entry")
    plt.grid(True, which="major", alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig(output_dir / f"contamination_diagonal_spectra.{suffix}", dpi=240)
    plt.close()


def plot_classification_distributions(records: list[dict], output_dir: Path) -> None:
    records = [
        record
        for record in records
        if record["method"] in {"EF-EWC", "EWC-DR", "IEWC"}
        and record["noise"] == 0.0
        and len(record.get("final_task_taw", [])) >= 10
    ]
    if not records:
        return
    colors = {"EF-EWC": "#8a3f3f", "EWC-DR": "#6b5aa6", "IEWC": "#2f7f75"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.35), sharex=True)
    for method in ["EF-EWC", "EWC-DR", "IEWC"]:
        items = [record for record in records if record["method"] == method]
        if not items:
            continue
        max_len = min(len(record["final_task_taw"]) for record in items)
        xs = np.arange(1, max_len + 1)
        acc = np.array([record["final_task_taw"][:max_len] for record in items])
        axes[0].plot(xs, acc.mean(axis=0), marker="o", markersize=2.6, linewidth=1.2, color=colors[method], label=method)
        if len(items) > 1:
            axes[0].fill_between(
                xs,
                acc.mean(axis=0) - acc.std(axis=0, ddof=1),
                acc.mean(axis=0) + acc.std(axis=0, ddof=1),
                color=colors[method],
                alpha=0.12,
                linewidth=0,
            )
        forgetting_items = [
            record["final_task_forgetting"][:max_len]
            for record in items
            if len(record.get("final_task_forgetting", [])) >= max_len
        ]
        if forgetting_items:
            forg = np.array(forgetting_items)
            axes[1].plot(xs, forg.mean(axis=0), marker="o", markersize=2.6, linewidth=1.2, color=colors[method], label=method)
            if len(forgetting_items) > 1:
                axes[1].fill_between(
                    xs,
                    forg.mean(axis=0) - forg.std(axis=0, ddof=1),
                    forg.mean(axis=0) + forg.std(axis=0, ddof=1),
                    color=colors[method],
                    alpha=0.12,
                    linewidth=0,
                )
    axes[0].set_ylabel("Final task-aware accuracy")
    axes[1].set_ylabel("Task-aware forgetting")
    for ax in axes:
        ax.set_xlabel("CIFAR-100 distribution index")
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"classification_distribution_retention.{suffix}", dpi=240)
    plt.close(fig)


def plot_tau_sensitivity(output_dir: Path) -> None:
    path = ARTIFACTS / "facil-prefix3-iewc-diag-tau-sensitivity.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    rows = [row for row in payload.get("rows", []) if row.get("status") == "ok"]
    if len(rows) < 3:
        return
    taus = np.array([float(row["tau"]) for row in rows])
    log_tau = np.log10(taus)
    final_acc = np.array([float(row["final_avg_taw"]) for row in rows])
    forgetting = np.array([float(row["avg_forgetting_taw"]) for row in rows])
    xs = np.linspace(log_tau.min(), log_tau.max(), 200)
    degree = min(3, len(rows) - 1)
    acc_fit = np.polyval(np.polyfit(log_tau, final_acc, degree), xs)
    forg_fit = np.polyval(np.polyfit(log_tau, forgetting, degree), xs)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharex=True)
    axes[0].scatter(taus, final_acc, color="#2f7f75", s=22, zorder=3)
    axes[0].plot(10**xs, acc_fit, color="#2f7f75", linewidth=1.2)
    axes[1].scatter(taus, forgetting, color="#8a3f3f", s=22, zorder=3)
    axes[1].plot(10**xs, forg_fit, color="#8a3f3f", linewidth=1.2)
    axes[0].set_ylabel("Final avg. TAw accuracy")
    axes[1].set_ylabel("Avg. TAw forgetting")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau$")
        ax.grid(True, which="major", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"tau_sensitivity_cifar100.{suffix}", dpi=240)
    plt.close(fig)


def contamination_spectrum_metrics(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        if record["method"] not in {"EF-EWC", "IEWC"}:
            continue
        if record["noise"] not in {0.0, 0.1, 0.25}:
            continue
        grouped[(record["method"], record["noise"])].append(record)
    metrics = []
    for method in ["EF-EWC", "IEWC"]:
        clean_records = grouped.get((method, 0.0), [])
        clean = [record["values"] for record in clean_records]
        if not clean:
            continue
        common_length = min(len(values) for values in clean)
        if common_length == 0:
            continue
        clean_mean = np.array(
            [mean(values[idx] for values in clean) for idx in range(common_length)]
        )
        clean_norm = clean_mean / max(float(clean_mean.sum()), 1e-300)
        for noise in [0.1, 0.25]:
            noisy_records = grouped.get((method, noise), [])
            noisy = [record["values"] for record in noisy_records]
            if not noisy:
                continue
            length = min(common_length, *(len(values) for values in noisy))
            noisy_mean = np.array(
                [mean(values[idx] for values in noisy) for idx in range(length)]
            )
            clean_eval = clean_mean[:length]
            noisy_norm = noisy_mean / max(float(noisy_mean.sum()), 1e-300)
            clean_eval_norm = clean_eval / max(float(clean_eval.sum()), 1e-300)
            clean_by_pair = {
                (record["seed"], record["task"]): record
                for record in clean_records
                if record.get("indices")
            }
            overlaps = []
            for record in noisy_records:
                indices = record.get("indices")
                clean_record = clean_by_pair.get((record["seed"], record["task"]))
                if not indices or not clean_record or not clean_record.get("indices"):
                    continue
                clean_top = set(clean_record["indices"][:length])
                noisy_top = set(indices[:length])
                union = clean_top | noisy_top
                if union:
                    overlaps.append(len(clean_top & noisy_top) / len(union))
            metrics.append(
                {
                    "method": method,
                    "noise": noise,
                    "entry_count": int(length),
                    "diagonal_tail_mass_ratio": float(noisy_mean.sum() / clean_eval.sum()),
                    "diagonal_tail_profile_l1": float(
                        np.abs(noisy_norm - clean_eval_norm).sum()
                    ),
                    "diagonal_tail_log_rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    np.log10(np.maximum(noisy_mean, 1e-300))
                                    - np.log10(np.maximum(clean_eval, 1e-300))
                                )
                                ** 2
                            )
                        )
                    ),
                    "diagonal_tail_coordinate_jaccard": mean(overlaps)
                    if overlaps
                    else None,
                }
            )
    return metrics


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(rows[0]))]
    output = []
    for idx, row in enumerate(rows):
        output.append(
            "| "
            + " | ".join(value.ljust(widths[col]) for col, value in enumerate(row))
            + " |"
        )
        if idx == 0:
            output.append("| " + " | ".join("-" * width for width in widths) + " |")
    return "\n".join(output) + "\n"


def metric_cell(row: dict | None, *, best: bool = False, digits: int = 4) -> str:
    if not row or not row.get("n"):
        return "--"
    value = pm(row["mean"], row["std"])
    return f"**{value}**" if best else value


def method_metric_rows(
    *,
    task: str,
    metric: str,
    values: dict[str, dict],
    direction: str,
    methods: list[str],
) -> list[str]:
    finite = {
        method: row["mean"]
        for method, row in values.items()
        if row and row.get("n") and math.isfinite(row["mean"])
    }
    if direction == "max":
        winning_value = max(finite.values()) if finite else math.nan
    elif direction == "min":
        winning_value = min(finite.values()) if finite else math.nan
    else:
        raise ValueError(f"Unknown direction: {direction}")
    cells = []
    for method in methods:
        row = values.get(method)
        best = bool(
            row
            and row.get("n")
            and math.isfinite(row["mean"])
            and abs(row["mean"] - winning_value) < 1e-12
        )
        cells.append(metric_cell(row, best=best))
    return [task, metric, *cells]


def write_outputs(data: dict, output: Path) -> None:
    lines = ["# Empirical Tables And Plots", ""]
    lines += ["## Main Results", ""]
    lines += [
        "Classification uses three seeds on the full ten-distribution FACIL CIFAR-100 task-aware protocol. "
        "Regression and diffusion use five seeds; segmentation uses three seeds on VOC2012 animal/vehicle class-set foreground masks. "
        "The main cross-domain rows use diagonal matrix surrogates.",
        "",
    ]
    methods = data["main_methods"]
    rows = [["Task type", "Metric", *methods]]
    rows.extend(data["main_metric_rows"])
    lines.append(markdown_table(rows))

    lines += ["## Contamination", ""]
    contam_rows, noises = data["contamination"]
    header = ["Method"] + [f"$p={noise:g}$" for noise in noises]
    rows = [header]
    for row in contam_rows:
        rows.append(
            [
                row["method"],
                *[
                    pm(row[noise]["mean"], row[noise]["std"])
                    if row[noise]["n"]
                    else ""
                    for noise in noises
                ],
            ]
        )
    lines.append(markdown_table(rows))

    lines += ["## Label-Noise Test of the Contamination Model", ""]
    rows = [
        [
            "Method",
            "Noise",
            "Entries",
            "Stored-tail mass ratio",
            "Stored-tail profile L1",
            "Stored-tail log-scale RMSE",
        ]
    ]
    include_coordinate_overlap = any(
        row.get("diagonal_tail_coordinate_jaccard") is not None
        for row in data["spectrum_metrics"]
    )
    if include_coordinate_overlap:
        rows[0].append("Stored-coordinate overlap")
    for row in data["spectrum_metrics"]:
        table_row = [
            row["method"],
            f"{row['noise']:g}",
            str(row["entry_count"]),
            fmt(row["diagonal_tail_mass_ratio"]),
            fmt(row["diagonal_tail_profile_l1"]),
            fmt(row["diagonal_tail_log_rmse"]),
        ]
        if include_coordinate_overlap:
            table_row.append(fmt(row.get("diagonal_tail_coordinate_jaccard")))
        rows.append(table_row)
    lines.append(markdown_table(rows))

    lines += ["## Diffusion Output Geometry", ""]
    rows = [
        [
            "Metric used in IEWC",
            "Seeds",
            "Old-output SW drift",
            "Old MSE increase",
            "New-distribution MSE",
        ]
    ]
    for metric, row in data["geometry_summary"].items():
        rows.append(
            [
                metric,
                str(row["n"]),
                pm(row["old_wasserstein_drift_mean"], row["old_wasserstein_drift_std"]),
                fmt(row["old_denoise_mse_increase_mean"]),
                fmt(row["task_b_denoise_mse_mean"]),
            ]
        )
    lines.append(markdown_table(rows))

    lines += ["## Plot Files", ""]
    paper_plot_names = {
        "classification_distribution_retention.png",
        "contamination_diagonal_spectra.png",
        "diffusion_generated_samples.png",
        "diffusion_wasserstein_geometry.png",
        "summand_norm_ecdf.png",
        "tau_sensitivity_cifar100.png",
    }
    for path in sorted(PLOTS.glob("*.png")):
        if path.name not in paper_plot_names:
            continue
        lines.append(f"- `{path.relative_to(ROOT)}`")
    output.write_text("\n".join(lines) + "\n")


def build_data() -> dict:
    clean = collect_paper_classification_records()
    clean = [record for record in clean if record["noise"] == 0.0]
    contam = collect_paper_contamination_records()
    facil_summary = summarize_facil_by_method(clean)

    methods = ["Sequential", "EF-EWC", "EWC-DR", "IEWC", "IEWC-SW"]
    main_metric_rows = []
    classification_values = {
        method: {
            "n": row["n"],
            "mean": row["final_avg_taw_mean"],
            "std": row["final_avg_taw_std"],
        }
        for method, row in facil_summary.items()
        if method in methods
    }
    classification_forgetting = {
        method: {
            "n": row["n"],
            "mean": row["final_avg_forgetting_mean"],
            "std": row["final_avg_forgetting_std"],
        }
        for method, row in facil_summary.items()
        if method in methods
    }
    main_metric_rows.append(
        method_metric_rows(
            task="Classification",
            metric="Final avg. TAw accuracy ↑",
            values=classification_values,
            direction="max",
            methods=methods,
        )
    )
    main_metric_rows.append(
        method_metric_rows(
            task="Classification",
            metric="Avg. TAw forgetting ↓",
            values=classification_forgetting,
            direction="min",
            methods=methods,
        )
    )

    task_specs = [
        (
            "Regression",
            "regression",
            collect_synthetic_results("final-synthetic-regression-seed*.json"),
            [
                (
                    "Old-distribution MSE after new distribution ↓",
                    "task_a_mse_after_task_b",
                    "min",
                ),
                ("Forgetting (MSE increase) ↓", "task_a_mse_increase", "min"),
                ("New-distribution MSE ↓", "task_b_mse_after_task_b", "min"),
            ],
        ),
        (
            "Diffusion",
            "diffusion",
            collect_synthetic_results("final-mnist-diffusion-seed*.json"),
            [
                (
                    "Old-distribution denoising MSE after new distribution ↓",
                    "task_a_denoise_mse_after_task_b",
                    "min",
                ),
                ("Forgetting (MSE increase) ↓", "task_a_denoise_mse_increase", "min"),
                ("New-distribution denoising MSE ↓", "task_b_denoise_mse_after_task_b", "min"),
            ],
        ),
    ]
    geometry_records = collect_diffusion_geometry()
    diffusion_ef_override = collect_synthetic_results(
        "final-mnist-diffusion-ef-lam1e4-seed*.json"
    )
    wasserstein_records = [
        {
            "file": "diffusion-geometry-wasserstein",
            "method": "IEWC-SW",
            "result": {
                "task_a_denoise_mse_increase": record["old_denoise_mse_increase"],
                "task_a_denoise_mse_after_task_b": record[
                    "task_a_denoise_mse_after_task_b"
                ],
                "task_b_denoise_mse_after_task_b": record["task_b_denoise_mse"],
            },
        }
        for record in geometry_records
        if record["metric"] == "Sliced-Wasserstein $G$"
    ]
    for task_name, sweep_key, records, metrics in task_specs:
        ef_records = (
            diffusion_ef_override
            if task_name == "Diffusion"
            else best_ef_sweep_records(sweep_key)
        )
        records_for_table = [
            record for record in records if record["method"] != "EF-EWC"
        ] + ef_records
        if task_name == "Diffusion":
            records_for_table = records_for_table + wasserstein_records
        for metric_label, result_key, direction in metrics:
            values = summarize_metric(
                records_for_table,
                lambda result, key=result_key: result.get(key),
            )
            main_metric_rows.append(
                method_metric_rows(
                    task=task_name,
                    metric=metric_label,
                    values=values,
                    direction=direction,
                    methods=methods,
                )
            )

    segmentation_records = collect_segmentation_records()
    segmentation_metrics = [
        ("Old class-set foreground IoU after new class set ↑", "old", "max"),
        ("Forgetting (IoU decrease) ↓", "forgetting", "min"),
        ("New class-set foreground IoU ↑", "new", "max"),
    ]
    for metric_label, record_key, direction in segmentation_metrics:
        values = summarize_metric(
            segmentation_records,
            lambda result, key=record_key: result.get(key),
        )
        main_metric_rows.append(
            method_metric_rows(
                task="Segmentation",
                metric=metric_label,
                values=values,
                direction=direction,
                methods=methods,
            )
        )

    return {
        "main_methods": methods,
        "main_metric_rows": main_metric_rows,
        "contamination": collect_contamination_table(clean, contam),
        "spectrum_metrics": contamination_spectrum_metrics(collect_spectra()),
        "geometry_summary": summarize_diffusion_geometry(geometry_records),
        "mnist": collect_mnist_low_loss(),
        "classification_records": clean,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS / "paper-tables-and-plots.md",
    )
    args = parser.parse_args()

    PLOTS.mkdir(parents=True, exist_ok=True)
    for obsolete in (
        "facil_final_accuracy.png",
        "facil_final_accuracy.pdf",
        "iewc_summand_scales.png",
        "summand_coefficient_ecdf.png",
        "summand_coefficient_ecdf.pdf",
        "loss_scale_medians.png",
        "synthetic_summand_coefficients.png",
        "stability_plasticity_pareto.png",
        "stability_plasticity_pareto.pdf",
    ):
        path = PLOTS / obsolete
        if path.exists():
            path.unlink()

    summand_records = (
        collect_facil_summand_norms()
        + collect_synthetic_summand_norms()
        + collect_segmentation_summand_norms()
    )
    plot_summand_norm_ecdf(summand_records, PLOTS)
    geometry_records = collect_diffusion_geometry()
    plot_diffusion_geometry(geometry_records, PLOTS)
    spectra = collect_spectra()
    plot_spectra(spectra, PLOTS)
    classification_records = collect_paper_classification_records()
    plot_classification_distributions(classification_records, PLOTS)
    plot_tau_sensitivity(PLOTS)

    data = build_data()
    write_outputs(data, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
