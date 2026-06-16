import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "empirical-evidence" / "artifacts"
OUT_ROOT = ARTIFACTS / "lambda-tradeoff-internal"
PLOTS = OUT_ROOT / "plots"
REPORT = OUT_ROOT / "lambda-tradeoff-internal.md"

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


def fmt(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    value = float(value)
    if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def read_matrix(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append([float(value) for value in line.split()])
    return rows


def latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def final_nonzero_values(row: list[float]) -> list[float]:
    return [value for value in row if value != 0.0]


def old_new_from_acc_file(path: Path) -> tuple[float, float] | None:
    rows = read_matrix(path)
    if not rows:
        return None
    values = final_nonzero_values(rows[-1])
    if len(values) < 2:
        return None
    return mean(values[:-1]), values[-1]


def parse_method(name: str) -> str:
    match = re.search(r"lambda_tradeoff_prefix3_(ef|ewcdr|iewc)_", name)
    if not match:
        return "Other"
    return {"ef": "EF-EWC", "ewcdr": "EWC-DR", "iewc": "IEWC"}[match.group(1)]


def parse_lambda(name: str) -> float | None:
    match = re.search(r"_lam([0-9epm]+)(?:_|$)", name)
    if not match:
        return None
    text = match.group(1).replace("p", ".").replace("m", "-")
    try:
        return float(text)
    except ValueError:
        return None


def collect_classification() -> list[dict]:
    records = []
    root = OUT_ROOT / "facil-classification"
    if root.exists():
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            method = parse_method(run_dir.name)
            lamb = parse_lambda(run_dir.name)
            if method == "Other" or lamb is None:
                continue
            acc_path = latest_file(run_dir / "results", "acc_taw-*.txt")
            if acc_path is None:
                continue
            old_new = old_new_from_acc_file(acc_path)
            if old_new is None:
                continue
            records.append(
                {
                    "panel": "Classification",
                    "method": method,
                    "lambda": lamb,
                    "old": old_new[0],
                    "new": old_new[1],
                    "old_label": "Old-distribution accuracy",
                    "new_label": "New-distribution accuracy",
                    "source": str(acc_path.relative_to(ROOT)),
                }
            )
    seq_path = (
        ARTIFACTS
        / "facil-prefix"
        / "cifar100_icarl_finetuning_prefix3_finetuning_seed0_e60_lr0p05"
        / "results"
    )
    acc_path = latest_file(seq_path, "acc_taw-*.txt")
    if acc_path is not None:
        old_new = old_new_from_acc_file(acc_path)
        if old_new is not None:
            records.append(
                {
                    "panel": "Classification",
                    "method": "Sequential",
                    "lambda": None,
                    "old": old_new[0],
                    "new": old_new[1],
                    "old_label": "Old-distribution accuracy",
                    "new_label": "New-distribution accuracy",
                    "source": str(acc_path.relative_to(ROOT)),
                }
            )
    return records


def collect_regression() -> list[dict]:
    grouped = defaultdict(list)
    for path in sorted((OUT_ROOT / "regression").glob("*.json")):
        payload = json.loads(path.read_text())
        lamb = float(payload["config"]["ewc_lambda"])
        for result in payload.get("results", []):
            method = {
                "sequential": "Sequential",
                "ef": "EF-EWC",
                "ief_diag": "IEWC",
            }.get(result.get("method"), result.get("method"))
            key = (method, None if method == "Sequential" else lamb)
            grouped[key].append(
                {
                    "old_mse": float(result["task_a_mse_after_task_b"]),
                    "new_mse": float(result["task_b_mse_after_task_b"]),
                    "source": str(path.relative_to(ROOT)),
                }
            )
    records = []
    for (method, lamb), items in sorted(
        grouped.items(),
        key=lambda item: (
            {"Sequential": 0, "EF-EWC": 1, "IEWC": 2}.get(item[0][0], 9),
            -math.inf if item[0][1] is None else item[0][1],
        ),
    ):
        old_mse = [item["old_mse"] for item in items]
        new_mse = [item["new_mse"] for item in items]
        records.append(
            {
                "panel": "Regression",
                "method": method,
                "lambda": lamb,
                "old": -mean(old_mse),
                "new": -mean(new_mse),
                "old_mse": mean(old_mse),
                "old_mse_std": std(old_mse),
                "new_mse": mean(new_mse),
                "new_mse_std": std(new_mse),
                "n": len(items),
                "old_label": "Old-distribution performance (-MSE)",
                "new_label": "New-distribution performance (-MSE)",
                "source": ", ".join(sorted({item["source"] for item in items})[:3]),
            }
        )
    return records


def lambda_label(value: float | None) -> str:
    if value is None:
        return "seq."
    if value != 0.0 and (abs(value) < 1e-2 or abs(value) >= 1e4):
        return f"{value:.0e}"
    return f"{value:g}"


def plot_tradeoff(records: list[dict]) -> None:
    if not records:
        return
    PLOTS.mkdir(parents=True, exist_ok=True)
    colors = {
        "Sequential": "#555555",
        "EF-EWC": "#8a3f3f",
        "EWC-DR": "#6b5aa6",
        "IEWC": "#2f7f75",
    }
    markers = {"Sequential": "x", "EF-EWC": "o", "EWC-DR": "s", "IEWC": "^"}
    panels = [panel for panel in ("Classification", "Regression") if any(r["panel"] == panel for r in records)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.35), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        panel_records = [record for record in records if record["panel"] == panel]
        for method in ("Sequential", "EF-EWC", "EWC-DR", "IEWC"):
            items = [record for record in panel_records if record["method"] == method]
            if not items:
                continue
            if method != "Sequential":
                items = sorted(items, key=lambda item: float(item["lambda"]))
                ax.plot(
                    [item["old"] for item in items],
                    [item["new"] for item in items],
                    color=colors[method],
                    linewidth=1.0,
                    alpha=0.65,
                    zorder=2,
                )
            ax.scatter(
                [item["old"] for item in items],
                [item["new"] for item in items],
                color=colors[method],
                marker=markers[method],
                s=24,
                label=method,
                zorder=3,
            )
            label_indices = {0, len(items) // 2, len(items) - 1}
            for item_index, item in enumerate(items):
                if method == "Sequential":
                    continue
                if item_index not in label_indices:
                    continue
                ax.annotate(
                    lambda_label(item["lambda"]),
                    (item["old"], item["new"]),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=5.8,
                    color=colors[method],
                )
        exemplar = panel_records[0]
        ax.set_xlabel(exemplar["old_label"])
        ax.set_ylabel(exemplar["new_label"])
        ax.grid(True, alpha=0.25)
        ax.text(
            0.04,
            0.94,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )
    axes[0, 0].legend(loc="best", frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(PLOTS / f"lambda_tradeoff_internal.{suffix}", dpi=240)
    plt.close(fig)


def write_report(classification: list[dict], regression: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Internal Lambda-Only Stability-Plasticity Sweep",
        "",
        "This is an internal, high-span lambda-only experiment. The classification panel uses the CIFAR-100 three-distribution FACIL prefix with seed 0. The regression panel uses the accepted phase-shift regression setup over five seeds. In both panels, only the regularization strength is changed within a mechanism; all other mechanism-specific settings are fixed.",
        "",
        "Lambda grids use factors `1e-3, 1e-1, 1, 1e1, 1e3` around each midpoint: classification EF-EWC/IEWC `10000`, classification EWC-DR `100`, regression EF-EWC `30`, and regression IEWC `1`.",
        "",
        "![Internal lambda tradeoff](plots/lambda_tradeoff_internal.png)",
        "",
        "## Classification",
        "",
        "| Method | Lambda | Old-distribution accuracy | New-distribution accuracy |",
        "| ------ | ------ | ------------------------- | ------------------------- |",
    ]
    for record in sorted(
        classification,
        key=lambda item: (
            {"Sequential": 0, "EF-EWC": 1, "EWC-DR": 2, "IEWC": 3}.get(item["method"], 9),
            -math.inf if item["lambda"] is None else float(item["lambda"]),
        ),
    ):
        lines.append(
            f"| {record['method']} | {lambda_label(record['lambda'])} | "
            f"{fmt(record['old'])} | {fmt(record['new'])} |"
        )
    lines.extend(
        [
            "",
            "## Regression",
            "",
            "| Method | Lambda | Seeds | Old-distribution MSE | New-distribution MSE |",
            "| ------ | ------ | ----- | -------------------- | -------------------- |",
        ]
    )
    for record in sorted(
        regression,
        key=lambda item: (
            {"Sequential": 0, "EF-EWC": 1, "IEWC": 2}.get(item["method"], 9),
            -math.inf if item["lambda"] is None else float(item["lambda"]),
        ),
    ):
        lines.append(
            f"| {record['method']} | {lambda_label(record['lambda'])} | "
            f"{record.get('n', 1)} | {fmt(record['old_mse'])} +/- {fmt(record['old_mse_std'])} | "
            f"{fmt(record['new_mse'])} +/- {fmt(record['new_mse_std'])} |"
        )
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    classification = collect_classification()
    regression = collect_regression()
    records = classification + regression
    plot_tradeoff(records)
    write_report(classification, regression)
    print(REPORT)


if __name__ == "__main__":
    main()
