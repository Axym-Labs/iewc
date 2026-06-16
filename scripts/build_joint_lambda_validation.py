import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "empirical-evidence" / "artifacts"
OUT_ROOT = ARTIFACTS / "joint-lambda-validation"
RAW = OUT_ROOT / "raw"
PLOTS = OUT_ROOT / "plots"
REPORT = OUT_ROOT / "joint-lambda-validation.md"

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


TASK_ORDER = ["Classification", "Regression", "Diffusion", "Segmentation"]
METHOD_ORDER = ["Sequential", "EF-EWC", "EWC-DR", "IEWC", "IEWC-SW"]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def std(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    value = float(value)
    if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def pm(mu: float, sigma: float, digits: int = 4) -> str:
    if not math.isfinite(mu):
        return "--"
    if sigma == 0.0:
        return fmt(mu, digits)
    return f"{fmt(mu, digits)} +/- {fmt(sigma, digits)}"


def lambda_label(value: float | None) -> str:
    if value is None:
        return "seq."
    if value != 0.0 and (abs(value) < 1e-2 or abs(value) >= 1e4):
        return f"{value:.0e}"
    return f"{value:g}"


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(rows[0]))]
    output = []
    for row_idx, row in enumerate(rows):
        output.append(
            "| "
            + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))
            + " |"
        )
        if row_idx == 0:
            output.append("| " + " | ".join("-" * width for width in widths) + " |")
    return "\n".join(output) + "\n"


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


def parse_lambda(name: str) -> float | None:
    match = re.search(r"_lam([0-9epm]+)(?:_|$)", name)
    if not match:
        return None
    text = match.group(1).replace("p", ".").replace("m", "-")
    try:
        return float(text)
    except ValueError:
        return None


def parse_classification_method(name: str) -> str | None:
    match = re.search(r"lambda_tradeoff_prefix3_(ef|ewcdr|iewc)_", name)
    if not match:
        return None
    return {"ef": "EF-EWC", "ewcdr": "EWC-DR", "iewc": "IEWC"}[match.group(1)]


def add_record(
    records: list[dict],
    *,
    task: str,
    method: str,
    lamb: float | None,
    joint: float,
    old: float | None,
    new: float | None,
    forgetting: float | None,
    direction: str,
    metric_label: str,
    source: Path,
):
    records.append(
        {
            "task": task,
            "method": method,
            "lambda": lamb,
            "joint": float(joint),
            "old": None if old is None else float(old),
            "new": None if new is None else float(new),
            "forgetting": None if forgetting is None else float(forgetting),
            "direction": direction,
            "metric_label": metric_label,
            "source": str(source.relative_to(ROOT)),
        }
    )


def collect_classification(records: list[dict]) -> None:
    root = ARTIFACTS / "lambda-tradeoff-internal" / "facil-classification"
    if root.exists():
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            method = parse_classification_method(run_dir.name)
            lamb = parse_lambda(run_dir.name)
            if method is None or lamb is None:
                continue
            acc_path = latest_file(run_dir / "results", "acc_taw-*.txt")
            if acc_path is None:
                continue
            values = final_nonzero_values(read_matrix(acc_path)[-1])
            if len(values) < 2:
                continue
            forg_path = latest_file(run_dir / "results", "forg_taw-*.txt")
            forgetting = None
            if forg_path is not None:
                fvals = final_nonzero_values(read_matrix(forg_path)[-1])
                if fvals:
                    forgetting = mean(fvals[:-1]) if len(fvals) > 1 else mean(fvals)
            add_record(
                records,
                task="Classification",
                method=method,
                lamb=lamb,
                joint=mean(values),
                old=mean(values[:-1]),
                new=values[-1],
                forgetting=forgetting,
                direction="max",
                metric_label="Final avg. TAw accuracy",
                source=acc_path,
            )
    seq_root = ARTIFACTS / "facil-prefix"
    if seq_root.exists():
        for run_dir in sorted(seq_root.glob("*finetuning_prefix3_finetuning_seed*")):
            acc_path = latest_file(run_dir / "results", "acc_taw-*.txt")
            if acc_path is None:
                continue
            values = final_nonzero_values(read_matrix(acc_path)[-1])
            add_record(
                records,
                task="Classification",
                method="Sequential",
                lamb=None,
                joint=mean(values),
                old=mean(values[:-1]),
                new=values[-1],
                forgetting=None,
                direction="max",
                metric_label="Final avg. TAw accuracy",
                source=acc_path,
            )


def collect_regression(records: list[dict]) -> None:
    for path in sorted((RAW / "regression").glob("*.json")):
        payload = json.loads(path.read_text())
        lamb = None
        if payload["config"]["methods"] != ["sequential"]:
            lamb = float(payload["config"]["ewc_lambda"])
        for result in payload.get("results", []):
            method = {
                "sequential": "Sequential",
                "ef": "EF-EWC",
                "ief_diag": "IEWC",
            }.get(result["method"], result["method"])
            add_record(
                records,
                task="Regression",
                method=method,
                lamb=None if method == "Sequential" else lamb,
                joint=float(result["final_avg_mse"]),
                old=float(result["old_avg_mse_after_final"]),
                new=float(result["new_task_mse_after_final"]),
                forgetting=float(result["avg_forgetting_mse"]),
                direction="min",
                metric_label="Final avg. MSE",
                source=path,
            )


def collect_diffusion(records: list[dict]) -> None:
    for path in sorted((RAW / "diffusion").glob("*.json")):
        payload = json.loads(path.read_text())
        lamb = None
        if payload["config"]["methods"] != ["sequential"]:
            lamb = float(payload["config"]["ewc_lambda"])
        for result in payload.get("results", []):
            method = {
                "sequential": "Sequential",
                "ef": "EF-EWC",
                "ief_diag": "IEWC",
            }.get(result["method"], result["method"])
            old = float(result["task_a_denoise_mse_after_task_b"])
            new = float(result["task_b_denoise_mse_after_task_b"])
            add_record(
                records,
                task="Diffusion",
                method=method,
                lamb=None if method == "Sequential" else lamb,
                joint=0.5 * (old + new),
                old=old,
                new=new,
                forgetting=float(result["task_a_denoise_mse_increase"]),
                direction="min",
                metric_label="Final avg. denoising MSE",
                source=path,
            )
    for path in sorted(ARTIFACTS.glob("diffusion-geometry-wasserstein-lam*seed*.json")):
        payload = json.loads(path.read_text())
        lamb = float(payload["config"]["ewc_lambda"])
        for result in payload.get("results", []):
            if result.get("method") != "ief_diag":
                continue
            old = float(result["task_a_denoise_mse_after_task_b"])
            new = float(result["task_b_denoise_mse_after_task_b"])
            add_record(
                records,
                task="Diffusion",
                method="IEWC-SW",
                lamb=lamb,
                joint=0.5 * (old + new),
                old=old,
                new=new,
                forgetting=float(result["task_a_denoise_mse_increase"]),
                direction="min",
                metric_label="Final avg. denoising MSE",
                source=path,
            )


def _single_result(path: Path, method: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    for result in payload.get("results", []):
        if result.get("method") == method:
            return result
    return None


def collect_segmentation(records: list[dict]) -> None:
    specs = [
        ("Sequential", None, [
            (ARTIFACTS / "voc-segmentation-pilot-seed0-lam10.json", "sequential"),
            (ARTIFACTS / "voc-segmentation-final-seed1-sequential.json", "sequential"),
            (ARTIFACTS / "voc-segmentation-final-seed2-sequential.json", "sequential"),
        ]),
        ("EF-EWC", 1.0, [
            (ARTIFACTS / "voc-segmentation-pilot-seed0-lam1.json", "ef"),
            (ARTIFACTS / "voc-segmentation-final-seed1-ef-lam1.json", "ef"),
            (ARTIFACTS / "voc-segmentation-final-seed2-ef-lam1.json", "ef"),
        ]),
        ("EF-EWC", 3.0, [
            (ARTIFACTS / "voc-segmentation-pilot-seed0-lam3.json", "ef"),
            (ARTIFACTS / "voc-segmentation-final-seed1-lam3.json", "ef"),
            (ARTIFACTS / "voc-segmentation-final-seed2-lam3.json", "ef"),
        ]),
        ("EF-EWC", 30.0, [
            (ARTIFACTS / "voc-segmentation-pilot-seed0-lam30.json", "ef"),
            (ARTIFACTS / "voc-segmentation-final-seed1-ef-lam30.json", "ef"),
            (ARTIFACTS / "voc-segmentation-final-seed2-ef-lam30.json", "ef"),
        ]),
        ("IEWC", 3.0, [
            (ARTIFACTS / "voc-segmentation-pilot-seed0-lam3.json", "ief_diag"),
            (ARTIFACTS / "voc-segmentation-final-seed1-lam3.json", "ief_diag"),
            (ARTIFACTS / "voc-segmentation-final-seed2-lam3.json", "ief_diag"),
        ]),
        ("IEWC", 10.0, [
            (ARTIFACTS / "voc-segmentation-pilot-seed0-lam10.json", "ief_diag"),
            (ARTIFACTS / "voc-segmentation-final-seed1-iewc-lam10.json", "ief_diag"),
            (ARTIFACTS / "voc-segmentation-final-seed2-iewc-lam10.json", "ief_diag"),
        ]),
        ("IEWC", 30.0, [
            (ARTIFACTS / "voc-segmentation-pilot-seed0-lam30.json", "ief_diag"),
            (ARTIFACTS / "voc-segmentation-final-seed1-iewc-lam30.json", "ief_diag"),
            (ARTIFACTS / "voc-segmentation-final-seed2-iewc-lam30.json", "ief_diag"),
        ]),
        ("IEWC", 100.0, [
            (ARTIFACTS / "voc-segmentation-pilot-seed0-lam100.json", "ief_diag"),
            (ARTIFACTS / "voc-segmentation-final-seed1-iewc-lam100.json", "ief_diag"),
            (ARTIFACTS / "voc-segmentation-final-seed2-iewc-lam100.json", "ief_diag"),
        ]),
    ]
    for method, lamb, paths in specs:
        for path, raw_method in paths:
            result = _single_result(path, raw_method)
            if result is None:
                continue
            old = float(result["task_a_iou_after_task_b"])
            new = float(result["task_b_iou_after_task_b"])
            add_record(
                records,
                task="Segmentation",
                method=method,
                lamb=lamb,
                joint=0.5 * (old + new),
                old=old,
                new=new,
                forgetting=float(result["task_a_iou_forgetting"]),
                direction="max",
                metric_label="Final avg. foreground IoU",
                source=path,
            )


def aggregate(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["task"], record["method"], record["lambda"])].append(record)
    rows = []
    for (task, method, lamb), items in grouped.items():
        rows.append(
            {
                "task": task,
                "method": method,
                "lambda": lamb,
                "direction": items[0]["direction"],
                "metric_label": items[0]["metric_label"],
                "n": len(items),
                "joint_mean": mean(item["joint"] for item in items),
                "joint_std": std(item["joint"] for item in items),
                "old_mean": mean(item["old"] for item in items if item["old"] is not None),
                "old_std": std(item["old"] for item in items if item["old"] is not None),
                "new_mean": mean(item["new"] for item in items if item["new"] is not None),
                "new_std": std(item["new"] for item in items if item["new"] is not None),
                "forgetting_mean": mean(
                    item["forgetting"] for item in items if item["forgetting"] is not None
                ),
                "forgetting_std": std(
                    item["forgetting"] for item in items if item["forgetting"] is not None
                ),
                "sources": sorted({item["source"] for item in items}),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            TASK_ORDER.index(row["task"]) if row["task"] in TASK_ORDER else 99,
            METHOD_ORDER.index(row["method"]) if row["method"] in METHOD_ORDER else 99,
            -math.inf if row["lambda"] is None else row["lambda"],
        ),
    )


def selected_rows(rows: list[dict]) -> list[dict]:
    selected = []
    by_pair = defaultdict(list)
    for row in rows:
        if row["method"] == "Sequential":
            row = dict(row)
            row["edge"] = False
            selected.append(row)
            continue
        by_pair[(row["task"], row["method"])].append(row)
    for (_task, _method), items in by_pair.items():
        direction = items[0]["direction"]
        key = (
            (lambda row: row["joint_mean"])
            if direction == "max"
            else (lambda row: -row["joint_mean"])
        )
        best = max(items, key=key)
        lambdas = sorted(row["lambda"] for row in items if row["lambda"] is not None)
        best = dict(best)
        best["edge"] = best["lambda"] in {lambdas[0], lambdas[-1]} if lambdas else False
        selected.append(best)
    return sorted(
        selected,
        key=lambda row: (
            TASK_ORDER.index(row["task"]) if row["task"] in TASK_ORDER else 99,
            METHOD_ORDER.index(row["method"]) if row["method"] in METHOD_ORDER else 99,
        ),
    )


def plot_method(rows: list[dict], method: str) -> Path | None:
    method_rows = [row for row in rows if row["method"] == method and row["lambda"] is not None]
    if not method_rows:
        return None
    PLOTS.mkdir(parents=True, exist_ok=True)
    tasks = [task for task in TASK_ORDER if any(row["task"] == task for row in method_rows)]
    fig, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(3.0 * len(tasks), 2.65),
        squeeze=False,
    )
    for ax, task in zip(axes[0], tasks):
        task_rows = sorted(
            [row for row in method_rows if row["task"] == task],
            key=lambda row: row["lambda"],
        )
        xs = np.array([row["lambda"] for row in task_rows], dtype=float)
        ys = np.array([row["joint_mean"] for row in task_rows], dtype=float)
        yerr = np.array([row["joint_std"] for row in task_rows], dtype=float)
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            color="#2f5f8f",
            marker="o",
            linestyle="none",
            elinewidth=1.0,
            markersize=3.2,
            capsize=2.0,
        )
        if len(xs) >= 2:
            degree = min(2, len(xs) - 1)
            log_x = np.log10(xs)
            grid = np.linspace(log_x.min(), log_x.max(), 200)
            coeffs = np.polyfit(log_x, ys, degree)
            ax.plot(
                np.power(10.0, grid),
                np.polyval(coeffs, grid),
                color="#8a3f3f",
                linewidth=1.0,
                alpha=0.75,
            )
        direction = task_rows[0]["direction"]
        best = max(task_rows, key=lambda row: row["joint_mean"]) if direction == "max" else min(task_rows, key=lambda row: row["joint_mean"])
        ax.scatter(
            [best["lambda"]],
            [best["joint_mean"]],
            color="#000000",
            marker="*",
            s=42,
            zorder=5,
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(task_rows[0]["metric_label"])
        ax.grid(True, which="major", alpha=0.25)
        ax.text(
            0.04,
            0.06,
            task,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    slug = method.lower().replace("-", "").replace(" ", "_")
    for suffix in ("png", "pdf"):
        fig.savefig(PLOTS / f"joint_lambda_{slug}.{suffix}", dpi=240)
    plt.close(fig)
    return PLOTS / f"joint_lambda_{slug}.png"


def write_report(rows: list[dict], selected: list[dict], plot_paths: list[Path]) -> None:
    lines = ["# Joint-Performance Lambda Validation", ""]
    lines += [
        "This artifact evaluates lambda selection with a joint final-performance objective. "
        "Classification uses the existing three-distribution CIFAR-100 FACIL prefix; regression uses a new five-distribution phase/amplitude stream; diffusion uses the accepted two-distribution DDPM denoising setup; segmentation uses the accepted two-step VOC animal/vehicle class-set setup.",
        "",
        "For accuracy and IoU, larger joint values are better. For regression and diffusion MSE, smaller joint values are better. If the best value is on a grid edge, the intended follow-up rule is to extend the grid in that direction up to three times, stopping once the optimum is bracketed.",
        "",
    ]
    lines += ["## Selected Lambda By Joint Objective", ""]
    selected_table = [[
        "Task",
        "Method",
        "Lambda",
        "Joint metric",
        "Old final",
        "New final",
        "Forgetting",
        "n",
        "Edge?",
    ]]
    for row in selected:
        selected_table.append(
            [
                row["task"],
                row["method"],
                lambda_label(row["lambda"]),
                pm(row["joint_mean"], row["joint_std"]),
                pm(row["old_mean"], row["old_std"]),
                pm(row["new_mean"], row["new_std"]),
                pm(row["forgetting_mean"], row["forgetting_std"]),
                str(row["n"]),
                "yes" if row["edge"] else "no",
            ]
        )
    lines.append(markdown_table(selected_table))
    lines += ["## Lambda Curves", ""]
    for path in plot_paths:
        lines += [f"![{path.stem}]({path.relative_to(OUT_ROOT)})", ""]
    lines += ["## Full Lambda Table", ""]
    detail_table = [[
        "Task",
        "Method",
        "Lambda",
        "Joint metric",
        "Old final",
        "New final",
        "Forgetting",
        "n",
    ]]
    for row in rows:
        detail_table.append(
            [
                row["task"],
                row["method"],
                lambda_label(row["lambda"]),
                pm(row["joint_mean"], row["joint_std"]),
                pm(row["old_mean"], row["old_std"]),
                pm(row["new_mean"], row["new_std"]),
                pm(row["forgetting_mean"], row["forgetting_std"]),
                str(row["n"]),
            ]
        )
    lines.append(markdown_table(detail_table))
    edge_rows = [row for row in selected if row["edge"]]
    if edge_rows:
        lines += ["## Edge Follow-Up", ""]
        for row in edge_rows:
            direction = "higher" if row["lambda"] == max(
                candidate["lambda"]
                for candidate in rows
                if candidate["task"] == row["task"]
                and candidate["method"] == row["method"]
                and candidate["lambda"] is not None
            ) else "lower"
            lines.append(
                f"- `{row['task']} / {row['method']}` selected the {direction} grid edge at `lambda={lambda_label(row['lambda'])}`."
            )
        lines.append("")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


def main() -> None:
    records: list[dict] = []
    collect_classification(records)
    collect_regression(records)
    collect_diffusion(records)
    collect_segmentation(records)
    rows = aggregate(records)
    selected = selected_rows(rows)
    plot_paths = []
    for method in METHOD_ORDER:
        path = plot_method(rows, method)
        if path is not None:
            plot_paths.append(path)
    write_report(rows, selected, plot_paths)


if __name__ == "__main__":
    main()
