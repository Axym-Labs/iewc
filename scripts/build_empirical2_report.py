import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "empirical-2" / "artifacts"
PLOTS = ARTIFACTS / "plots"
REPORT = ARTIFACTS / "internal-report.md"


def load_results() -> list[dict]:
    results = []
    for path in sorted(ARTIFACTS.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        data["_path"] = path.name
        results.append(data)
    return results


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows yet._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def plot_metric(results: list[dict], experiment: str, metric: str, ylabel: str, filename: str) -> str | None:
    rows = [result for result in results if result.get("experiment") == experiment and metric in result]
    if not rows:
        return None
    labels = []
    seen = defaultdict(int)
    for result in rows:
        method = str(result.get("method", "method"))
        config = result.get("config", {})
        suffix = ""
        if experiment == "empirical2_vision_cl":
            suffix = str(config.get("adaptation", ""))
        elif experiment == "empirical2_nlp_cl":
            suffix = str(config.get("adaptation", ""))
        label = f"{method}\n{suffix}" if suffix else method
        seen[label] += 1
        if seen[label] > 1:
            label = f"{label}\nseed {config.get('seed', seen[label] - 1)}"
        labels.append(label)
    values = [float(result[metric]) for result in rows]
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(4.0, 1.2 * len(rows)), 2.6))
    ax.bar(range(len(rows)), values, color="#4477AA")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    out = PLOTS / filename
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return f"plots/{filename}"


def plot_forecast_rollouts(results: list[dict]) -> list[str]:
    outputs = []
    for result in results:
        if result.get("experiment") != "empirical2_m4_forecasting_cl":
            continue
        method = result.get("method", "method")
        rollouts = result.get("rollouts", {})
        if not rollouts:
            continue
        frequencies = list(rollouts)[:3]
        if not frequencies:
            continue
        PLOTS.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(len(frequencies), 1, figsize=(5.2, 1.8 * len(frequencies)), squeeze=False)
        for ax, frequency in zip(axes[:, 0], frequencies):
            rollout = rollouts[frequency][0]
            context = rollout["context"]
            target = rollout["target"]
            prediction = rollout["prediction"]
            x_context = list(range(len(context)))
            x_future = list(range(len(context), len(context) + len(target)))
            ax.plot(x_context, context, color="#666666", linewidth=1.0, label="context")
            ax.plot(x_future, target, color="#228833", marker="o", markersize=2.5, linewidth=1.0, label="target")
            ax.plot(x_future, prediction, color="#CC6677", marker="s", markersize=2.5, linewidth=1.0, label="prediction")
            ax.text(0.01, 0.05, frequency, transform=ax.transAxes, fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
        fig.tight_layout()
        filename = f"forecast_rollouts_{method}_{result.get('_path', 'result').replace('.json', '')}.png"
        out = PLOTS / filename
        fig.savefig(out, dpi=180)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
        outputs.append(f"plots/{filename}")
    return outputs


def build_report(results: list[dict], plot_paths: dict[str, str | None], rollout_paths: list[str]) -> str:
    grouped = defaultdict(list)
    for result in results:
        grouped[result.get("experiment", "unknown")].append(result)

    sections = [
        "# Empirical-2 Internal Report",
        "",
        "This report collects exploratory evidence for larger continual-learning settings. It is intentionally internal until the setups are validated.",
        "",
        "## Current Artifact Index",
        "",
        md_table(
            ["file", "experiment", "method", "key metric"],
            [
                [
                    result.get("_path", ""),
                    result.get("experiment", ""),
                    result.get("method", ""),
                    fmt(
                        result.get(
                            "final_avg_accuracy",
                            result.get("final_avg_mse", ""),
                        )
                    ),
                ]
                for result in results
            ],
        ),
    ]

    for experiment, rows in sorted(grouped.items()):
        sections.extend(["", f"## {experiment}", ""])
        table_rows = []
        for result in rows:
            table_rows.append(
                [
                    result.get("_path", ""),
                    result.get("method", ""),
                    fmt(result.get("final_avg_accuracy", result.get("final_avg_mse", ""))),
                    fmt(result.get("avg_forgetting", result.get("avg_forgetting_mse", ""))),
                    fmt(result.get("n_trainable_parameters", result.get("n_parameters", ""))),
                ]
            )
        sections.append(
            md_table(
                ["file", "method", "joint metric", "forgetting", "trainable/params"],
                table_rows,
            )
        )

    sections.extend(["", "## Plots", ""])
    for label, path in plot_paths.items():
        if path:
            sections.extend([f"### {label}", "", f"![{label}]({path})", ""])
    for path in rollout_paths:
        sections.extend(["### Forecast Rollout", "", f"![Forecast rollout]({path})", ""])

    sections.extend(
        [
            "## Decisions And Guard Notes",
            "",
            "- Smoke-test artifacts are implementation checks only; they are not evidence for method quality.",
            "- Before heavy runs, require non-degenerate sequential learning and visible separation among methods or lambdas.",
            "- For task groups where all methods are nearly tied, revise task difficulty or reject the setup as uninformative.",
            "",
        ]
    )
    return "\n".join(sections)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    results = load_results()
    plot_paths = {
        "Vision Final Average Accuracy": plot_metric(
            results,
            "empirical2_vision_cl",
            "final_avg_accuracy",
            "final avg. accuracy",
            "vision_final_avg_accuracy.png",
        ),
        "NLP Final Average Accuracy": plot_metric(
            results,
            "empirical2_nlp_cl",
            "final_avg_accuracy",
            "final avg. accuracy",
            "nlp_final_avg_accuracy.png",
        ),
        "Forecasting Final Average MSE": plot_metric(
            results,
            "empirical2_m4_forecasting_cl",
            "final_avg_mse",
            "final avg. MSE",
            "forecast_final_avg_mse.png",
        ),
    }
    rollout_paths = plot_forecast_rollouts(results)
    REPORT.write_text(build_report(results, plot_paths, rollout_paths))
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
