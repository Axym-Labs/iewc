import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "empirical-2" / "artifacts"
RAW_LEDGER = ROOT / "docs" / "empirical-2" / "run_tables.md"
RESULT_TABLES = ARTIFACTS / "result-tables.md"


METHOD_LABELS = {
    "sequential": "Sequential",
    "ef": "EF-EWC",
    "ewc_dr": "EWC-DR",
    "iewc": "IEWC",
    "iewc_gss": "IEWC-GSS",
    "iewc_fromp": "IEWC-FROMP",
}


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_path"] = str(path)
    data["_artifact"] = path.name
    return data


def training_runs() -> list[dict[str, Any]]:
    runs = []
    for path in sorted(ARTIFACTS.glob("*.json")):
        if path.name.endswith("-tuning-summary.json"):
            continue
        data = read_json(path)
        if data.get("experiment"):
            runs.append(data)
    return runs


def tuning_summaries() -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted(ARTIFACTS.glob("*-tuning-summary.json"))]


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return str(value)
    if abs(x) >= 10_000 or (0 < abs(x) < 1e-3):
        return f"{x:.3g}"
    text = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def fmt_metric(value: float, sd: float | None = None) -> str:
    if sd is None:
        return fmt_num(value)
    return f"{fmt_num(value)} +- {fmt_num(sd)}"


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def config(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("config", {})


def lambda_value(data: dict[str, Any]) -> str:
    if data.get("method") == "sequential":
        return "-"
    return fmt_num(config(data).get("ewc_lambda"))


def vision_rows(runs: list[dict[str, Any]]) -> list[str]:
    rows = [
        "## Vision Classification",
        "",
        f"Total runs: `{len(runs)}`.",
        "",
        "| artifact | method | seed | lambda | dataset | model | adaptation | distributions | evaluation | epochs | samples/class | final avg accuracy | forgetting |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for data in runs:
        cfg = config(data)
        samples = f"train={cfg.get('train_samples_per_class', '-')}, test={cfg.get('test_samples_per_class', '-')}"
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{data['_artifact']}`",
                    method_label(data.get("method", "")),
                    str(cfg.get("seed", "-")),
                    lambda_value(data),
                    str(cfg.get("dataset", "-")),
                    str(cfg.get("model_name", "-")),
                    str(cfg.get("adaptation", "-")),
                    f"{cfg.get('n_tasks', '-') }x{cfg.get('classes_per_task', '-')}",
                    str(cfg.get("evaluation", "-")),
                    str(cfg.get("epochs_per_task", "-")),
                    samples,
                    fmt_num(data.get("final_avg_accuracy")),
                    fmt_num(data.get("avg_forgetting")),
                ]
            )
            + " |"
        )
    rows.append("")
    return rows


def forecast_model_name(cfg: dict[str, Any]) -> str:
    if cfg.get("model_type") == "patchtst":
        return (
            f"PatchTST d={cfg.get('d_model')}, L={cfg.get('n_layers')}, "
            f"H={cfg.get('n_heads')}, ff={cfg.get('dim_feedforward')}, "
            f"p={cfg.get('patch_length')}, s={cfg.get('patch_stride')}"
        )
    return (
        f"TransformerEncoder d={cfg.get('d_model')}, L={cfg.get('n_layers')}, "
        f"H={cfg.get('n_heads')}, ff={cfg.get('dim_feedforward')}"
    )


def forecasting_rows(runs: list[dict[str, Any]]) -> list[str]:
    rows = [
        "## Time-Series Forecasting / Regression",
        "",
        f"Total runs: `{len(runs)}`.",
        "",
        "| artifact | method | seed | lambda | dataset | distributions | model | context->horizon | windows | epochs | normalization | final avg MSE | forgetting MSE |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for data in runs:
        cfg = config(data)
        freqs = ",".join(str(v) for v in data.get("frequencies", cfg.get("frequencies", [])))
        windows = f"train={cfg.get('windows_per_series', '-')}, eval={cfg.get('eval_windows_per_series', '-')}"
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{data['_artifact']}`",
                    method_label(data.get("method", "")),
                    str(cfg.get("seed", "-")),
                    lambda_value(data),
                    str(cfg.get("dataset", "-")),
                    freqs,
                    forecast_model_name(cfg),
                    f"{cfg.get('context_length', '-')}->{cfg.get('horizon', '-')}",
                    windows,
                    str(cfg.get("epochs_per_task", "-")),
                    str(cfg.get("normalization", "-")),
                    fmt_num(data.get("final_avg_mse")),
                    fmt_num(data.get("avg_forgetting_mse")),
                ]
            )
            + " |"
        )
    rows.append("")
    return rows


def nlp_rows(runs: list[dict[str, Any]]) -> list[str]:
    rows = [
        "## NLP Text Classification",
        "",
        f"Total runs: `{len(runs)}`.",
        "",
        "| artifact | method | seed | lambda | model | adaptation | distributions | train/eval cap | epochs | batch | final avg accuracy | forgetting |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for data in runs:
        cfg = config(data)
        tasks = "->".join(str(v) for v in cfg.get("tasks", []))
        caps = f"train={cfg.get('max_train_samples', '-')}, eval={cfg.get('max_eval_samples', '-')}"
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{data['_artifact']}`",
                    method_label(data.get("method", "")),
                    str(cfg.get("seed", "-")),
                    lambda_value(data),
                    str(cfg.get("model_name", "-")),
                    str(cfg.get("adaptation", "-")),
                    tasks,
                    caps,
                    str(cfg.get("epochs_per_task", "-")),
                    str(cfg.get("batch_size", "-")),
                    fmt_num(data.get("final_avg_accuracy")),
                    fmt_num(data.get("avg_forgetting")),
                ]
            )
            + " |"
        )
    rows.append("")
    return rows


def trace_rows(runs: list[dict[str, Any]]) -> list[str]:
    rows = [
        "## TRACE Decoder-Only LLM",
        "",
        f"Total runs: `{len(runs)}`.",
        "",
        "| artifact | method | seed | lambda | model | LoRA rank | answer mode | distributions | train/eval cap | epochs | batch/accum | final avg score | forgetting | final avg NLL |",
        "| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for data in runs:
        cfg = config(data)
        tasks = "->".join(str(v) for v in cfg.get("tasks", []))
        caps = f"train={cfg.get('max_train_samples', '-')}, eval={cfg.get('max_eval_samples', '-')}"
        batch = f"{cfg.get('batch_size', '-')}/{cfg.get('gradient_accumulation_steps', '-')}"
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{data['_artifact']}`",
                    method_label(data.get("method", "")),
                    str(cfg.get("seed", "-")),
                    lambda_value(data),
                    str(cfg.get("model_name", "-")),
                    str(cfg.get("lora_rank", "-")),
                    str(cfg.get("answer_mode", "-")),
                    tasks,
                    caps,
                    str(cfg.get("epochs_per_task", "-")),
                    batch,
                    fmt_num(data.get("final_avg_score")),
                    fmt_num(data.get("avg_forgetting_score")),
                    fmt_num(data.get("final_avg_nll")),
                ]
            )
            + " |"
        )
    rows.append("")
    return rows


def build_raw_ledger() -> None:
    runs = training_runs()
    vision = [run for run in runs if run.get("experiment") == "empirical2_vision_cl"]
    forecasting = [
        run
        for run in runs
        if run.get("experiment") in {"empirical2_forecasting_cl", "empirical2_m4_forecasting_cl"}
    ]
    nlp = [run for run in runs if run.get("experiment") == "empirical2_nlp_cl"]
    trace = [run for run in runs if run.get("experiment") == "empirical2_trace_cl"]

    lines = [
        "# Raw Empirical Training Runs",
        "",
        "This tracked ledger records every individual empirical-2 training-run JSON currently present in `docs/empirical-2/artifacts/`, excluding only `*-tuning-summary.json` files because those summarize multiple runs rather than train a model. Raw artifacts and plots are ignored, but this per-run index is committed for auditability.",
        "",
        f"Total logged training runs: `{len(vision) + len(forecasting) + len(nlp) + len(trace)}`.",
        "",
    ]
    lines.extend(vision_rows(vision))
    lines.extend(forecasting_rows(forecasting))
    lines.extend(nlp_rows(nlp))
    lines.extend(trace_rows(trace))
    RAW_LEDGER.write_text("\n".join(lines), encoding="utf-8")


def family_name(summary: dict[str, Any]) -> str:
    tag = summary.get("tag", "")
    cfg = summary.get("config", {})
    if tag == "smoke-vit-tune":
        return "smoke: ImageNet-R ViT-tiny LoRA, 2x3, task-aware"
    if tag == "vitb-imagenetr-10x20-full-e2-classinc":
        return "ImageNet-R ViT-B/16 LoRA, 10x20, class-incremental"
    if tag == "vitb-imagenetr-5x20-full-e2-classinc":
        return "ImageNet-R ViT-B/16 LoRA, 5x20, class-incremental"
    if tag == "ett-4task-transformer-d128-e10-lr3e4-series":
        return "ETT transformer forecaster, 4 distributions"
    if tag == "longhorizon-patchtst-3task-guard":
        return "guard: Long-horizon PatchTST, Weather -> ECL -> Traffic"
    if tag == "longhorizon-patchtst-3task-scaled":
        return "Long-horizon PatchTST, Weather -> ECL -> Traffic"
    if tag == "t5-lora-glue-sst2-mrpc-qqp-1024-e3":
        return "T5-small LoRA GLUE guard, SST-2 -> MRPC -> QQP"
    if tag == "t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda":
        return "T5-base LoRA GLUE, SST-2 -> MRPC -> QQP"
    if tag.startswith("qwen05-trace-choiceanswer-3task"):
        return "Qwen2.5-0.5B LoRA TRACE, C-STANCE -> FOMC -> ScienceQA"
    if tag.startswith("qwen15-trace-choiceanswer"):
        return "Qwen2.5-1.5B LoRA TRACE, C-STANCE -> FOMC -> ScienceQA"
    if tag.startswith("qwen3-trace-choiceanswer"):
        return "Qwen2.5-3B LoRA TRACE, C-STANCE -> FOMC -> ScienceQA"
    if cfg.get("frequencies"):
        return f"{tag}: {' -> '.join(cfg['frequencies'])}"
    if cfg.get("tasks") and summary.get("group") == "trace":
        return f"{tag}: {' -> '.join(cfg['tasks'])}"
    return tag


def metric_keys(group: str) -> tuple[str, str, bool]:
    if group in {"vision", "nlp"}:
        return "final_avg_accuracy", "avg_forgetting", True
    if group == "trace":
        return "final_avg_score", "avg_forgetting_score", True
    return "final_avg_mse", "avg_forgetting_mse", False


def load_result(path: str) -> dict[str, Any]:
    return read_json(Path(path))


def selected_tune_path(selection: dict[str, Any]) -> str:
    selected = float(selection["selected_lambda"])
    for record in selection.get("records", []):
        if float(record["lambda"]) == selected:
            return record["path"]
    raise ValueError(f"Could not find selected lambda for {selection.get('method')}")


def aggregate_records(summary: dict[str, Any], method: str, selection: dict[str, Any] | None) -> tuple[list[dict[str, Any]], str, str, str]:
    records = []
    if method == "sequential":
        records.append(load_result(summary["sequential"]["path"]))
        lamb = "-"
        grid_pts = "-"
        edge_ext = "-"
    else:
        assert selection is not None
        records.append(load_result(selected_tune_path(selection)))
        lamb = fmt_num(selection["selected_lambda"])
        grid_pts = str(len(selection.get("records", [])))
        edge_ext = str(selection.get("edge_extensions", 0))
    for final in summary.get("final_records", []):
        if final["method"] == method:
            records.append(load_result(final["result"]))
    return records, lamb, grid_pts, edge_ext


def selected_result_row(summary: dict[str, Any], method: str, selection: dict[str, Any] | None) -> list[str]:
    records, lamb, grid_pts, edge_ext = aggregate_records(summary, method, selection)
    metric_key, forgetting_key, _ = metric_keys(summary["group"])
    seeds = ",".join(str(config(record).get("seed")) for record in records)
    values = [float(record[metric_key]) for record in records]
    forget = [float(record[forgetting_key]) for record in records]
    metric_sd = statistics.stdev(values) if len(values) > 1 else None
    forget_sd = statistics.stdev(forget) if len(forget) > 1 else None
    return [
        family_name(summary),
        method_label(method),
        lamb,
        grid_pts,
        edge_ext,
        seeds,
        fmt_metric(sum(values) / len(values), metric_sd),
        fmt_metric(sum(forget) / len(forget), forget_sd),
    ]


def selected_sections(summaries: list[dict[str, Any]], group: str, heading: str) -> list[str]:
    metric_key, forgetting_key, higher_is_better = metric_keys(group)
    if group == "trace":
        metric_label = "Final avg score"
        forgetting_label = "Forgetting score"
    else:
        metric_label = "Final avg accuracy" if higher_is_better else "Final avg MSE"
        forgetting_label = "Forgetting" if higher_is_better else "Forgetting MSE"
    lines = [
        f"## {heading}",
        "",
        f"| Run family | Method | lambda | grid pts | edge ext. | seeds | {metric_label} | {forgetting_label} |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for summary in [item for item in summaries if item.get("group") == group]:
        rows = [selected_result_row(summary, "sequential", None)]
        for selection in summary.get("selections", []):
            rows.append(selected_result_row(summary, selection["method"], selection))
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def build_selected_tables() -> None:
    summaries = tuning_summaries()
    lines = [
        "# Empirical-2 Selected Result Tables",
        "",
        "These tables summarize selected/tuned run families. The tracked per-training-run ledger is `docs/empirical-2/run_tables.md`.",
        "",
    ]
    lines.extend(selected_sections(summaries, "vision", "Vision Classification"))
    lines.extend(selected_sections(summaries, "forecasting", "Time-Series Forecasting / Regression"))
    lines.extend(selected_sections(summaries, "nlp", "NLP Text Classification"))
    lines.extend(selected_sections(summaries, "trace", "TRACE Decoder-Only LLM"))
    RESULT_TABLES.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    build_raw_ledger()
    build_selected_tables()
    print(f"Wrote {RAW_LEDGER.relative_to(ROOT)}")
    print(f"Wrote {RESULT_TABLES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
