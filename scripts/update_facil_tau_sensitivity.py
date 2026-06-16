import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "empirical-evidence" / "artifacts"
FACIL_PREFIX = ARTIFACTS / "facil-prefix"
OUTPUT = ARTIFACTS / "facil-prefix3-iewc-diag-tau-sensitivity.json"


def read_matrix(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append([float(value) for value in line.split()])
    return rows


def latest_result(run_dir: Path, prefix: str) -> Path | None:
    paths = sorted((run_dir / "results").glob(f"{prefix}-*.txt"))
    return paths[-1] if paths else None


def nonzero_values(row: list[float]) -> list[float]:
    return [value for value in row if value != 0.0]


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def parse_tau(name: str) -> float | None:
    match = re.search(r"_tau([^_]+)$", name)
    if not match:
        return None
    token = match.group(1)
    try:
        return float(token)
    except ValueError:
        return None


def tag_for_tau(tau: float) -> str:
    return f"{tau:.0e}".replace("+0", "").replace("+", "")


def row_from_run(run_dir: Path) -> dict | None:
    tau = parse_tau(run_dir.name)
    if tau is None:
        return None
    acc_path = latest_result(run_dir, "acc_taw")
    forg_path = latest_result(run_dir, "forg_taw")
    if acc_path is None or forg_path is None:
        return None
    acc = read_matrix(acc_path)
    forg = read_matrix(forg_path)
    if not acc or not forg:
        return None
    final_task_row = nonzero_values(acc[-1])
    stats_path = run_dir / "importance_stats.json"
    importance_stats = json.loads(stats_path.read_text()) if stats_path.exists() else []
    slim_stats = []
    for item in importance_stats:
        slim_stats.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"top_diagonal_eigenvalues", "top_diagonal_indices"}
            }
        )
    return {
        "avg_forgetting_taw": round(mean(nonzero_values(forg[-1])), 6),
        "completed_tasks": len(acc),
        "final_avg_taw": round(mean(final_task_row), 6),
        "final_task_row_taw": final_task_row,
        "importance_stats": slim_stats,
        "mean_ef_scale_factor": mean(
            item["ef_scale_factor"]
            for item in importance_stats
            if "ef_scale_factor" in item
        ),
        "mean_loss_scale_mean": mean(
            item["loss_scale_mean"]
            for item in importance_stats
            if "loss_scale_mean" in item
        ),
        "mean_loss_scale_median": mean(
            item["loss_scale_median"]
            for item in importance_stats
            if "loss_scale_median" in item
        ),
        "status": "ok",
        "tag": tag_for_tau(tau),
        "tau": tau,
    }


def main() -> None:
    rows = []
    pattern = "cifar100_icarl_iewc_prefix3_iewc_diag_matchfacil_seed0_e60_lr0p05_lam10000_tau*"
    for run_dir in sorted(FACIL_PREFIX.glob(pattern)):
        if not run_dir.is_dir():
            continue
        row = row_from_run(run_dir)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda row: float(row["tau"]))
    payload = {
        "benchmark": "FACIL CIFAR-100 iCaRL order, ResNet-32, first 3 tasks of 10-task task-aware protocol",
        "epochs": 60,
        "lr": 0.05,
        "method": "IEWC diagonal surrogate, EF-scale matched, lambda=10000, 512-sample sketches",
        "optimizer": "SGD momentum=0.9 weight_decay=0.0002",
        "rows": rows,
        "seed": 0,
        "status": "exploratory_diagonal_iewc_tau_sensitivity",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
