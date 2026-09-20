"""Compute-versus-performance Pareto analysis from observed ledger records only."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_pareto_frontier(costs: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """Return the non-dominated points while minimizing both cost and error."""
    if len(costs) != len(errors) or len(costs) == 0:
        raise ValueError("costs and errors must have the same non-zero length")
    return np.array([not np.any((costs <= cost) & (errors <= error) & ((costs < cost) | (errors < error))) for cost, error in zip(costs, errors)])


def generate_pareto_analysis(df_results: pd.DataFrame, output_dir: Path, task_id: str | None = "t1_aomic_age") -> pd.DataFrame:
    required = {"data_status", "model_id", "unseen_mae", "total_gpu_hours"}
    missing = required.difference(df_results.columns)
    if missing:
        raise ValueError(f"Results ledger is missing required columns: {sorted(missing)}")
    data = df_results.copy()
    data = data[data["data_status"] == "observed"]
    if "record_type" in data.columns:
        data = data[data["record_type"] == "final_evaluation"]
    if task_id and "task_id" in data.columns:
        data = data[data["task_id"] == task_id]
    data["unseen_mae"] = pd.to_numeric(data["unseen_mae"], errors="coerce")
    data["total_gpu_hours"] = pd.to_numeric(data["total_gpu_hours"], errors="coerce")
    data = data.dropna(subset=["unseen_mae", "total_gpu_hours"])
    if data.empty:
        raise ValueError("No observed results are available. Mock results and representative profiles are excluded.")
    summary = data.groupby("model_id", as_index=False).agg(
        gpu_hours=("total_gpu_hours", "sum"), unseen_mae=("unseen_mae", "mean"), n_cells=("model_id", "size")
    )
    summary = summary[summary.gpu_hours > 0].copy()
    if summary.empty:
        raise ValueError("Observed results have no positive compute accounting.")
    summary["is_pareto"] = compute_pareto_frontier(summary.gpu_hours.to_numpy(), summary.unseen_mae.to_numpy())
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "pareto_summary.csv", index=False)
    (output_dir / "pareto_provenance.json").write_text(
        json.dumps({"data_status": "observed", "task_id": task_id}), encoding="utf-8"
    )

    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    colors = np.where(summary.is_pareto, "#2980b9", "#95a5a6")
    plt.scatter(summary.gpu_hours, summary.unseen_mae, c=colors, s=95, edgecolor="black")
    frontier = summary[summary.is_pareto].sort_values("gpu_hours")
    plt.plot(frontier.gpu_hours, frontier.unseen_mae, "k--", alpha=.6, label="Pareto frontier")
    for _, row in summary.iterrows():
        plt.annotate(row.model_id, (row.gpu_hours, row.unseen_mae), textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.xscale("log")
    plt.xlabel("Observed total compute (GPU-hours, log scale)")
    plt.ylabel("Observed unseen MAE (lower is better)")
    plt.title("Observed performance–compute Pareto frontier")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "figure4_pareto_frontier.png", dpi=300)
    plt.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Pareto frontier from observed ledger results.")
    parser.add_argument("--results_csv", default="results/results.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--task", default="t1_aomic_age")
    args = parser.parse_args()
    path = Path(args.results_csv)
    if not path.exists():
        raise SystemExit(f"Results file not found: {path}. No representative profile fallback is used.")
    try:
        summary = generate_pareto_analysis(pd.read_csv(path), Path(args.output_dir), args.task)
    except ValueError as exc:
        raise SystemExit(f"Cannot compute Pareto frontier: {exc}") from None
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
