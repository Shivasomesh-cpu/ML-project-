"""Observed-data label-equivalence analysis."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fit_curve(n: np.ndarray, error: np.ndarray) -> tuple[float, float]:
    if len(n) < 2 or len(np.unique(n)) < 2:
        raise ValueError("At least two distinct label budgets are required per model and cohort.")
    return tuple(np.polyfit(np.log(n.astype(float)), error.astype(float), 1))


def _required_n(target_error: float, a: float, b: float) -> float | None:
    if a >= 0:
        return None
    estimate = float(np.exp((target_error - b) / a))
    return estimate if np.isfinite(estimate) and estimate > 0 else None


def compute_label_equivalence(df: pd.DataFrame, fm_model: str = "brainiac", base_model: str = "resnet18_scratch", eval_n: int = 50, task_id: str | None = "t1_aomic_age") -> tuple[Dict[str, Any], Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    required = {"data_status", "model_id", "task_id", "n_labels", "seen_mae", "unseen_mae"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Results ledger is missing required columns: {sorted(missing)}")
    data = df[(df["data_status"] == "observed") & df["model_id"].isin([fm_model, base_model])].copy()
    if "record_type" in data.columns:
        data = data[data["record_type"] == "final_evaluation"]
    if task_id:
        data = data[data["task_id"] == task_id]
    result: Dict[str, Any] = {"data_status": "observed", "evaluated_budget_N": eval_n, "task_id": task_id}
    fit_data = {}
    for cohort, col in (("seen_cohort", "seen_mae"), ("unseen_cohort", "unseen_mae")):
        grouped = data.groupby(["model_id", "n_labels"], as_index=False)[col].mean()
        fm = grouped[grouped.model_id == fm_model].sort_values("n_labels")
        base = grouped[grouped.model_id == base_model].sort_values("n_labels")
        common = sorted(set(fm.n_labels).intersection(base.n_labels))
        if len(common) < 2:
            raise ValueError(f"{cohort} needs two shared budgets for {fm_model} and {base_model}.")
        n = np.asarray(common, dtype=float)
        fm_err = fm.set_index("n_labels").loc[common, col].to_numpy(float)
        base_err = base.set_index("n_labels").loc[common, col].to_numpy(float)
        fa, fb = _fit_curve(n, fm_err)
        ba, bb = _fit_curve(n, base_err)
        fm_at_n = float(fa * np.log(eval_n) + fb)
        baseline_required = _required_n(fm_at_n, ba, bb)
        result[cohort] = {
            "fm_mae": fm_at_n,
            "n_baseline_required": baseline_required,
            "N_eq": None if baseline_required is None else baseline_required - eval_n,
            "n_budgets": len(n),
        }
        fit_data[cohort] = (n, base_err, fm_err, np.array([ba, bb]), np.array([fa, fb]))
    return result, fit_data


def plot_label_equivalence(fit_data: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], result: Dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, cohort in zip(axes, ("seen_cohort", "unseen_cohort")):
        n, base, fm, base_fit, fm_fit = fit_data[cohort]
        dense = np.linspace(n.min(), n.max(), 200)
        ax.scatter(n, base, label="Baseline", color="#e67e22")
        ax.scatter(n, fm, label="Foundation model", color="#3498db")
        ax.plot(dense, base_fit[0] * np.log(dense) + base_fit[1], color="#e67e22", linestyle="--")
        ax.plot(dense, fm_fit[0] * np.log(dense) + fm_fit[1], color="#3498db")
        neq = result[cohort]["N_eq"]
        label = "not estimable" if neq is None else f"{neq:+.1f}"
        ax.set_title(f"{cohort.replace('_', ' ').title()}\nN_eq({result['evaluated_budget_N']}) = {label}")
        ax.set_xlabel("Labels (N)")
        ax.grid(alpha=.25)
        ax.legend()
    axes[0].set_ylabel("MAE (lower is better)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit observed label-equivalence curves.")
    parser.add_argument("--results_csv", default="results/results.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--fm_model", default="brainiac")
    parser.add_argument("--base_model", default="resnet18_scratch")
    parser.add_argument("--task", default="t1_aomic_age")
    parser.add_argument("--n_labels", type=int, default=50)
    args = parser.parse_args()
    path = Path(args.results_csv)
    if not path.exists():
        raise SystemExit(f"Results file not found: {path}. No synthetic fallback is used.")
    try:
        result, fit_data = compute_label_equivalence(pd.read_csv(path), args.fm_model, args.base_model, args.n_labels, args.task)
    except ValueError as exc:
        raise SystemExit(f"Cannot compute label equivalence: {exc}") from None
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_equivalence_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_label_equivalence(fit_data, result, out / "figure3_label_equivalence.png")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
