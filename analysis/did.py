"""Difference-in-differences analysis for completed, real evaluation cells.

This module deliberately refuses to invent results. Every DiD estimate is paired by
task, label budget, seed, and trial budget so unlike-for-like runs cannot be compared.
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REQUIRED_COLUMNS = {"data_status", "model_id", "task_id", "n_labels", "seed", "trial_budget", "seen_mae", "unseen_mae"}
PAIR_COLUMNS = ["task_id", "n_labels", "seed", "trial_budget"]


def _paired_gaps(df: pd.DataFrame, fm_model: str, base_model: str, task_id: str | None) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Results ledger is missing required columns: {sorted(missing)}")
    subset = df[(df["data_status"] == "observed") & df["model_id"].isin([fm_model, base_model])].copy()
    if "record_type" in subset.columns:
        subset = subset[subset["record_type"] == "final_evaluation"]
    if task_id:
        subset = subset[subset["task_id"] == task_id]
    if subset.empty:
        raise ValueError("No rows match the requested models/task.")
    subset["gap"] = pd.to_numeric(subset["unseen_mae"], errors="coerce") - pd.to_numeric(subset["seen_mae"], errors="coerce")
    subset = subset.dropna(subset=["gap"])
    grouped = subset.groupby(PAIR_COLUMNS + ["model_id"], as_index=False)["gap"].mean()
    wide = grouped.pivot(index=PAIR_COLUMNS, columns="model_id", values="gap").reset_index()
    missing_models = [m for m in (fm_model, base_model) if m not in wide.columns]
    if missing_models:
        raise ValueError(f"No usable paired results for: {', '.join(missing_models)}")
    wide = wide.dropna(subset=[fm_model, base_model]).copy()
    if wide.empty:
        raise ValueError("No matched cells exist. Run both models with identical task, N, seed, and trial budget.")
    wide["did"] = wide[fm_model] - wide[base_model]
    return wide


def compute_did_bootstrap(df_results: pd.DataFrame, fm_model: str = "brainiac", base_model: str = "resnet18_scratch", n_bootstrap: int = 2000, seed: int = 42, task_id: str | None = "t1_aomic_age") -> Dict[str, Any]:
    """Compute paired DiD, a bootstrap interval and two-sided bootstrap test.

    The exact sign-flip test is retained as a separately reported sensitivity check.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    pairs = _paired_gaps(df_results, fm_model, base_model, task_id)
    dids = pairs["did"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(dids, size=len(dids), replace=True).mean() for _ in range(n_bootstrap)])
    point = float(dids.mean())

    # Center the paired cell effects under H0, then use a two-tailed bootstrap test.
    centered = dids - point
    null_boot = np.array([rng.choice(centered, size=len(centered), replace=True).mean() for _ in range(n_bootstrap)])
    extreme = int(np.count_nonzero(np.abs(null_boot) >= abs(point)))
    bootstrap_p = float((extreme + 1) / (n_bootstrap + 1))

    # Secondary exact sign-flip sensitivity check for the small seed counts used here.
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(dids))))
    sign_flip_null = (signs * dids).mean(axis=1)
    sign_flip_p = float(np.mean(np.abs(sign_flip_null) >= abs(point)))
    ci_lower, ci_upper = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {
        "data_status": "observed",
        "fm_model": fm_model,
        "base_model": base_model,
        "task_id": task_id,
        "n_paired_cells": int(len(pairs)),
        "fm_mean_gap": float(pairs[fm_model].mean()),
        "base_mean_gap": float(pairs[base_model].mean()),
        "did_point_estimate": point,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value_bootstrap_two_sided": bootstrap_p,
        "p_value_sign_flip_secondary": sign_flip_p,
        "h1_supported": bool(ci_lower > 0 and bootstrap_p < 0.05),
        "paired_cells": pairs.to_dict(orient="records"),
    }


def plot_amsterdam_natural_experiment(did_res: Dict[str, Any], output_path: Path) -> None:
    pairs = pd.DataFrame(did_res["paired_cells"])
    fm, base = did_res["fm_model"], did_res["base_model"]
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    plt.bar([base, fm], [pairs[base].mean(), pairs[fm].mean()], color=["#e67e22", "#3498db"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Unseen − seen MAE (years; positive favours seen)")
    plt.title("Observed cohort performance gaps")
    plt.text(0.02, 0.98, f"DiD = {did_res['did_point_estimate']:+.3f}\n95% CI [{did_res['ci_95_lower']:+.3f}, {did_res['ci_95_upper']:+.3f}]\nBootstrap p = {did_res['p_value_bootstrap_two_sided']:.4f}", transform=plt.gca().transAxes, va="top", bbox={"boxstyle": "round", "facecolor": "white"})
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DiD from observed ledger results.")
    parser.add_argument("--results_csv", default="results/results.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--fm_model", default="brainiac")
    parser.add_argument("--base_model", default="resnet18_scratch")
    parser.add_argument("--task", default="t1_aomic_age")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    results_path = Path(args.results_csv)
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}. No synthetic fallback is used.")
    try:
        result = compute_did_bootstrap(pd.read_csv(results_path), args.fm_model, args.base_model, args.bootstrap, task_id=args.task)
    except ValueError as exc:
        raise SystemExit(f"Cannot compute DiD: {exc}") from None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "did_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_amsterdam_natural_experiment(result, output_dir / "figure2_amsterdam_did.png")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
