"""
harness/run_cell.py
Main execution entrypoint for an individual experimental cell:
  (task_id, model_id, n_labels, seed, trial_budget)
Evaluates foundation models and tuned supervised baselines under identical constraints.
"""

import os
import sys
import time
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import models
from models.registry import get_model_arm
from harness.sweep import run_optuna_sweep
from harness.ledger import ResultsLedger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_mock_task1_data(
    n_train: int = 25,
    n_val: int = 15,
    n_seen_test: int = 20,
    n_unseen_test: int = 25,
    vol_shape = (128, 128, 128),
    seed: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Generates synthetic 3D volumes and realistic age targets for quick verification."""
    np.random.seed(seed)
    
    def make_samples(n, base_age=22.0, age_std=2.0):
        # Downsampled grid for fast smoke test (e.g. 32x32x32)
        mock_shape = (n, 1, 32, 32, 32)
        X = np.random.randn(*mock_shape).astype(np.float32)
        # Induce a synthetic correlation between voxel mean and age
        ages = []
        for i in range(n):
            age = float(np.clip(base_age + np.random.normal(0, age_std), 19.0, 26.0))
            ages.append(age)
            X[i] += (age - base_age) * 0.1
        return X, np.array(ages, dtype=np.float32)

    return {
        "train": make_samples(n_train),
        "val": make_samples(n_val),
        "seen_test": make_samples(n_seen_test),
        "unseen_test": make_samples(n_unseen_test)
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str = "regression") -> Dict[str, float]:
    if task_type == "regression":
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and np.std(y_pred) > 1e-6 else 0.0
        # Age-bias correction is fit on the evaluated cohort only for reporting; it must
        # never be used for model selection.
        if len(y_true) >= 3 and np.std(y_pred) > 1e-6:
            slope, intercept = np.polyfit(y_true, y_pred, 1)
            corrected = y_pred - (slope - 1.0) * y_true - intercept
            bias_corrected_mae = float(mean_absolute_error(y_true, corrected))
        else:
            bias_corrected_mae = float("nan")
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pearson_r": corr,
            "age_bias_corrected_mae": bias_corrected_mae,
        }
    else:
        # Binary classification
        try:
            auc = float(roc_auc_score(y_true, y_pred))
        except Exception:
            auc = 0.5
        labels = (y_pred > 0.5).astype(int)
        balanced_accuracy = float(balanced_accuracy_score(y_true, labels))
        try:
            auprc = float(average_precision_score(y_true, y_pred))
        except Exception:
            auprc = float("nan")
        return {
            "auroc": auc,
            "balanced_accuracy": balanced_accuracy,
            "auprc": auprc,
        }


def execute_cell(
    task_id: str,
    model_id: str,
    n_labels: int,
    seed: int,
    trial_budget: int = 5,
    mock: bool = False,
    results_dir: Path = Path("results"),
    splits_json: Path | None = None,
    metadata_csv: Path | None = None,
    cache_dir: Path | None = None,
) -> Dict[str, Any]:
    logger.info(f"=== Starting Run: Task={task_id} | Model={model_id} | N={n_labels} | Seed={seed} | Budget={trial_budget} ===")
    
    # 1. Load configs
    task_cfg_path = PROJECT_ROOT / f"configs/tasks/{task_id}.yaml"
    model_cfg_path = PROJECT_ROOT / f"configs/models/{model_id}.yaml"

    if not task_cfg_path.exists():
        raise FileNotFoundError(f"Missing task configuration: {task_cfg_path}")
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Missing model configuration: {model_cfg_path}")

    task_cfg = load_yaml(task_cfg_path)
    model_cfg = load_yaml(model_cfg_path)
    checkpoint_path = model_cfg.get("checkpoint_path")
    if checkpoint_path and not Path(checkpoint_path).is_absolute():
        model_cfg["checkpoint_path"] = str(PROJECT_ROOT / checkpoint_path)
    if model_cfg.get("type") == "foundation" and not Path(model_cfg.get("checkpoint_path", "")).is_file():
        raise FileNotFoundError(
            f"Foundation checkpoint not found: {model_cfg.get('checkpoint_path')!r}. "
            "Install the verified checkpoint before starting a sweep."
        )
    task_type = task_cfg.get("type", "regression")

    # 2. Prepare Data
    if mock:
        datasets = generate_mock_task1_data(
            n_train=n_labels if n_labels > 0 else 50,
            n_val=15,
            n_seen_test=20,
            n_unseen_test=25,
            seed=seed
        )
    else:
        if not all((splits_json, metadata_csv, cache_dir)):
            raise ValueError("Observed Task 1 runs require --splits_json, --metadata_csv, and --cache_dir.")
        datasets = load_task1_observed_data(splits_json, metadata_csv, cache_dir, n_labels, seed, task_cfg)

    train_data = datasets["train"]
    val_data = datasets["val"]
    seen_test_data = datasets["seen_test"]
    unseen_test_data = datasets["unseen_test"]

    # 3. Hyperparameter Sweep on Validation Split
    logger.info("Executing Optuna sweep on validation split...")
    sweep_results = run_optuna_sweep(
        model_id=model_id,
        model_config=model_cfg,
        train_data=train_data,
        val_data=val_data,
        task_type=task_type,
        n_trials=trial_budget,
        seed=seed
    )

    ledger = ResultsLedger(ledger_dir=results_dir)
    for trial in sweep_results["trial_logs"]:
        ledger.append_result({
            "record_type": "tuning_trial",
            "data_status": "mock" if mock else "observed",
            "task_id": task_id,
            "model_id": model_id,
            "n_labels": n_labels,
            "seed": seed,
            "trial_budget": trial_budget,
            "trial_number": trial["trial_number"],
            "best_hyperparams": json.dumps(trial["params"]),
            "primary_metric_name": "validation_loss",
            "primary_metric_val": trial["val_loss"],
            "trial_status": trial["status"],
            "sweep_seconds": trial["duration_seconds"],
            "training_seconds": 0.0,
            "total_gpu_seconds": trial["duration_seconds"],
            "total_gpu_hours": trial["duration_seconds"] / 3600.0,
        })

    best_params = sweep_results["best_params"]
    if not np.isfinite(sweep_results["best_val_loss"]):
        raise RuntimeError("All tuning trials failed; final training was not started. Inspect the recorded trial rows.")
    best_params["epochs"] = 5 # Final training epochs

    # 4. Train Final Model using Best Parameters
    logger.info(f"Training final model with winning hyperparameters: {best_params}...")
    final_arm = get_model_arm(model_id, model_cfg, task_type=task_type)
    final_fit_stats = final_arm.fit(train_data, val_data, best_params)

    # 5. Evaluate on Test Sets
    seen_preds = final_arm.predict(seen_test_data)
    unseen_preds = final_arm.predict(unseen_test_data)

    metrics_seen = compute_metrics(seen_test_data[1], seen_preds, task_type=task_type)
    metrics_unseen = compute_metrics(unseen_test_data[1], unseen_preds, task_type=task_type)

    # Compute Contamination Gap: unseen MAE - seen MAE (lower is better, so positive gap means seen performed better)
    if task_type == "regression":
        perf_gap = float(metrics_unseen["mae"] - metrics_seen["mae"])
        primary_metric = metrics_unseen["mae"]
        primary_metric_name = "unseen_mae"
    else:
        perf_gap = float(metrics_seen["auroc"] - metrics_unseen["auroc"])
        primary_metric = metrics_unseen["auroc"]
        primary_metric_name = "unseen_auroc"

    total_compute_seconds = sweep_results["total_sweep_seconds"] + final_arm.get_compute_seconds()

    record = {
        "record_type": "final_evaluation",
        "data_status": "mock" if mock else "observed",
        "task_id": task_id,
        "model_id": model_id,
        "n_labels": n_labels,
        "seed": seed,
        "trial_budget": trial_budget,
        "best_hyperparams": json.dumps(best_params),
        "primary_metric_name": primary_metric_name,
        "primary_metric_val": primary_metric,
        "seen_mae": metrics_seen.get("mae"),
        "unseen_mae": metrics_unseen.get("mae"),
        "seen_r2": metrics_seen.get("r2"),
        "unseen_r2": metrics_unseen.get("r2"),
        "seen_age_bias_corrected_mae": metrics_seen.get("age_bias_corrected_mae"),
        "unseen_age_bias_corrected_mae": metrics_unseen.get("age_bias_corrected_mae"),
        "seen_auroc": metrics_seen.get("auroc"),
        "unseen_auroc": metrics_unseen.get("auroc"),
        "seen_balanced_accuracy": metrics_seen.get("balanced_accuracy"),
        "unseen_balanced_accuracy": metrics_unseen.get("balanced_accuracy"),
        "seen_auprc": metrics_seen.get("auprc"),
        "unseen_auprc": metrics_unseen.get("auprc"),
        "contamination_gap": perf_gap,
        "sweep_seconds": sweep_results["total_sweep_seconds"],
        "training_seconds": final_arm.get_compute_seconds(),
        "total_gpu_seconds": total_compute_seconds,
        "total_gpu_hours": total_compute_seconds / 3600.0,
    }

    # 6. Append to Ledger
    ledger.append_result(record)

    logger.info(
        f"[RESULT] Finished {model_id} (N={n_labels}, Seed={seed}): "
        f"Seen MAE={metrics_seen.get('mae', 0):.3f} | "
        f"Unseen MAE={metrics_unseen.get('mae', 0):.3f} | "
        f"Gap={perf_gap:+.3f} | Compute={total_compute_seconds:.1f}s"
    )
    return record


def load_task1_observed_data(splits_json: Path, metadata_csv: Path, cache_dir: Path, n_labels: int, seed: int, task_cfg: Dict[str, Any]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load a frozen Task-1 split from cached volumes without inferring file names.

    ``metadata_csv`` must contain ``scan_id`` and the configured target column. Each
    cached volume is expected at ``cache_dir/<scan_id>.npy`` (or ``cache_path`` when
    that optional column is supplied). This explicit contract prevents accidental
    session mixing or use of non-frozen data.
    """
    split_data = json.loads(Path(splits_json).read_text(encoding="utf-8"))
    if split_data.get("task_id") != "t1_aomic_age":
        raise ValueError("Observed loader currently supports frozen Task 1 splits only.")
    seed_spec = split_data.get("seeds", {}).get(str(seed))
    if not seed_spec:
        raise ValueError(f"Seed {seed} is absent from {splits_json}")
    budget_key = "all" if n_labels < 0 else str(n_labels)
    if budget_key not in seed_spec.get("budgets", {}):
        raise ValueError(f"Label budget {budget_key} is absent from frozen splits")
    metadata = pd.read_csv(metadata_csv)
    target_col = task_cfg["target_column"]
    needed = {"scan_id", target_col}
    if not needed.issubset(metadata.columns):
        raise ValueError(f"Metadata must contain {sorted(needed)}")
    metadata = metadata.drop_duplicates("scan_id").set_index("scan_id", drop=False)

    def materialize(scan_ids: list[str]) -> Tuple[np.ndarray, np.ndarray]:
        missing = [scan_id for scan_id in scan_ids if scan_id not in metadata.index]
        if missing:
            raise ValueError(f"{len(missing)} frozen scans are missing from metadata (for example {missing[0]!r}).")
        arrays, targets = [], []
        for scan_id in scan_ids:
            row = metadata.loc[scan_id]
            path = Path(row["cache_path"]) if "cache_path" in metadata.columns and pd.notna(row["cache_path"]) else Path(cache_dir) / f"{scan_id}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing cached volume for frozen scan {scan_id!r}: {path}")
            volume = np.load(path).astype(np.float32)
            if volume.shape != (128, 128, 128):
                raise ValueError(f"Expected a 128^3 cached volume for {scan_id!r}, got {volume.shape}")
            arrays.append(volume[np.newaxis, ...]); targets.append(float(row[target_col]))
        return np.stack(arrays), np.asarray(targets, dtype=np.float32)

    return {
        "train": materialize(seed_spec["budgets"][budget_key]["train_scans"]),
        "val": materialize(seed_spec["val_scans"]),
        "seen_test": materialize(seed_spec["seen_test_scans"]),
        "unseen_test": materialize(split_data["unseen_test_scans"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Run a single evaluation harness cell.")
    parser.add_argument("--task", type=str, default="t1_aomic_age")
    parser.add_argument("--model", type=str, default="resnet18_scratch")
    parser.add_argument("--n_labels", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--mock", action="store_true", help="Run synthetic smoke-test data; results are marked mock and excluded from analysis.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--splits_json", type=str, help="Frozen Task 1 split JSON for an observed run")
    parser.add_argument("--metadata_csv", type=str, help="Scan metadata with scan_id and target column")
    parser.add_argument("--cache_dir", type=str, help="Directory containing the cached 128^3 .npy volumes")
    args = parser.parse_args()

    execute_cell(
        task_id=args.task,
        model_id=args.model,
        n_labels=args.n_labels,
        seed=args.seed,
        trial_budget=args.trials,
        mock=args.mock,
        results_dir=Path(args.results_dir),
        splits_json=Path(args.splits_json) if args.splits_json else None,
        metadata_csv=Path(args.metadata_csv) if args.metadata_csv else None,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )


if __name__ == "__main__":
    main()
