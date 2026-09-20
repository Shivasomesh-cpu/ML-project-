"""
harness/sweep.py
Optuna hyperparameter optimization engine.
Logs GPU-seconds for every trial (including failed trials).
Selects the optimal configuration on the validation fold before test evaluation.
"""

import time
import logging
import random
from typing import Dict, Any, Tuple
import numpy as np
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False

from models.registry import get_model_arm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_optuna_sweep(
    model_id: str,
    model_config: Dict[str, Any],
    train_data: Any,
    val_data: Any,
    task_type: str = "regression",
    n_trials: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes an Optuna hyperparameter sweep for a specified model arm and trial budget.
    Measures exact wall-clock/GPU compute time across all trials.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    sweep_start = time.perf_counter()
    trial_logs = []

    search_space = model_config.get("optuna_search_space", {})
    adaptation_options = search_space.get("adaptation_mode", ["linear_probe"])
    
    def objective(trial: Any) -> float:
        trial_start = time.perf_counter()
        
        # Sample hyperparameters
        params = {}
        if "learning_rate" in search_space:
            lr_range = search_space["learning_rate"]
            params["learning_rate"] = trial.suggest_float("learning_rate", lr_range[0], lr_range[1], log=True)
        else:
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        if "weight_decay" in search_space:
            wd_range = search_space["weight_decay"]
            params["weight_decay"] = trial.suggest_float("weight_decay", wd_range[0], wd_range[1], log=True)

        if "batch_size" in search_space:
            params["batch_size"] = trial.suggest_categorical("batch_size", search_space["batch_size"])
        else:
            params["batch_size"] = 4

        if "adaptation_mode" in search_space:
            params["adaptation_mode"] = trial.suggest_categorical("adaptation_mode", adaptation_options)

        if "model_type" in search_space:
            params["model_type"] = trial.suggest_categorical("model_type", search_space["model_type"])

        params["epochs"] = 3 # Fast epochs per trial during search

        # Instantiate fresh model arm for trial
        arm = get_model_arm(model_id, model_config, task_type=task_type)
        try:
            fit_result = arm.fit(train_data, val_data, params)
            val_loss = fit_result.get("best_val_loss", float("inf"))
            status = "SUCCESS"
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            val_loss = float("inf")
            status = "FAILED"

        trial_duration = time.perf_counter() - trial_start
        trial_logs.append({
            "trial_number": trial.number,
            "params": params,
            "val_loss": val_loss,
            "duration_seconds": trial_duration,
            "status": status
        })

        return val_loss

    if HAS_OPTUNA:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, timeout=None)
        best_params = study.best_params if len(study.trials) > 0 else {}
        best_value = study.best_value if len(study.trials) > 0 else float("inf")
    else:
        logger.info("Optuna not installed; using reproducible seeded random parameter sampler.")
        rng = random.Random(seed)
        best_value = float("inf")
        best_params = {}

        class MockTrial:
            def __init__(self, number):
                self.number = number
            def suggest_float(self, name, low, high, log=False):
                low_f = float(low)
                high_f = float(high)
                if log:
                    import math
                    return float(10 ** rng.uniform(math.log10(low_f), math.log10(high_f)))
                return float(rng.uniform(low_f, high_f))
            def suggest_categorical(self, name, choices):
                return rng.choice(choices)

        for t_idx in range(n_trials):
            mt = MockTrial(t_idx)
            val_loss = objective(mt)
            if val_loss < best_value:
                best_value = val_loss
                best_params = trial_logs[-1]["params"]

    total_sweep_seconds = time.perf_counter() - sweep_start

    logger.info(
        f"[SWEEP COMPLETE] Model={model_id} | Trials={n_trials} | "
        f"Best Val Loss={best_value:.4f} | Sweep Duration={total_sweep_seconds:.1f}s"
    )

    return {
        "best_params": best_params,
        "best_val_loss": best_value,
        "total_sweep_seconds": total_sweep_seconds,
        "trial_logs": trial_logs
    }
