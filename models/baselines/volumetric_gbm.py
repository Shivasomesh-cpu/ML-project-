"""
models/baselines/volumetric_gbm.py
Implements Volumetric Regional Features + LightGBM / Ridge regression/classification.
As described in SEEN-IT-BEFORE-README.md Section 4.3:
Cheap, interpretable, and competitive baseline on brain age (T1/T2) and dementia (T3).
"""

import time
import logging
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression

from models.registry import ModelArm, register_model_arm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_fast_regional_features(volume_batch: np.ndarray) -> np.ndarray:
    """
    Extracts summary volumetric and morphological regional features:
    - Total brain volume (mask count)
    - Gray matter / white matter intensity approximations (percentiles and quantiles)
    - Hemispheric symmetry ratios
    - Quadrant regional volume proxies
    """
    if volume_batch.ndim == 5:
        volume_batch = volume_batch[:, 0, :, :, :]
    
    n_samples = volume_batch.shape[0]
    features = []

    for i in range(n_samples):
        v = volume_batch[i]
        brain_mask = v > 0
        total_vol = float(np.sum(brain_mask))

        if total_vol == 0:
            features.append(np.zeros(16, dtype=np.float32))
            continue

        vox = v[brain_mask]
        mean_val = float(np.mean(vox))
        std_val = float(np.std(vox))
        p10 = float(np.percentile(vox, 10))
        p25 = float(np.percentile(vox, 25))
        p50 = float(np.percentile(vox, 50))
        p75 = float(np.percentile(vox, 75))
        p90 = float(np.percentile(vox, 90))

        # Hemispheric split (left vs right)
        mid_x = v.shape[0] // 2
        vol_left = float(np.sum(v[:mid_x, :, :] > 0))
        vol_right = float(np.sum(v[mid_x:, :, :] > 0))
        asymmetry = (vol_left - vol_right) / (vol_left + vol_right + 1e-6)

        # Anterior vs Posterior
        mid_y = v.shape[1] // 2
        vol_ant = float(np.sum(v[:, :mid_y, :] > 0))
        vol_post = float(np.sum(v[:, mid_y:, :] > 0))
        ap_ratio = (vol_ant - vol_post) / (vol_ant + vol_post + 1e-6)

        # Superior vs Inferior
        mid_z = v.shape[2] // 2
        vol_sup = float(np.sum(v[:, :, mid_z:] > 0))
        vol_inf = float(np.sum(v[:, :, :mid_z] > 0))
        si_ratio = (vol_sup - vol_inf) / (vol_sup + vol_inf + 1e-6)

        feats = [
            total_vol, mean_val, std_val, p10, p25, p50, p75, p90,
            vol_left, vol_right, asymmetry,
            vol_ant, vol_post, ap_ratio,
            vol_sup, vol_inf
        ]
        features.append(feats)

    return np.array(features, dtype=np.float32)


@register_model_arm("volumetric_gbm")
class VolumetricGBMArm(ModelArm):
    def __init__(self, model_id: str, config: Dict[str, Any], task_type: str = "regression"):
        super().__init__(model_id, config, task_type)
        self.model = None

    def fit(self, train_data: Any, val_data: Any, trial_params: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        # Extract features
        X_train_raw, y_train = self._unwrap_data(train_data)
        X_val_raw, y_val = self._unwrap_data(val_data)

        X_train = extract_fast_regional_features(X_train_raw)
        X_val = extract_fast_regional_features(X_val_raw)

        model_type = trial_params.get("model_type", "lightgbm")
        
        try:
            if model_type == "lightgbm":
                import lightgbm as lgb
                if self.task_type == "regression":
                    self.model = lgb.LGBMRegressor(
                        n_estimators=trial_params.get("n_estimators", 100),
                        learning_rate=trial_params.get("learning_rate", 0.05),
                        num_leaves=trial_params.get("num_leaves", 31),
                        random_state=42,
                        verbosity=-1
                    )
                else:
                    self.model = lgb.LGBMClassifier(
                        n_estimators=trial_params.get("n_estimators", 100),
                        learning_rate=trial_params.get("learning_rate", 0.05),
                        num_leaves=trial_params.get("num_leaves", 31),
                        random_state=42,
                        verbosity=-1
                    )
            else:
                raise ImportError("Fallback to Ridge")
        except Exception:
            if self.task_type == "regression":
                self.model = Ridge(alpha=trial_params.get("alpha", 1.0))
            else:
                self.model = LogisticRegression(C=1.0 / max(trial_params.get("alpha", 1.0), 1e-4), max_iter=200)

        self.model.fit(X_train, y_train)

        # Evaluate on validation
        val_preds = self.model.predict(X_val)
        if self.task_type == "regression":
            best_val_loss = float(np.mean((val_preds - y_val) ** 2))
        else:
            best_val_loss = float(1.0 - np.mean((val_preds > 0.5) == y_val))

        elapsed = time.perf_counter() - start_time
        self.total_compute_seconds += elapsed
        self.is_fitted = True

        return {
            "best_val_loss": best_val_loss,
            "training_seconds": elapsed,
            "n_features": X_train.shape[1]
        }

    def predict(self, test_data: Any) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_test_raw, _ = self._unwrap_data(test_data)
        X_test = extract_fast_regional_features(X_test_raw)

        if self.task_type == "classification" and hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_test)
            return probs[:, 1] if probs.ndim == 2 else probs
        return self.model.predict(X_test)

    def _unwrap_data(self, data: Any) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(data, tuple):
            return data[0], data[1]
        from torch.utils.data import DataLoader
        if isinstance(data, DataLoader):
            all_x, all_y = [], []
            for bx, by in data:
                all_x.append(bx.cpu().numpy())
                all_y.append(by.cpu().numpy())
            return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)
        raise ValueError(f"Cannot unwrap data type {type(data)}")
