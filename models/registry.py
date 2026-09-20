"""
models/registry.py
Unified model interface and factory registry as specified in SEEN-IT-BEFORE-README.md Section 8.
Every model arm (baselines, foundation models, tabular models) implements the exact same contract:
  fit(train_data, val_data, trial_params) -> fit_stats
  predict(test_data) -> np.ndarray
  get_compute_seconds() -> float
"""

import time
import abc
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_MODEL_REGISTRY: Dict[str, Any] = {}


def register_model_arm(name: str):
    """Decorator to register model arm implementations."""
    def decorator(cls):
        _MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


class ModelArm(abc.ABC):
    """
    Abstract base class for all model arms.
    Ensures identical evaluation across foundation models and supervised baselines.
    """
    def __init__(self, model_id: str, config: Dict[str, Any], task_type: str = "regression"):
        self.model_id = model_id
        self.config = config
        self.task_type = task_type
        self.total_compute_seconds: float = 0.0
        self.is_fitted: bool = False

    @abc.abstractmethod
    def fit(self, train_data: Any, val_data: Any, trial_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fits the model within the given trial hyperparameter budget.
        Returns a dictionary containing training curves, best val loss, and epoch count.
        """
        pass

    @abc.abstractmethod
    def predict(self, test_data: Any) -> np.ndarray:
        """
        Generates predictions for the test dataset.
        For regression: returns 1D array of continuous predictions (e.g. predicted age).
        For classification: returns 1D array of predicted probabilities.
        """
        pass

    def get_compute_seconds(self) -> float:
        """Returns total wall-clock GPU/CPU seconds consumed by this model arm."""
        return self.total_compute_seconds


def get_model_arm(model_id: str, config: Dict[str, Any], task_type: str = "regression") -> ModelArm:
    """Factory method to instantiate a model arm by model_id."""
    key = model_id.lower().strip()
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model arm '{model_id}'. Registered models: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[key](model_id, config, task_type)


def list_registered_models():
    return list(_MODEL_REGISTRY.keys())
