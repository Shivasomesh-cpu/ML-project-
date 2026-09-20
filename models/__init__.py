"""
models package initialization.
Imports all baseline and foundation arms to register them into the unified factory.
"""

from models.registry import ModelArm, get_model_arm, list_registered_models, register_model_arm
import models.baselines.monai_models
import models.baselines.volumetric_gbm
import models.foundation.brainiac_adapter
import models.foundation.fomo_adapter

__all__ = ["ModelArm", "get_model_arm", "list_registered_models", "register_model_arm"]
