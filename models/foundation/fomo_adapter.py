"""
models/foundation/fomo_adapter.py
Foundation model adapter for FOMO25 baseline checkpoints (UNet-B & UNet-XL).
Supports pretraining checkpoint loading and the full 4-tier adaptation ladder.
"""

import time
import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.registry import ModelArm, register_model_arm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class UNetEncoder3D(nn.Module):
    """
    3D UNet Feature Encoder for FOMO25 UNet-B and UNet-XL architectures.
    """
    def __init__(self, model_variant: str = "unet_b", out_classes: int = 1):
        super().__init__()
        base_channels = 32 if model_variant == "unet_b" else 64
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(base_channels * 8, out_classes)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        feat = self.gap(b).flatten(1)
        return self.head(feat)


class FOMOUNetAdapter(ModelArm):
    def __init__(self, model_id: str, config: Dict[str, Any], task_type: str = "regression", variant: str = "unet_b"):
        super().__init__(model_id, config, task_type)
        self.variant = variant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _setup_adaptation(self, model: nn.Module, adaptation_mode: str, num_classes: int = 1):
        if adaptation_mode in ["linear_probe", "mlp_probe"]:
            for param in model.parameters():
                param.requires_grad = False
            
            in_feat = model.head.in_features
            if adaptation_mode == "linear_probe":
                model.head = nn.Linear(in_feat, num_classes)
            else: # mlp_probe
                model.head = nn.Sequential(
                    nn.Linear(in_feat, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(128, num_classes)
                )
            for param in model.head.parameters():
                param.requires_grad = True

        elif adaptation_mode == "last_block_finetune":
            in_feat = model.head.in_features
            model.head = nn.Linear(in_feat, num_classes)
            for param in model.parameters():
                param.requires_grad = False
            for param in model.bottleneck.parameters():
                param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True

        elif adaptation_mode == "full_finetune":
            in_feat = model.head.in_features
            model.head = nn.Linear(in_feat, num_classes)
            for param in model.parameters():
                param.requires_grad = True

    def fit(self, train_data: Any, val_data: Any, trial_params: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        adaptation_mode = trial_params.get("adaptation_mode", "linear_probe")
        lr = trial_params.get("learning_rate", 1e-4)
        wd = trial_params.get("weight_decay", 1e-4)
        batch_size = trial_params.get("batch_size", 4)
        epochs = trial_params.get("epochs", 5)

        self.model = UNetEncoder3D(model_variant=self.variant, out_classes=1)

        # A foundation-model arm must never silently run with random weights.
        ckpt_path = self.config.get("checkpoint_path")
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"FOMO25 checkpoint not found: {ckpt_path!r}. Refusing to label random weights as pretrained.")
        logger.info(f"Loading pretrained FOMO25 {self.variant} weights from {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        loaded = len(self.model.state_dict()) - len(incompatible.missing_keys)
        if loaded == 0:
            raise ValueError("FOMO25 checkpoint has no compatible parameters for this adapter architecture.")

        self._setup_adaptation(self.model, adaptation_mode, num_classes=1)
        self.model.to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
        criterion = nn.MSELoss() if self.task_type == "regression" else nn.BCEWithLogitsLoss()

        train_loader = self._to_loader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = self._to_loader(val_data, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        history = []

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx = bx.to(self.device)
                by = by.to(self.device).float()
                optimizer.zero_grad()
                pred = self.model(bx).squeeze(-1)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(bx)
            train_loss /= max(len(train_loader.dataset), 1)

            # Val
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(self.device)
                    by = by.to(self.device).float()
                    pred = self.model(bx).squeeze(-1)
                    val_loss += criterion(pred, by).item() * len(bx)
            val_loss /= max(len(val_loader.dataset), 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        elapsed = time.perf_counter() - start_time
        self.total_compute_seconds += elapsed
        self.is_fitted = True

        return {
            "best_val_loss": best_val_loss,
            "training_seconds": elapsed,
            "adaptation_mode": adaptation_mode,
            "history": history
        }

    def predict(self, test_data: Any) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("FOMO UNet adapter has not been fitted yet.")
        self.model.eval()
        loader = self._to_loader(test_data, batch_size=4, shuffle=False)
        preds = []
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(self.device)
                p = self.model(bx).squeeze(-1)
                if self.task_type == "classification":
                    p = torch.sigmoid(p)
                preds.extend(p.cpu().numpy().tolist())
        return np.array(preds)

    def _to_loader(self, data: Any, batch_size: int = 4, shuffle: bool = False) -> DataLoader:
        if isinstance(data, DataLoader):
            return data
        if isinstance(data, tuple):
            x, y = data
            if isinstance(x, np.ndarray):
                if x.ndim == 4:
                    x = x[:, np.newaxis, ...]
                x_tensor = torch.tensor(x, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.float32)
                dataset = TensorDataset(x_tensor, y_tensor)
                drop_last = True if (shuffle and len(dataset) > batch_size) else False
                return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        raise ValueError(f"Unsupported data format: {type(data)}")


@register_model_arm("fomo_unet_b")
class FOMOUNetBAdapter(FOMOUNetAdapter):
    def __init__(self, model_id: str, config: Dict[str, Any], task_type: str = "regression"):
        super().__init__(model_id, config, task_type, variant="unet_b")


@register_model_arm("fomo_unet_xl")
class FOMOUNetXLAdapter(FOMOUNetAdapter):
    def __init__(self, model_id: str, config: Dict[str, Any], task_type: str = "regression"):
        super().__init__(model_id, config, task_type, variant="unet_xl")
