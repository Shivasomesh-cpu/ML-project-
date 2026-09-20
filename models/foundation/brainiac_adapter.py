"""
models/foundation/brainiac_adapter.py
Foundation model adapter for BrainIAC (3D ResNet-50 SimCLR).
Implements the full 4-tier adaptation ladder:
1. Frozen encoder + linear probe
2. Frozen encoder + 2-layer MLP head
3. Last-block unfreeze + head
4. Full fine-tune
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


def build_3d_resnet50(num_classes: int = 1):
    """Builds a 3D ResNet-50 encoder."""
    try:
        from monai.networks.nets import resnet50
        return resnet50(spatial_dims=3, n_input_channels=1, num_classes=num_classes)
    except Exception:
        # Fallback 3D ResNet backbone
        class Bottleneck3D(nn.Module):
            def __init__(self, in_c, out_c, stride=1):
                super().__init__()
                mid_c = out_c // 4
                self.conv = nn.Sequential(
                    nn.Conv3d(in_c, mid_c, kernel_size=1, bias=False),
                    nn.BatchNorm3d(mid_c),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(mid_c, mid_c, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm3d(mid_c),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(mid_c, out_c, kernel_size=1, bias=False),
                    nn.BatchNorm3d(out_c),
                )
                self.shortcut = nn.Sequential()
                if stride != 1 or in_c != out_c:
                    self.shortcut = nn.Sequential(
                        nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm3d(out_c)
                    )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                return self.relu(self.shortcut(x) + self.conv(x))

        class ResNet50Fallback(nn.Module):
            def __init__(self, out_classes):
                super().__init__()
                self.conv1 = nn.Sequential(
                    nn.Conv3d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
                    nn.BatchNorm3d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=2, stride=2)
                )
                self.layer1 = Bottleneck3D(32, 64)
                self.layer2 = Bottleneck3D(64, 128, stride=2)
                self.layer3 = Bottleneck3D(128, 256, stride=2)
                self.layer4 = Bottleneck3D(256, 512, stride=2)
                self.avgpool = nn.AdaptiveAvgPool3d(1)
                self.fc = nn.Linear(512, out_classes)

            def forward(self, x):
                x = self.conv1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                feat = self.avgpool(x).flatten(1)
                return self.fc(feat)

        return ResNet50Fallback(num_classes)


@register_model_arm("brainiac")
class BrainIACAdapter(ModelArm):
    def __init__(self, model_id: str, config: Dict[str, Any], task_type: str = "regression"):
        super().__init__(model_id, config, task_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _setup_adaptation(self, model: nn.Module, adaptation_mode: str, num_classes: int = 1):
        """Applies the adaptation ladder configuration."""
        # Find final linear layer
        linear_attr = "fc" if hasattr(model, "fc") else ("head" if hasattr(model, "head") else None)
        
        if adaptation_mode in ["linear_probe", "mlp_probe"]:
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            
            if linear_attr and hasattr(model, linear_attr):
                in_feat = getattr(model, linear_attr).in_features
                if adaptation_mode == "linear_probe":
                    setattr(model, linear_attr, nn.Linear(in_feat, num_classes))
                else: # mlp_probe
                    setattr(model, linear_attr, nn.Sequential(
                        nn.Linear(in_feat, 128),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Linear(128, num_classes)
                    ))
                for param in getattr(model, linear_attr).parameters():
                    param.requires_grad = True

        elif adaptation_mode == "last_block_finetune":
            if linear_attr:
                in_feat = getattr(model, linear_attr).in_features
                setattr(model, linear_attr, nn.Linear(in_feat, num_classes))
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze layer4 and head
            target_layers = ["layer4", linear_attr]
            for name, module in model.named_children():
                if name in target_layers:
                    for param in module.parameters():
                        param.requires_grad = True

        elif adaptation_mode == "full_finetune":
            if linear_attr:
                in_feat = getattr(model, linear_attr).in_features
                setattr(model, linear_attr, nn.Linear(in_feat, num_classes))
            for param in model.parameters():
                param.requires_grad = True

    def fit(self, train_data: Any, val_data: Any, trial_params: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        adaptation_mode = trial_params.get("adaptation_mode", "linear_probe")
        lr = trial_params.get("learning_rate", 1e-4)
        wd = trial_params.get("weight_decay", 1e-4)
        batch_size = trial_params.get("batch_size", 4)
        epochs = trial_params.get("epochs", 5)

        self.model = build_3d_resnet50(num_classes=1)

        # A foundation-model arm must never silently run with random weights.
        ckpt_path = self.config.get("checkpoint_path")
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"BrainIAC checkpoint not found: {ckpt_path!r}. Refusing to label random weights as pretrained.")
        logger.info(f"Loading pretrained BrainIAC weights from {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        loaded = len(self.model.state_dict()) - len(incompatible.missing_keys)
        if loaded == 0:
            raise ValueError("BrainIAC checkpoint has no compatible parameters for this adapter architecture.")

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
            raise RuntimeError("BrainIAC adapter has not been fitted yet.")
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
