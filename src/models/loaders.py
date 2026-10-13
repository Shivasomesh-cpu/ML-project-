import os
import sys
import types
import torch
import torch.nn as nn
from monai.networks.nets import ViT, SwinUNETR, resnet18

# Patch for AnatCL module structure compatibility if needed
try:
    import anatcl.models as _anatcl_models
    class _AgeEstimator:
        pass
    _estimators_mod = types.ModuleType('models.estimators')
    _estimators_mod.AgeEstimator = _AgeEstimator
    sys.modules['models'] = _anatcl_models
    sys.modules['models.estimators'] = _estimators_mod
    from anatcl import AnatCL
    ANATCL_AVAILABLE = True
except Exception:
    ANATCL_AVAILABLE = False


def load_brainiac_vit(ckpt_path=None, device="cpu"):
    """
    Initializes BrainIAC ViT architecture (96x96x96, patch 16, hidden 768).
    Loads checkpoint if available.
    """
    vit = ViT(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        save_attn=True
    )
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        pe_key = "backbone.patch_embedding.position_embeddings"
        state = ckpt.get("state_dict", ckpt)
        if pe_key in state:
            vit.patch_embedding.position_embeddings = nn.Parameter(state[pe_key].clone())
        backbone_state = {
            k.replace("backbone.", ""): v
            for k, v in state.items()
            if k.startswith("backbone.")
        }
        vit.load_state_dict(backbone_state, strict=False)
        print(f"Loaded BrainIAC weights from {ckpt_path}")
    else:
        if ckpt_path:
            print(f"BrainIAC checkpoint not found at {ckpt_path}; initialized with default weights.")
    return vit.to(device).eval()


def load_swinbrain(ckpt_path=None, device="cpu"):
    """
    Initializes SwinBrain (SwinUNETR with 128x128x64 input, 3 channels).
    Loads checkpoint if available.
    """
    import inspect
    sig = inspect.signature(SwinUNETR.__init__)
    kwargs = {
        "in_channels": 3,
        "out_channels": 3,
        "spatial_dims": 3,
        "feature_size": 24,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0
    }
    if "img_size" in sig.parameters:
        kwargs["img_size"] = (128, 128, 64)

    model = SwinUNETR(**kwargs)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        clean_state = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(clean_state, strict=True)
        print(f"Loaded SwinBrain weights from {ckpt_path}")
    else:
        if ckpt_path:
            print(f"SwinBrain checkpoint not found at {ckpt_path}; initialized with default weights.")
    return model.to(device).eval()


def load_simclr_encoder(ckpt_path=None, device="cpu"):
    """
    Initializes 3D-Neuro-SimCLR ResNet-18 backbone (spatial_dims=3, 1-channel).
    Loads checkpoint if available.
    """
    encoder = resnet18(spatial_dims=3, n_input_channels=1, num_classes=0)
    encoder.fc = nn.Identity()
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = {
            k.replace("module.encoder.", ""): v
            for k, v in ckpt.get('model_state_dict', ckpt).items()
            if "encoder" in k and "projector" not in k
        }
        encoder.load_state_dict(state_dict, strict=True)
        print(f"Loaded 3D-Neuro-SimCLR weights from {ckpt_path}")
    else:
        if ckpt_path:
            print(f"3D-Neuro-SimCLR checkpoint not found at {ckpt_path}; initialized with default weights.")
    return encoder.to(device).eval()


def load_anatcl_encoder(descriptor="global", fold=0, weights_dir=None, device="cpu"):
    """
    Initializes AnatCL encoder for CAT12 VBM inputs.
    """
    if not ANATCL_AVAILABLE:
        raise ImportError("AnatCL is not installed or available.")

    encoder = AnatCL(descriptor=descriptor, fold=fold, pretrained=False).to(device).eval()
    if weights_dir:
        path = os.path.join(weights_dir, f"weights_{descriptor}_fold{fold}.pth")
        if os.path.exists(path):
            state = torch.load(path, map_location=device, weights_only=False)
            model_weights = state.get('model', state)
            encoder.backbone.load_state_dict(model_weights)
            print(f"Loaded AnatCL {descriptor} fold {fold} weights from {path}")
        else:
            print(f"AnatCL weights not found at {path}; using uninitialized weights.")
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder
