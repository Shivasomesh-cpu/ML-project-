"""
data/qc/montage.py
Generates 3-plane (axial, sagittal, coronal) mid-slice QC montages for preprocessed volumes.
Maintains a persistent exclusion ledger at data/qc/exclusions.csv.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_slice_montage(vol_path: Path, output_png_path: Path, title: str = None):
    """
    Renders mid-axial, mid-sagittal, and mid-coronal slices side-by-side.
    """
    if str(vol_path).endswith(".npy"):
        vol = np.load(vol_path)
    else:
        import nibabel as nib
        vol = nib.load(str(vol_path)).get_fdata()

    if vol.ndim == 4:
        vol = vol[0]

    d_x, d_y, d_z = vol.shape
    mid_x = d_x // 2
    mid_y = d_y // 2
    mid_z = d_z // 2

    # Slices
    sagittal = np.rot90(vol[mid_x, :, :])
    coronal = np.rot90(vol[:, mid_y, :])
    axial = np.rot90(vol[:, :, mid_z])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    if title is None:
        title = vol_path.stem
    fig.suptitle(f"QC Slice Montage: {title}", fontsize=12, weight="bold")

    axes[0].imshow(sagittal, cmap="gray")
    axes[0].set_title(f"Sagittal (x={mid_x})", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(coronal, cmap="gray")
    axes[1].set_title(f"Coronal (y={mid_y})", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(axial, cmap="gray")
    axes[2].set_title(f"Axial (z={mid_z})", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved QC montage: {output_png_path}")


def log_exclusion(subject_id: str, reason: str, ledger_path: Path = Path("data/qc/exclusions.csv")):
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "subject_id": subject_id,
        "reason": reason,
        "excluded_at": pd.Timestamp.now().isoformat()
    }
    df_new = pd.DataFrame([record])
    if ledger_path.exists():
        df_new.to_csv(ledger_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(ledger_path, index=False)
    logger.info(f"Excluded subject {subject_id}: {reason}")


def main():
    parser = argparse.ArgumentParser(description="QC montage generator.")
    parser.add_argument("--vol", type=str, default="data/cache/smoke_test/synthetic_T1w.npy")
    parser.add_argument("--out", type=str, default="data/qc/montages/synthetic_montage.png")
    args = parser.parse_args()

    vpath = Path(args.vol)
    if not vpath.exists():
        logger.error(f"File {vpath} does not exist.")
        return

    generate_slice_montage(vpath, Path(args.out))


if __name__ == "__main__":
    main()
