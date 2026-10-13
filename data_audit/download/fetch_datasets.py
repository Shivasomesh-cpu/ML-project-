"""
data/download/fetch_datasets.py
CLI tool to download benchmark MRI datasets:
- AOMIC PIOP1 (ds002785) via openneuro-py
- AOMIC PIOP2 (ds002790) via openneuro-py
- AOMIC ID1000 via openneuro-py
- IXI via direct URL downloads
"""

import os
import sys
import shutil
import argparse
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IXI_URLS = {
    "T1": "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar",
    "T2": "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar",
}

OPENNEURO_DATASETS = {
    "piop1": "ds002785",
    "piop2": "ds002790",
}


def download_openneuro(dataset_id: str, target_dir: Path, include_pattern: str = "sub-*/anat/*T1w.nii.gz"):
    logger.info(f"Initiating OpenNeuro download for {dataset_id} into {target_dir}...")
    try:
        import subprocess
        executable = shutil.which("openneuro-py")
        cmd = ([executable, "download"] if executable else [
            sys.executable, "-m", "openneuro_py", "download"
        ]) + [
            f"--dataset={dataset_id}",
            f"--include={include_pattern}",
            f"--target-dir={str(target_dir)}"
        ]
        logger.info(f"Running command: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip() or "openneuro-py failed")
        logger.info(f"Successfully downloaded {dataset_id}")
    except Exception as e:
        raise RuntimeError(f"Failed OpenNeuro download for {dataset_id}: {e}") from e


def download_openneuro_participants(dataset_id: str, target_dir: Path):
    """Fallback: downloads participants.tsv from github/openneuro raw URL."""
    target_dir.mkdir(parents=True, exist_ok=True)
    tsv_url = f"https://raw.githubusercontent.com/OpenNeuroDatasets/{dataset_id}/master/participants.tsv"
    dest_path = target_dir / "participants.tsv"
    try:
        logger.info(f"Downloading {tsv_url} -> {dest_path}")
        urllib.request.urlretrieve(tsv_url, str(dest_path))
        logger.info(f"Downloaded participants.tsv to {dest_path}")
    except Exception as e:
        logger.warning(f"Could not download participants.tsv directly: {e}")


def download_ixi(target_dir: Path, modality: str = "T1"):
    target_dir.mkdir(parents=True, exist_ok=True)
    url = IXI_URLS.get(modality)
    if not url:
        raise ValueError(f"Unknown IXI modality: {modality}")

    tar_path = target_dir / f"IXI-{modality}.tar"
    logger.info(f"Downloading IXI {modality} from {url} to {tar_path}...")
    try:
        urllib.request.urlretrieve(url, str(tar_path))
        logger.info(f"Downloaded IXI tar to {tar_path}")
    except Exception as e:
        logger.error(f"Error downloading IXI: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fetch benchmark MRI datasets.")
    parser.add_argument("--dataset", type=str, required=True, choices=["piop1", "piop2", "id1000", "ixi", "all"])
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--id1000_openneuro_id", type=str, help="Explicit OpenNeuro accession for ID1000, if available")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.dataset in ["piop1", "all"]:
        download_openneuro("ds002785", data_root / "piop1")
    if args.dataset in ["piop2", "all"]:
        download_openneuro("ds002790", data_root / "piop2")
    if args.dataset in ["id1000", "all"]:
        if not args.id1000_openneuro_id:
            parser.error("ID1000 has no hard-coded accession; provide --id1000_openneuro_id explicitly.")
        download_openneuro(args.id1000_openneuro_id, data_root / "id1000")
    if args.dataset in ["ixi", "all"]:
        download_ixi(data_root / "ixi", modality="T1")


if __name__ == "__main__":
    main()
