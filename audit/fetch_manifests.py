"""
audit/fetch_manifests.py
Pulls FOMO50K manifest files (mapping.tsv, participants.tsv, mri_info.tsv) from Hugging Face.
Supports HF_TOKEN from environment or prompt, with automated fallback to FOMO45K / FOMO260K.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PRIMARY_REPO = "FOMO-MRI/FOMO50K"
FALLBACK_REPOS = ["FOMO-MRI/FOMO45K", "FOMO-MRI/FOMO260K"]
TARGET_FILES = ["mapping.tsv", "participants.tsv", "mri_info.tsv"]


def fetch_file(repo_id: str, filename: str, local_dir: Path, token: str = None) -> Path:
    from huggingface_hub import hf_hub_download
    logger.info(f"Attempting to download {filename} from {repo_id}...")
    dest = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=token,
    )
    logger.info(f"Successfully downloaded {filename} to {dest}")
    return Path(dest)


def create_mock_manifest(dest_dir: Path):
    """
    Creates a valid representative mock manifest for testing and offline environments.
    Reflects the documented constituent datasets of FOMO50K:
    OASIS1, OASIS2, IXI, AOMIC ID1000, BraTS-GEN, etc.
    """
    logger.info("Generating realistic mock manifests in %s for local audit verification...", dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    mapping_data = []
    # Mock AOMIC ID1000 subjects (seen)
    for i in range(1, 929):
        subj_id = f"sub-{i:04d}"
        mapping_data.append({
            "fomo_id": f"FOMO_AOMIC_{i:04d}",
            "original_dataset": "AOMIC_ID1000",
            "original_subject_id": subj_id,
            "session_id": "ses-1",
            "modality": "T1w",
            "pretraining_split": "train"
        })
    
    # Mock IXI subjects (seen)
    for i in range(1, 580):
        site = "Guys" if i % 3 == 0 else ("HH" if i % 3 == 1 else "IOP")
        subj_id = f"IXI{i:03d}-{site}-{1000+i:04d}"
        mapping_data.append({
            "fomo_id": f"FOMO_IXI_{i:04d}",
            "original_dataset": "IXI",
            "original_subject_id": subj_id,
            "session_id": "ses-01",
            "modality": "T1w",
            "pretraining_split": "train"
        })
        
    # Mock OASIS-1 subjects (seen)
    for i in range(1, 417):
        subj_id = f"OAS1_{i:04d}_MR1"
        mapping_data.append({
            "fomo_id": f"FOMO_OAS1_{i:04d}",
            "original_dataset": "OASIS1",
            "original_subject_id": subj_id,
            "session_id": "MR1",
            "modality": "T1w",
            "pretraining_split": "train"
        })

    # Mock OASIS-2 subjects (seen)
    for i in range(1, 151):
        subj_id = f"OAS2_{i:04d}_MR1"
        mapping_data.append({
            "fomo_id": f"FOMO_OAS2_{i:04d}",
            "original_dataset": "OASIS2",
            "original_subject_id": subj_id,
            "session_id": "MR1",
            "modality": "T1w",
            "pretraining_split": "train"
        })

    df_mapping = pd.DataFrame(mapping_data)
    df_mapping.to_csv(dest_dir / "mapping.tsv", sep="\t", index=False)
    
    # Create participants.tsv
    participants_data = []
    for row in mapping_data:
        participants_data.append({
            "fomo_id": row["fomo_id"],
            "dataset": row["original_dataset"],
            "subject_id": row["original_subject_id"],
            "age": 22.5 if "AOMIC" in row["original_dataset"] else (65.0 if "OAS" in row["original_dataset"] else 45.0),
            "sex": "M" if hash(row["original_subject_id"]) % 2 == 0 else "F"
        })
    df_part = pd.DataFrame(participants_data)
    df_part.to_csv(dest_dir / "participants.tsv", sep="\t", index=False)

    # Create empty/minimal mri_info.tsv
    df_mri = pd.DataFrame([{"fomo_id": r["fomo_id"], "magnetic_field_strength": 3.0} for r in mapping_data])
    df_mri.to_csv(dest_dir / "mri_info.tsv", sep="\t", index=False)
    
    logger.info("Mock manifests successfully created: mapping.tsv (%d records)", len(df_mapping))


def main():
    parser = argparse.ArgumentParser(description="Fetch FOMO manifests from Hugging Face.")
    parser.add_argument("--output_dir", type=str, default="data/manifests", help="Local directory to store manifests")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token (optional)")
    parser.add_argument("--mock_if_failed", action="store_true", help="Create a clearly labelled mock manifest if all downloads fail")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    dest_dir = Path(args.output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    repos_to_try = [PRIMARY_REPO] + FALLBACK_REPOS
    downloaded = False
    source_repo = None

    for repo in repos_to_try:
        try:
            logger.info("Trying repository: %s", repo)
            for fname in TARGET_FILES:
                fetch_file(repo, fname, dest_dir, token=token)
            downloaded = True
            source_repo = repo
            logger.info("All manifest files successfully downloaded from %s", repo)
            break
        except Exception as e:
            logger.warning("Could not download from %s: %s", repo, str(e))

    if not downloaded:
        if args.mock_if_failed:
            logger.warning("Falling back to a mock manifest requested explicitly by --mock_if_failed.")
            create_mock_manifest(dest_dir)
            source_repo = "mock"
        else:
            logger.error("Failed to download manifests. Set HF_TOKEN, use an available fallback repository, or explicitly pass --mock_if_failed for a smoke test.")
            sys.exit(1)

    provenance = {
        "data_status": "observed" if downloaded else "mock",
        "source_repo": source_repo,
        "requested_primary_repo": PRIMARY_REPO,
        "files": TARGET_FILES,
    }
    (dest_dir / "manifest_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
