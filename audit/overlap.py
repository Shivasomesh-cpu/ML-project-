"""Subject-level overlap between supplied benchmark cohorts and a FOMO manifest."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import pandas as pd


def normalize_id(identifier: object) -> str:
    """Normalize familiar BIDS, IXI, and OASIS subject/session decorations."""
    value = str(identifier).strip().lower()
    ixi = re.search(r"(ixi\d{3})", value)
    if ixi:
        return ixi.group(1)
    value = re.sub(r"(_ses-[^_]+|_mri|_mr\d+)$", "", value)
    return re.sub(r"^sub[_-]?", "", value)


def load_fomo_manifest(manifest_path: Path) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path, sep="\t")
    columns = {str(c).lower(): c for c in df.columns}
    subject_column = columns.get("original_subject_id") or columns.get("subject_id") or columns.get("participant_id")
    dataset_column = columns.get("original_dataset") or columns.get("dataset") or columns.get("source")
    if not subject_column:
        raise ValueError(f"No recognized subject-ID column in manifest: {list(df.columns)}")
    grouped: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        key = str(row[dataset_column]).strip().upper() if dataset_column else "UNKNOWN"
        grouped.setdefault(key, set()).add(normalize_id(row[subject_column]))
    grouped["_ALL_"] = set().union(*grouped.values()) if grouped else set()
    return df, grouped


def audit_cohort(
    cohort_name: str,
    cohort_subjects: Iterable[object],
    fomo_subjects_by_dataset: Dict[str, Set[str]],
    expected_fomo_dataset_key: str | None = None,
    *,
    seen_threshold_pct: float,
    unseen_threshold_pct: float,
) -> Dict[str, object]:
    if not (0 <= unseen_threshold_pct < seen_threshold_pct <= 100):
        raise ValueError("Thresholds must satisfy 0 <= unseen threshold < seen threshold <= 100.")
    subjects = {normalize_id(subject) for subject in cohort_subjects}
    if not subjects:
        raise ValueError(f"Cohort {cohort_name!r} has no subject IDs")
    all_overlap = subjects.intersection(fomo_subjects_by_dataset["_ALL_"])
    matching_sources = set()
    if expected_fomo_dataset_key:
        for dataset, ids in fomo_subjects_by_dataset.items():
            if dataset != "_ALL_" and expected_fomo_dataset_key.upper() in dataset:
                matching_sources.update(ids)
    source_overlap = subjects.intersection(matching_sources)
    percentage = 100 * len(all_overlap) / len(subjects)
    return {
        "cohort": cohort_name,
        "total_subjects": len(subjects),
        "pretraining_seen_count": len(all_overlap),
        "contamination_pct": round(percentage, 2),
        "status": "SEEN (Contaminated)" if percentage >= seen_threshold_pct else "UNSEEN (Clean)" if percentage <= unseen_threshold_pct else "PARTIAL",
        "seen_threshold_pct": seen_threshold_pct,
        "unseen_threshold_pct": unseen_threshold_pct,
        "dataset_key_match_count": len(source_overlap),
    }


def load_cohorts(cohorts_json: Path) -> Dict[str, Dict[str, object]]:
    """Load an explicit cohort manifest.

    JSON format: {"AOMIC ID1000": {"subjects": ["sub-0001"], "dataset_key": "AOMIC"}}.
    A list of IDs is also accepted as shorthand for ``subjects``.
    """
    payload = json.loads(cohorts_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Cohort JSON must be an object mapping cohort names to subject specifications.")
    result = {}
    for name, spec in payload.items():
        spec = {"subjects": spec} if isinstance(spec, list) else spec
        if not isinstance(spec, dict) or not isinstance(spec.get("subjects"), list):
            raise ValueError(f"Cohort {name!r} must contain a subjects list.")
        result[name] = spec
    return result


def run_full_audit(manifest_path: Path, cohorts_json: Path, *, seen_threshold_pct: float, unseen_threshold_pct: float) -> pd.DataFrame:
    if not (0 <= unseen_threshold_pct < seen_threshold_pct <= 100):
        raise ValueError("Thresholds must satisfy 0 <= unseen threshold < seen threshold <= 100.")
    _, fomo_sets = load_fomo_manifest(manifest_path)
    cohorts = load_cohorts(cohorts_json)
    return pd.DataFrame([
        audit_cohort(
            name, spec["subjects"], fomo_sets, str(spec.get("dataset_key", "")) or None,
            seen_threshold_pct=seen_threshold_pct, unseen_threshold_pct=unseen_threshold_pct,
        )
        for name, spec in cohorts.items()
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an overlap audit on explicitly supplied cohort subject IDs.")
    parser.add_argument("--manifest_path", default="data/manifests/mapping.tsv")
    parser.add_argument("--cohorts_json", required=True, help="JSON mapping cohort names to actual subject IDs")
    parser.add_argument("--seen_threshold", required=True, type=float, help="Pre-registered minimum overlap percentage for a seen label")
    parser.add_argument("--unseen_threshold", required=True, type=float, help="Pre-registered maximum overlap percentage for an unseen label")
    args = parser.parse_args()
    print(run_full_audit(
        Path(args.manifest_path), Path(args.cohorts_json),
        seen_threshold_pct=args.seen_threshold, unseen_threshold_pct=args.unseen_threshold,
    ).to_string(index=False))


if __name__ == "__main__":
    main()
