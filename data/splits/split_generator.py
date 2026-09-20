"""
data/splits/split_generator.py
Generates deterministic, subject-level stratified train/val/test splits.
Enforces:
1. Strict 19.0 - 26.0 year age window restriction for the AOMIC natural experiment
2. Age band & sex stratification
3. Zero subject leakage (longitudinal sessions strictly grouped together)
4. Frozen splits across the 5 pre-registered seeds: [42, 1337, 2026, 7, 99]
5. Label subsamples for N in {25, 50, 100, 250, 500, -1}
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 1337, 2026, 7, 99]
LABEL_BUDGETS = [25, 50, 100, 250, 500, -1]
AOMIC_AGE_WINDOW = (19.0, 26.0)


def create_synthetic_cohort_metadata(output_csv: Path, cohort: str, n_subjects: int = 200) -> pd.DataFrame:
    """Generates synthetic cohort demographics table for testing."""
    np.random.seed(42)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    records = []
    
    if cohort == "id1000":
        # Broader age: 18 to 35
        ages = np.random.uniform(18.0, 35.0, size=n_subjects)
    elif "piop" in cohort:
        # Student age: 19 to 25
        ages = np.random.uniform(19.0, 25.0, size=n_subjects)
    elif "ixi" in cohort:
        # Wide adult: 20 to 80
        ages = np.random.uniform(20.0, 80.0, size=n_subjects)
    elif "oasis" in cohort:
        # Elderly: 60 to 95, with dementia CDR
        ages = np.random.uniform(60.0, 95.0, size=n_subjects)
    else:
        ages = np.random.uniform(20.0, 30.0, size=n_subjects)

    for i in range(n_subjects):
        subj_id = f"{cohort}_sub_{i:04d}"
        # Repeat sessions simulation (e.g. 2 scans for some subjects)
        n_scans = 2 if (i % 5 == 0) else 1
        for s in range(n_scans):
            rec = {
                "subject_id": subj_id,
                "scan_id": f"{subj_id}_ses_{s}",
                "age": float(np.round(ages[i], 1)),
                "sex": "F" if (i % 2 == 0) else "M",
                "cohort": cohort,
                "cdr": float(np.random.choice([0.0, 0.5, 1.0, 2.0], p=[0.6, 0.2, 0.15, 0.05])) if "oasis" in cohort else 0.0
            }
            records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    return df


def generate_task1_splits(
    id1000_csv: Path,
    piop_csv: Path,
    output_json: Path,
    seeds: List[int] = SEEDS,
    label_budgets: List[int] = LABEL_BUDGETS
):
    """
    Builds frozen splits for Task T1 (Amsterdam Natural Experiment).
    - Filters both ID1000 and PIOP to 19.0 <= age <= 26.0.
    - Preserves test sets: held-out ID1000 (seen test) and PIOP1+PIOP2 (unseen test).
    - Subsamples training set at each N budget and seed.
    """
    logger.info("Generating Task 1 (AOMIC age) splits with age window [19.0, 26.0]...")
    df_id1000 = pd.read_csv(id1000_csv)
    df_piop = pd.read_csv(piop_csv)

    # 1. Apply age-window restriction
    df_id1000_matched = df_id1000[
        (df_id1000["age"] >= AOMIC_AGE_WINDOW[0]) & (df_id1000["age"] <= AOMIC_AGE_WINDOW[1])
    ].copy()
    df_piop_matched = df_piop[
        (df_piop["age"] >= AOMIC_AGE_WINDOW[0]) & (df_piop["age"] <= AOMIC_AGE_WINDOW[1])
    ].copy()

    logger.info(
        f"Age matching applied: ID1000 {len(df_id1000)} -> {len(df_id1000_matched)} scans | "
        f"PIOP {len(df_piop)} -> {len(df_piop_matched)} scans."
    )

    # Create age bands for stratification
    df_id1000_matched["age_band"] = pd.qcut(df_id1000_matched["age"], q=3, labels=["young", "mid", "older"])
    df_id1000_matched["strat_key"] = df_id1000_matched["sex"].astype(str) + "_" + df_id1000_matched["age_band"].astype(str)

    # PIOP serves as the primary unseen test set (uncontaminated evaluation)
    piop_test_scans = df_piop_matched["scan_id"].tolist()

    all_splits: Dict[str, Any] = {
        "task_id": "t1_aomic_age",
        "age_window": list(AOMIC_AGE_WINDOW),
        "unseen_test_scans": piop_test_scans,
        "seeds": {}
    }

    unique_subjects = df_id1000_matched.drop_duplicates(subset=["subject_id"])

    for seed in seeds:
        seed_data: Dict[str, Any] = {"budgets": {}}
        
        # Split ID1000 subjects into 70% train+val, 30% seen_test
        train_val_subj, seen_test_subj = train_test_split(
            unique_subjects,
            test_size=0.30,
            random_state=seed,
            stratify=unique_subjects["strat_key"]
        )

        # Split train_val into train (80%) and val (20%)
        train_subj, val_subj = train_test_split(
            train_val_subj,
            test_size=0.20,
            random_state=seed,
            stratify=train_val_subj["strat_key"]
        )

        df_train = df_id1000_matched[df_id1000_matched["subject_id"].isin(train_subj["subject_id"])]
        df_val = df_id1000_matched[df_id1000_matched["subject_id"].isin(val_subj["subject_id"])]
        df_seen_test = df_id1000_matched[df_id1000_matched["subject_id"].isin(seen_test_subj["subject_id"])]

        val_scans = df_val["scan_id"].tolist()
        seen_test_scans = df_seen_test["scan_id"].tolist()

        seed_data["val_scans"] = val_scans
        seed_data["seen_test_scans"] = seen_test_scans

        train_subj_unique = df_train.drop_duplicates(subset=["subject_id"])

        for budget in label_budgets:
            if budget == -1 or budget >= len(train_subj_unique):
                chosen_subj = train_subj_unique["subject_id"].tolist()
            else:
                chosen_subj, _ = train_test_split(
                    train_subj_unique,
                    train_size=budget,
                    random_state=seed,
                    stratify=train_subj_unique["strat_key"]
                )
                chosen_subj = chosen_subj["subject_id"].tolist()

            chosen_scans = df_train[df_train["subject_id"].isin(chosen_subj)]["scan_id"].tolist()
            b_key = "all" if budget == -1 else str(budget)
            seed_data["budgets"][b_key] = {
                "n_subjects": len(chosen_subj),
                "train_scans": chosen_scans
            }

        all_splits["seeds"][str(seed)] = seed_data

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_splits, f, indent=2)

    logger.info(f"Saved frozen splits to {output_json}")
    return all_splits


def main():
    parser = argparse.ArgumentParser(description="Generate frozen experimental splits.")
    parser.add_argument("--mock_metadata", action="store_true", help="Create synthetic metadata for a smoke test")
    parser.add_argument("--id1000_csv", type=str, help="Real ID1000 scan-level metadata CSV")
    parser.add_argument("--piop_csv", type=str, help="Real combined PIOP scan-level metadata CSV")
    parser.add_argument("--output_json", type=str, default="data/splits/t1_aomic_age_splits.json")
    args = parser.parse_args()

    meta_dir = Path("data/metadata")
    splits_dir = Path("data/splits")

    if args.mock_metadata:
        id1000_csv = meta_dir / "id1000_participants.csv"
        piop_csv = meta_dir / "piop_participants.csv"
        create_synthetic_cohort_metadata(id1000_csv, cohort="id1000", n_subjects=300)
        create_synthetic_cohort_metadata(piop_csv, cohort="piop", n_subjects=150)
        
        task1_splits_file = Path(args.output_json)
        generate_task1_splits(id1000_csv, piop_csv, task1_splits_file)
    else:
        if not args.id1000_csv or not args.piop_csv:
            parser.error("Provide both --id1000_csv and --piop_csv, or pass --mock_metadata for a smoke test.")
        generate_task1_splits(Path(args.id1000_csv), Path(args.piop_csv), Path(args.output_json))


if __name__ == "__main__":
    main()
