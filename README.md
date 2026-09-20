# Seen It Before: Do Brain-MRI Foundation Models Generalize, or Have They Already Met Your Test Set?

[![Pre-registered](https://img.shields.io/badge/Pre--registered-HYPOTHESIS.md-blue.svg)](./HYPOTHESIS.md)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

A contamination-audited, label-budget-matched evaluation harness for 3D brain-MRI foundation models (BrainIAC, FOMO25) against well-tuned supervised baselines (MONAI ResNet-18, DenseNet-121, Volumetric + LightGBM).

---

## Key Features

1. **Contamination Audit Suite (`audit/`)**:
   - Subject-level set intersection against FOMO50K/FOMO60K pretraining manifests (`mapping.tsv`, `participants.tsv`).
   - Automatically generates Table 1 quantifying contamination percentages across standard benchmarks.

2. **The Amsterdam Natural Experiment (`configs/tasks/t1_aomic_age.yaml`)**:
   - Compares **AOMIC ID1000** (pretraining seen) vs **AOMIC PIOP1 & PIOP2** (unseen) on the exact same scanner (Philips Achieva 3T at UvA) and acquisition protocol.
   - Enforces strict age matching (19–26 years) and sex stratification.
   - Estimates Difference-in-Differences ($\text{DiD}$) with 2,000 bootstrap resamples.

3. **Standardized Adaptation Ladder (`models/foundation/`)**:
   - Evaluates foundation models across a 4-tier adaptation ladder:
     1. Frozen encoder + Linear Probe
     2. Frozen encoder + 2-layer MLP head
     3. Last-block fine-tuning + head
     4. Full fine-tuning

4. **Rigorous Experimental Hygiene**:
   - Matched label budgets: $N \in \{25, 50, 100, 250, 500, \text{all}\}$.
   - Matched Optuna tuning budgets with exact GPU-second accounting.
   - Common Pipeline B preprocessing (2mm isotropic, $128^3$, in-mask Z-score, float16 `.npy` caching). Template registration must be completed and documented upstream; the preprocessing script no longer implies it has registered scans.
   - Append-only results ledger (`results.parquet`).

---

## Directory Layout

```
seen-it-before/
├── HYPOTHESIS.md              # Pre-registered predictions and statistical protocol
├── env/
│   └── requirements.txt       # Core dependencies
├── configs/
│   ├── tasks/                 # Task specs (T1: AOMIC age, T2: IXI->PIOP, T3: OASIS dementia)
│   ├── models/                # Model hyperparameters and search spaces
│   └── budgets.yaml           # Label budgets, seeds, Optuna search budgets
├── audit/
│   ├── fetch_manifests.py     # Download FOMO50K metadata manifests
│   ├── overlap.py             # Match subject IDs across cohorts
│   └── report_contamination.py# Generate Table 1 and contamination figures
├── data/
│   ├── download/              # Per-dataset fetchers (OpenNeuro, IXI, OASIS)
│   ├── preprocess/            # Pipeline A & Pipeline B implementations
│   ├── qc/                    # Slice montage generator & exclusion logging
│   └── splits/                # Frozen, age/sex stratified subject-level splits
├── models/
│   ├── registry.py            # Unified ModelArm ABC: fit(loader, budget) / predict(loader)
│   ├── baselines/             # MONAI ResNet-18, DenseNet-121, Volumetric+GBM
│   └── foundation/            # BrainIAC and FOMO25 adapters
├── harness/
│   ├── run_cell.py            # CLI to run a single (task, model, N, seed, budget) cell
│   ├── sweep.py               # Optuna hyperparameter optimization engine
│   └── ledger.py              # Thread-safe append-only results.parquet logger
├── analysis/
│   ├── did.py                 # Difference-in-Differences + bootstrap CIs
│   ├── label_equivalence.py   # Curve fitting for N_eq
│   └── pareto.py              # Performance vs GPU-hours Pareto plots
└── results/
    └── results.parquet        # Source-of-truth results store
```

---

## Quickstart

### 1. Environment Setup
```bash
pip install -r env/requirements.txt
```

### 2. Run the Contamination Audit
```bash
# Pulls FOMO50K/45K manifests and checks subject overlap with evaluation benchmarks
python audit/fetch_manifests.py
python audit/report_contamination.py --cohorts_json data/metadata/audit_cohorts.json --output_dir results/
```

### 3. Run a Smoke Test on the Harness
```bash
# Runs a micro cell: Task T1, ResNet18 baseline, N=25, seed=42, 5 Optuna trials
python harness/run_cell.py --task t1_aomic_age --model resnet18_scratch --n_labels 25 --seed 42 --trials 5 --mock
```

`--mock` is only a plumbing check: its ledger rows are marked `mock` and analysis
scripts exclude them. For a real Task 1 run, provide frozen splits, scan metadata,
and a directory of preprocessed `128^3` caches:

```bash
python harness/run_cell.py --task t1_aomic_age --model resnet18_scratch --n_labels 25 --seed 42 --trials 5 \
  --splits_json data/splits/t1_aomic_age_splits.json \
  --metadata_csv data/metadata/aomic_scans.csv --cache_dir data/cache/aomic
```
