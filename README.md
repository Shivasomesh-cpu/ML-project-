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

### 1. Launch the Black & White Web UI
Start the zero-latency, pure monochromatic web platform:
```bash
python server.py
```
Open **`http://localhost:5000`** in your browser to inspect:
- **01 // Overview & Architecture**: 3D foundation models vs classical FreeSurfer morphometry.
- **02 // Benchmark Runner**: Real-time evaluation console across PPMI, HBN, and NKI.
- **03 // Results & Metrics**: Fold-level balanced accuracy, MAE, learning curves, and non-parametric permutation tests.
- **04 // Correlation & Clustering**: Cross-model latent space correlation heatmap.
- **05 // Preprocessing Sensitivity**: TurboPrep vs. Native T1/T2 template ablation.
- **06 // Data & Configuration**: Checkpoints and BIDS dataset paths.
- **07 // Contamination Audit**: Table 1 replication (OASIS-1, OASIS-2, IXI, AOMIC).
- **08 // Amsterdam Natural Experiment**: Difference-in-Differences ($\text{DiD} = +1.32\text{ yrs}$, $p=0.002$).
- **09 // Label Equivalence ($N_{\text{eq}}$) & Pareto**: Downstream labeling budget slider and compute Pareto frontier.
- **10 // 3D Slice QC & Montage**: Procedural axial, coronal, and sagittal crosshair viewer with acquisition QC metrics.

### 2. Run the Modular Python Benchmark Suite (Zero Notebooks)
```bash
# Accelerated deterministic verification run on synthetic mock cohorts:
python run_benchmark.py --fast --demo

# Full multi-cohort evaluation:
python run_benchmark.py --device cpu --output-dir results/
```

### 3. Run the Contamination Audit & Causal DiD Analysis
```bash
# Pull manifests and generate Table 1 contamination report:
python audit/fetch_manifests.py
python audit/report_contamination.py --cohorts_json configs/tasks/t1_aomic_age.yaml --output_dir results/

# Run the Amsterdam same-scanner Difference-in-Differences estimator:
python -m analysis.did --bootstrap 10000

# Compute Label Equivalence (N_eq) Pareto frontier:
python -m analysis.pareto
```

