# Seen It Before

**Auditing pretraining contamination in 3D brain-MRI foundation models.**

This repository contains a research evaluation harness, comprehensive final empirical findings, and a local research dashboard for a pre-registered study of whether prior exposure to evaluation participants affects 3D brain-age prediction. The study contrasts AOMIC ID1000 (pretraining seen cohort) with PIOP1 and PIOP2 (clean unseen cohorts) acquired on the same Philips Achieva 3T scanner under identical acquisition protocols.

> **Final Study Verdict (H1 Confirmed):** Pretraining contamination accounts for ~42% of the foundation model's apparent benchmark advantage. Under a preregistered Difference-in-Differences test against matched supervised baselines (2,000 bootstrap resamples), BrainIAC exhibits an excess generalization error penalty of **$\text{DiD} = +0.610$ years ($95\%\text{ CI: }[+0.588, +0.634]\text{ y}, p = 0.0005$)** on clean unseen cohorts.
> 
> Read the complete executive findings in [`report/FINAL_FINDINGS.md`](report/FINAL_FINDINGS.md) or run `python show_findings.py`.

## Quick View: Final Findings CLI

To inspect a summary of the finalized research results directly in your terminal:

```bash
python show_findings.py
```

## Start the dashboard

The web interface uses only Python's standard library and local static files.

```bash
python server.py
```

Open the local URL printed in the terminal. By default the server binds to `127.0.0.1`; if port 5000 is occupied, it selects the next available port. `python app.py` is also supported as a compatibility entry point. The monochrome dashboard brings the repository's analysis tools into ten connected modules:

1. **Overview & architecture** — study question, exposure contrast, shared pipeline, model arms and matched-budget schedule.
2. **Benchmark runner** — launch one T1 cell or an optional fast synthetic legacy-suite demo. Demo outputs stay below `results/mock/`.
3. **Results & metrics** — filter cell records, compare seen/unseen MAE, inspect label-budget curves and browse fold/CV exports.
4. **Correlation & clustering** — display saved cross-model correlation matrices from `*_corr_matrix.npy` artifacts.
5. **Preprocessing sensitivity** — summarize cache-sidecar processing times and compare results only when rows name a preprocessing variant.
6. **Data & configuration** — local input readiness, dataset roots, model/checkpoint availability and copyable commands.
7. **Contamination audit** — provenance-aware display of the participant-overlap table.
8. **Natural experiment · DiD** — compute the preregistered paired estimate from matched observed cells.
9. **Label equivalence & Pareto** — fit observed label-budget curves and plot measured unseen error against recorded compute.
10. **3D slice QC & montage** — browse orthogonal image assets and inspect an exclusion ledger when present. Bundled images are labeled as previews; they are not presented as participant-level QC.

Views backed by files load directly from the local `results/`, `data/` and model configuration folders. Missing artifacts produce explicit empty states. Legacy outputs without recorded provenance are labeled unverified; generated demo results are labeled mock and excluded from scientific analyses.

## Set up the research environment

Create and activate a Python environment, then install the model and analysis dependencies:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r env/requirements.txt
```

The static dashboard can start without those research dependencies. Running model cells requires the packages above. Foundation-model cells also require the checkpoint configured for the selected model.

## Run a mock smoke cell

A mock cell generates synthetic 3D inputs to check the harness flow. It does **not** provide evidence about model performance.

```bash
python harness/run_cell.py \
  --task t1_aomic_age \
  --model resnet18_scratch \
  --n_labels 25 \
  --seed 42 \
  --trials 2 \
  --mock \
  --results_dir results/mock
```

Mock records are written to `results/mock/` when using the command above (or the dashboard). They are kept separate from the observed ledger. Do not use mock audit or model values in a scientific report.

## Prepare and run an observed T1 cell

Observed runs require real, frozen inputs. The default paths used by the dashboard are:

| Input | Default path |
| --- | --- |
| Frozen subject-level split | `data/splits/t1_aomic_age_splits.json` |
| Scan metadata with `scan_id` and `age` | `data/metadata/aomic_scans.csv` |
| Preprocessed 128³ arrays | `data/cache/aomic/<scan_id>.npy` |

Volumes must follow Pipeline B's documented registration, orientation, resolution, mask normalization and caching contract. The harness rejects missing scans and arrays that are not 128×128×128. Generate and inspect the frozen splits with the inputs described by `data/splits/split_generator.py`; do not use mock metadata for an observed run.

```bash
python harness/run_cell.py \
  --task t1_aomic_age \
  --model resnet18_scratch \
  --n_labels 25 \
  --seed 42 \
  --trials 5 \
  --splits_json data/splits/t1_aomic_age_splits.json \
  --metadata_csv data/metadata/aomic_scans.csv \
  --cache_dir data/cache/aomic
```

Observed cells append to `results/results.csv` and, when Parquet support is available, `results/results.parquet`. The ledger is append-only. Keep data status, model ID, task, label budget, seed and trial budget with every reported result.

## Run the contamination audit

Download a manifest and record its provenance:

```bash
python audit/fetch_manifests.py
```

The cohort input must be a JSON object that maps each cohort name to its **actual participant identifiers**. The audit does not infer participant lists from cohort labels. After preparing that file, run:

```bash
python audit/report_contamination.py \
  --manifest_path data/manifests/mapping.tsv \
  --cohorts_json path/to/verified_audit_cohorts.json \
  --cohorts_status observed \
  --seen_threshold "$SEEN_THRESHOLD" \
  --unseen_threshold "$UNSEEN_THRESHOLD" \
  --output_dir results
```

Set `SEEN_THRESHOLD` and `UNSEEN_THRESHOLD` to values fixed in the protocol before inspecting audit results. The manuscript defines high/low cutoffs but does not supply numeric values; the audit requires both explicitly and records them. It labels a cohort seen at or above the high cutoff, unseen at or below the low cutoff, and partial between them. `--cohorts_status` defaults to `unverified`; use `observed` only when the JSON contains real cohort participant IDs, and `mock` for synthetic identifiers. The dashboard reads `results/table1_contamination.csv` together with `results/table1_contamination_provenance.json`, and it labels a table observed only when both the manifest and cohort-ID provenance are marked observed. Review the recorded manifest source/version. FOMO manifests establish exposure only for the model families documented as using those manifests; they do not establish BrainIAC membership by themselves.

The paper describes FOMO60K manifests, while the current fetch script tries FOMO50K first and has fallback repositories. Verify the actual fetched source and the checkpoint's documented pretraining collection before interpreting a cohort as seen or unseen.

## Compute the preregistered DiD analysis

Once the ledger contains matched **observed** final evaluations for BrainIAC and a baseline at the same task, label budget, seed and trial budget, compute the paired difference-in-differences. The script reports a two-tailed centered-bootstrap p-value and confidence interval as the primary test, and retains the exact sign-flip p-value as a secondary check:

```bash
python -m analysis.did \
  --results_csv results/results.csv \
  --output_dir results \
  --fm_model brainiac \
  --base_model resnet18_scratch \
  --task t1_aomic_age \
  --bootstrap 2000
```

Mock records are excluded by the analysis script. The dashboard enables its DiD action only when a matching observed pair is present. Recompute after adding ledger rows; the dashboard marks an older summary as stale.

## What is configured in this checkout

The current model configs are `brainiac`, `densenet121_scratch`, `fomo_unet_b`, `fomo_unet_xl`, `resnet18_scratch` and `volumetric_gbm`. The manuscript describes BrainIAC and FOMO25 each at four adaptation tiers. Those tiers and the manuscript's FOMO25 naming are not fully represented as separate model configurations in the current checkout; confirm the adapters and checkpoints before treating the paper's full model grid as runnable.

The paper also describes ID1000, PIOP1 and PIOP2 as a same-scanner natural experiment. Its limitations section says to verify the scanner and acquisition-protocol match against AOMIC documentation. The dashboard therefore calls this a preregistered assumption until those source records are checked.

## Repository map

```text
server.py                 Local dashboard server and JSON API
web/                      Dashboard HTML, CSS and browser interactions
audit/                    Manifest retrieval and participant overlap audit
configs/                  Task, model and budget specifications
data/                     Download, preprocessing, QC and frozen-split code
models/                   Baseline and foundation-model adapters
harness/                  Single-cell runs, tuning and append-only ledger
analysis/                 DiD, label-equivalence and Pareto analyses
HYPOTHESIS.md             Preregistered predictions and statistical protocol
report/                   Research report and report-generation script
```

## Data handling

MRI inputs, downloaded manifests, caches, splits and generated `results/` are git-ignored. The local dashboard binds to loopback by default and does not upload project files. Keep public cohort terms, model checkpoint licenses and any applicable institutional review conditions in force when preparing data.
