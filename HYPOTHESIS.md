# Pre-Registered Hypothesis: "Seen It Before" Evaluation

**Date of Pre-Registration:** 2026-09-20  
**Status:** Frozen prior to model training and evaluation

---

## 1. Background & Scope

Public 3D brain-MRI foundation models (such as BrainIAC and the FOMO25 UNet checkpoints) have demonstrated strong downstream performance, particularly in low-label regimes. However, the pretraining corpus (e.g., FOMO50K / FOMO60K) includes public datasets that are standard benchmarks in the field, including OASIS-1, OASIS-2, IXI, and AOMIC ID1000.

This evaluation leverages a natural experiment within the Amsterdam Open MRI Collection (AOMIC):
- **AOMIC ID1000**: Acquired on Philips Achieva 3T at LAB neuroimaging centre (UvA), BIDS formatted, healthy young adults. **Included in FOMO50K pretraining data (SEEN).**
- **AOMIC PIOP1 & PIOP2**: Acquired on the exact same Philips Achieva 3T scanner at UvA, identical protocol, healthy young adults. **Excluded from FOMO50K pretraining data (UNSEEN).**

By comparing performance differences across seen and unseen cohorts between foundation models and from-scratch supervised baselines, we control for scanner, site, and protocol confounds.

---

## 2. Formal Hypotheses

### Primary Hypothesis ($H_1$)
Foundation models derive a statistically significant performance advantage from pretraining benchmark contamination rather than pure out-of-distribution architectural representation learning. Specifically, the drop in performance from the seen cohort to the unseen cohort will be significantly larger for the foundation model than for a tuned supervised baseline trained from scratch.

Let:
$$\Delta_{\text{model}} = \text{Metric}(\text{model}, \text{Seen Cohort}) - \text{Metric}(\text{model}, \text{Unseen Cohort})$$

For performance metrics where lower is better (e.g., Mean Absolute Error in brain age prediction, MAE):
$$\text{Performance Gap} = \text{MAE}_{\text{unseen}} - \text{MAE}_{\text{seen}}$$

We define the Difference-in-Differences statistic ($\text{DiD}$):
$$\text{DiD} = \Delta_{\text{FM}} - \Delta_{\text{baseline}} = (\text{MAE}_{\text{FM, unseen}} - \text{MAE}_{\text{FM, seen}}) - (\text{MAE}_{\text{Base, unseen}} - \text{MAE}_{\text{Base, seen}})$$

- **$H_1$ (Contamination Hypothesis):** $\text{DiD} > 0$ with $p < 0.05$ (two-tailed bootstrap test, 2,000 resamples).
- **$H_0$ (Generalization / Invariance Hypothesis):** $\text{DiD} \le 0$. If $\text{DiD} \approx 0$, it provides empirical proof that the foundation model representation generalizes out-of-distribution without reliance on pretraining memorization.

---

## 3. Mandatory Experimental Controls

To eliminate confounding variables, all evaluations must satisfy:
1. **Age-Window Restriction:** Both AOMIC ID1000 and PIOP cohorts must be restricted to the exact overlapping age window: **19.0 to 26.0 years**.
2. **Demographic Balance:** Evaluation splits must be stratified by sex and age histogram.
3. **Subject-Level Split Isolation:** No subject or longitudinal session from the same subject may appear in both train and validation/test splits.
4. **Matched Label Budgets:** Evaluated at fixed sample sizes $N \in \{25, 50, 100, 250, 500, \text{all}\}$.
5. **Matched Tuning Budget:** Baselines and foundation model adaptation heads must receive identical Optuna trial budgets (5, 20, 50 trials) and identical search spaces.
6. **Unified Preprocessing:** Evaluated on Common Pipeline B (2mm isotropic, $128^3$ MNI152 space, RAS, in-mask Z-score).

---

## 4. Secondary Pre-Registered Metrics

1. **Label Equivalence ($N_{\text{eq}}$):**
   $$N_{\text{eq}}(N) = N_{\text{baseline required to match FM at } N} - N$$
   Assessed separately for seen and unseen cohorts.
2. **Compute-Matched Pareto Frontier:**
   Performance plotted against total logged GPU-seconds (preprocessing + tuning + training + inference).
3. **Rank Stability:**
   Frequency of model rank flips across 5 random seeds ($42, 1337, 2026, 7, 99$).

---

## 5. Statistical Protocol & Stopping Rules

- **Number of seeds:** 5 random seeds per cell $(task, model, N)$.
- **Resampling:** 2,000 bootstrap iterations for 95% confidence intervals.
- **Multiple testing correction:** Holm-Bonferroni correction across task suite T1–T3.
- **Test Set Quarantine:** Test sets are evaluated exactly once at final benchmark freeze; all hyperparameter selection is strictly confined to validation folds.
