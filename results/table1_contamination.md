# Table 1: Benchmark Pretraining Contamination Audit
**Manifest Version:** FOMO60K (Release 2025/2026) | **Status:** Observed

**Pre-registered status cutoffs:** seen >= 95.0% | unseen <= 5.0% | partial between

| Evaluation Cohort | Total Subjects ($N$) | Found in Pretraining ($N_{\text{seen}}$) | Contamination Rate (%) | Pretraining Status |
| :--- | :---: | :---: | :---: | :--- |
| AOMIC ID1000 | 928 | 928 | 100.0% | **SEEN (Contaminated)** |
| AOMIC PIOP1 | 216 | 0 | 0.0% | **UNSEEN (Clean)** |
| AOMIC PIOP2 | 226 | 0 | 0.0% | **UNSEEN (Clean)** |
| IXI | 568 | 568 | 100.0% | **SEEN (Contaminated)** |
| OASIS-1 | 416 | 416 | 100.0% | **SEEN (Contaminated)** |
| OASIS-2 | 150 | 150 | 100.0% | **SEEN (Contaminated)** |

*Verification note: AOMIC ID1000 is 100% represented in FOMO60K pretraining manifests, whereas PIOP1 and PIOP2 are completely absent (0% overlap), satisfying the prerequisite for the same-scanner natural experiment.*