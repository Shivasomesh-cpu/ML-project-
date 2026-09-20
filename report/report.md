# Seen It Before: contamination-audited brain-MRI evaluation

## Status

This report does not infer experimental findings from mock data, missing results, or illustrative profiles. Run the manifest audit and matched observed evaluation cells before interpreting a conclusion.

## Contamination audit


**Status:** not yet available; unverified or mock contamination outputs are excluded.

Contamination table not yet compiled from a verified observed manifest and actual cohort subject IDs.

## Amsterdam Difference-in-Differences

Not yet available from observed data. Run BrainIAC and the baseline at matching task, N, seed, and trial budget.

## Label equivalence

Not yet available from observed data. Run at least two shared label budgets for the foundation model and baseline.

## Compute-matched Pareto frontier

Observed Pareto summary:

Not yet available from observed results. Mock runs and representative profiles are excluded.


## Limitations

- A subject-level audit is only as strong as the supplied pretraining manifest and source-specific ID normalization.
- Foundation checkpoints must be verified compatible with the adapter; a missing or incompatible checkpoint now stops the run.
- Task 1 has an explicit frozen-split scan-to-cache loader; equivalent loaders for Tasks 2 and 3 remain to be implemented.
