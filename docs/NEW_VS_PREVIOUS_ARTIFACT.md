# New Artifact Versus the Previous Submission Artifact

## Why the Previous Material Is Still Included

The revised study preserves the same two-stage empirical foundation. The
original Stage A and Stage B scripts/results are therefore still needed to
support the integrated experiment, ablations, adapted comparators, and earlier
selector analyses.

The new repository includes those materials so reviewers can inspect one
self-contained artifact. It does not require the previous repository.

## Preserved Supporting Experiments

### Stage A

- Initial trace-budget experiment.
- Random versus latency-prioritized comparison.
- Richer trace/span budget and structured-selector experiment.
- Early/late timing pilot.
- Stored-output bootstrap uncertainty analysis.

### Stage B

- GAIA integrated budget experiment and aggregates.
- Context ablation.
- Scoring-weight sensitivity.
- Anchored-service random baseline.
- Adapted BARO-style aggregate baseline.
- Adapted MRCA-style aggregate baseline.

## New Post-Submission Experiments

### Stage A Dense Budget Sweep

- Nine budgets rather than four.
- Ten random seeds.
- 180 cases and 16,200 records.
- Explicit nominal-versus-realized budget semantics.

### Stage A Policy Expansion

- Ten policy families.
- Five budgets.
- 17,100 records.
- Case-level bootstrap deltas against random.

### Stage B Weak-Anchor Robustness

- Anchor corruption extended through 50%.
- Two multi-service budgets.
- Five repetitions.
- Direct comparison with anchored random, random multibudget, and full
  reference.

### Learned Downstream Evidence Scoring

- Deterministic, logistic, random-forest, and histogram-gradient scorers.
- Five case-disjoint folds.
- Explicit root/alert indicator excluded from the main learned feature set.

## Excluded Materials

- Raw public datasets.
- Manuscript drafts and personal research notes.
- Duplicate empty output folders.
- Python caches and local logs.
- Absolute local paths and personal machine information.

