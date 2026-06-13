# Anonymous Artifact: Adaptive Runtime Instrumentation Under Telemetry Budgets

This repository contains the scripts, derived results, paper-facing figures,
and verification documentation for a two-stage empirical study of budgeted
runtime instrumentation in microservice systems.

The artifact treats fault localization as a downstream measure of whether
retained runtime evidence remains useful. It does **not** present Stage B as
blind end-to-end root-cause localization.

## Artifact Scope

- **Stage A: budgeted trace instrumentation.** RCAEval RE2-OB and RE2-TT are
  used to characterize retained post-injection trace budgets and compare trace
  selectors under a fixed lightweight trace-only downstream ranker.
- **Stage B: context-conditioned instrumentation.** GAIA MicroSS is used in an
  explicitly post-alert setting to allocate telemetry across services, time
  windows, and endpoints, followed by ablation, weak-anchor, and
  downstream-scorer sensitivity analyses.

The repository integrates the complete supporting experiments and the broader
budget, selector, robustness, and scorer-sensitivity analyses. See
[`docs/ARTIFACT_COVERAGE.md`](docs/ARTIFACT_COVERAGE.md).

## Quick Start: Inspect Results Without Rerunning Experiments

The derived outputs are included because complete reruns require large public
datasets and can be time-consuming.

Start with:

1. [`docs/RQ_TO_ARTIFACT.md`](docs/RQ_TO_ARTIFACT.md) for the research-question
   mapping.
2. [`docs/KEY_RESULTS.md`](docs/KEY_RESULTS.md) for the principal verified
   values.
3. [`docs/GAIA_RECONSTRUCTION.md`](docs/GAIA_RECONSTRUCTION.md) for the Stage B
   reconstruction and case-set boundary.
4. [`figures/`](figures/) and [`tables/`](tables/) for paper-facing summaries.
5. [`results/stage_a/`](results/stage_a/) and
   [`results/stage_b/`](results/stage_b/) for case-level and aggregate outputs.

Then run the lightweight bundled-result check:

```bash
python scripts/verify_key_results.py
```

## Repository Layout

```text
README.md
DATA_SOURCES.txt
LICENSE
requirements.txt

docs/
  KEY_RESULTS.md
  NEW_VS_PREVIOUS_ARTIFACT.md
  REPRODUCIBILITY.md
  RQ_TO_ARTIFACT.md
  DECISION_TIME_AND_LIMITATIONS.md

data/
  RE2/README.md
  GAIA/README.md

scripts/
  stage_a/
  stage_b/

results/
  stage_a/
  stage_b/

figures/
tables/
```

## Main Extension Experiments

### Dense Stage A Budget Sweep

- Script: `scripts/stage_a/run_stage_a_dense_budget.py`
- Results: `results/stage_a/stage_a_dense_budget/`
- Design: 180 cases, nine budgets, ten random seeds, 16,200 records.

### Expanded Stage A Selector Comparison

- Script: `scripts/stage_a/run_stage_a_policy_expansion.py`
- Results: `results/stage_a/stage_a_policy_expansion/`
- Design: 180 cases, ten selectors, five budgets, 17,100 records.

### Stage B Weak-Anchor Robustness

- Script: `scripts/stage_b/run_stage_b_weak_anchor.py`
- Results: `results/stage_b/stage_b_weak_anchor/`
- Design: 890 cases, two budgets, six corruption levels, five repetitions.

### Stage B Case-Resampling and Fault-Macro Summary

- Script: `scripts/stage_b/make_stage_b_case_resampling_summary.py`
- Results: `results/stage_b/stage_b_case_resampling/`
- Design: derived analysis over the zero-corruption weak-anchor case outputs;
  randomized policies are averaged per case before 10,000 bootstrap replicates.

### GAIA Learned Evidence-Scorer Sensitivity

- Script: `scripts/stage_b/run_gaia_ml_evidence_scorer.py`
- Results: `results/stage_b/gaia_ml_evidence_scorer/`
- Design: adaptive selection held fixed; deterministic and learned scorers
  compared over five case-disjoint folds.

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduction Levels

- **Level 1: inspect bundled outputs.** No dataset download is needed.
- **Level 2: rerun Stage B follow-ups.** The bundled
  `results/stage_b/gaia_integrated_experiment/` aggregates support the Stage B
  ablation, weak-anchor, adapted-baseline, and learned-scorer scripts.
- **Level 3: full reproduction from raw public data.** Download RCAEval RE2 and
  GAIA using `DATA_SOURCES.txt`, then follow
  [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Important Interpretation Boundaries

- Stage A retains all pre-injection traces; its nominal budget applies only to
  post-injection traces.
- Stage A fixes a lightweight trace-only ranker to isolate evidence-retention
  effects.
- Stage B is post-alert and service-conditioned. In the reconstructed
  artifacts, `root_service` serves as both the selection anchor and evaluation
  label.
- The BARO- and MRCA-style scripts are aggregate-level adaptations, not native
  reproductions.
- Retained rows, samples, and bytes are monitoring-cost proxies, not direct
  measurements of runtime overhead.

See [`docs/DECISION_TIME_AND_LIMITATIONS.md`](docs/DECISION_TIME_AND_LIMITATIONS.md)
for the complete interpretation and leakage boundary.

Instructions for publishing this local artifact as a separate GitHub
repository are in [`docs/PUBLISHING.md`](docs/PUBLISHING.md).
