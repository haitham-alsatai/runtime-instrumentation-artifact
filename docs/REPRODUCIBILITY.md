# Reproducibility Guide

## 1. Environment

Create and activate a Python environment, then install:

```bash
pip install -r requirements.txt
```

All commands below are run from the repository root.

## 2. Fast Verification Using Bundled Outputs

No experiment rerun is needed to inspect the principal values:

- Stage A dense budgets:
  `results/stage_a/stage_a_dense_budget/summary_overall.csv`
- Stage A selector deltas:
  `results/stage_a/stage_a_policy_expansion/policy_vs_random_delta.csv`
- Stage A selector uncertainty:
  `results/stage_a/stage_a_policy_expansion/policy_bootstrap_delta.csv`
- Stage B ablation:
  `results/stage_b/gaia_ablation_experiment/summary_overall.csv`
- Stage B weak anchor:
  `results/stage_b/stage_b_weak_anchor/summary_overall.csv`
- Learned scorer sensitivity:
  `results/stage_b/gaia_ml_evidence_scorer/summary_overall.csv`

## 3. Stage B Follow-Up Reruns From Bundled Aggregates

The following scripts consume the bundled
`results/stage_b/gaia_integrated_experiment/` aggregates:

```bash
python scripts/stage_b/gaia_ablation_eval.py
python scripts/stage_b/gaia_weight_sensitivity.py
python scripts/stage_b/gaia_anchor_random_baseline.py
python scripts/stage_b/gaia_baro_baseline.py
python scripts/stage_b/gaia_mrca_baseline.py
python scripts/stage_b/run_stage_b_weak_anchor.py
python scripts/stage_b/run_gaia_ml_evidence_scorer.py
```

These commands overwrite the corresponding result folders in a working copy.
Preserve the bundled outputs before rerunning if exact comparison is required.

## 4. Full Stage A Reruns

Download RCAEval RE2 and place it under `data/RE2/`, then run:

```bash
python scripts/stage_a/run_stage_a_dense_budget.py
python scripts/stage_a/run_stage_a_policy_expansion.py
```

The dense and expanded studies process 180 cases and may take substantial time.
Earlier supporting Stage A scripts are also available in `scripts/stage_a/`.

## 5. Full Integrated Stage B Rerun

Arrange the GAIA July 2021 data as described in `data/GAIA/README.md`, then run:

```bash
python scripts/stage_b/run_gaia_integrated_budget_experiment.py
```

The integrated experiment creates the aggregate files consumed by the Stage B
follow-ups.

## 6. Reproducibility Scope

The bundled outputs are the authoritative derived outputs used for claim
verification. Full reruns depend on public dataset versions, hardware,
available memory, and library versions. The artifact does not redistribute raw
public data.

