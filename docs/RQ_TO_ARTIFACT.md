# Research Question to Artifact Mapping

| RQ | Purpose | Primary script(s) | Primary results | Paper-facing outputs |
|---|---|---|---|---|
| RQ1 | Budgeted trace-instrumentation behavior across systems and faults | `scripts/stage_a/run_stage_a_dense_budget.py` | `results/stage_a/stage_a_dense_budget/` | `figures/stage_a_dense_budget_curves.*`, `tables/stage_a_dense_budget_summary.md`, `tables/stage_a_dense_min_budget_by_fault.md` |
| RQ2 | Whether structured selectors preserve more useful evidence than random | `scripts/stage_a/run_stage_a_policy_expansion.py` | `results/stage_a/stage_a_policy_expansion/` | `figures/stage_a_policy_delta_heatmap_*.{png,pdf}`, `figures/stage_a_policy_bootstrap_deltas.*`, `tables/stage_a_policy_expansion_*.md` |
| RQ3 | Post-alert allocation across where, when, and what | `scripts/stage_b/run_gaia_integrated_budget_experiment.py`, `scripts/stage_b/gaia_anchor_random_baseline.py`, `scripts/stage_b/gaia_ablation_eval.py`, `scripts/stage_b/make_stage_b_case_resampling_summary.py` | `results/stage_b/gaia_integrated_experiment/`, `results/stage_b/gaia_anchor_random_baseline/`, `results/stage_b/gaia_ablation_experiment/`, `results/stage_b/stage_b_case_resampling/` | `figures/stage_b_cost_utility_frontier.*` and Stage B summary CSVs |
| RQ4 | Operating boundaries and downstream sensitivity | `scripts/stage_b/run_stage_b_weak_anchor.py`, `scripts/stage_b/gaia_weight_sensitivity.py`, `scripts/stage_b/run_gaia_ml_evidence_scorer.py` | `results/stage_b/stage_b_weak_anchor/`, `results/stage_b/gaia_weight_sensitivity/`, `results/stage_b/gaia_ml_evidence_scorer/` | `figures/stage_b_weak_anchor_avg5_curve.*`, `figures/gaia_ml_evidence_scorer_avg5.*`, corresponding tables |

## Supporting and Contextual Experiments

| Experiment | Script | Results |
|---|---|---|
| Initial Stage A sparse budget study | `scripts/stage_a/run_trace_budget_experiment.py` | `results/stage_a/trace_budget_experiment/` |
| Random versus latency top-k | `scripts/stage_a/run_trace_policy_comparison_experiment.py` | `results/stage_a/trace_policy_experiment/` |
| Earlier structured selectors | `scripts/stage_a/run_trace_good_extensions_experiment.py` | `results/stage_a/trace_good_extensions_experiment/` |
| Timing pilot | `scripts/stage_a/run_trace_when_to_trace_pilot.py` | `results/stage_a/trace_when_pilot_experiment/` |
| Stage A stored-output bootstrap | `scripts/stage_a/stage_a_bootstrap_uncertainty.py` | `results/stage_a/stage_a_bootstrap_uncertainty/` |
| Adapted BARO-style reference | `scripts/stage_b/gaia_baro_baseline.py` | `results/stage_b/gaia_baro_baseline/` |
| Adapted MRCA-style reference | `scripts/stage_b/gaia_mrca_baseline.py` | `results/stage_b/gaia_mrca_baseline/` |
