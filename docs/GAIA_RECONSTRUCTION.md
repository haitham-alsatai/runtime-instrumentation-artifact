# GAIA Reconstruction and Case-Set Notes

## Raw Integrated Extraction

The integrated Stage B script reconstructs service-minute trace and metric
aggregates from the July 2021 GAIA MicroSS data and run table. The integrated
manifest records 900 discovered run-level cases.

## Complete Evaluation Subset

The paper-facing Stage B follow-up analyses use the 890-case subset with the
complete aggregate and cost information required by the reconstructed
evaluation:

- `gaia_ablation_experiment`
- `gaia_anchor_random_baseline`
- `gaia_baro_baseline`
- `gaia_mrca_baseline`
- `gaia_weight_sensitivity`
- `stage_b_weak_anchor`
- `gaia_ml_evidence_scorer`

The bundled `gaia_integrated_experiment/case_results_with_costs.csv` contains
890 unique case identifiers. The distinction between 900 discovered runs and
890 complete evaluation cases should remain explicit in reporting.

## Service Anchor

In the reconstructed case-level outputs, `root_service` functions as:

1. the correct post-alert service anchor in the uncorrupted
   service-conditioned setting; and
2. the ground-truth service label used for ranking evaluation.

Accordingly, Stage B is a post-alert service-conditioned instrumentation study,
not an independent test of blind alerted-service/root-cause mismatch.

## Aggregate Representation

The reconstruction creates:

- `trace_service_minute_endpoint.csv`
- `trace_service_minute.csv`
- `metric_service_minute.csv`
- case-level rankings and cost summaries

These aggregate files are included because they support transparent inspection
and rerunning of the Stage B follow-ups without redistributing raw GAIA data.

