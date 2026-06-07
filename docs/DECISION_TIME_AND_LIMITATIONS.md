# Decision-Time Information and Interpretation Boundaries

## Decision-Time Information

| Policy/analysis | Service anchor | Fault category | Time-window context | Unselected-service scores used during selection? | Withheld evaluation label used during selection? |
|---|---:|---:|---:|---:|---:|
| Stage A selectors | No | No | Injection split only | No | No |
| Stage B random multibudget | No | No | Budgeted random window | No | No |
| Stage B anchored random | Yes | No | Random within budget | No | The reconstructed correct-anchor condition uses `root_service` as the post-alert anchor |
| Stage B adaptive | Yes | Yes | Yes | No | The reconstructed correct-anchor condition uses `root_service` as the post-alert anchor |
| Weak-anchor adaptive | Corrupted decision-time anchor | Yes | Yes | No | True `root_service` is retained only for evaluation |
| Learned evidence scorers | Selection already fixed | Fault category feature | Selected aggregate context | No additional evidence selected | Test labels used only for evaluation |

## Core Limitations

1. Stage A is trace-only and uses a fixed lightweight downstream ranker.
2. Stage A retains all pre-injection traces; nominal budgets apply only to
   post-injection traces.
3. Stage A bootstrap intervals resample stored case outputs rather than rerun
   telemetry collection.
4. Stage B is post-alert and service-conditioned, not blind localization.
5. In the reconstructed Stage B setting, `root_service` is both the correct
   service anchor and the evaluation label.
6. Stage B uses reconstructed trace/metric aggregates and not the complete
   business-log layer.
7. BARO- and MRCA-style references are aggregate-level adaptations rather than
   native raw-telemetry reproductions.
8. Cost is represented by retained traces, rows, samples, bytes, time, and
   service coverage rather than measured runtime overhead.
9. The learned scorers are a downstream-sensitivity analysis specific to the
   retained GAIA aggregate evidence.
10. External validity is limited to RCAEval RE2-OB, RCAEval RE2-TT, and the
    evaluated GAIA reconstruction.

## Claims This Artifact Does Not Support

- A new state-of-the-art RCA method.
- Blind end-to-end localization in Stage B.
- Universal superiority of collecting less telemetry.
- Equal importance of where, when, and what context.
- Native reproduction or general defeat of BARO or MRCA.
- Direct runtime-overhead savings equal to the retained-volume reductions.

