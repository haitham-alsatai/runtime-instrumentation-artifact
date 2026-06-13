# Key Derived Results

All values below are available in the corresponding CSV files under
`results/`.

## Stage A Dense Budget Sweep

Overall Avg@5:

| System | 5% | 10% | 25% | 50% | 100% |
|---|---:|---:|---:|---:|---:|
| RE2-OB | 0.612 | 0.617 | 0.624 | 0.621 | 0.624 |
| RE2-TT | 0.946 | 0.951 | 0.955 | 0.955 | 0.956 |

The nominal budget is the retained post-injection fraction. At a nominal 10%
budget, realized total-trace reduction is approximately 42.4% for RE2-OB and
43.9% for RE2-TT because pre-injection traces are retained in full.

## Stage A Policy Expansion

- RE2-OB at 10%: early-window delta versus random is +0.014 Avg@5, with
  bootstrap interval `[-0.013, 0.039]`.
- RE2-TT at 10%: every tested non-random selector is below random.
- RE2-TT at 10%: abnormality top-k delta is -0.129, interval
  `[-0.201, -0.066]`.
- RE2-TT at 10%: coverage-aware delta is -0.260, interval
  `[-0.350, -0.177]`.

## Stage B Main and Ablation Results

- Full adaptive context: Avg@5 is 0.826 at 25% and 0.642 at 50%.
- Adaptive 95% case-resampling intervals are `[0.815, 0.837]` at 25% and
  `[0.623, 0.660]` at 50%.
- Paired adaptive-minus-anchored-random Avg@5 intervals remain positive:
  `[0.011, 0.032]` at 25% and `[0.007, 0.036]` at 50%.
- Adaptive fault-type macro Avg@5 is 0.835 at 25% and 0.665 at 50%.
- No alerted-service context: Avg@5 is 0.213 at 25% and 0.229 at 50%.
- Removing fault-type or time-window context leaves performance close to full
  adaptive context.
- Moderate scoring-weight perturbations change Avg@5 by about +/-0.003 at the
  two multi-service budgets.

## Stage B Weak-Anchor Robustness

Adaptive Avg@5:

| Budget | 0% corruption | 10% | 20% | 30% | 40% | 50% |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 0.826 | 0.765 | 0.707 | 0.641 | 0.592 | 0.529 |
| 50% | 0.642 | 0.603 | 0.566 | 0.523 | 0.492 | 0.452 |

The incremental adaptive advantage over anchored random becomes small around
30% corruption.

## Learned Downstream Evidence Scoring

| Scorer | 25% Avg@5 | 25% Top-1 | 50% Avg@5 | 50% Top-1 |
|---|---:|---:|---:|---:|
| Deterministic formula | 0.826 | 0.418 | 0.642 | 0.249 |
| Logistic regression | 0.921 | 0.745 | 0.728 | 0.402 |
| Random forest | 0.938 | 0.840 | 0.809 | 0.665 |
| Histogram gradient boosting | 0.938 | 0.837 | 0.802 | 0.602 |

These models change only downstream scoring after adaptive selection; they do
not remove the post-alert service-anchor assumption.
