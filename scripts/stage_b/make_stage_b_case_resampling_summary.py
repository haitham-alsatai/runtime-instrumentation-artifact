#!/usr/bin/env python3
"""Derive Stage B case-resampling intervals and fault-macro summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "stage_b" / "stage_b_weak_anchor" / "case_results.csv"
OUTPUT = ROOT / "results" / "stage_b" / "stage_b_case_resampling"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEEDS = {0.25: 94, 0.50: 190}
POLICIES = [
    "adaptive_corrupted_anchor",
    "anchored_random_rest",
    "random_multibudget",
    "full_reference",
]
METRICS = ["avg_at_5", "top1", "top3", "top5"]


def bootstrap_interval(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    distribution = values[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    rows = pd.read_csv(INPUT)
    rows = rows[
        (rows["anchor_corruption"] == 0.0)
        & rows["budget"].isin(BOOTSTRAP_SEEDS)
        & rows["policy"].isin(POLICIES)
    ].copy()

    # Randomized policies are averaged within case before cases are resampled.
    cases = (
        rows.groupby(
            ["case_id", "root_service", "fault_type", "service_family", "budget", "policy"],
            as_index=False,
        )[METRICS]
        .mean()
        .sort_values(["budget", "policy", "case_id"])
    )

    overall = cases.groupby(["policy", "budget"], as_index=False)[METRICS].mean()
    fault_wise = cases.groupby(["policy", "budget", "fault_type"], as_index=False)[METRICS].mean()
    macro = fault_wise.groupby(["policy", "budget"], as_index=False)[METRICS].mean()

    bootstrap_rows: list[dict[str, float | int | str]] = []
    for budget, seed in BOOTSTRAP_SEEDS.items():
        budget_cases = cases[cases["budget"] == budget]
        pivots = {
            metric: budget_cases.pivot(index="case_id", columns="policy", values=metric)
            for metric in METRICS
        }
        case_count = len(pivots["avg_at_5"])
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, case_count, size=(BOOTSTRAP_REPLICATES, case_count))

        for metric in ["avg_at_5", "top1"]:
            values = pivots[metric]["adaptive_corrupted_anchor"].to_numpy()
            low, high = bootstrap_interval(values, indices)
            bootstrap_rows.append(
                {
                    "comparison": "adaptive",
                    "budget": budget,
                    "metric": metric,
                    "cases": case_count,
                    "replicates": BOOTSTRAP_REPLICATES,
                    "seed": seed,
                    "point_estimate": float(values.mean()),
                    "ci_low": low,
                    "ci_high": high,
                }
            )

        delta = (
            pivots["avg_at_5"]["adaptive_corrupted_anchor"]
            - pivots["avg_at_5"]["anchored_random_rest"]
        ).to_numpy()
        low, high = bootstrap_interval(delta, indices)
        bootstrap_rows.append(
            {
                "comparison": "adaptive_minus_anchored_random",
                "budget": budget,
                "metric": "avg_at_5",
                "cases": case_count,
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": seed,
                "point_estimate": float(delta.mean()),
                "ci_low": low,
                "ci_high": high,
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases.to_csv(OUTPUT / "case_results_repetition_averaged.csv", index=False)
    overall.to_csv(OUTPUT / "summary_case_weighted.csv", index=False)
    fault_wise.to_csv(OUTPUT / "summary_by_fault_type.csv", index=False)
    macro.to_csv(OUTPUT / "summary_fault_macro.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(OUTPUT / "bootstrap_intervals.csv", index=False)

    (OUTPUT / "README.md").write_text(
        "# Stage B Case-Resampling and Fault-Macro Summary\n\n"
        "This folder derives the paper's Stage B uncertainty and subgroup summaries "
        "from the stored zero-corruption weak-anchor case outputs.\n\n"
        "- Randomized policies are averaged per case across five repetitions.\n"
        "- Bootstrap intervals use 10,000 nonparametric case-resampling replicates.\n"
        "- Fault-macro values assign equal weight to each observed fault type.\n"
        "- These are derived analyses over stored outputs, not experiment reruns.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
