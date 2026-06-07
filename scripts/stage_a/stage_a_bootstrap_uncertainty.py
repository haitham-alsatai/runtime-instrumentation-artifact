#!/usr/bin/env python3
"""Bootstrap uncertainty summaries for Stage A RCAEval case-level results.

This is not a multi-seed rerun of the original Stage A policies. The stored
artifact snapshot only preserves one realization of the random policies, but it
does preserve per-case outputs. We therefore estimate uncertainty by
nonparametric case resampling within each experiment group.
"""

from __future__ import annotations

import csv
import math
import os
import random
from collections import defaultdict
from statistics import mean


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_ROOT = os.path.join(ROOT, "results", "stage_a")
OUT_DIR = os.path.join(RESULTS_ROOT, "stage_a_bootstrap_uncertainty")

BOOTSTRAP_REPS = 1000
SEED = 20260327


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv(path: str):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def percentile(sorted_values, q: float) -> float:
    if not sorted_values:
        return math.nan
    idx = (len(sorted_values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(sorted_values[lo])
    frac = idx - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def bootstrap_mean_ci(values, rng: random.Random, reps: int = BOOTSTRAP_REPS):
    n = len(values)
    if n == 0:
        return math.nan, math.nan, math.nan
    means = []
    for _ in range(reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    point = sum(values) / n
    return point, percentile(means, 0.025), percentile(means, 0.975)


def bootstrap_delta_ci(values_a, values_b, rng: random.Random, reps: int = BOOTSTRAP_REPS):
    n = min(len(values_a), len(values_b))
    if n == 0:
        return math.nan, math.nan, math.nan
    deltas = []
    for _ in range(reps):
        idxs = [rng.randrange(n) for _ in range(n)]
        a = sum(values_a[i] for i in idxs) / n
        b = sum(values_b[i] for i in idxs) / n
        deltas.append(a - b)
    deltas.sort()
    point = (sum(values_a[:n]) / n) - (sum(values_b[:n]) / n)
    return point, percentile(deltas, 0.025), percentile(deltas, 0.975)


def write_csv(path: str, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_trace_budget(rng: random.Random):
    rows = read_csv(os.path.join(RESULTS_ROOT, "trace_budget_experiment", "case_results.csv"))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["budget"])].append(float(row["avg5"]))

    out = []
    for (dataset, budget), values in sorted(groups.items()):
        point, lo, hi = bootstrap_mean_ci(values, rng)
        out.append(
            {
                "dataset": dataset,
                "budget": budget,
                "cases": len(values),
                "avg5": f"{point:.6f}",
                "ci95_low": f"{lo:.6f}",
                "ci95_high": f"{hi:.6f}",
                "ci95_half_width": f"{((hi - lo) / 2):.6f}",
            }
        )
    write_csv(
        os.path.join(OUT_DIR, "trace_budget_avg5_bootstrap.csv"),
        out,
        ["dataset", "budget", "cases", "avg5", "ci95_low", "ci95_high", "ci95_half_width"],
    )


def summarize_trace_policy(rng: random.Random):
    rows = read_csv(os.path.join(RESULTS_ROOT, "trace_policy_experiment", "case_results.csv"))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["policy"], row["budget"])].append(float(row["avg5"]))

    summary_rows = []
    for (dataset, policy, budget), values in sorted(grouped.items()):
        point, lo, hi = bootstrap_mean_ci(values, rng)
        summary_rows.append(
            {
                "dataset": dataset,
                "policy": policy,
                "budget": budget,
                "cases": len(values),
                "avg5": f"{point:.6f}",
                "ci95_low": f"{lo:.6f}",
                "ci95_high": f"{hi:.6f}",
            }
        )
    write_csv(
        os.path.join(OUT_DIR, "trace_policy_avg5_bootstrap.csv"),
        summary_rows,
        ["dataset", "policy", "budget", "cases", "avg5", "ci95_low", "ci95_high"],
    )

    delta_rows = []
    for dataset in sorted({k[0] for k in grouped}):
        for budget in sorted({k[2] for k in grouped if k[0] == dataset}, key=float):
            if (dataset, "latency_topk", budget) in grouped and (dataset, "random", budget) in grouped:
                point, lo, hi = bootstrap_delta_ci(
                    grouped[(dataset, "latency_topk", budget)],
                    grouped[(dataset, "random", budget)],
                    rng,
                )
                delta_rows.append(
                    {
                        "dataset": dataset,
                        "budget": budget,
                        "comparison": "latency_topk_minus_random",
                        "delta_avg5": f"{point:.6f}",
                        "ci95_low": f"{lo:.6f}",
                        "ci95_high": f"{hi:.6f}",
                    }
                )
    write_csv(
        os.path.join(OUT_DIR, "trace_policy_delta_bootstrap.csv"),
        delta_rows,
        ["dataset", "budget", "comparison", "delta_avg5", "ci95_low", "ci95_high"],
    )


def summarize_trace_when(rng: random.Random):
    rows = read_csv(os.path.join(RESULTS_ROOT, "trace_when_pilot_experiment", "case_results.csv"))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["policy"], row["budget"])].append(float(row["avg5"]))

    summary_rows = []
    for (dataset, policy, budget), values in sorted(grouped.items()):
        point, lo, hi = bootstrap_mean_ci(values, rng)
        summary_rows.append(
            {
                "dataset": dataset,
                "policy": policy,
                "budget": budget,
                "cases": len(values),
                "avg5": f"{point:.6f}",
                "ci95_low": f"{lo:.6f}",
                "ci95_high": f"{hi:.6f}",
            }
        )
    write_csv(
        os.path.join(OUT_DIR, "trace_when_avg5_bootstrap.csv"),
        summary_rows,
        ["dataset", "policy", "budget", "cases", "avg5", "ci95_low", "ci95_high"],
    )

    delta_rows = []
    for dataset in sorted({k[0] for k in grouped}):
        for budget in sorted({k[2] for k in grouped if k[0] == dataset}, key=float):
            for policy in ("early_window", "late_window"):
                if (dataset, policy, budget) in grouped and (dataset, "random", budget) in grouped:
                    point, lo, hi = bootstrap_delta_ci(
                        grouped[(dataset, policy, budget)],
                        grouped[(dataset, "random", budget)],
                        rng,
                    )
                    delta_rows.append(
                        {
                            "dataset": dataset,
                            "budget": budget,
                            "comparison": f"{policy}_minus_random",
                            "delta_avg5": f"{point:.6f}",
                            "ci95_low": f"{lo:.6f}",
                            "ci95_high": f"{hi:.6f}",
                        }
                    )
    write_csv(
        os.path.join(OUT_DIR, "trace_when_delta_bootstrap.csv"),
        delta_rows,
        ["dataset", "budget", "comparison", "delta_avg5", "ci95_low", "ci95_high"],
    )


def summarize_good_extensions(rng: random.Random):
    rows = read_csv(os.path.join(RESULTS_ROOT, "trace_good_extensions_experiment", "case_results.csv"))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["budget_type"], row["dataset"], row["policy"], row["budget"])].append(float(row["avg5"]))

    summary_rows = []
    for (budget_type, dataset, policy, budget), values in sorted(grouped.items()):
        point, lo, hi = bootstrap_mean_ci(values, rng)
        summary_rows.append(
            {
                "budget_type": budget_type,
                "dataset": dataset,
                "policy": policy,
                "budget": budget,
                "cases": len(values),
                "avg5": f"{point:.6f}",
                "ci95_low": f"{lo:.6f}",
                "ci95_high": f"{hi:.6f}",
            }
        )
    write_csv(
        os.path.join(OUT_DIR, "trace_good_extensions_avg5_bootstrap.csv"),
        summary_rows,
        ["budget_type", "dataset", "policy", "budget", "cases", "avg5", "ci95_low", "ci95_high"],
    )

    delta_rows = []
    for budget_type, dataset in sorted({(k[0], k[1]) for k in grouped}):
        budgets = sorted({k[3] for k in grouped if k[0] == budget_type and k[1] == dataset}, key=float)
        for budget in budgets:
            base_key = (budget_type, dataset, "random", budget)
            if base_key not in grouped:
                continue
            for policy in ("abnormality_topk", "service_aware_abnormality"):
                key = (budget_type, dataset, policy, budget)
                if key in grouped:
                    point, lo, hi = bootstrap_delta_ci(grouped[key], grouped[base_key], rng)
                    delta_rows.append(
                        {
                            "budget_type": budget_type,
                            "dataset": dataset,
                            "budget": budget,
                            "comparison": f"{policy}_minus_random",
                            "delta_avg5": f"{point:.6f}",
                            "ci95_low": f"{lo:.6f}",
                            "ci95_high": f"{hi:.6f}",
                        }
                    )
    write_csv(
        os.path.join(OUT_DIR, "trace_good_extensions_delta_bootstrap.csv"),
        delta_rows,
        ["budget_type", "dataset", "budget", "comparison", "delta_avg5", "ci95_low", "ci95_high"],
    )


def main():
    ensure_dir(OUT_DIR)
    rng = random.Random(SEED)
    summarize_trace_budget(rng)
    summarize_trace_policy(rng)
    summarize_trace_when(rng)
    summarize_good_extensions(rng)

    readme = os.path.join(OUT_DIR, "README.txt")
    with open(readme, "w") as f:
        f.write(
            "Stage A bootstrap uncertainty summaries\\n"
            "=====================================\\n\\n"
            f"Replicates: {BOOTSTRAP_REPS}\\n"
            f"Seed: {SEED}\\n\\n"
            "These summaries are based on nonparametric case resampling over the stored\\n"
            "case-level Stage A outputs. They are not multi-seed reruns of the original\\n"
            "custom trace-retention scripts, which are not preserved in this artifact\\n"
            "snapshot. Use them as uncertainty estimates for the observed case-level\\n"
            "means and policy deltas in the stored results.\\n"
        )


if __name__ == "__main__":
    main()
