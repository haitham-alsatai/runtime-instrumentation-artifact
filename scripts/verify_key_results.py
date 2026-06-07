#!/usr/bin/env python3
"""Verify the principal bundled result values without rerunning experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 5e-4


def read_rows(relative_path: str) -> list[dict[str, str]]:
    path = ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], **conditions: object) -> dict[str, str]:
    for row in rows:
        if all(str(row[key]) == str(value) for key, value in conditions.items()):
            return row
    raise AssertionError(f"No row matched {conditions}")


def check(
    relative_path: str,
    value_column: str,
    expected: float,
    **conditions: object,
) -> None:
    row = find_row(read_rows(relative_path), **conditions)
    actual = float(row[value_column])
    if not math.isclose(actual, expected, abs_tol=TOLERANCE):
        raise AssertionError(
            f"{relative_path}: expected {expected} for {conditions}, got {actual}"
        )
    print(f"PASS {relative_path}: {conditions} -> {actual:.6f}")


def main() -> None:
    dense = "results/stage_a/stage_a_dense_budget/summary_overall.csv"
    check(dense, "avg5_mean", 0.611556, dataset="RE2-OB", budget=0.05)
    check(dense, "avg5_mean", 0.956000, dataset="RE2-TT", budget=1.0)

    policies = "results/stage_a/stage_a_policy_expansion/policy_bootstrap_delta.csv"
    check(
        policies,
        "delta_avg5",
        0.014222,
        dataset="RE2-OB",
        budget=0.1,
        policy="early_window",
    )
    check(
        policies,
        "delta_avg5",
        -0.259556,
        dataset="RE2-TT",
        budget=0.1,
        policy="coverage_aware",
    )

    weak_anchor = "results/stage_b/stage_b_weak_anchor/summary_overall.csv"
    check(
        weak_anchor,
        "avg_at_5",
        0.825618,
        policy="adaptive_corrupted_anchor",
        budget=0.25,
        anchor_corruption=0.0,
    )
    check(
        weak_anchor,
        "avg_at_5",
        0.529213,
        policy="adaptive_corrupted_anchor",
        budget=0.25,
        anchor_corruption=0.5,
    )

    learned = "results/stage_b/gaia_ml_evidence_scorer/summary_overall.csv"
    check(
        learned,
        "avg_at_5",
        0.937528,
        model="random_forest_evidence",
        budget=0.25,
    )
    check(
        learned,
        "avg_at_5",
        0.808764,
        model="random_forest_evidence",
        budget=0.5,
    )

    print("All principal bundled-result checks passed.")


if __name__ == "__main__":
    main()

