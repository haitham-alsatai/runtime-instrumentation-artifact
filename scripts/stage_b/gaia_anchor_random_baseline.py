#!/usr/bin/env python3
"""Structured non-random GAIA baseline: anchored service, random remaining choices.

This baseline keeps the service anchor used by the adaptive policy but replaces
the minute and endpoint choices with random selection. It is intended to test
how much of the Stage B gain comes from alerted-service prioritization alone
versus the additional when/what allocation logic.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import mean

import gaia_ablation_eval as gaia


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "stage_b" / "gaia_anchor_random_baseline"

BUDGETS = [0.1, 0.25, 0.5]
POLICY_NAME = "anchored_service_random_rest"
SEED = 7


def rng_for(case_id: str, budget: float, salt: str) -> random.Random:
    return random.Random(f"{SEED}:{case_id}:{budget}:{salt}")


def choose_services(case, budget, neutral_order):
    keep_services = max(1, math.ceil(len(gaia.SERVICE_ORDER) * budget))
    rest = [service for service in neutral_order if service != case.root_service]
    rng = rng_for(case.case_id, budget, "services")
    rng.shuffle(rest)
    ordered = [case.root_service] + rest
    return ordered[:keep_services], ordered


def choose_random_minutes(case, budget):
    candidate = gaia.minute_range(case.alert_time, case.full_minutes)
    keep = max(1, math.ceil(len(candidate) * budget))
    rng = rng_for(case.case_id, budget, "minutes")
    selected = sorted(rng.sample(candidate, keep))
    return selected


def choose_random_endpoints(case, service, selected_minutes, budget, trace_endpoint):
    candidates = sorted(
        {
            endpoint
            for minute in selected_minutes
            for endpoint in trace_endpoint[service].get(minute, {}).keys()
        }
    )
    if not candidates:
        return []
    keep = max(1, math.ceil(len(candidates) * budget))
    rng = rng_for(case.case_id, budget, f"endpoints:{service}")
    if keep >= len(candidates):
        chosen = list(candidates)
    else:
        chosen = sorted(rng.sample(candidates, keep))
    return chosen


def evaluate_case(case, budget, trace_endpoint, metric_service, neutral_order):
    selected_services, service_order = choose_services(case, budget, neutral_order)
    selected_minutes = choose_random_minutes(case, budget)
    base_minutes = gaia.baseline_minutes(selected_minutes)

    score_map = {}
    trace_rows_kept = 0.0
    trace_bytes_kept = 0.0
    metric_samples_kept = 0.0
    metric_bytes_kept = 0.0

    for service in gaia.SERVICE_ORDER:
        if service not in selected_services:
            score_map[service] = 0.0
            continue

        endpoints = choose_random_endpoints(case, service, selected_minutes, budget, trace_endpoint)
        obs_trace = gaia.aggregate_trace_selected(service, selected_minutes, endpoints, trace_endpoint)
        base_trace = gaia.aggregate_trace_selected(service, base_minutes, endpoints, trace_endpoint)
        obs_metric_means, _obs_metric_stds, obs_metric_samples, obs_metric_bytes = gaia.aggregate_metric(
            service, selected_minutes, metric_service
        )
        base_metric_means, base_metric_stds, _base_metric_samples, _base_metric_bytes = gaia.aggregate_metric(
            service, base_minutes, metric_service
        )

        score = gaia.trace_score(obs_trace, base_trace) + 0.8 * gaia.metric_score(
            obs_metric_means, base_metric_means, base_metric_stds
        )
        score_map[service] = score

        trace_rows_kept += obs_trace["trace_rows"]
        trace_bytes_kept += obs_trace["trace_bytes"]
        metric_samples_kept += obs_metric_samples
        metric_bytes_kept += obs_metric_bytes

    ranking = gaia.rank_services(score_map, service_order)
    rank = ranking.index(case.root_service) + 1
    combined_kept = trace_bytes_kept + metric_bytes_kept
    full_combined = case.full_trace_bytes + case.full_metric_bytes

    return {
        "case_id": case.case_id,
        "root_service": case.root_service,
        "fault_type": case.fault_type,
        "service_family": case.service_family,
        "budget": budget,
        "policy": POLICY_NAME,
        "rank": rank,
        "top1": 1 if rank <= 1 else 0,
        "top3": 1 if rank <= 3 else 0,
        "top5": 1 if rank <= 5 else 0,
        "avg_at_5": sum(1 if rank <= cutoff else 0 for cutoff in range(1, 6)) / 5.0,
        "trace_rows_kept": trace_rows_kept,
        "trace_bytes_kept": trace_bytes_kept,
        "metric_samples_kept": metric_samples_kept,
        "metric_bytes_kept": metric_bytes_kept,
        "minutes_kept": len(selected_minutes),
        "services_kept": len(selected_services),
        "combined_reduction": 0.0 if full_combined == 0 else 1.0 - (combined_kept / full_combined),
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    trace_endpoint = gaia.load_trace_endpoint()
    _trace_service, neutral_order = gaia.load_trace_service()
    metric_service = gaia.load_metric_service()
    cases = gaia.load_cases()

    rows = []
    for case in cases:
        for budget in BUDGETS:
            rows.append(evaluate_case(case, budget, trace_endpoint, metric_service, neutral_order))

    overall = []
    for budget in BUDGETS:
        subset = [row for row in rows if row["budget"] == budget]
        overall.append(
            {
                "policy": POLICY_NAME,
                "budget": budget,
                "top1": mean(row["top1"] for row in subset),
                "top3": mean(row["top3"] for row in subset),
                "top5": mean(row["top5"] for row in subset),
                "avg_at_5": mean(row["avg_at_5"] for row in subset),
                "combined_reduction": mean(row["combined_reduction"] for row in subset),
            }
        )

    by_fault_25 = []
    for fault in sorted(set(row["fault_type"] for row in rows)):
        subset = [row for row in rows if row["budget"] == 0.25 and row["fault_type"] == fault]
        by_fault_25.append(
            {
                "fault_type": fault,
                "budget": 0.25,
                "avg_at_5": mean(row["avg_at_5"] for row in subset),
                "top1": mean(row["top1"] for row in subset),
            }
        )

    manifest = {
        "policy": POLICY_NAME,
        "budgets": BUDGETS,
        "seed": SEED,
        "notes": [
            "Service selection is anchored on the root_service field, with the remaining service order randomized.",
            "Minute and endpoint choices are random within the retained budget.",
        ],
    }

    write_csv(OUTPUT_DIR / "case_results.csv", rows)
    write_csv(OUTPUT_DIR / "summary_overall.csv", overall)
    write_csv(OUTPUT_DIR / "summary_fault_budget_025.csv", by_fault_25)
    with (OUTPUT_DIR / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
