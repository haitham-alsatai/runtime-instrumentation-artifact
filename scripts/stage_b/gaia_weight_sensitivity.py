#!/usr/bin/env python3
"""GAIA scoring-weight sensitivity analysis using processed aggregate artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import gaia_ablation_eval as gaia


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "stage_b" / "gaia_weight_sensitivity"


WEIGHT_CONFIGS = [
    {
        "name": "baseline",
        "trace_latency_weight": 2.5,
        "trace_error_weight": 4.0,
        "trace_volume_weight": 0.6,
        "metric_weight": 0.8,
    },
    {
        "name": "trace_heavier_metric_lighter",
        "trace_latency_weight": 3.75,
        "trace_error_weight": 6.0,
        "trace_volume_weight": 0.9,
        "metric_weight": 0.4,
    },
    {
        "name": "trace_lighter_metric_heavier",
        "trace_latency_weight": 1.25,
        "trace_error_weight": 2.0,
        "trace_volume_weight": 0.3,
        "metric_weight": 1.2,
    },
    {
        "name": "error_lighter",
        "trace_latency_weight": 2.5,
        "trace_error_weight": 2.0,
        "trace_volume_weight": 0.6,
        "metric_weight": 0.8,
    },
    {
        "name": "error_heavier",
        "trace_latency_weight": 2.5,
        "trace_error_weight": 6.0,
        "trace_volume_weight": 0.6,
        "metric_weight": 0.8,
    },
]

BUDGETS = [0.1, 0.25, 0.5]
VARIANT = "adaptive_full_context"


def trace_score(obs, base, weights):
    latency_term = weights["trace_latency_weight"] * max(
        0.0, (obs["mean_latency"] - base["mean_latency"]) / max(base["mean_latency"], 1.0)
    )
    error_term = weights["trace_error_weight"] * max(0.0, obs["error_rate"] - base["error_rate"])
    volume_term = weights["trace_volume_weight"] * abs(obs["trace_rows"] - base["trace_rows"]) / max(base["trace_rows"], 1.0)
    return latency_term + error_term + volume_term


def evaluate_case_with_weights(case, budget, weights, trace_endpoint, metric_service, neutral_order):
    service_order = gaia.choose_service_order(case, VARIANT, neutral_order)
    keep_services = max(1, gaia.math.ceil(len(gaia.SERVICE_ORDER) * budget))
    selected_services = service_order[:keep_services]
    selected_minutes = gaia.choose_minutes(case, budget, VARIANT)
    base_minutes = gaia.baseline_minutes(selected_minutes)

    score_map = {}
    for service in gaia.SERVICE_ORDER:
        if service not in selected_services:
            score_map[service] = 0.0
            continue
        endpoints = gaia.pick_endpoints(service, selected_minutes, budget, trace_endpoint)
        obs_trace = gaia.aggregate_trace_selected(service, selected_minutes, endpoints, trace_endpoint)
        base_trace = gaia.aggregate_trace_selected(service, base_minutes, endpoints, trace_endpoint)
        obs_metric_means, _obs_metric_stds, _obs_metric_samples, _obs_metric_bytes = gaia.aggregate_metric(
            service, selected_minutes, metric_service
        )
        base_metric_means, base_metric_stds, _base_metric_samples, _base_metric_bytes = gaia.aggregate_metric(
            service, base_minutes, metric_service
        )
        score_map[service] = trace_score(obs_trace, base_trace, weights) + weights["metric_weight"] * gaia.metric_score(
            obs_metric_means, base_metric_means, base_metric_stds
        )

    ranking = gaia.rank_services(score_map, service_order)
    rank = ranking.index(case.root_service) + 1
    return {
        "case_id": case.case_id,
        "fault_type": case.fault_type,
        "service_family": case.service_family,
        "budget": budget,
        "rank": rank,
        "top1": 1 if rank <= 1 else 0,
        "top3": 1 if rank <= 3 else 0,
        "top5": 1 if rank <= 5 else 0,
        "avg_at_5": sum(1 if rank <= cutoff else 0 for cutoff in range(1, 6)) / 5.0,
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

    case_rows = []
    summary_rows = []

    baseline_map = {}

    for config in WEIGHT_CONFIGS:
        rows = []
        for case in cases:
            for budget in BUDGETS:
                result = evaluate_case_with_weights(
                    case=case,
                    budget=budget,
                    weights=config,
                    trace_endpoint=trace_endpoint,
                    metric_service=metric_service,
                    neutral_order=neutral_order,
                )
                result["config"] = config["name"]
                rows.append(result)
                case_rows.append(result)

        for budget in BUDGETS:
            subset = [row for row in rows if row["budget"] == budget]
            summary = {
                "config": config["name"],
                "budget": budget,
                "top1": mean(row["top1"] for row in subset),
                "top3": mean(row["top3"] for row in subset),
                "top5": mean(row["top5"] for row in subset),
                "avg_at_5": mean(row["avg_at_5"] for row in subset),
                "trace_latency_weight": config["trace_latency_weight"],
                "trace_error_weight": config["trace_error_weight"],
                "trace_volume_weight": config["trace_volume_weight"],
                "metric_weight": config["metric_weight"],
            }
            summary_rows.append(summary)
            if config["name"] == "baseline":
                baseline_map[budget] = summary["avg_at_5"]

    deltas = []
    for row in summary_rows:
        deltas.append(
            {
                **row,
                "delta_vs_baseline": row["avg_at_5"] - baseline_map[row["budget"]],
            }
        )

    manifest = {
        "variant": VARIANT,
        "budgets": BUDGETS,
        "configs": WEIGHT_CONFIGS,
        "notes": [
            "Sensitivity analysis uses the reconstructed GAIA evaluator on processed aggregate CSVs.",
            "The policy variant is the reconstructed adaptive full-context setting only.",
        ],
    }

    write_csv(OUTPUT_DIR / "case_results.csv", case_rows)
    write_csv(OUTPUT_DIR / "summary_overall.csv", summary_rows)
    write_csv(OUTPUT_DIR / "summary_with_deltas.csv", deltas)
    with (OUTPUT_DIR / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
