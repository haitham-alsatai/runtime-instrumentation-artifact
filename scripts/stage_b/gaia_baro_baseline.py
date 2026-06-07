#!/usr/bin/env python3
"""BARO-style full-data baseline on GAIA minute-level aggregate artifacts.

This uses the public BARO scoring idea on the GAIA full-retention windows:
for each service-feature time series, compare the post-alert window against an
equal-length pre-alert baseline using a robust z-score, then rank services by
their strongest feature-level anomaly score.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median

import gaia_ablation_eval as gaia


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "stage_b" / "gaia_baro_baseline"

FEATURES = (
    "cpu",
    "memory",
    "net_in_err",
    "net_out_err",
    "trace_rows",
    "mean_latency_ms",
    "error_rows",
)


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def robust_baro_score(normal_values, anomalous_values):
    med = median(normal_values)
    q1 = percentile(normal_values, 0.25)
    q3 = percentile(normal_values, 0.75)
    iqr = q3 - q1
    scale = iqr if iqr > 1e-6 else 1.0
    zscores = [(value - med) / scale for value in anomalous_values]
    return max(zscores) if zscores else 0.0


def feature_series(service, minutes, trace_service, metric_service):
    series = {feature: [] for feature in FEATURES}
    for minute in minutes:
        trace = trace_service[service].get(
            minute,
            {
                "trace_rows": 0.0,
                "latency_sum": 0.0,
                "max_latency_ms": 0.0,
                "error_rows": 0.0,
                "mean_latency_ms": 0.0,
                "trace_bytes": 0.0,
            },
        )
        metric = metric_service[service].get(
            minute,
            {
                "cpu": 0.0,
                "memory": 0.0,
                "net_in_err": 0.0,
                "net_out_err": 0.0,
                "metric_samples": 0.0,
                "metric_bytes": 0.0,
            },
        )
        series["cpu"].append(metric["cpu"])
        series["memory"].append(metric["memory"])
        series["net_in_err"].append(metric["net_in_err"])
        series["net_out_err"].append(metric["net_out_err"])
        series["trace_rows"].append(trace["trace_rows"])
        series["mean_latency_ms"].append(trace["mean_latency_ms"])
        series["error_rows"].append(trace["error_rows"])
    return series


def rank_services_for_case(case, trace_service, metric_service):
    anomalous_minutes = gaia.minute_range(case.alert_time, case.full_minutes)
    normal_minutes = gaia.baseline_minutes(anomalous_minutes)

    service_scores = {}
    feature_scores = []

    for service in gaia.SERVICE_ORDER:
        normal = feature_series(service, normal_minutes, trace_service, metric_service)
        anomal = feature_series(service, anomalous_minutes, trace_service, metric_service)
        scores = {feature: robust_baro_score(normal[feature], anomal[feature]) for feature in FEATURES}
        service_scores[service] = max(scores.values()) if scores else 0.0
        for feature, score in scores.items():
            feature_scores.append(
                {
                    "case_id": case.case_id,
                    "service": service,
                    "feature": feature,
                    "score": score,
                }
            )

    ranking = sorted(gaia.SERVICE_ORDER, key=lambda s: (-service_scores[s], s))
    return ranking, service_scores, feature_scores


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    trace_service, _neutral = gaia.load_trace_service()
    metric_service = gaia.load_metric_service()
    cases = gaia.load_cases()

    case_rows = []
    feature_rows = []

    for case in cases:
        ranking, service_scores, feature_scores = rank_services_for_case(case, trace_service, metric_service)
        feature_rows.extend(feature_scores)
        rank = ranking.index(case.root_service) + 1
        case_rows.append(
            {
                "case_id": case.case_id,
                "root_service": case.root_service,
                "fault_type": case.fault_type,
                "service_family": case.service_family,
                "rank": rank,
                "top1": 1 if rank <= 1 else 0,
                "top3": 1 if rank <= 3 else 0,
                "top5": 1 if rank <= 5 else 0,
                "avg_at_5": sum(1 if rank <= cutoff else 0 for cutoff in range(1, 6)) / 5.0,
                "top5_services": str(ranking[:5]),
                "root_service_score": service_scores[case.root_service],
            }
        )

    summary_overall = [
        {
            "policy": "baro_full_baseline",
            "cases": len(case_rows),
            "top1": mean(row["top1"] for row in case_rows),
            "top3": mean(row["top3"] for row in case_rows),
            "top5": mean(row["top5"] for row in case_rows),
            "avg_at_5": mean(row["avg_at_5"] for row in case_rows),
        }
    ]

    by_fault = []
    for fault_type in sorted({row["fault_type"] for row in case_rows}):
        subset = [row for row in case_rows if row["fault_type"] == fault_type]
        by_fault.append(
            {
                "fault_type": fault_type,
                "cases": len(subset),
                "top1": mean(row["top1"] for row in subset),
                "top3": mean(row["top3"] for row in subset),
                "top5": mean(row["top5"] for row in subset),
                "avg_at_5": mean(row["avg_at_5"] for row in subset),
            }
        )

    by_family = []
    for family in sorted({row["service_family"] for row in case_rows}):
        subset = [row for row in case_rows if row["service_family"] == family]
        by_family.append(
            {
                "service_family": family,
                "cases": len(subset),
                "top1": mean(row["top1"] for row in subset),
                "top3": mean(row["top3"] for row in subset),
                "top5": mean(row["top5"] for row in subset),
                "avg_at_5": mean(row["avg_at_5"] for row in subset),
            }
        )

    manifest = {
        "method": "BARO-style robust z-score ranking over full GAIA minute-level aggregates",
        "features": list(FEATURES),
        "notes": [
            "Uses equal-length pre-alert and post-alert windows for each case.",
            "Ranks services by the strongest feature-level BARO score across metric and trace aggregates.",
            "Evaluates on the same 890-case GAIA aggregate subset used by the paper's Stage B analysis.",
        ],
    }

    write_csv(OUTPUT_DIR / "case_results.csv", case_rows)
    write_csv(OUTPUT_DIR / "feature_scores.csv", feature_rows)
    write_csv(OUTPUT_DIR / "summary_overall.csv", summary_overall)
    write_csv(OUTPUT_DIR / "summary_by_fault.csv", by_fault)
    write_csv(OUTPUT_DIR / "summary_by_family.csv", by_family)
    with (OUTPUT_DIR / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
