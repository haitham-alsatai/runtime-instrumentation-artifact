#!/usr/bin/env python3
"""MRCA-inspired full-data baseline on GAIA minute-level aggregate artifacts.

This is an adaptation, not a native reproduction of ASE 2024 MRCA.
We only have service-minute trace and metric aggregates locally, so we emulate:
1. abnormal-service ranking from multi-signal anomaly evidence
2. pruning by anomaly timing / precedence among top abnormal services
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import gaia_ablation_eval as gaia


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "stage_b" / "gaia_mrca_baseline"

FEATURES = (
    "cpu",
    "memory",
    "net_in_err",
    "net_out_err",
    "trace_rows",
    "mean_latency_ms",
    "error_rows",
)
TOP_K_SERVICES = 5
ANOMALY_THRESHOLD = 3.0


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def robust_stats(values):
    med = median(values)
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    return med, iqr if iqr > 1e-6 else 1.0


def service_series(service, minutes, trace_service, metric_service):
    rows = []
    for minute in minutes:
        trace = trace_service[service].get(
            minute,
            {
                "trace_rows": 0.0,
                "mean_latency_ms": 0.0,
                "error_rows": 0.0,
            },
        )
        metric = metric_service[service].get(
            minute,
            {
                "cpu": 0.0,
                "memory": 0.0,
                "net_in_err": 0.0,
                "net_out_err": 0.0,
            },
        )
        rows.append(
            {
                "cpu": metric["cpu"],
                "memory": metric["memory"],
                "net_in_err": metric["net_in_err"],
                "net_out_err": metric["net_out_err"],
                "trace_rows": trace["trace_rows"],
                "mean_latency_ms": trace["mean_latency_ms"],
                "error_rows": trace["error_rows"],
            }
        )
    return rows


def feature_anomaly_profile(normal_rows, anomal_rows):
    feature_scores = {}
    feature_first_hit = {}
    for feature in FEATURES:
        normal_values = [row[feature] for row in normal_rows]
        anomal_values = [row[feature] for row in anomal_rows]
        med, scale = robust_stats(normal_values)
        zscores = [(value - med) / scale for value in anomal_values]
        feature_scores[feature] = max(zscores) if zscores else 0.0
        hit_index = None
        for idx, zscore in enumerate(zscores):
            if zscore >= ANOMALY_THRESHOLD:
                hit_index = idx
                break
        feature_first_hit[feature] = hit_index
    return feature_scores, feature_first_hit


def anomaly_probability(service_scores):
    # squash strongest feature evidence into a bounded abnormal-service score
    strongest = max(service_scores.values()) if service_scores else 0.0
    return 1.0 / (1.0 + math.exp(-strongest))


def root_likeness(top_services, service_prob, first_hits):
    root_score = {service: service_prob[service] for service in top_services}
    for src in top_services:
        src_hits = [hit for hit in first_hits[src].values() if hit is not None]
        if not src_hits:
            continue
        src_first = min(src_hits)
        for dst in top_services:
            if src == dst:
                continue
            dst_hits = [hit for hit in first_hits[dst].values() if hit is not None]
            if not dst_hits:
                continue
            dst_first = min(dst_hits)
            if src_first < dst_first:
                root_score[src] += 0.10
                root_score[dst] -= 0.05
    return root_score


def evaluate_case(case, trace_service, metric_service):
    anomalous_minutes = gaia.minute_range(case.alert_time, case.full_minutes)
    normal_minutes = gaia.baseline_minutes(anomalous_minutes)

    service_feature_scores = {}
    first_hits = {}
    service_prob = {}
    feature_rows = []

    for service in gaia.SERVICE_ORDER:
        normal_rows = service_series(service, normal_minutes, trace_service, metric_service)
        anomal_rows = service_series(service, anomalous_minutes, trace_service, metric_service)
        scores, hits = feature_anomaly_profile(normal_rows, anomal_rows)
        service_feature_scores[service] = scores
        first_hits[service] = hits
        service_prob[service] = anomaly_probability(scores)
        for feature in FEATURES:
            feature_rows.append(
                {
                    "case_id": case.case_id,
                    "service": service,
                    "feature": feature,
                    "score": scores[feature],
                    "first_hit_index": "" if hits[feature] is None else hits[feature],
                }
            )

    top_services = sorted(gaia.SERVICE_ORDER, key=lambda s: (-service_prob[s], s))[:TOP_K_SERVICES]
    root_scores = root_likeness(top_services, service_prob, first_hits)

    final_ranking = sorted(
        gaia.SERVICE_ORDER,
        key=lambda s: (-(root_scores[s] if s in root_scores else service_prob[s]), -service_prob[s], s),
    )
    rank = final_ranking.index(case.root_service) + 1
    return {
        "case_id": case.case_id,
        "root_service": case.root_service,
        "fault_type": case.fault_type,
        "service_family": case.service_family,
        "rank": rank,
        "top1": 1 if rank <= 1 else 0,
        "top3": 1 if rank <= 3 else 0,
        "top5": 1 if rank <= 5 else 0,
        "avg_at_5": sum(1 if rank <= cutoff else 0 for cutoff in range(1, 6)) / 5.0,
        "top5_services": str(final_ranking[:5]),
        "root_service_prob": service_prob[case.root_service],
    }, feature_rows


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
        row, feature_part = evaluate_case(case, trace_service, metric_service)
        case_rows.append(row)
        feature_rows.extend(feature_part)

    summary_overall = [
        {
            "policy": "mrca_style_full_baseline",
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
        "method": "MRCA-inspired aggregate baseline",
        "features": list(FEATURES),
        "top_k_services_for_pruning": TOP_K_SERVICES,
        "anomaly_threshold": ANOMALY_THRESHOLD,
        "notes": [
            "Adaptation only: uses GAIA service-minute trace and metric aggregates, not the full MRCA trace-log-metric stack.",
            "Approximates abnormal-service ranking with multi-signal anomaly scores and anomaly-order pruning among top abnormal services.",
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
