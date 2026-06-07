#!/usr/bin/env python3
"""Lightweight GAIA ablation evaluator rebuilt from processed aggregate CSVs.

This script reconstructs the minimum viable decision-time context ablation
using only the artifacts already present in the workspace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "stage_b" / "gaia_integrated_experiment"
OUTPUT_DIR = ROOT / "results" / "stage_b" / "gaia_ablation_experiment"

TRACE_ENDPOINT_FILE = RESULTS_DIR / "trace_service_minute_endpoint.csv"
TRACE_SERVICE_FILE = RESULTS_DIR / "trace_service_minute.csv"
METRIC_SERVICE_FILE = RESULTS_DIR / "metric_service_minute.csv"
CASE_RESULTS_FILE = RESULTS_DIR / "case_results.csv"
CASE_RESULTS_COST_FILE = RESULTS_DIR / "case_results_with_costs.csv"
STORED_SUMMARY_FILE = RESULTS_DIR / "summary_overall.csv"

SERVICE_ORDER = [
    "webservice1",
    "webservice2",
    "mobservice1",
    "mobservice2",
    "dbservice1",
    "dbservice2",
    "redisservice1",
    "redisservice2",
    "logservice1",
    "logservice2",
]

FAMILY_MAP = {
    "webservice1": "web",
    "webservice2": "web",
    "mobservice1": "mobile",
    "mobservice2": "mobile",
    "dbservice1": "database",
    "dbservice2": "database",
    "redisservice1": "cache",
    "redisservice2": "cache",
    "logservice1": "logging",
    "logservice2": "logging",
}

NEIGHBORS = {
    "webservice1": ["dbservice1", "dbservice2", "redisservice1", "redisservice2", "logservice1", "logservice2", "mobservice1", "mobservice2", "webservice2"],
    "webservice2": ["dbservice1", "dbservice2", "redisservice1", "redisservice2", "logservice1", "logservice2", "mobservice1", "mobservice2", "webservice1"],
    "mobservice1": ["webservice1", "webservice2", "dbservice1", "dbservice2", "redisservice1", "redisservice2", "logservice1", "logservice2", "mobservice2"],
    "mobservice2": ["webservice1", "webservice2", "dbservice1", "dbservice2", "redisservice1", "redisservice2", "logservice1", "logservice2", "mobservice1"],
    "dbservice1": ["webservice1", "webservice2", "redisservice1", "redisservice2", "logservice1", "logservice2", "mobservice1", "mobservice2", "dbservice2"],
    "dbservice2": ["webservice1", "webservice2", "redisservice1", "redisservice2", "logservice1", "logservice2", "mobservice1", "mobservice2", "dbservice1"],
    "redisservice1": ["webservice1", "webservice2", "dbservice1", "dbservice2", "logservice1", "logservice2", "mobservice1", "mobservice2", "redisservice2"],
    "redisservice2": ["webservice1", "webservice2", "dbservice1", "dbservice2", "logservice1", "logservice2", "mobservice1", "mobservice2", "redisservice1"],
    "logservice1": ["webservice1", "webservice2", "dbservice1", "dbservice2", "redisservice1", "redisservice2", "mobservice1", "mobservice2", "logservice2"],
    "logservice2": ["webservice1", "webservice2", "dbservice1", "dbservice2", "redisservice1", "redisservice2", "mobservice1", "mobservice2", "logservice1"],
}

VARIANTS = {
    "adaptive_full_context",
    "adaptive_no_fault_type",
    "adaptive_no_time_window",
    "adaptive_no_alerted_service",
}

RESOURCE_FAULTS = {"memory", "cpu", "file_move"}
INTERACTION_FAULTS = {"errno111", "qr_expired"}


@dataclass(frozen=True)
class Case:
    case_id: str
    root_service: str
    fault_type: str
    service_family: str
    alert_time: datetime
    full_minutes: int
    full_services: int
    full_trace_rows: float
    full_trace_bytes: float
    full_metric_samples: float
    full_metric_bytes: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budgets",
        default="0.1,0.25,0.5",
        help="Comma-separated budgets to evaluate.",
    )
    return parser.parse_args()


def parse_minute(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def load_trace_endpoint():
    data = defaultdict(lambda: defaultdict(dict))
    with TRACE_ENDPOINT_FILE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            service = row["service"]
            minute = parse_minute(row["minute"])
            endpoint = row["endpoint"]
            data[service][minute][endpoint] = {
                "trace_rows": float(row["trace_rows"]),
                "latency_sum": float(row["latency_sum"]),
                "max_latency_ms": float(row["max_latency_ms"]),
                "error_rows": float(row["error_rows"]),
                "mean_latency_ms": float(row["mean_latency_ms"]),
                "trace_bytes": float(row["trace_bytes"]),
            }
    return data


def load_trace_service():
    data = defaultdict(dict)
    totals = defaultdict(float)
    with TRACE_SERVICE_FILE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            service = row["service"]
            minute = parse_minute(row["minute"])
            values = {
                "trace_rows": float(row["trace_rows"]),
                "latency_sum": float(row["latency_sum"]),
                "max_latency_ms": float(row["max_latency_ms"]),
                "error_rows": float(row["error_rows"]),
                "mean_latency_ms": float(row["mean_latency_ms"]),
                "trace_bytes": float(row["trace_bytes"]),
            }
            data[service][minute] = values
            totals[service] += values["trace_rows"]
    neutral_order = sorted(SERVICE_ORDER, key=lambda s: (-totals[s], s))
    return data, neutral_order


def load_metric_service():
    data = defaultdict(dict)
    with METRIC_SERVICE_FILE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            service = row["service"]
            minute = parse_minute(row["minute"])
            data[service][minute] = {
                "cpu": float(row["cpu"]),
                "memory": float(row["memory"]),
                "net_in_err": float(row["net_in_err"]),
                "net_out_err": float(row["net_out_err"]),
                "metric_samples": float(row["metric_samples"]),
                "metric_bytes": float(row["metric_bytes"]),
            }
    return data


def load_cases():
    cost_by_case = {}
    with CASE_RESULTS_COST_FILE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["policy"] == "full" and row["budget"] == "1.0":
                cost_by_case[row["case_id"]] = row

    seen = set()
    cases = []
    with CASE_RESULTS_FILE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] in seen or row["policy"] != "full" or row["budget"] != "1.0":
                continue
            seen.add(row["case_id"])
            timestamp = int(row["case_id"].split("_")[-1])
            alert_time = datetime.fromtimestamp(timestamp, timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
            costs = cost_by_case[row["case_id"]]
            cases.append(
                Case(
                    case_id=row["case_id"],
                    root_service=row["root_service"],
                    fault_type=row["fault_type"],
                    service_family=row["service_family"],
                    alert_time=alert_time,
                    full_minutes=int(float(row["minutes_kept"])),
                    full_services=int(float(row["services_kept"])),
                    full_trace_rows=float(costs["full_trace_rows"] or 0.0),
                    full_trace_bytes=float(costs["full_trace_bytes"] or 0.0),
                    full_metric_samples=float(costs["full_metric_samples"] or 0.0),
                    full_metric_bytes=float(costs["full_metric_bytes"] or 0.0),
                )
            )
    return cases


def minute_range(end_minute: datetime, length: int):
    start = end_minute - timedelta(minutes=length - 1)
    return [start + timedelta(minutes=idx) for idx in range(length)]


def centered_slice(minutes, keep):
    if keep >= len(minutes):
        return list(minutes)
    start = max(0, (len(minutes) - keep) // 2)
    return list(minutes[start : start + keep])


def choose_candidate_window(case: Case, variant: str):
    if variant == "adaptive_no_time_window":
        generic_len = max(case.full_minutes, 11)
        return minute_range(case.alert_time, generic_len)
    return minute_range(case.alert_time, case.full_minutes)


def choose_service_order(case: Case, variant: str, neutral_order):
    if variant == "adaptive_no_alerted_service":
        return list(neutral_order)
    seen = {case.root_service}
    ordered = [case.root_service]
    for service in NEIGHBORS.get(case.root_service, []):
        if service not in seen:
            ordered.append(service)
            seen.add(service)
    for service in neutral_order:
        if service not in seen:
            ordered.append(service)
            seen.add(service)
    return ordered


def choose_minutes(case: Case, budget: float, variant: str):
    candidate = choose_candidate_window(case, variant)
    keep = max(1, math.ceil(len(candidate) * budget))
    if variant == "adaptive_no_fault_type":
        return centered_slice(candidate, keep)
    if case.fault_type in RESOURCE_FAULTS:
        return candidate[:keep]
    return centered_slice(candidate, keep)


def baseline_minutes(selected_minutes):
    if not selected_minutes:
        return []
    first = selected_minutes[0]
    return [first - timedelta(minutes=len(selected_minutes) - idx) for idx in range(len(selected_minutes))]


def pick_endpoints(service, selected_minutes, budget, trace_endpoint):
    endpoint_scores = defaultdict(float)
    for minute in selected_minutes:
        for endpoint, values in trace_endpoint[service].get(minute, {}).items():
            endpoint_scores[endpoint] += values["trace_rows"] * values["mean_latency_ms"]
    if not endpoint_scores:
        return []
    ordered = sorted(endpoint_scores.items(), key=lambda item: (-item[1], item[0]))
    keep = max(1, math.ceil(len(ordered) * budget))
    return [endpoint for endpoint, _ in ordered[:keep]]


def aggregate_trace_selected(service, minutes, endpoints, trace_endpoint):
    total = {
        "trace_rows": 0.0,
        "latency_sum": 0.0,
        "error_rows": 0.0,
        "trace_bytes": 0.0,
    }
    for minute in minutes:
        endpoint_values = trace_endpoint[service].get(minute, {})
        for endpoint in endpoints:
            values = endpoint_values.get(endpoint)
            if not values:
                continue
            total["trace_rows"] += values["trace_rows"]
            total["latency_sum"] += values["latency_sum"]
            total["error_rows"] += values["error_rows"]
            total["trace_bytes"] += values["trace_bytes"]
    if total["trace_rows"] > 0:
        total["mean_latency"] = total["latency_sum"] / total["trace_rows"]
        total["error_rate"] = total["error_rows"] / total["trace_rows"]
    else:
        total["mean_latency"] = 0.0
        total["error_rate"] = 0.0
    return total


def aggregate_metric(service, minutes, metric_service):
    observations = {name: [] for name in ("cpu", "memory", "net_in_err", "net_out_err")}
    total_samples = 0.0
    total_bytes = 0.0
    for minute in minutes:
        values = metric_service[service].get(minute)
        if not values:
            continue
        for name in observations:
            observations[name].append(values[name])
        total_samples += values["metric_samples"]
        total_bytes += values["metric_bytes"]
    means = {name: (mean(values) if values else 0.0) for name, values in observations.items()}
    stds = {
        name: (pstdev(values) if len(values) > 1 else 0.0)
        for name, values in observations.items()
    }
    return means, stds, total_samples, total_bytes


def trace_score(obs, base):
    latency_term = 2.5 * max(0.0, (obs["mean_latency"] - base["mean_latency"]) / max(base["mean_latency"], 1.0))
    error_term = 4.0 * max(0.0, obs["error_rate"] - base["error_rate"])
    volume_term = 0.6 * abs(obs["trace_rows"] - base["trace_rows"]) / max(base["trace_rows"], 1.0)
    return latency_term + error_term + volume_term


def metric_score(obs_means, base_means, base_stds):
    best = 0.0
    for metric in ("cpu", "memory", "net_in_err", "net_out_err"):
        score = abs(obs_means[metric] - base_means[metric]) / max(base_stds[metric], 1e-6)
        best = max(best, score)
    return best


def rank_services(score_map, service_order):
    order_index = {service: idx for idx, service in enumerate(service_order)}
    return sorted(
        SERVICE_ORDER,
        key=lambda service: (-score_map.get(service, 0.0), order_index.get(service, 10_000)),
    )


def evaluate_case(case: Case, budget: float, variant: str, trace_endpoint, metric_service, neutral_order):
    service_order = choose_service_order(case, variant, neutral_order)
    keep_services = max(1, math.ceil(len(SERVICE_ORDER) * budget))
    selected_services = service_order[:keep_services]
    selected_minutes = choose_minutes(case, budget, variant)
    base_minutes = baseline_minutes(selected_minutes)

    score_map = {}
    trace_rows_kept = 0.0
    trace_bytes_kept = 0.0
    metric_samples_kept = 0.0
    metric_bytes_kept = 0.0

    for service in SERVICE_ORDER:
        if service not in selected_services:
            score_map[service] = 0.0
            continue

        endpoints = pick_endpoints(service, selected_minutes, budget, trace_endpoint)
        obs_trace = aggregate_trace_selected(service, selected_minutes, endpoints, trace_endpoint)
        base_trace = aggregate_trace_selected(service, base_minutes, endpoints, trace_endpoint)

        obs_metric_means, _obs_metric_stds, obs_metric_samples, obs_metric_bytes = aggregate_metric(service, selected_minutes, metric_service)
        base_metric_means, base_metric_stds, _base_metric_samples, _base_metric_bytes = aggregate_metric(service, base_minutes, metric_service)

        score = trace_score(obs_trace, base_trace) + 0.8 * metric_score(obs_metric_means, base_metric_means, base_metric_stds)
        score_map[service] = score

        trace_rows_kept += obs_trace["trace_rows"]
        trace_bytes_kept += obs_trace["trace_bytes"]
        metric_samples_kept += obs_metric_samples
        metric_bytes_kept += obs_metric_bytes

    ranking = rank_services(score_map, service_order)
    rank = ranking.index(case.root_service) + 1
    combined_kept = trace_bytes_kept + metric_bytes_kept
    full_combined = case.full_trace_bytes + case.full_metric_bytes

    return {
        "case_id": case.case_id,
        "root_service": case.root_service,
        "fault_type": case.fault_type,
        "service_family": case.service_family,
        "budget": budget,
        "policy": variant,
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
        "full_trace_rows": case.full_trace_rows,
        "full_trace_bytes": case.full_trace_bytes,
        "full_metric_samples": case.full_metric_samples,
        "full_metric_bytes": case.full_metric_bytes,
        "combined_bytes_kept": combined_kept,
        "full_combined_bytes": full_combined,
        "combined_reduction": 0.0 if full_combined == 0 else 1.0 - (combined_kept / full_combined),
        "trace_row_reduction": 0.0 if case.full_trace_rows == 0 else 1.0 - (trace_rows_kept / case.full_trace_rows),
        "metric_sample_reduction": 0.0 if case.full_metric_samples == 0 else 1.0 - (metric_samples_kept / case.full_metric_samples),
        "time_reduction": 0.0 if case.full_minutes == 0 else 1.0 - (len(selected_minutes) / case.full_minutes),
        "where_reduction": 0.0 if case.full_services == 0 else 1.0 - (len(selected_services) / case.full_services),
    }


def summarize(rows, budgets):
    overall = []
    by_fault = []
    by_family = []
    for variant in sorted(VARIANTS):
        for budget in budgets:
            subset = [row for row in rows if row["policy"] == variant and row["budget"] == budget]
            overall.append(
                {
                    "policy": variant,
                    "budget": budget,
                    "top1": mean(row["top1"] for row in subset),
                    "top3": mean(row["top3"] for row in subset),
                    "top5": mean(row["top5"] for row in subset),
                    "avg_at_5": mean(row["avg_at_5"] for row in subset),
                    "combined_reduction": mean(row["combined_reduction"] for row in subset),
                    "trace_row_reduction": mean(row["trace_row_reduction"] for row in subset),
                    "metric_sample_reduction": mean(row["metric_sample_reduction"] for row in subset),
                    "time_reduction": mean(row["time_reduction"] for row in subset),
                    "where_reduction": mean(row["where_reduction"] for row in subset),
                }
            )
    main_budget = min(budgets)
    for grouping, field, target in (
        ("fault", "fault_type", by_fault),
        ("family", "service_family", by_family),
    ):
        keys = sorted(set(row[field] for row in rows))
        for key in keys:
            for variant in sorted(VARIANTS):
                subset = [
                    row
                    for row in rows
                    if row["budget"] == main_budget and row["policy"] == variant and row[field] == key
                ]
                target.append(
                    {
                        field: key,
                        "policy": variant,
                        "avg_at_5": mean(row["avg_at_5"] for row in subset),
                        "top3": mean(row["top3"] for row in subset),
                    }
                )
    return overall, by_fault, by_family


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(cases, budgets):
    manifest = {
        "cases": len(cases),
        "variants": sorted(VARIANTS),
        "budgets": budgets,
        "source_results_dir": str(RESULTS_DIR.relative_to(ROOT)),
        "notes": [
            "Reconstructed from processed GAIA aggregate CSV files.",
            "Case set taken from the stored full-policy GAIA results.",
            "Anomalous window approximated as the trailing full-window length ending at the alert minute.",
        ],
    }
    with (OUTPUT_DIR / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


def compare_to_stored(budgets, summary_rows):
    stored = {}
    with STORED_SUMMARY_FILE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["policy"] == "adaptive_multibudget":
                stored[float(row["budget"])] = float(row["avg_at_5"])
    comparison = []
    for row in summary_rows:
        if row["policy"] != "adaptive_full_context":
            continue
        budget = row["budget"]
        comparison.append(
            {
                "budget": budget,
                "reconstructed_avg_at_5": row["avg_at_5"],
                "stored_adaptive_avg_at_5": stored.get(budget, ""),
                "delta": "" if budget not in stored else row["avg_at_5"] - stored[budget],
            }
        )
    write_csv(OUTPUT_DIR / "validation_vs_stored_adaptive.csv", comparison)


def main():
    args = parse_args()
    budgets = [float(part) for part in args.budgets.split(",") if part.strip()]

    trace_endpoint = load_trace_endpoint()
    _trace_service, neutral_order = load_trace_service()
    metric_service = load_metric_service()
    cases = load_cases()

    rows = []
    for case in cases:
        for budget in budgets:
            for variant in sorted(VARIANTS):
                rows.append(
                    evaluate_case(
                        case=case,
                        budget=budget,
                        variant=variant,
                        trace_endpoint=trace_endpoint,
                        metric_service=metric_service,
                        neutral_order=neutral_order,
                    )
                )

    overall, by_fault, by_family = summarize(rows, budgets)
    write_csv(OUTPUT_DIR / "case_results.csv", rows)
    write_csv(OUTPUT_DIR / "summary_overall.csv", overall)
    write_csv(OUTPUT_DIR / f"summary_fault_budget_{str(min(budgets)).replace('.', '')}.csv", by_fault)
    write_csv(OUTPUT_DIR / f"summary_family_budget_{str(min(budgets)).replace('.', '')}.csv", by_family)
    write_manifest(cases, budgets)
    compare_to_stored(budgets, overall)


if __name__ == "__main__":
    main()
