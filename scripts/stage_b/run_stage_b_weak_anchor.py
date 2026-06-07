#!/usr/bin/env python3
"""RV 2026 Stage B weak-anchor robustness experiment.

This script evaluates how GAIA post-alert telemetry allocation degrades when the
service anchor used at decision time is corrupted. It reads the existing GAIA
aggregate artifacts and writes only to the RV extension workspace.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean


EXT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = EXT_ROOT / "results" / "stage_b" / "gaia_integrated_experiment"
OUTPUT_DIR = EXT_ROOT / "results" / "stage_b" / "stage_b_weak_anchor"
FIGURE_DIR = EXT_ROOT / "figures"
TABLE_DIR = EXT_ROOT / "tables"

TRACE_ENDPOINT_FILE = INPUT_DIR / "trace_service_minute_endpoint.csv"
TRACE_SERVICE_FILE = INPUT_DIR / "trace_service_minute.csv"
METRIC_SERVICE_FILE = INPUT_DIR / "metric_service_minute.csv"
CASE_RESULTS_FILE = INPUT_DIR / "case_results.csv"
CASE_RESULTS_COST_FILE = INPUT_DIR / "case_results_with_costs.csv"

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

RESOURCE_FAULTS = {"memory", "cpu", "file_move"}
BUDGETS = [0.25, 0.50]
CORRUPTION_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
REPETITIONS = 5
BASE_SEED = 20260602


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


def parse_minute(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def minute_range(end_minute: datetime, length: int) -> list[datetime]:
    start = end_minute - timedelta(minutes=length - 1)
    return [start + timedelta(minutes=idx) for idx in range(length)]


def baseline_minutes(selected_minutes: list[datetime]) -> list[datetime]:
    if not selected_minutes:
        return []
    first = selected_minutes[0]
    return [first - timedelta(minutes=len(selected_minutes) - idx) for idx in range(len(selected_minutes))]


def centered_slice(minutes: list[datetime], keep: int) -> list[datetime]:
    if keep >= len(minutes):
        return list(minutes)
    start = max(0, (len(minutes) - keep) // 2)
    return list(minutes[start : start + keep])


def rng_for(*parts: object) -> random.Random:
    return random.Random(":".join(str(part) for part in (BASE_SEED, *parts)))


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
                "error_rows": float(row["error_rows"]),
                "mean_latency_ms": float(row["mean_latency_ms"]),
                "trace_bytes": float(row["trace_bytes"]),
            }
    return data


def load_trace_service():
    totals = defaultdict(float)
    with TRACE_SERVICE_FILE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            service = row["service"]
            totals[service] += float(row["trace_rows"])
    return sorted(SERVICE_ORDER, key=lambda service: (-totals[service], service))


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


def load_cases() -> list[Case]:
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


def load_full_baseline_rows() -> list[dict]:
    rows = []
    seen = set()
    with CASE_RESULTS_FILE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] in seen or row["policy"] != "full" or row["budget"] != "1.0":
                continue
            seen.add(row["case_id"])
            rank = int(float(row["rank"]))
            rows.append(
                {
                    "case_id": row["case_id"],
                    "root_service": row["root_service"],
                    "fault_type": row["fault_type"],
                    "service_family": row["service_family"],
                    "rank": rank,
                    "top1": 1 if rank <= 1 else 0,
                    "top3": 1 if rank <= 3 else 0,
                    "top5": 1 if rank <= 5 else 0,
                    "avg_at_5": sum(1 if rank <= cutoff else 0 for cutoff in range(1, 6)) / 5.0,
                    "combined_reduction": 0.0,
                    "trace_row_reduction": 0.0,
                    "metric_sample_reduction": 0.0,
                    "anchor_corrupted": 0,
                    "decision_anchor": row["root_service"],
                }
            )
    return rows


def choose_decision_anchor(case: Case, corruption: float, repetition: int) -> tuple[str, int]:
    rng = rng_for("anchor", case.case_id, corruption, repetition)
    if rng.random() >= corruption:
        return case.root_service, 0
    candidates = [service for service in SERVICE_ORDER if service != case.root_service]
    return rng.choice(candidates), 1


def service_order_from_anchor(anchor: str, neutral_order: list[str]) -> list[str]:
    seen = {anchor}
    ordered = [anchor]
    for service in NEIGHBORS.get(anchor, []):
        if service not in seen:
            ordered.append(service)
            seen.add(service)
    for service in neutral_order:
        if service not in seen:
            ordered.append(service)
            seen.add(service)
    return ordered


def adaptive_minutes(case: Case, budget: float) -> list[datetime]:
    candidate = minute_range(case.alert_time, case.full_minutes)
    keep = max(1, math.ceil(len(candidate) * budget))
    if case.fault_type in RESOURCE_FAULTS:
        return candidate[:keep]
    return centered_slice(candidate, keep)


def random_minutes(case: Case, budget: float, policy: str, repetition: int) -> list[datetime]:
    candidate = minute_range(case.alert_time, case.full_minutes)
    keep = max(1, math.ceil(len(candidate) * budget))
    rng = rng_for(policy, "minutes", case.case_id, budget, repetition)
    if keep >= len(candidate):
        return list(candidate)
    return sorted(rng.sample(candidate, keep))


def adaptive_endpoints(service: str, selected_minutes: list[datetime], budget: float, trace_endpoint) -> list[str]:
    endpoint_scores = defaultdict(float)
    for minute in selected_minutes:
        for endpoint, values in trace_endpoint[service].get(minute, {}).items():
            endpoint_scores[endpoint] += values["trace_rows"] * values["mean_latency_ms"]
    if not endpoint_scores:
        return []
    ordered = sorted(endpoint_scores.items(), key=lambda item: (-item[1], item[0]))
    keep = max(1, math.ceil(len(ordered) * budget))
    return [endpoint for endpoint, _ in ordered[:keep]]


def random_endpoints(case: Case, service: str, selected_minutes: list[datetime], budget: float, trace_endpoint, policy: str, repetition: int) -> list[str]:
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
    if keep >= len(candidates):
        return list(candidates)
    rng = rng_for(policy, "endpoints", case.case_id, service, budget, repetition)
    return sorted(rng.sample(candidates, keep))


def aggregate_trace_selected(service: str, minutes: list[datetime], endpoints: list[str], trace_endpoint) -> dict:
    total = {"trace_rows": 0.0, "latency_sum": 0.0, "error_rows": 0.0, "trace_bytes": 0.0}
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


def aggregate_metric(service: str, minutes: list[datetime], metric_service) -> tuple[dict, dict, float, float]:
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
    stds = {}
    for name, values in observations.items():
        if len(values) <= 1:
            stds[name] = 0.0
        else:
            mu = means[name]
            stds[name] = math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))
    return means, stds, total_samples, total_bytes


def trace_score(obs: dict, base: dict) -> float:
    latency_term = 2.5 * max(0.0, (obs["mean_latency"] - base["mean_latency"]) / max(base["mean_latency"], 1.0))
    error_term = 4.0 * max(0.0, obs["error_rate"] - base["error_rate"])
    volume_term = 0.6 * abs(obs["trace_rows"] - base["trace_rows"]) / max(base["trace_rows"], 1.0)
    return latency_term + error_term + volume_term


def metric_score(obs_means: dict, base_means: dict, base_stds: dict) -> float:
    best = 0.0
    for metric in ("cpu", "memory", "net_in_err", "net_out_err"):
        score = abs(obs_means[metric] - base_means[metric]) / max(base_stds[metric], 1e-6)
        best = max(best, score)
    return best


def rank_services(score_map: dict[str, float], service_order: list[str]) -> list[str]:
    order_index = {service: idx for idx, service in enumerate(service_order)}
    return sorted(SERVICE_ORDER, key=lambda service: (-score_map.get(service, 0.0), order_index.get(service, 10_000)))


def selected_services_for_policy(case: Case, budget: float, policy: str, decision_anchor: str, neutral_order: list[str], repetition: int) -> tuple[list[str], list[str]]:
    keep = max(1, math.ceil(len(SERVICE_ORDER) * budget))
    if policy == "random_multibudget":
        rng = rng_for(policy, "services", case.case_id, budget, repetition)
        ordered = list(neutral_order)
        rng.shuffle(ordered)
        return ordered[:keep], ordered
    if policy == "anchored_random_rest":
        rest = [service for service in neutral_order if service != decision_anchor]
        rng = rng_for(policy, "services", case.case_id, budget, repetition)
        rng.shuffle(rest)
        ordered = [decision_anchor] + rest
        return ordered[:keep], ordered
    ordered = service_order_from_anchor(decision_anchor, neutral_order)
    return ordered[:keep], ordered


def evaluate_case(case: Case, budget: float, policy: str, corruption: float, repetition: int, trace_endpoint, metric_service, neutral_order: list[str]) -> dict:
    decision_anchor, anchor_corrupted = choose_decision_anchor(case, corruption, repetition)
    selected_services, service_order = selected_services_for_policy(case, budget, policy, decision_anchor, neutral_order, repetition)
    if policy == "adaptive_corrupted_anchor":
        selected_minutes = adaptive_minutes(case, budget)
    else:
        selected_minutes = random_minutes(case, budget, policy, repetition)
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
        if policy == "adaptive_corrupted_anchor":
            endpoints = adaptive_endpoints(service, selected_minutes, budget, trace_endpoint)
        else:
            endpoints = random_endpoints(case, service, selected_minutes, budget, trace_endpoint, policy, repetition)

        obs_trace = aggregate_trace_selected(service, selected_minutes, endpoints, trace_endpoint)
        base_trace = aggregate_trace_selected(service, base_minutes, endpoints, trace_endpoint)
        obs_metric_means, _obs_metric_stds, obs_metric_samples, obs_metric_bytes = aggregate_metric(service, selected_minutes, metric_service)
        base_metric_means, base_metric_stds, _base_metric_samples, _base_metric_bytes = aggregate_metric(service, base_minutes, metric_service)

        score_map[service] = trace_score(obs_trace, base_trace) + 0.8 * metric_score(obs_metric_means, base_metric_means, base_metric_stds)
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
        "decision_anchor": decision_anchor,
        "anchor_corrupted": anchor_corrupted,
        "fault_type": case.fault_type,
        "service_family": case.service_family,
        "budget": budget,
        "anchor_corruption": corruption,
        "repetition": repetition,
        "policy": policy,
        "rank": rank,
        "top1": 1 if rank <= 1 else 0,
        "top3": 1 if rank <= 3 else 0,
        "top5": 1 if rank <= 5 else 0,
        "avg_at_5": sum(1 if rank <= cutoff else 0 for cutoff in range(1, 6)) / 5.0,
        "combined_reduction": 0.0 if full_combined == 0 else 1.0 - (combined_kept / full_combined),
        "trace_row_reduction": 0.0 if case.full_trace_rows == 0 else 1.0 - (trace_rows_kept / case.full_trace_rows),
        "metric_sample_reduction": 0.0 if case.full_metric_samples == 0 else 1.0 - (metric_samples_kept / case.full_metric_samples),
        "services_kept": len(selected_services),
        "minutes_kept": len(selected_minutes),
    }


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    keys = sorted({(row["policy"], row["budget"], row["anchor_corruption"]) for row in rows})
    for policy, budget, corruption in keys:
        subset = [row for row in rows if row["policy"] == policy and row["budget"] == budget and row["anchor_corruption"] == corruption]
        summary.append(
            {
                "policy": policy,
                "budget": budget,
                "anchor_corruption": corruption,
                "rows": len(subset),
                "cases_per_repetition": len({row["case_id"] for row in subset}),
                "repetitions": len({row["repetition"] for row in subset}),
                "actual_anchor_corruption": mean(row["anchor_corrupted"] for row in subset),
                "top1": mean(row["top1"] for row in subset),
                "top3": mean(row["top3"] for row in subset),
                "top5": mean(row["top5"] for row in subset),
                "avg_at_5": mean(row["avg_at_5"] for row in subset),
                "combined_reduction": mean(row["combined_reduction"] for row in subset),
                "trace_row_reduction": mean(row["trace_row_reduction"] for row in subset),
                "metric_sample_reduction": mean(row["metric_sample_reduction"] for row in subset),
            }
        )
    return summary


def summarize_by_repetition(rows: list[dict]) -> list[dict]:
    summary = []
    keys = sorted({(row["policy"], row["budget"], row["anchor_corruption"], row["repetition"]) for row in rows})
    for policy, budget, corruption, repetition in keys:
        subset = [
            row
            for row in rows
            if row["policy"] == policy
            and row["budget"] == budget
            and row["anchor_corruption"] == corruption
            and row["repetition"] == repetition
        ]
        summary.append(
            {
                "policy": policy,
                "budget": budget,
                "anchor_corruption": corruption,
                "repetition": repetition,
                "actual_anchor_corruption": mean(row["anchor_corrupted"] for row in subset),
                "avg_at_5": mean(row["avg_at_5"] for row in subset),
                "top1": mean(row["top1"] for row in subset),
                "top3": mean(row["top3"] for row in subset),
                "top5": mean(row["top5"] for row in subset),
            }
        )
    return summary


def paired_policy_deltas(rows: list[dict]) -> list[dict]:
    by_key = {
        (row["case_id"], row["budget"], row["anchor_corruption"], row["repetition"], row["policy"]): row["avg_at_5"]
        for row in rows
    }
    deltas = []
    for budget in BUDGETS:
        for corruption in CORRUPTION_LEVELS:
            for repetition in range(REPETITIONS):
                values = []
                wins = losses = ties = 0
                for case_id in sorted({row["case_id"] for row in rows if row["budget"] == budget}):
                    adaptive = by_key.get((case_id, budget, corruption, repetition, "adaptive_corrupted_anchor"))
                    anchored = by_key.get((case_id, budget, corruption, repetition, "anchored_random_rest"))
                    if adaptive is None or anchored is None:
                        continue
                    delta = adaptive - anchored
                    values.append(delta)
                    if delta > 0:
                        wins += 1
                    elif delta < 0:
                        losses += 1
                    else:
                        ties += 1
                deltas.append(
                    {
                        "budget": budget,
                        "anchor_corruption": corruption,
                        "repetition": repetition,
                        "comparison": "adaptive_minus_anchored_random",
                        "mean_delta_avg_at_5": mean(values) if values else 0.0,
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                    }
                )
    return deltas


def add_full_reference(full_rows: list[dict]) -> list[dict]:
    rows = []
    for budget in BUDGETS:
        for corruption in CORRUPTION_LEVELS:
            for repetition in range(REPETITIONS):
                for row in full_rows:
                    rows.append(
                        {
                            **row,
                            "budget": budget,
                            "anchor_corruption": corruption,
                            "repetition": repetition,
                            "policy": "full_reference",
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [
        row
        for row in rows
        if row["policy"] in {"adaptive_corrupted_anchor", "anchored_random_rest", "random_multibudget", "full_reference"}
    ]
    lines = [
        "| policy | budget | corruption | Avg@5 | Top-1 | Top-3 | actual corruption |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['policy']} | {row['budget']:.2f} | {row['anchor_corruption']:.2f} | "
            f"{row['avg_at_5']:.3f} | {row['top1']:.3f} | {row['top3']:.3f} | {row['actual_anchor_corruption']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(cases: list[Case]) -> None:
    manifest = {
        "experiment": "stage_b_weak_anchor_rv2026",
        "created_date": "2026-06-02",
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "budgets": BUDGETS,
        "anchor_corruption_levels": CORRUPTION_LEVELS,
        "repetitions": REPETITIONS,
        "base_seed": BASE_SEED,
        "cases": len(cases),
        "policies": [
            "adaptive_corrupted_anchor",
            "anchored_random_rest",
            "random_multibudget",
            "full_reference",
        ],
        "decision_time_safety": [
            "Corruption changes only the service anchor used for allocation.",
            "Evaluation always uses the true root_service label.",
            "No withheld ranking label is used when selecting services, minutes, or endpoints.",
            "Full_reference is a constant baseline copied from stored full-policy rankings.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GAIA aggregate inputs...")
    trace_endpoint = load_trace_endpoint()
    neutral_order = load_trace_service()
    metric_service = load_metric_service()
    cases = load_cases()
    full_rows = load_full_baseline_rows()

    print(f"Cases: {len(cases)}")
    rows = []
    policies = ["adaptive_corrupted_anchor", "anchored_random_rest", "random_multibudget"]
    for corruption in CORRUPTION_LEVELS:
        print(f"Evaluating corruption={corruption:.2f}")
        for repetition in range(REPETITIONS):
            for budget in BUDGETS:
                for case in cases:
                    for policy in policies:
                        rows.append(
                            evaluate_case(
                                case=case,
                                budget=budget,
                                policy=policy,
                                corruption=corruption,
                                repetition=repetition,
                                trace_endpoint=trace_endpoint,
                                metric_service=metric_service,
                                neutral_order=neutral_order,
                            )
                        )

    rows_with_full = rows + add_full_reference(full_rows)
    summary = summarize(rows_with_full)
    by_rep = summarize_by_repetition(rows_with_full)
    deltas = paired_policy_deltas(rows)

    write_csv(OUTPUT_DIR / "case_results.csv", rows_with_full)
    write_csv(OUTPUT_DIR / "summary_overall.csv", summary)
    write_csv(OUTPUT_DIR / "summary_by_repetition.csv", by_rep)
    write_csv(OUTPUT_DIR / "adaptive_vs_anchored_deltas.csv", deltas)
    write_markdown_table(TABLE_DIR / "stage_b_weak_anchor_summary.md", summary)
    write_manifest(cases)

    print("Completed Stage B weak-anchor experiment.")
    print(f"Wrote: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
