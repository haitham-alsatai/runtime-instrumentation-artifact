from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
GAIA_DATA = ROOT / "data" / "GAIA"
TRACE_DIR = GAIA_DATA / "GAIA_extracted" / "trace" / "trace"
METRIC_DIR = GAIA_DATA / "GAIA_extracted" / "metric" / "metric"
RUN_FILE = GAIA_DATA / "gaia_run" / "run" / "run_table_2021-07.csv"
OUT_DIR = ROOT / "results" / "stage_b" / "gaia_integrated_experiment"

RANDOM_SEED = 7
TRACE_CHUNK = 200_000
SERVICE_BUDGETS = [0.10, 0.25, 0.50, 1.00]
TRACE_COST_COLUMNS = ["trace_rows", "trace_bytes"]
METRIC_COST_COLUMNS = ["metric_samples", "metric_bytes"]
ALL_SERVICES = [
    "dbservice1",
    "dbservice2",
    "logservice1",
    "logservice2",
    "mobservice1",
    "mobservice2",
    "redisservice1",
    "redisservice2",
    "webservice1",
    "webservice2",
]
ENDPOINT_ALLOWLIST = {
    "dbservice1": {"/db_login_methods"},
    "dbservice2": {"/db_login_methods"},
    "logservice1": {"/login_model_implement", "/login_query_redis_info"},
    "logservice2": {"/login_model_implement", "/login_query_redis_info"},
    "mobservice1": {"/mob_info_to_redis"},
    "mobservice2": {"/mob_info_to_redis"},
    "redisservice1": {
        "/get_value_from_redis",
        "/set_key_value_into_redis",
        "/keys_existence_check",
    },
    "redisservice2": {
        "/get_value_from_redis",
        "/set_key_value_into_redis",
        "/keys_existence_check",
    },
    "webservice1": {"/web_login_service"},
    "webservice2": {"/web_login_service"},
}
SERVICE_NEIGHBORS = {
    "mobservice1": ["mobservice1", "webservice1", "redisservice1"],
    "mobservice2": ["mobservice2", "webservice2", "redisservice2"],
    "webservice1": ["webservice1", "dbservice1", "logservice1"],
    "webservice2": ["webservice2", "dbservice2", "logservice2"],
    "dbservice1": ["dbservice1", "logservice1", "webservice1"],
    "dbservice2": ["dbservice2", "logservice2", "webservice2"],
    "redisservice1": ["redisservice1", "mobservice1", "webservice1"],
    "redisservice2": ["redisservice2", "mobservice2", "webservice2"],
    "logservice1": ["logservice1", "dbservice1", "webservice1"],
    "logservice2": ["logservice2", "dbservice2", "webservice2"],
}
METRIC_SPECS = {
    "cpu": "docker_cpu_total_pct",
    "memory": "docker_memory_usage_pct",
    "net_in_err": "docker_network_in_errors",
    "net_out_err": "docker_network_out_errors",
}


@dataclass
class Case:
    case_id: str
    service: str
    fault_type: str
    log_time: pd.Timestamp
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    service_family: str


def ensure_dirs() -> None:
    OUT_DIR.mkdir(exist_ok=True)


def service_family(service: str) -> str:
    if service.startswith("mob"):
        return "mobile"
    if service.startswith("web"):
        return "web"
    if service.startswith("db"):
        return "database"
    if service.startswith("redis"):
        return "cache"
    return "logging"


def parse_message_time(message: str) -> Optional[pd.Timestamp]:
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", str(message))
    if not m:
        return None
    return pd.to_datetime(m.group(1))


def parse_case_row(row: pd.Series) -> Optional[Case]:
    message = str(row["message"])
    service = str(row["service"])
    log_ts = parse_message_time(message)
    if log_ts is None:
        return None

    fault_type = None
    start_ts = None
    end_ts = None

    if "[memory_anomalies]" in message:
        fault_type = "memory"
        m = re.search(r"start at ([\d\-: .]+) and lasts ([\d.]+) seconds", message)
        if m:
            start_ts = pd.to_datetime(m.group(1))
            duration = min(float(m.group(2)), 600.0)
            end_ts = start_ts + pd.Timedelta(seconds=duration)
    elif "[cpu_anomalies]" in message:
        fault_type = "cpu"
        m = re.search(r"start at ([\d\-: .]+) and lasts ([\d.]+) seconds", message)
        if m:
            start_ts = pd.to_datetime(m.group(1))
            duration = min(float(m.group(2)), 600.0)
            end_ts = start_ts + pd.Timedelta(seconds=duration)
    elif "qr code expired" in message.lower():
        fault_type = "qr_expired"
        start_ts = log_ts
        end_ts = log_ts + pd.Timedelta(seconds=30)
    elif "file moving program" in message.lower():
        fault_type = "file_move"
        m = re.search(r"start with ([\d\-: .]+), last for ([\d.]+) seconds", message)
        if m:
            start_ts = pd.to_datetime(m.group(1))
            duration = min(float(m.group(2)), 600.0)
            end_ts = start_ts + pd.Timedelta(seconds=duration)
    elif "Errno 111" in message:
        fault_type = "errno111"
        start_ts = log_ts
        end_ts = log_ts + pd.Timedelta(minutes=5)

    if fault_type is None or start_ts is None or end_ts is None:
        return None

    return Case(
        case_id=f"{service}_{fault_type}_{int(log_ts.timestamp())}",
        service=service,
        fault_type=fault_type,
        log_time=log_ts,
        start_time=start_ts.floor("min"),
        end_time=end_ts.ceil("min"),
        service_family=service_family(service),
    )


def load_cases() -> List[Case]:
    df = pd.read_csv(RUN_FILE)
    cases = [parse_case_row(row) for _, row in df.iterrows()]
    cases = [c for c in cases if c is not None]

    grouped: Dict[str, List[Case]] = defaultdict(list)
    for case in cases:
        grouped[case.fault_type].append(case)

    rng = random.Random(RANDOM_SEED)
    selected: List[Case] = []
    for fault_type, fault_cases in grouped.items():
        fault_cases = sorted(fault_cases, key=lambda c: c.log_time)
        if fault_type == "qr_expired":
            # Downsample the very dense QR-expired class while preserving both mobile services.
            per_service: Dict[str, List[Case]] = defaultdict(list)
            for case in fault_cases:
                per_service[case.service].append(case)
            for service_cases in per_service.values():
                step = max(1, len(service_cases) // 80)
                selected.extend(service_cases[::step][:80])
        else:
            selected.extend(fault_cases)

    selected = sorted(selected, key=lambda c: c.log_time)
    return selected


def safe_endpoint(url: str) -> str:
    try:
        return urlsplit(str(url)).path or "/"
    except Exception:
        return "/"


def build_trace_aggregates() -> Tuple[pd.DataFrame, pd.DataFrame]:
    service_frames: List[pd.DataFrame] = []
    endpoint_frames: List[pd.DataFrame] = []

    for service in ALL_SERVICES:
        path = TRACE_DIR / f"trace_table_{service}_2021-07.csv"
        if not path.exists():
            continue

        service_chunks: List[pd.DataFrame] = []
        endpoint_chunks: List[pd.DataFrame] = []
        for chunk in pd.read_csv(
            path,
            usecols=["timestamp", "start_time", "end_time", "url", "status_code"],
            chunksize=TRACE_CHUNK,
        ):
            ts = pd.to_datetime(chunk["timestamp"], errors="coerce")
            start = pd.to_datetime(chunk["start_time"], errors="coerce")
            end = pd.to_datetime(chunk["end_time"], errors="coerce")
            latency_ms = (end - start).dt.total_seconds().fillna(0) * 1000.0
            endpoint = chunk["url"].map(safe_endpoint)

            frame = pd.DataFrame(
                {
                    "service": service,
                    "minute": ts.dt.floor("min"),
                    "endpoint": endpoint,
                    "latency_ms": latency_ms,
                    "latency_sum": latency_ms,
                    "is_error": pd.to_numeric(chunk["status_code"], errors="coerce")
                    .fillna(200)
                    .astype(float)
                    .ge(400)
                    .astype(int),
                }
            ).dropna(subset=["minute"])

            svc_group = (
                frame.groupby(["service", "minute"], observed=True)
                .agg(
                    trace_rows=("latency_ms", "size"),
                    latency_sum=("latency_sum", "sum"),
                    max_latency_ms=("latency_ms", "max"),
                    error_rows=("is_error", "sum"),
                )
                .reset_index()
            )
            service_chunks.append(svc_group)

            ep_group = (
                frame.groupby(["service", "minute", "endpoint"], observed=True)
                .agg(
                    trace_rows=("latency_ms", "size"),
                    latency_sum=("latency_sum", "sum"),
                    max_latency_ms=("latency_ms", "max"),
                    error_rows=("is_error", "sum"),
                )
                .reset_index()
            )
            endpoint_chunks.append(ep_group)

        service_df = pd.concat(service_chunks, ignore_index=True)
        service_df = (
            service_df.groupby(["service", "minute"], observed=True, as_index=False)
            .agg(
                trace_rows=("trace_rows", "sum"),
                latency_sum=("latency_sum", "sum"),
                max_latency_ms=("max_latency_ms", "max"),
                error_rows=("error_rows", "sum"),
            )
        )
        service_df["mean_latency_ms"] = service_df["latency_sum"] / service_df["trace_rows"].clip(lower=1)

        endpoint_df = pd.concat(endpoint_chunks, ignore_index=True)
        endpoint_df = (
            endpoint_df.groupby(["service", "minute", "endpoint"], observed=True, as_index=False)
            .agg(
                trace_rows=("trace_rows", "sum"),
                latency_sum=("latency_sum", "sum"),
                max_latency_ms=("max_latency_ms", "max"),
                error_rows=("error_rows", "sum"),
            )
        )
        endpoint_df["mean_latency_ms"] = endpoint_df["latency_sum"] / endpoint_df["trace_rows"].clip(lower=1)
        service_frames.append(service_df)
        endpoint_frames.append(endpoint_df)

    all_service = pd.concat(service_frames, ignore_index=True)
    all_endpoint = pd.concat(endpoint_frames, ignore_index=True)

    mean_trace_bytes = 180.0
    all_service["trace_bytes"] = all_service["trace_rows"] * mean_trace_bytes
    all_endpoint["trace_bytes"] = all_endpoint["trace_rows"] * mean_trace_bytes
    return all_service, all_endpoint


def load_or_build_trace_aggregates() -> Tuple[pd.DataFrame, pd.DataFrame]:
    svc_path = OUT_DIR / "trace_service_minute.csv"
    ep_path = OUT_DIR / "trace_service_minute_endpoint.csv"
    if svc_path.exists() and ep_path.exists():
        svc_df = pd.read_csv(svc_path, parse_dates=["minute"])
        ep_df = pd.read_csv(ep_path, parse_dates=["minute"])
        return svc_df, ep_df
    svc_df, ep_df = build_trace_aggregates()
    svc_df.to_csv(svc_path, index=False)
    ep_df.to_csv(ep_path, index=False)
    return svc_df, ep_df


def metric_files_for_service(service: str) -> Dict[str, List[Path]]:
    files = list(METRIC_DIR.glob(f"{service}_*_2021-07*.csv"))
    buckets: Dict[str, List[Path]] = defaultdict(list)
    for file in files:
        name = file.name
        for key, pattern in METRIC_SPECS.items():
            if pattern in name:
                buckets[key].append(file)
    return buckets


def build_metric_aggregates() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for service in ALL_SERVICES:
        spec_files = metric_files_for_service(service)
        series_list: List[pd.DataFrame] = []
        for metric_name, files in spec_files.items():
            metric_frames: List[pd.DataFrame] = []
            for file in files:
                df = pd.read_csv(file)
                df["minute"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce").dt.floor(
                    "min"
                )
                df = df.dropna(subset=["minute"])
                grouped = (
                    df.groupby("minute", as_index=False)
                    .agg(value=("value", "mean"), samples=("value", "size"))
                )
                grouped["metric_name"] = metric_name
                metric_frames.append(grouped)
            if not metric_frames:
                continue
            cat = pd.concat(metric_frames, ignore_index=True)
            cat = (
                cat.groupby(["minute", "metric_name"], as_index=False)
                .agg(value=("value", "mean"), samples=("samples", "sum"))
            )
            cat["service"] = service
            series_list.append(cat)

        if not series_list:
            continue

        service_metrics = pd.concat(series_list, ignore_index=True)
        pivot = service_metrics.pivot_table(
            index=["service", "minute"],
            columns="metric_name",
            values="value",
            aggfunc="mean",
        ).reset_index()
        sample_pivot = service_metrics.pivot_table(
            index=["service", "minute"],
            columns="metric_name",
            values="samples",
            aggfunc="sum",
        ).reset_index()
        merged = pivot.merge(sample_pivot, on=["service", "minute"], suffixes=("", "_samples"))
        frames.append(merged)

    all_metrics = pd.concat(frames, ignore_index=True).fillna(0)
    sample_cols = [c for c in all_metrics.columns if c.endswith("_samples")]
    all_metrics["metric_samples"] = all_metrics[sample_cols].sum(axis=1)
    all_metrics["metric_bytes"] = all_metrics["metric_samples"] * 24.0
    return all_metrics


def load_or_build_metric_aggregates() -> pd.DataFrame:
    path = OUT_DIR / "metric_service_minute.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["minute"])
    metric_df = build_metric_aggregates()
    metric_df.to_csv(path, index=False)
    return metric_df


def split_by_service(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for service in ALL_SERVICES:
        sub = df[df["service"] == service].copy()
        if not sub.empty:
            out[service] = sub
    return out


def baseline_window(case: Case) -> Tuple[pd.Timestamp, pd.Timestamp]:
    width = case.end_time - case.start_time
    start = case.start_time - width
    end = case.start_time
    return start, end


def minute_range(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    if end <= start:
        return [start.floor("min")]
    return list(pd.date_range(start=start, end=end, freq="min", inclusive="left"))


def choose_services(case: Case, budget: float, policy: str, rng: random.Random) -> List[str]:
    k = max(1, math.ceil(len(ALL_SERVICES) * budget))
    if policy == "full":
        return ALL_SERVICES[:]
    if policy == "adaptive_multibudget":
        neighbors = SERVICE_NEIGHBORS[case.service]
        ordered = neighbors + [s for s in ALL_SERVICES if s not in neighbors]
        return ordered[:k]
    sampled = ALL_SERVICES[:]
    rng.shuffle(sampled)
    return sampled[:k]


def choose_minutes(case: Case, budget: float, policy: str, rng: random.Random) -> List[pd.Timestamp]:
    minutes = minute_range(case.start_time, case.end_time)
    if policy == "full":
        return minutes
    k = max(1, math.ceil(len(minutes) * budget))
    if policy == "adaptive_multibudget":
        if case.fault_type in {"memory", "cpu", "file_move"}:
            return minutes[:k]
        center = max(0, len(minutes) // 2)
        lo = max(0, center - (k // 2))
        hi = min(len(minutes), lo + k)
        return minutes[lo:hi]
    sampled = minutes[:]
    rng.shuffle(sampled)
    return sorted(sampled[:k])


def choose_endpoints(
    case: Case,
    service: str,
    budget: float,
    policy: str,
    service_ep_df: pd.DataFrame,
    selected_minutes: Sequence[pd.Timestamp],
    rng: random.Random,
) -> List[str]:
    all_eps = sorted(ENDPOINT_ALLOWLIST.get(service, {"/"}))
    if policy == "full" or len(all_eps) == 1:
        return all_eps

    k = max(1, math.ceil(len(all_eps) * budget))
    if policy == "adaptive_multibudget":
        subset = service_ep_df[
            (service_ep_df["minute"].isin(selected_minutes))
            & (service_ep_df["endpoint"].isin(all_eps))
        ]
        if subset.empty:
            return all_eps[:k]
        ranked = (
            subset.groupby("endpoint", as_index=False)
            .agg(trace_rows=("trace_rows", "sum"), mean_latency_ms=("mean_latency_ms", "mean"))
        )
        ranked["score"] = ranked["trace_rows"] * ranked["mean_latency_ms"]
        ranked = ranked.sort_values(["score", "trace_rows"], ascending=False)
        return ranked["endpoint"].head(k).tolist()

    sampled = all_eps[:]
    rng.shuffle(sampled)
    return sorted(sampled[:k])


def summarize_trace_score(
    service: str,
    selected_minutes: Sequence[pd.Timestamp],
    selected_endpoints: Sequence[str],
    case: Case,
    service_ep_df: pd.DataFrame,
) -> Tuple[float, Dict[str, float]]:
    obs = service_ep_df[
        (service_ep_df["minute"].isin(selected_minutes))
        & (service_ep_df["endpoint"].isin(selected_endpoints))
    ]
    base_start, base_end = baseline_window(case)
    baseline_minutes = minute_range(base_start, base_end)
    base = service_ep_df[
        (service_ep_df["minute"].isin(baseline_minutes))
        & (service_ep_df["endpoint"].isin(selected_endpoints))
    ]

    obs_rows = float(obs["trace_rows"].sum())
    obs_latency = float(obs["mean_latency_ms"].mean()) if not obs.empty else 0.0
    obs_error = float(obs["error_rows"].sum()) / max(obs_rows, 1.0)
    base_rows = float(base["trace_rows"].sum())
    base_latency = float(base["mean_latency_ms"].mean()) if not base.empty else 1.0
    base_error = float(base["error_rows"].sum()) / max(base_rows, 1.0)

    latency_gain = max(0.0, (obs_latency - base_latency) / max(base_latency, 1.0))
    error_gain = max(0.0, obs_error - base_error)
    volume_gain = abs(obs_rows - base_rows) / max(base_rows, 1.0)
    trace_score = 2.5 * latency_gain + 4.0 * error_gain + 0.6 * volume_gain
    return trace_score, {
        "trace_rows": obs_rows,
        "trace_bytes": float(obs["trace_bytes"].sum()),
    }


def summarize_metric_score(
    service: str,
    selected_minutes: Sequence[pd.Timestamp],
    case: Case,
    service_metric_df: pd.DataFrame,
) -> Tuple[float, Dict[str, float]]:
    obs = service_metric_df[service_metric_df["minute"].isin(selected_minutes)]
    base_start, base_end = baseline_window(case)
    baseline_minutes = minute_range(base_start, base_end)
    base = service_metric_df[service_metric_df["minute"].isin(baseline_minutes)]

    if obs.empty:
        return 0.0, {"metric_samples": 0.0, "metric_bytes": 0.0}

    scores = []
    for metric_name in METRIC_SPECS:
        if metric_name not in obs.columns:
            continue
        obs_val = float(obs[metric_name].mean())
        if base.empty or metric_name not in base.columns:
            z = 0.0
        else:
            base_series = base[metric_name].astype(float)
            mu = float(base_series.mean())
            sigma = float(base_series.std(ddof=0))
            z = abs(obs_val - mu) / max(sigma, 1e-6)
        scores.append(z)
    metric_score = float(max(scores) if scores else 0.0)
    return metric_score, {
        "metric_samples": float(obs["metric_samples"].sum()),
        "metric_bytes": float(obs["metric_bytes"].sum()),
    }


def evaluate_case(
    case: Case,
    budget: float,
    policy: str,
    rng: random.Random,
    ep_map: Dict[str, pd.DataFrame],
    metric_map: Dict[str, pd.DataFrame],
) -> Dict[str, object]:
    if policy == "single_budget_random":
        selected_services = choose_services(case, budget, "random_multibudget", rng)
        selected_minutes = choose_minutes(case, budget, "random_multibudget", rng)
    else:
        selected_services = choose_services(case, budget, policy, rng)
        selected_minutes = choose_minutes(case, budget, policy, rng)

    rows: List[Dict[str, object]] = []
    cost_totals = Counter()
    for service in selected_services:
        service_ep_df = ep_map.get(service, pd.DataFrame(columns=["minute", "endpoint", "trace_rows", "trace_bytes", "mean_latency_ms", "error_rows"]))
        service_metric_df = metric_map.get(service, pd.DataFrame(columns=["minute", "metric_samples", "metric_bytes"]))
        service_budget = 1.0 if policy == "single_budget_random" else budget
        selected_endpoints = choose_endpoints(
            case, service, service_budget, policy, service_ep_df, selected_minutes, rng
        )
        trace_score, trace_cost = summarize_trace_score(
            service, selected_minutes, selected_endpoints, case, service_ep_df
        )
        metric_score, metric_cost = summarize_metric_score(
            service, selected_minutes, case, service_metric_df
        )
        score = trace_score + 0.8 * metric_score
        rows.append({"service": service, "score": score})
        cost_totals.update(trace_cost)
        cost_totals.update(metric_cost)

    ranking = sorted(rows, key=lambda x: x["score"], reverse=True)
    ranked_services = [row["service"] for row in ranking]
    rank = ranked_services.index(case.service) + 1 if case.service in ranked_services else 999

    return {
        "case_id": case.case_id,
        "root_service": case.service,
        "fault_type": case.fault_type,
        "service_family": case.service_family,
        "budget": budget,
        "policy": policy,
        "rank": rank,
        "top1": 1 if rank <= 1 else 0,
        "top3": 1 if rank <= 3 else 0,
        "top5": 1 if rank <= 5 else 0,
        "avg_at_5": sum(1 for k in range(rank, 6)) / 5.0 if rank <= 5 else 0.0,
        "trace_rows_kept": float(cost_totals["trace_rows"]),
        "trace_bytes_kept": float(cost_totals["trace_bytes"]),
        "metric_samples_kept": float(cost_totals["metric_samples"]),
        "metric_bytes_kept": float(cost_totals["metric_bytes"]),
        "minutes_kept": len(selected_minutes),
        "services_kept": len(selected_services),
    }


def summarize_results(results: pd.DataFrame, full_costs: pd.DataFrame) -> pd.DataFrame:
    merged = results.merge(
        full_costs[
            [
                "case_id",
                "trace_rows_kept",
                "trace_bytes_kept",
                "metric_samples_kept",
                "metric_bytes_kept",
                "minutes_kept",
                "services_kept",
            ]
        ].rename(
            columns={
                "trace_rows_kept": "full_trace_rows",
                "trace_bytes_kept": "full_trace_bytes",
                "metric_samples_kept": "full_metric_samples",
                "metric_bytes_kept": "full_metric_bytes",
                "minutes_kept": "full_minutes",
                "services_kept": "full_services",
            }
        ),
        on="case_id",
        how="left",
    )
    merged["combined_bytes_kept"] = merged["trace_bytes_kept"] + merged["metric_bytes_kept"]
    merged["full_combined_bytes"] = merged["full_trace_bytes"] + merged["full_metric_bytes"]
    merged["combined_reduction"] = 1.0 - (
        merged["combined_bytes_kept"] / merged["full_combined_bytes"].replace(0, np.nan)
    )
    merged["trace_row_reduction"] = 1.0 - (
        merged["trace_rows_kept"] / merged["full_trace_rows"].replace(0, np.nan)
    )
    merged["metric_sample_reduction"] = 1.0 - (
        merged["metric_samples_kept"] / merged["full_metric_samples"].replace(0, np.nan)
    )
    merged["time_reduction"] = 1.0 - (merged["minutes_kept"] / merged["full_minutes"].replace(0, np.nan))
    merged["where_reduction"] = 1.0 - (
        merged["services_kept"] / merged["full_services"].replace(0, np.nan)
    )

    summary = (
        merged.groupby(["policy", "budget"], as_index=False)
        .agg(
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            avg_at_5=("avg_at_5", "mean"),
            combined_reduction=("combined_reduction", "mean"),
            trace_row_reduction=("trace_row_reduction", "mean"),
            metric_sample_reduction=("metric_sample_reduction", "mean"),
            time_reduction=("time_reduction", "mean"),
            where_reduction=("where_reduction", "mean"),
        )
        .sort_values(["policy", "budget"])
    )
    return merged, summary


def fault_summary(merged: pd.DataFrame, budget: float) -> pd.DataFrame:
    sub = merged[merged["budget"] == budget]
    out = (
        sub.groupby(["fault_type", "policy"], as_index=False)
        .agg(avg_at_5=("avg_at_5", "mean"), top3=("top3", "mean"))
        .sort_values(["fault_type", "policy"])
    )
    return out


def family_summary(merged: pd.DataFrame, budget: float) -> pd.DataFrame:
    sub = merged[merged["budget"] == budget]
    out = (
        sub.groupby(["service_family", "policy"], as_index=False)
        .agg(avg_at_5=("avg_at_5", "mean"), top3=("top3", "mean"))
        .sort_values(["service_family", "policy"])
    )
    return out


def main() -> None:
    ensure_dirs()
    rng = random.Random(RANDOM_SEED)
    print("Loading cases...")
    cases = load_cases()
    print(f"Selected {len(cases)} cases")

    print("Building trace aggregates...")
    svc_df, ep_df = load_or_build_trace_aggregates()

    print("Building metric aggregates...")
    metric_df = load_or_build_metric_aggregates()

    ep_map = split_by_service(ep_df)
    metric_map = split_by_service(metric_df)

    policies = ["full", "single_budget_random", "random_multibudget", "adaptive_multibudget"]
    results: List[Dict[str, object]] = []
    for policy in policies:
        for budget in SERVICE_BUDGETS:
            if policy == "full" and budget != 1.0:
                continue
            print(f"Running {policy} @ {budget:.2f}")
            for case in cases:
                results.append(evaluate_case(case, budget, policy, rng, ep_map, metric_map))

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "case_results.csv", index=False)

    full_costs = results_df[results_df["policy"] == "full"].copy()
    merged, summary = summarize_results(results_df, full_costs)
    merged.to_csv(OUT_DIR / "case_results_with_costs.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_overall.csv", index=False)

    fault10 = fault_summary(merged, 0.10)
    fault10.to_csv(OUT_DIR / "summary_fault_budget_010.csv", index=False)

    family10 = family_summary(merged, 0.10)
    family10.to_csv(OUT_DIR / "summary_family_budget_010.csv", index=False)

    manifest = {
        "cases": len(cases),
        "fault_counts": Counter([case.fault_type for case in cases]),
        "service_counts": Counter([case.service for case in cases]),
        "policies": policies,
        "budgets": SERVICE_BUDGETS,
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=lambda x: dict(x))

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
