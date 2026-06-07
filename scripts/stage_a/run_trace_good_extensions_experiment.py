from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_CANDIDATES = [
    ROOT / "data" / "RE2",
]
OUT_DIR = ROOT / "results" / "stage_a" / "trace_good_extensions_experiment"
OUT_DIR.mkdir(exist_ok=True)

DATASETS = ["RE2-OB", "RE2-TT"]
BUDGET_TYPES = ["trace_fraction", "span_fraction"]
POLICIES = ["random", "abnormality_topk", "service_aware_abnormality"]
BUDGETS = [0.10, 0.25, 0.50, 1.00]
RANDOM_SEED = 2026
MAX_WORKERS = 4


@dataclass(frozen=True)
class CaseInfo:
    dataset: str
    service: str
    fault: str
    instance: str
    path: str


def resolve_data_root() -> Path:
    for candidate in DATA_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find RE2 data in any candidate path: {DATA_ROOT_CANDIDATES}")


def discover_cases(data_root: Path) -> list[CaseInfo]:
    cases: list[CaseInfo] = []
    for dataset in DATASETS:
        for traces_file in sorted((data_root / dataset).glob("*/*/traces.csv")):
            instance_dir = traces_file.parent
            if not instance_dir.name.isdigit():
                continue
            fault_dir = instance_dir.parent.name
            service, fault = fault_dir.rsplit("_", 1)
            cases.append(
                CaseInfo(
                    dataset=dataset,
                    service=service,
                    fault=fault,
                    instance=instance_dir.name,
                    path=str(instance_dir),
                )
            )
    return cases


def trace_budget_rank_services(df: pd.DataFrame, inject_time_us: int) -> list[str]:
    data = df.copy()
    data["methodName"] = data["methodName"].fillna(data["operationName"])
    data["operation"] = data["serviceName"] + "_" + data["methodName"]
    data["endTime"] = data["startTime"] + data["duration"]

    normal = data[data["endTime"] < inject_time_us]
    anomalous = data[data["endTime"] >= inject_time_us].copy()
    if normal.empty or anomalous.empty:
        return []

    op_stats = normal.groupby("operation")["duration"].agg(["mean", "std"]).rename(
        columns={"mean": "op_mean", "std": "op_std"}
    )
    svc_stats = normal.groupby("serviceName")["duration"].agg(["mean", "std"]).rename(
        columns={"mean": "svc_mean", "std": "svc_std"}
    )
    global_mean = normal["duration"].mean()
    global_std = normal["duration"].std()

    anomalous = anomalous.merge(op_stats, left_on="operation", right_index=True, how="left")
    anomalous = anomalous.merge(svc_stats, left_on="serviceName", right_index=True, how="left")
    anomalous["mean"] = anomalous["op_mean"].fillna(anomalous["svc_mean"]).fillna(global_mean)
    anomalous["std"] = anomalous["op_std"].fillna(anomalous["svc_std"]).fillna(global_std).fillna(0)
    anomalous["abnormal"] = anomalous["duration"] >= anomalous["mean"] + 3 * anomalous["std"]

    total_abnormal = int(anomalous["abnormal"].sum())
    if total_abnormal == 0:
        return anomalous.groupby("serviceName").size().sort_values(ascending=False).index.tolist()

    op_aggs = anomalous.groupby(["serviceName", "operation"]).agg(
        total=("abnormal", "size"),
        abnormal=("abnormal", "sum"),
    )
    op_aggs["support"] = op_aggs["abnormal"] / total_abnormal
    op_aggs["confidence"] = op_aggs["abnormal"] / op_aggs["total"]
    denom = op_aggs["support"] + op_aggs["confidence"]
    op_aggs["ji"] = 0.0
    valid = denom > 0
    op_aggs.loc[valid, "ji"] = (
        2 * op_aggs.loc[valid, "support"] * op_aggs.loc[valid, "confidence"] / denom[valid]
    )

    return op_aggs.groupby(level=0)["ji"].max().sort_values(ascending=False).index.tolist()


def avg_at_5_from_services(predicted: list[str], answer: str) -> float:
    return sum(1.0 if answer in predicted[:k] else 0.0 for k in range(1, 6)) / 5.0


def prepare_anomalous_trace_meta(df: pd.DataFrame, inject_time_us: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["methodName"] = data["methodName"].fillna(data["operationName"])
    data["operation"] = data["serviceName"] + "_" + data["methodName"]
    data["endTime"] = data["startTime"] + data["duration"]

    normal = data[data["endTime"] < inject_time_us]
    anomalous = data[data["endTime"] >= inject_time_us].copy()

    op_stats = normal.groupby("operation")["duration"].agg(["mean", "std"]).rename(
        columns={"mean": "op_mean", "std": "op_std"}
    )
    svc_stats = normal.groupby("serviceName")["duration"].agg(["mean", "std"]).rename(
        columns={"mean": "svc_mean", "std": "svc_std"}
    )
    global_mean = normal["duration"].mean()
    global_std = normal["duration"].std()

    anomalous = anomalous.merge(op_stats, left_on="operation", right_index=True, how="left")
    anomalous = anomalous.merge(svc_stats, left_on="serviceName", right_index=True, how="left")
    anomalous["mean"] = anomalous["op_mean"].fillna(anomalous["svc_mean"]).fillna(global_mean)
    anomalous["std"] = anomalous["op_std"].fillna(anomalous["svc_std"]).fillna(global_std).fillna(0)
    anomalous["threshold"] = anomalous["mean"] + 3 * anomalous["std"]
    anomalous["abnormal"] = anomalous["duration"] >= anomalous["threshold"]
    anomalous["excess"] = (anomalous["duration"] - anomalous["threshold"]).clip(lower=0)

    trace_stats = (
        data.groupby("traceID", sort=False)
        .agg(
            startTime=("startTime", "min"),
            endTime=("endTime", "max"),
            total_span_count=("duration", "size"),
        )
        .reset_index()
    )
    trace_stats["trace_duration"] = trace_stats["endTime"] - trace_stats["startTime"]

    anom_trace = (
        anomalous.groupby("traceID", sort=False)
        .agg(
            anomalous_span_count=("duration", "size"),
            abnormal_span_count=("abnormal", "sum"),
            abnormal_excess_sum=("excess", "sum"),
            anomalous_service_count=("serviceName", "nunique"),
        )
        .reset_index()
    )
    anom_trace = anom_trace.merge(trace_stats, on="traceID", how="left")

    services_by_trace = anomalous.groupby("traceID")["serviceName"].agg(lambda x: sorted(set(x))).rename("services")
    anom_trace = anom_trace.merge(services_by_trace, on="traceID", how="left")
    anom_trace["utility_score"] = (
        1000 * anom_trace["abnormal_span_count"]
        + anom_trace["abnormal_excess_sum"]
        + 0.001 * anom_trace["trace_duration"]
    )
    return data, anom_trace


def select_random(anom_trace: pd.DataFrame) -> pd.DataFrame:
    return anom_trace.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def select_abnormality_topk(anom_trace: pd.DataFrame) -> pd.DataFrame:
    return (
        anom_trace.sort_values(
            ["abnormal_span_count", "abnormal_excess_sum", "trace_duration", "anomalous_service_count"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )


def take_by_budget(ordered: pd.DataFrame, budget: float, budget_type: str) -> set[str]:
    if ordered.empty:
        return set()

    if budget >= 1.0:
        return set(ordered["traceID"].tolist())

    if budget_type == "trace_fraction":
        capacity = max(1, round(len(ordered) * budget))
        return set(ordered.head(capacity)["traceID"].tolist())

    if budget_type == "span_fraction":
        capacity = max(1, round(float(ordered["anomalous_span_count"].sum()) * budget))
        kept: list[str] = []
        used = 0
        for row in ordered.itertuples(index=False):
            cost = int(row.anomalous_span_count)
            if not kept:
                kept.append(row.traceID)
                used += cost
                continue
            if used + cost > capacity:
                continue
            kept.append(row.traceID)
            used += cost
        return set(kept)

    raise ValueError(f"Unknown budget_type: {budget_type}")


def select_service_aware(anom_trace: pd.DataFrame, budget: float, budget_type: str) -> set[str]:
    if anom_trace.empty:
        return set()
    if budget >= 1.0:
        return set(anom_trace["traceID"].tolist())

    ordered = select_abnormality_topk(anom_trace)
    if budget_type == "trace_fraction":
        capacity = max(1, round(len(ordered) * budget))
    elif budget_type == "span_fraction":
        capacity = max(1, round(float(ordered["anomalous_span_count"].sum()) * budget))
    else:
        raise ValueError(f"Unknown budget_type: {budget_type}")

    selected: list[str] = []
    selected_set: set[str] = set()
    covered_services: set[str] = set()
    all_services = set(service for services in ordered["services"] for service in services)

    def cost_of(row) -> int:
        return 1 if budget_type == "trace_fraction" else int(row.anomalous_span_count)

    used = 0
    remaining = ordered.copy()
    while covered_services != all_services:
        best_idx = None
        best_tuple = None
        for idx, row in remaining.iterrows():
            cost = cost_of(row)
            if used + cost > capacity:
                continue
            services = set(row["services"])
            new_coverage = len(services - covered_services)
            if new_coverage <= 0:
                continue
            score_tuple = (new_coverage, row["abnormal_span_count"], row["abnormal_excess_sum"], row["trace_duration"])
            if best_tuple is None or score_tuple > best_tuple:
                best_tuple = score_tuple
                best_idx = idx
        if best_idx is None:
            if not selected and not remaining.empty:
                fallback = remaining.iloc[0]
                selected.append(fallback["traceID"])
                selected_set.add(fallback["traceID"])
                used += cost_of(fallback)
            break
        row = remaining.loc[best_idx]
        trace_id = row["traceID"]
        selected.append(trace_id)
        selected_set.add(trace_id)
        used += cost_of(row)
        covered_services.update(row["services"])
        remaining = remaining[remaining["traceID"] != trace_id]

    for row in remaining.itertuples(index=False):
        trace_id = row.traceID
        if trace_id in selected_set:
            continue
        cost = cost_of(row)
        if used + cost > capacity:
            continue
        selected.append(trace_id)
        selected_set.add(trace_id)
        used += cost

    return set(selected)


def choose_anomalous_ids(anom_trace: pd.DataFrame, budget: float, budget_type: str, policy: str) -> set[str]:
    if policy == "random":
        return take_by_budget(select_random(anom_trace), budget, budget_type)
    if policy == "abnormality_topk":
        return take_by_budget(select_abnormality_topk(anom_trace), budget, budget_type)
    if policy == "service_aware_abnormality":
        return select_service_aware(anom_trace, budget, budget_type)
    raise ValueError(f"Unknown policy: {policy}")


def evaluate_case_all(case: CaseInfo) -> list[dict]:
    case_dir = Path(case.path)
    inject_time_us = int((case_dir / "inject_time.txt").read_text().strip()) * 1_000_000
    df = pd.read_csv(
        case_dir / "traces.csv",
        usecols=["traceID", "serviceName", "methodName", "operationName", "startTime", "duration"],
    )
    data, anom_trace = prepare_anomalous_trace_meta(df, inject_time_us)
    normal_trace_ids = set(data.loc[data["endTime"] < inject_time_us, "traceID"].unique().tolist())
    anomalous_trace_end = data.groupby("traceID", sort=False)["endTime"].max()
    full_trace_count = int(data["traceID"].nunique())
    full_anomalous_trace_count = int((anomalous_trace_end >= inject_time_us).sum())
    full_anomalous_span_count = int(len(data[data["endTime"] >= inject_time_us]))

    records: list[dict] = []
    for budget_type in BUDGET_TYPES:
        for policy in POLICIES:
            for budget in BUDGETS:
                kept_anomalous = choose_anomalous_ids(anom_trace, budget, budget_type, policy)
                kept_ids = normal_trace_ids | kept_anomalous
                sampled = df[df["traceID"].isin(kept_ids)].copy()
                ranked_services = trace_budget_rank_services(sampled, inject_time_us)
                kept_anomalous_span_count = int(
                    anom_trace.loc[anom_trace["traceID"].isin(kept_anomalous), "anomalous_span_count"].sum()
                )
                records.append(
                    {
                        "dataset": case.dataset,
                        "service": case.service,
                        "fault": case.fault,
                        "instance": case.instance,
                        "budget_type": budget_type,
                        "policy": policy,
                        "budget": budget,
                        "top1": float(case.service in ranked_services[:1]),
                        "top3": float(case.service in ranked_services[:3]),
                        "top5": float(case.service in ranked_services[:5]),
                        "avg5": avg_at_5_from_services(ranked_services, case.service),
                        "full_trace_count": full_trace_count,
                        "full_anomalous_trace_count": full_anomalous_trace_count,
                        "full_anomalous_span_count": full_anomalous_span_count,
                        "kept_trace_count": int(len(kept_ids)),
                        "kept_anomalous_trace_count": int(len(kept_anomalous)),
                        "kept_anomalous_span_count": kept_anomalous_span_count,
                    }
                )
    return records


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = (
        results.groupby(["budget_type", "policy", "dataset", "budget"], as_index=False)
        .agg(
            cases=("top1", "size"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            avg5=("avg5", "mean"),
            kept_trace_sum=("kept_trace_count", "sum"),
            full_trace_sum=("full_trace_count", "sum"),
            kept_anom_trace_sum=("kept_anomalous_trace_count", "sum"),
            full_anom_trace_sum=("full_anomalous_trace_count", "sum"),
            kept_anom_span_sum=("kept_anomalous_span_count", "sum"),
            full_anom_span_sum=("full_anomalous_span_count", "sum"),
        )
    )
    overall["trace_reduction"] = 1.0 - (overall["kept_trace_sum"] / overall["full_trace_sum"])
    overall["anom_trace_reduction"] = 1.0 - (overall["kept_anom_trace_sum"] / overall["full_anom_trace_sum"])
    overall["anom_span_reduction"] = 1.0 - (overall["kept_anom_span_sum"] / overall["full_anom_span_sum"])

    by_fault = (
        results.groupby(["budget_type", "policy", "dataset", "fault", "budget"], as_index=False)
        .agg(
            cases=("top1", "size"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            avg5=("avg5", "mean"),
        )
    )

    comparison = (
        overall.pivot_table(
            index=["budget_type", "dataset", "budget"],
            columns="policy",
            values=["avg5", "top3", "trace_reduction", "anom_trace_reduction", "anom_span_reduction"],
        )
        .sort_index(axis=1)
    )
    comparison.columns = [f"{metric}_{policy}" for metric, policy in comparison.columns]
    comparison = comparison.reset_index()
    comparison["avg5_gain_abnormality_vs_random"] = (
        comparison["avg5_abnormality_topk"] - comparison["avg5_random"]
    )
    comparison["avg5_gain_service_aware_vs_random"] = (
        comparison["avg5_service_aware_abnormality"] - comparison["avg5_random"]
    )

    fault_comparison = (
        by_fault[by_fault["budget"] == 0.10]
        .pivot_table(
            index=["budget_type", "dataset", "fault"],
            columns="policy",
            values="avg5",
        )
        .reset_index()
    )
    fault_comparison["gain_abnormality_vs_random"] = (
        fault_comparison["abnormality_topk"] - fault_comparison["random"]
    )
    fault_comparison["gain_service_aware_vs_random"] = (
        fault_comparison["service_aware_abnormality"] - fault_comparison["random"]
    )

    return overall, by_fault, comparison, fault_comparison


def main() -> None:
    data_root = resolve_data_root()
    cases = discover_cases(data_root)
    records: list[dict] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_case_all, case): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                records.extend(future.result())
            except Exception as exc:
                records.append(
                    {
                        "dataset": case.dataset,
                        "service": case.service,
                        "fault": case.fault,
                        "instance": case.instance,
                        "error": str(exc),
                    }
                )
                print(f"FAILED {case.dataset} {case.service} {case.fault} {case.instance}: {exc}")

    results = pd.DataFrame(records)
    if "error" in results.columns:
        failures = results[results["error"].notna()]
        clean = results[results["error"].isna()]
    else:
        failures = pd.DataFrame()
        clean = results

    overall, by_fault, comparison, fault_comparison = summarize(clean)

    clean.to_csv(OUT_DIR / "case_results.csv", index=False)
    overall.to_csv(OUT_DIR / "summary_overall.csv", index=False)
    by_fault.to_csv(OUT_DIR / "summary_by_fault.csv", index=False)
    comparison.to_csv(OUT_DIR / "summary_policy_budget_comparison.csv", index=False)
    fault_comparison.to_csv(OUT_DIR / "summary_fault_comparison_10pct.csv", index=False)
    failures_path = OUT_DIR / "failures.csv"
    if not failures.empty:
        failures.to_csv(failures_path, index=False)
    elif failures_path.exists():
        failures_path.unlink()

    payload = {
        "datasets": DATASETS,
        "budget_types": BUDGET_TYPES,
        "policies": POLICIES,
        "budgets": BUDGETS,
        "random_seed": RANDOM_SEED,
        "max_workers": MAX_WORKERS,
        "data_root": str(data_root),
        "num_cases": len(cases),
        "num_runs": int(len(clean)),
        "num_failures": int(len(failures)),
        "note": "Good-next-extensions experiment with multiple budgets, multiple budget definitions, single-budget and multi-budget informed trace-selection policies.",
    }
    (OUT_DIR / "experiment_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Completed Good Next Extensions experiment.")
    print(json.dumps(payload, indent=2))
    print("\nOverall comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
