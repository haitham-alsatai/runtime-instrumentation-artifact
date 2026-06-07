from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "RE2"
OUT_DIR = ROOT / "results" / "stage_a" / "trace_budget_experiment"
OUT_DIR.mkdir(exist_ok=True)

DATASETS = ["RE2-OB", "RE2-TT"]
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


def discover_cases() -> list[CaseInfo]:
    cases: list[CaseInfo] = []
    for dataset in DATASETS:
        for traces_file in sorted((DATA_ROOT / dataset).glob("*/*/traces.csv")):
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


def keep_trace_ids(trace_end_times: pd.Series, inject_time_us: int, budget: float, seed: int) -> set[str]:
    normal_ids = trace_end_times[trace_end_times < inject_time_us].index.to_list()
    anomalous_ids = trace_end_times[trace_end_times >= inject_time_us].index.to_list()

    if not anomalous_ids:
        return set(normal_ids)

    if budget >= 1.0:
        kept_anomalous = anomalous_ids
    else:
        kept_count = max(1, round(len(anomalous_ids) * budget))
        kept_anomalous = (
            pd.Series(anomalous_ids)
            .sample(n=kept_count, replace=False, random_state=seed)
            .tolist()
        )

    return set(normal_ids) | set(kept_anomalous)


def service_ranks_from_operations(ranks: Iterable[str]) -> list[str]:
    services: list[str] = []
    seen: set[str] = set()
    for op in ranks:
        service = str(op).split("_", 1)[0]
        if service not in seen:
            services.append(service)
            seen.add(service)
    return services


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
    score = 0.0
    for k in range(1, 6):
        score += 1.0 if answer in predicted[:k] else 0.0
    return score / 5.0


def evaluate_case(case: CaseInfo, budget: float) -> dict:
    case_dir = Path(case.path)
    inject_time_us = int((case_dir / "inject_time.txt").read_text().strip()) * 1_000_000

    df = pd.read_csv(
        case_dir / "traces.csv",
        usecols=["traceID", "serviceName", "methodName", "operationName", "startTime", "duration"],
    )
    trace_end_times = (
        df.assign(endTime=df["startTime"] + df["duration"])
        .groupby("traceID", sort=False)["endTime"]
        .max()
    )
    kept_ids = keep_trace_ids(trace_end_times, inject_time_us, budget, RANDOM_SEED)
    sampled = df[df["traceID"].isin(kept_ids)].copy()
    ranked_services = trace_budget_rank_services(sampled, inject_time_us)

    return {
        "dataset": case.dataset,
        "service": case.service,
        "fault": case.fault,
        "instance": case.instance,
        "budget": budget,
        "top1": float(case.service in ranked_services[:1]),
        "top3": float(case.service in ranked_services[:3]),
        "top5": float(case.service in ranked_services[:5]),
        "avg5": avg_at_5_from_services(ranked_services, case.service),
        "full_trace_count": int(len(trace_end_times)),
        "full_anomalous_trace_count": int((trace_end_times >= inject_time_us).sum()),
        "kept_trace_count": int(len(kept_ids)),
        "kept_anomalous_trace_count": int(
            sum(1 for trace_id in kept_ids if trace_end_times.loc[trace_id] >= inject_time_us)
        ),
        "top5_services": ranked_services[:5],
    }


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_fault = (
        results.groupby(["dataset", "fault", "budget"], as_index=False)
        .agg(
            cases=("top1", "size"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            avg5=("avg5", "mean"),
            kept_anomalous_mean=("kept_anomalous_trace_count", "mean"),
            full_anomalous_mean=("full_anomalous_trace_count", "mean"),
        )
    )

    overall = (
        results.groupby(["dataset", "budget"], as_index=False)
        .agg(
            cases=("top1", "size"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            avg5=("avg5", "mean"),
        )
    )

    min_budget_rows: list[dict] = []
    for (dataset, fault), group in by_fault.groupby(["dataset", "fault"]):
        ordered = group.sort_values("budget")
        full_avg5 = float(ordered.loc[ordered["budget"] == 1.0, "avg5"].iloc[0])
        target = 0.95 * full_avg5
        eligible = ordered[ordered["avg5"] >= target]
        min_budget_rows.append(
            {
                "dataset": dataset,
                "fault": fault,
                "full_budget_avg5": full_avg5,
                "target_95pct": target,
                "min_budget_for_95pct_full": float(eligible["budget"].iloc[0]) if not eligible.empty else None,
                "gain_10_to_full": float(
                    ordered.loc[ordered["budget"] == 1.0, "avg5"].iloc[0]
                    - ordered.loc[ordered["budget"] == 0.10, "avg5"].iloc[0]
                ),
            }
        )

    min_budget = pd.DataFrame(min_budget_rows).sort_values(["dataset", "gain_10_to_full"], ascending=[True, False])
    return by_fault, overall, min_budget


def main() -> None:
    cases = discover_cases()
    jobs = [(case, budget) for case in cases for budget in BUDGETS]
    records: list[dict] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_case, case, budget): (case, budget) for case, budget in jobs}
        for future in as_completed(futures):
            case, budget = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                records.append(
                    {
                        "dataset": case.dataset,
                        "service": case.service,
                        "fault": case.fault,
                        "instance": case.instance,
                        "budget": budget,
                        "error": str(exc),
                    }
                )
                print(f"FAILED {case.dataset} {case.service} {case.fault} {case.instance} budget={budget}: {exc}")

    results = pd.DataFrame(records).sort_values(["dataset", "fault", "service", "instance", "budget"])
    if "error" in results.columns:
        failures = results[results["error"].notna()]
        clean = results[results["error"].isna()]
    else:
        failures = pd.DataFrame()
        clean = results

    by_fault, overall, min_budget = summarize(clean)

    results.to_csv(OUT_DIR / "case_results.csv", index=False)
    by_fault.to_csv(OUT_DIR / "summary_by_fault.csv", index=False)
    overall.to_csv(OUT_DIR / "summary_overall.csv", index=False)
    min_budget.to_csv(OUT_DIR / "minimum_budget_by_fault.csv", index=False)
    failures_path = OUT_DIR / "failures.csv"
    if not failures.empty:
        failures.to_csv(failures_path, index=False)
    elif failures_path.exists():
        failures_path.unlink()

    payload = {
        "datasets": DATASETS,
        "budgets": BUDGETS,
        "random_seed": RANDOM_SEED,
        "max_workers": MAX_WORKERS,
        "method": "trace_budget_rank_services",
        "method_note": "Lightweight service-level trace ranking inspired by TraceRCA with service-level fallbacks for unseen post-fault operations.",
        "num_cases": len(cases),
        "num_runs": len(jobs),
        "num_failures": int(len(failures)),
    }
    (OUT_DIR / "experiment_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Completed.")
    print(json.dumps(payload, indent=2))
    print("\nOverall results:")
    print(overall.to_string(index=False))
    print("\nMinimum budget by fault:")
    print(min_budget.to_string(index=False))


if __name__ == "__main__":
    main()
