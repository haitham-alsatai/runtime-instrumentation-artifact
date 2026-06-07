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
OUT_DIR = ROOT / "results" / "stage_a" / "trace_when_pilot_experiment"
OUT_DIR.mkdir(exist_ok=True)

DATASETS = ["RE2-OB", "RE2-TT"]
POLICIES = ["random", "early_window", "late_window"]
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


def choose_trace_ids(trace_stats: pd.DataFrame, inject_time_us: int, budget: float, policy: str) -> set[str]:
    normal_ids = trace_stats.loc[trace_stats["endTime"] < inject_time_us, "traceID"].tolist()
    anomalous = trace_stats.loc[trace_stats["endTime"] >= inject_time_us].copy()
    anomalous_ids = anomalous["traceID"].tolist()

    if not anomalous_ids:
        return set(normal_ids)
    if budget >= 1.0:
        return set(normal_ids) | set(anomalous_ids)

    kept_count = max(1, round(len(anomalous_ids) * budget))
    if policy == "random":
        kept = anomalous["traceID"].sample(n=kept_count, replace=False, random_state=RANDOM_SEED).tolist()
    elif policy == "early_window":
        kept = anomalous.sort_values(["endTime", "trace_duration"], ascending=[True, False]).head(kept_count)["traceID"].tolist()
    elif policy == "late_window":
        kept = anomalous.sort_values(["endTime", "trace_duration"], ascending=[False, False]).head(kept_count)["traceID"].tolist()
    else:
        raise ValueError(f"Unknown policy: {policy}")

    return set(normal_ids) | set(kept)


def evaluate_case_all(case: CaseInfo) -> list[dict]:
    case_dir = Path(case.path)
    inject_time_us = int((case_dir / "inject_time.txt").read_text().strip()) * 1_000_000
    df = pd.read_csv(
        case_dir / "traces.csv",
        usecols=["traceID", "serviceName", "methodName", "operationName", "startTime", "duration"],
    )
    trace_stats = (
        df.assign(endTime=df["startTime"] + df["duration"])
        .groupby("traceID", sort=False)
        .agg(startTime=("startTime", "min"), endTime=("endTime", "max"), span_count=("duration", "size"))
        .reset_index()
    )
    trace_stats["trace_duration"] = trace_stats["endTime"] - trace_stats["startTime"]
    trace_end_map = trace_stats.set_index("traceID")["endTime"]
    full_anom_count = int((trace_stats["endTime"] >= inject_time_us).sum())

    records: list[dict] = []
    for policy in POLICIES:
        for budget in BUDGETS:
            kept_ids = choose_trace_ids(trace_stats, inject_time_us, budget, policy)
            sampled = df[df["traceID"].isin(kept_ids)].copy()
            ranked_services = trace_budget_rank_services(sampled, inject_time_us)
            records.append(
                {
                    "dataset": case.dataset,
                    "service": case.service,
                    "fault": case.fault,
                    "instance": case.instance,
                    "policy": policy,
                    "budget": budget,
                    "top1": float(case.service in ranked_services[:1]),
                    "top3": float(case.service in ranked_services[:3]),
                    "top5": float(case.service in ranked_services[:5]),
                    "avg5": avg_at_5_from_services(ranked_services, case.service),
                    "full_trace_count": int(len(trace_stats)),
                    "kept_trace_count": int(len(kept_ids)),
                    "full_anomalous_trace_count": full_anom_count,
                    "kept_anomalous_trace_count": int(
                        sum(1 for trace_id in kept_ids if trace_end_map.loc[trace_id] >= inject_time_us)
                    ),
                }
            )
    return records


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        results.groupby(["policy", "dataset", "budget"], as_index=False)
        .agg(
            cases=("top1", "size"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            avg5=("avg5", "mean"),
            kept_trace_sum=("kept_trace_count", "sum"),
            full_trace_sum=("full_trace_count", "sum"),
            kept_anom_sum=("kept_anomalous_trace_count", "sum"),
            full_anom_sum=("full_anomalous_trace_count", "sum"),
        )
    )
    overall["trace_reduction"] = 1.0 - (overall["kept_trace_sum"] / overall["full_trace_sum"])
    overall["anom_trace_reduction"] = 1.0 - (overall["kept_anom_sum"] / overall["full_anom_sum"])

    comparison = (
        overall.pivot_table(
            index=["dataset", "budget"],
            columns="policy",
            values=["avg5", "top3", "trace_reduction"],
        )
        .sort_index(axis=1)
    )
    comparison.columns = [f"{metric}_{policy}" for metric, policy in comparison.columns]
    comparison = comparison.reset_index()
    comparison["avg5_gain_early_vs_random"] = comparison["avg5_early_window"] - comparison["avg5_random"]
    comparison["avg5_gain_late_vs_random"] = comparison["avg5_late_window"] - comparison["avg5_random"]
    return overall, comparison


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

    overall, comparison = summarize(clean)
    clean.to_csv(OUT_DIR / "case_results.csv", index=False)
    overall.to_csv(OUT_DIR / "summary_overall.csv", index=False)
    comparison.to_csv(OUT_DIR / "summary_policy_comparison.csv", index=False)
    failures_path = OUT_DIR / "failures.csv"
    if not failures.empty:
        failures.to_csv(failures_path, index=False)
    elif failures_path.exists():
        failures_path.unlink()

    payload = {
        "datasets": DATASETS,
        "policies": POLICIES,
        "budgets": BUDGETS,
        "random_seed": RANDOM_SEED,
        "max_workers": MAX_WORKERS,
        "data_root": str(data_root),
        "num_cases": len(cases),
        "num_runs": int(len(clean)),
        "num_failures": int(len(failures)),
        "note": "Longer-term pilot for the 'when to trace' dimension using temporal window selection.",
    }
    (OUT_DIR / "experiment_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Completed when-to-trace pilot.")
    print(json.dumps(payload, indent=2))
    print("\nOverall comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
