from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "RE2"
OUT_DIR = ROOT / "results" / "stage_a" / "stage_a_policy_expansion"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
NOTES = ROOT / "docs" / "generated"

DATASETS = ["RE2-OB", "RE2-TT"]
BUDGETS = [0.05, 0.10, 0.25, 0.50, 1.00]
RANDOM_SEEDS = list(range(2026, 2036))
DETERMINISTIC_SEED_LABEL = -1
MAX_WORKERS = 4

POLICIES = [
    "random",
    "latency_topk",
    "abnormality_topk",
    "service_aware_abnormality",
    "early_window",
    "late_window",
    "latency_service_diversity",
    "abnormality_service_diversity",
    "coverage_aware",
    "hybrid_abnormality_latency",
]


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
            service, fault = instance_dir.parent.name.rsplit("_", 1)
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


def normalize(series: pd.Series) -> pd.Series:
    max_value = series.max()
    min_value = series.min()
    if pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def prepare_trace_meta(df: pd.DataFrame, inject_time_us: int) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    data = df.copy()
    data["methodName"] = data["methodName"].fillna(data["operationName"])
    data["operation"] = data["serviceName"] + "_" + data["methodName"]
    data["endTime"] = data["startTime"] + data["duration"]

    normal = data[data["endTime"] < inject_time_us]
    anomalous = data[data["endTime"] >= inject_time_us].copy()
    normal_trace_ids = set(normal["traceID"].unique().tolist())

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
    anomalous["excess"] = (anomalous["duration"] - anomalous["threshold"]).clip(lower=0)
    anomalous["abnormal"] = anomalous["excess"] > 0

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

    services = anomalous.groupby("traceID")["serviceName"].agg(lambda values: tuple(sorted(set(values))))
    primary_service = anomalous.groupby("traceID")["serviceName"].agg(lambda values: values.value_counts().idxmax())
    anom_trace = anom_trace.merge(services.rename("services"), on="traceID", how="left")
    anom_trace = anom_trace.merge(primary_service.rename("primary_service"), on="traceID", how="left")

    anom_trace["latency_score"] = normalize(anom_trace["trace_duration"])
    anom_trace["abnormality_score"] = normalize(anom_trace["abnormal_excess_sum"]) + normalize(
        anom_trace["abnormal_span_count"]
    )
    anom_trace["coverage_score"] = normalize(anom_trace["anomalous_service_count"])
    anom_trace["hybrid_score"] = (
        0.45 * normalize(anom_trace["abnormal_excess_sum"])
        + 0.25 * normalize(anom_trace["abnormal_span_count"])
        + 0.20 * normalize(anom_trace["trace_duration"])
        + 0.10 * normalize(anom_trace["anomalous_service_count"])
    )
    return data, anom_trace, normal_trace_ids


def capacity(meta: pd.DataFrame, budget: float) -> int:
    if meta.empty:
        return 0
    if budget >= 1.0:
        return len(meta)
    return max(1, round(len(meta) * budget))


def take_sorted(meta: pd.DataFrame, budget: float, columns: list[str], ascending: list[bool]) -> set[str]:
    n = capacity(meta, budget)
    if n == 0:
        return set()
    return set(meta.sort_values(columns, ascending=ascending).head(n)["traceID"].tolist())


def select_service_diversity(meta: pd.DataFrame, budget: float, score_col: str) -> set[str]:
    n = capacity(meta, budget)
    if n == 0:
        return set()
    if n >= len(meta):
        return set(meta["traceID"].tolist())

    selected: list[str] = []
    remaining = meta.sort_values([score_col, "trace_duration"], ascending=[False, False]).copy()
    for _, group in remaining.groupby("primary_service", sort=False):
        if len(selected) >= n:
            break
        selected.append(group.iloc[0]["traceID"])

    selected_set = set(selected)
    for row in remaining.itertuples(index=False):
        if len(selected_set) >= n:
            break
        if row.traceID not in selected_set:
            selected_set.add(row.traceID)
    return selected_set


def select_coverage_aware(meta: pd.DataFrame, budget: float, score_col: str = "coverage_score") -> set[str]:
    n = capacity(meta, budget)
    if n == 0:
        return set()
    if n >= len(meta):
        return set(meta["traceID"].tolist())

    selected: set[str] = set()
    covered_services: set[str] = set()
    ordered = meta.sort_values(
        [score_col, "abnormality_score", "latency_score", "anomalous_service_count"],
        ascending=[False, False, False, False],
    )
    for row in ordered.itertuples(index=False):
        if len(selected) >= n:
            break
        services = set(row.services)
        if services - covered_services:
            selected.add(row.traceID)
            covered_services.update(services)
    for row in ordered.itertuples(index=False):
        if len(selected) >= n:
            break
        if row.traceID not in selected:
            selected.add(row.traceID)
    return selected


def choose_anomalous_ids(meta: pd.DataFrame, budget: float, policy: str, seed: int) -> set[str]:
    if meta.empty:
        return set()
    if budget >= 1.0:
        return set(meta["traceID"].tolist())
    if policy == "random":
        n = capacity(meta, budget)
        return set(meta["traceID"].sample(n=n, replace=False, random_state=seed).tolist())
    if policy == "latency_topk":
        return take_sorted(meta, budget, ["trace_duration", "endTime"], [False, False])
    if policy == "abnormality_topk":
        return take_sorted(
            meta,
            budget,
            ["abnormal_span_count", "abnormal_excess_sum", "trace_duration", "anomalous_service_count"],
            [False, False, False, False],
        )
    if policy == "service_aware_abnormality":
        return select_coverage_aware(meta, budget, score_col="abnormality_score")
    if policy == "early_window":
        return take_sorted(meta, budget, ["endTime", "trace_duration"], [True, False])
    if policy == "late_window":
        return take_sorted(meta, budget, ["endTime", "trace_duration"], [False, False])
    if policy == "latency_service_diversity":
        return select_service_diversity(meta, budget, score_col="latency_score")
    if policy == "abnormality_service_diversity":
        return select_service_diversity(meta, budget, score_col="abnormality_score")
    if policy == "coverage_aware":
        return select_coverage_aware(meta, budget, score_col="coverage_score")
    if policy == "hybrid_abnormality_latency":
        return take_sorted(meta, budget, ["hybrid_score", "trace_duration"], [False, False])
    raise ValueError(f"Unknown policy: {policy}")


def evaluate_case_all(case: CaseInfo) -> list[dict]:
    case_dir = Path(case.path)
    inject_time_us = int((case_dir / "inject_time.txt").read_text().strip()) * 1_000_000
    df = pd.read_csv(
        case_dir / "traces.csv",
        usecols=["traceID", "serviceName", "methodName", "operationName", "startTime", "duration"],
    )
    data, meta, normal_trace_ids = prepare_trace_meta(df, inject_time_us)
    full_trace_count = int(data["traceID"].nunique())
    full_anomalous_trace_count = int(len(meta))

    records: list[dict] = []
    for budget in BUDGETS:
        for seed in RANDOM_SEEDS:
            kept_anomalous = choose_anomalous_ids(meta, budget, "random", seed)
            records.append(evaluate_selection(case, df, meta, normal_trace_ids, inject_time_us, budget, "random", seed, kept_anomalous, full_trace_count, full_anomalous_trace_count))

        for policy in POLICIES:
            if policy == "random":
                continue
            kept_anomalous = choose_anomalous_ids(meta, budget, policy, DETERMINISTIC_SEED_LABEL)
            records.append(
                evaluate_selection(
                    case,
                    df,
                    meta,
                    normal_trace_ids,
                    inject_time_us,
                    budget,
                    policy,
                    DETERMINISTIC_SEED_LABEL,
                    kept_anomalous,
                    full_trace_count,
                    full_anomalous_trace_count,
                )
            )
    return records


def evaluate_selection(
    case: CaseInfo,
    df: pd.DataFrame,
    meta: pd.DataFrame,
    normal_trace_ids: set[str],
    inject_time_us: int,
    budget: float,
    policy: str,
    seed: int,
    kept_anomalous: set[str],
    full_trace_count: int,
    full_anomalous_trace_count: int,
) -> dict:
    kept_ids = normal_trace_ids | kept_anomalous
    sampled = df[df["traceID"].isin(kept_ids)]
    ranked_services = trace_budget_rank_services(sampled, inject_time_us)
    kept_anomalous_span_count = int(
        meta.loc[meta["traceID"].isin(kept_anomalous), "anomalous_span_count"].sum()
    )
    full_anomalous_span_count = int(meta["anomalous_span_count"].sum())
    return {
        "dataset": case.dataset,
        "service": case.service,
        "fault": case.fault,
        "instance": case.instance,
        "budget": budget,
        "policy": policy,
        "seed": seed,
        "top1": float(case.service in ranked_services[:1]),
        "top3": float(case.service in ranked_services[:3]),
        "top5": float(case.service in ranked_services[:5]),
        "avg5": avg_at_5_from_services(ranked_services, case.service),
        "full_trace_count": full_trace_count,
        "full_anomalous_trace_count": full_anomalous_trace_count,
        "kept_trace_count": int(len(kept_ids)),
        "kept_anomalous_trace_count": int(len(kept_anomalous)),
        "full_anomalous_span_count": full_anomalous_span_count,
        "kept_anomalous_span_count": kept_anomalous_span_count,
        "total_reduction": 1.0 - (len(kept_ids) / full_trace_count) if full_trace_count else 0.0,
        "post_injection_trace_reduction": 1.0 - (len(kept_anomalous) / full_anomalous_trace_count)
        if full_anomalous_trace_count
        else 0.0,
        "post_injection_span_reduction": 1.0 - (kept_anomalous_span_count / full_anomalous_span_count)
        if full_anomalous_span_count
        else 0.0,
        "top5_services": ranked_services[:5],
    }


def bootstrap_delta_ci(case_policy: pd.DataFrame, iterations: int = 2000, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    key_cols = ["dataset", "budget", "policy"]
    for (dataset, budget, policy), group in case_policy.groupby(key_cols):
        if policy == "random":
            continue
        random_group = case_policy[
            (case_policy["dataset"] == dataset)
            & (case_policy["budget"] == budget)
            & (case_policy["policy"] == "random")
        ]
        merged = group.merge(
            random_group[
                ["dataset", "service", "fault", "instance", "budget", "avg5_case_policy"]
            ].rename(columns={"avg5_case_policy": "avg5_random_case"}),
            on=["dataset", "service", "fault", "instance", "budget"],
            how="inner",
        )
        if merged.empty:
            continue
        deltas = (merged["avg5_case_policy"] - merged["avg5_random_case"]).to_numpy()
        sampled = rng.choice(deltas, size=(iterations, len(deltas)), replace=True).mean(axis=1)
        rows.append(
            {
                "dataset": dataset,
                "budget": budget,
                "policy": policy,
                "cases": len(deltas),
                "delta_avg5": float(deltas.mean()),
                "ci95_low": float(np.quantile(sampled, 0.025)),
                "ci95_high": float(np.quantile(sampled, 0.975)),
                "wins": int((deltas > 0).sum()),
                "losses": int((deltas < 0).sum()),
                "ties": int((deltas == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    case_policy = (
        results.groupby(["dataset", "service", "fault", "instance", "budget", "policy"], as_index=False)
        .agg(
            avg5_case_policy=("avg5", "mean"),
            top1_case_policy=("top1", "mean"),
            top3_case_policy=("top3", "mean"),
            top5_case_policy=("top5", "mean"),
            total_reduction=("total_reduction", "mean"),
            post_injection_trace_reduction=("post_injection_trace_reduction", "mean"),
            post_injection_span_reduction=("post_injection_span_reduction", "mean"),
        )
    )
    overall = (
        case_policy.groupby(["policy", "dataset", "budget"], as_index=False)
        .agg(
            cases=("avg5_case_policy", "size"),
            top1=("top1_case_policy", "mean"),
            top3=("top3_case_policy", "mean"),
            top5=("top5_case_policy", "mean"),
            avg5=("avg5_case_policy", "mean"),
            total_reduction=("total_reduction", "mean"),
            post_injection_trace_reduction=("post_injection_trace_reduction", "mean"),
            post_injection_span_reduction=("post_injection_span_reduction", "mean"),
        )
    )
    by_fault = (
        case_policy.groupby(["policy", "dataset", "fault", "budget"], as_index=False)
        .agg(
            cases=("avg5_case_policy", "size"),
            top1=("top1_case_policy", "mean"),
            top3=("top3_case_policy", "mean"),
            top5=("top5_case_policy", "mean"),
            avg5=("avg5_case_policy", "mean"),
            total_reduction=("total_reduction", "mean"),
            post_injection_trace_reduction=("post_injection_trace_reduction", "mean"),
            post_injection_span_reduction=("post_injection_span_reduction", "mean"),
        )
    )
    pivot = overall.pivot_table(index=["dataset", "budget"], columns="policy", values="avg5").reset_index()
    comparison_rows: list[dict] = []
    for row in pivot.itertuples(index=False):
        dataset = row.dataset
        budget = row.budget
        random_value = getattr(row, "random")
        for policy in POLICIES:
            value = getattr(row, policy)
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "policy": policy,
                    "avg5": float(value),
                    "avg5_random": float(random_value),
                    "delta_vs_random": float(value - random_value),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    bootstrap = bootstrap_delta_ci(case_policy)
    return {
        "case_policy": case_policy,
        "overall": overall,
        "by_fault": by_fault,
        "comparison": comparison,
        "bootstrap": bootstrap,
    }


def write_md_table(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(comparison: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    policies = [p for p in POLICIES if p != "random"]
    display_policy = {
        "latency_topk": "latency",
        "abnormality_topk": "abnormality",
        "service_aware_abnormality": "service-aware\nabnormality",
        "early_window": "early",
        "late_window": "late",
        "latency_service_diversity": "latency+\ndiversity",
        "abnormality_service_diversity": "abnormality+\ndiversity",
        "coverage_aware": "coverage",
        "hybrid_abnormality_latency": "hybrid",
    }

    for budget in [0.10, 0.25, 0.50]:
        sub = comparison[(comparison["budget"] == budget) & (comparison["policy"].isin(policies))]
        matrix = (
            sub.pivot_table(index="policy", columns="dataset", values="delta_vs_random")
            .reindex(policies)
            .reindex(columns=DATASETS)
        )
        fig, ax = plt.subplots(figsize=(8.4, 5.3))
        im = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-0.18, vmax=0.18)
        ax.set_aspect("auto")
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels(DATASETS)
        ax.set_yticks(range(len(policies)))
        ax.set_yticklabels([display_policy[p] for p in policies])
        ax.set_title(f"Stage A policy delta vs random at {100 * budget:.0f}% budget")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Delta Avg@5")
        fig.tight_layout()
        fig.savefig(FIGURES / f"stage_a_policy_delta_heatmap_{int(100 * budget)}pct.png", dpi=220, bbox_inches="tight")
        fig.savefig(FIGURES / f"stage_a_policy_delta_heatmap_{int(100 * budget)}pct.pdf", bbox_inches="tight")
        plt.close(fig)

    focus = bootstrap[(bootstrap["budget"].isin([0.10, 0.25, 0.50])) & (bootstrap["policy"].isin(policies))]
    focus = focus.sort_values(["dataset", "budget", "delta_avg5"], ascending=[True, True, False])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = focus[focus["dataset"] == dataset].copy()
        labels = [f"{row.policy}\n{100 * row.budget:.0f}%" for row in sub.itertuples()]
        y = np.arange(len(labels))
        ax.barh(y, sub["delta_avg5"], color=["#4c78a8" if v >= 0 else "#d95f02" for v in sub["delta_avg5"]])
        ax.errorbar(
            sub["delta_avg5"],
            y,
            xerr=[sub["delta_avg5"] - sub["ci95_low"], sub["ci95_high"] - sub["delta_avg5"]],
            fmt="none",
            ecolor="black",
            capsize=2,
            linewidth=0.8,
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(dataset)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    axes[0].set_xlabel("Delta Avg@5 vs random")
    axes[1].set_xlabel("Delta Avg@5 vs random")
    fig.suptitle("Expanded Stage A policies: bootstrap deltas vs random", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage_a_policy_bootstrap_deltas.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / "stage_a_policy_bootstrap_deltas.pdf", bbox_inches="tight")
    plt.close(fig)


def make_tables_and_note(summaries: dict[str, pd.DataFrame], cases: list[CaseInfo]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    comparison = summaries["comparison"]
    bootstrap = summaries["bootstrap"]
    overall = summaries["overall"]

    focus = comparison[comparison["budget"].isin([0.10, 0.25, 0.50])].copy()
    focus = focus.sort_values(["dataset", "budget", "delta_vs_random"], ascending=[True, True, False])
    focus["budget"] = focus["budget"].map(lambda value: f"{100 * value:.0f}%")
    for col in ["avg5", "avg5_random", "delta_vs_random"]:
        focus[col] = focus[col].map(lambda value: f"{value:.3f}")
    write_md_table(
        focus[["dataset", "budget", "policy", "avg5", "avg5_random", "delta_vs_random"]],
        TABLES / "stage_a_policy_expansion_summary.md",
    )

    boot = bootstrap[bootstrap["budget"].isin([0.10, 0.25, 0.50])].copy()
    boot = boot.sort_values(["dataset", "budget", "delta_avg5"], ascending=[True, True, False])
    boot["budget"] = boot["budget"].map(lambda value: f"{100 * value:.0f}%")
    for col in ["delta_avg5", "ci95_low", "ci95_high"]:
        boot[col] = boot[col].map(lambda value: f"{value:.3f}")
    write_md_table(
        boot[["dataset", "budget", "policy", "delta_avg5", "ci95_low", "ci95_high", "wins", "losses", "ties"]],
        TABLES / "stage_a_policy_expansion_bootstrap.md",
    )

    best_rows = []
    for (dataset, budget), group in comparison[comparison["policy"] != "random"].groupby(["dataset", "budget"]):
        row = group.sort_values("delta_vs_random", ascending=False).iloc[0]
        best_rows.append(row.to_dict())
    best = pd.DataFrame(best_rows)
    lines = [
        "# Stage A Policy Expansion Result Note (2026-06-02)",
        "",
        "## Scope",
        "",
        f"- Usable cases: {len(cases)} total; 90 per system.",
        f"- Budgets: {', '.join(f'{100 * b:.0f}%' for b in BUDGETS)} retained post-injection traces.",
        f"- Policies: {', '.join(POLICIES)}.",
        f"- Random retention repeated across {len(RANDOM_SEEDS)} seeds; deterministic policies evaluated once per case/budget.",
        "- Pre-injection traces are retained in full.",
        "- The fixed lightweight Stage A ranker is unchanged.",
        "",
        "## Best non-random policy by dataset/budget",
        "",
    ]
    for row in best.sort_values(["dataset", "budget"]).itertuples(index=False):
        lines.append(
            f"- {row.dataset} {100 * row.budget:.0f}%: {row.policy} "
            f"Avg@5={row.avg5:.3f}, random={row.avg5_random:.3f}, delta={row.delta_vs_random:+.3f}."
        )

    random_overall = overall[overall["policy"] == "random"]
    lines.extend(["", "## Random baseline means", ""])
    for row in random_overall.sort_values(["dataset", "budget"]).itertuples(index=False):
        lines.append(f"- {row.dataset} {100 * row.budget:.0f}%: Avg@5={row.avg5:.3f}.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This experiment strengthens RQ2 by comparing random retention with latency, abnormality, timing, coverage, diversity, and hybrid trace-selection policies.",
            "",
            "The paper-facing interpretation should depend on the sign and consistency of the deltas rather than on any single best policy. If the best policy changes by dataset or budget, the result supports the claim that one-dimensional intuitive retention rules are not reliably dominant.",
            "",
            "## Generated artifacts",
            "",
            "- `results/stage_a_policy_expansion_rv2026/case_results.csv`",
            "- `results/stage_a_policy_expansion_rv2026/case_policy_results.csv`",
            "- `results/stage_a_policy_expansion_rv2026/summary_overall.csv`",
            "- `results/stage_a_policy_expansion_rv2026/summary_by_fault.csv`",
            "- `results/stage_a_policy_expansion_rv2026/policy_vs_random_delta.csv`",
            "- `results/stage_a_policy_expansion_rv2026/policy_bootstrap_delta.csv`",
            "- `figures/stage_a_policy_delta_heatmap_10pct.png` and `.pdf`",
            "- `figures/stage_a_policy_delta_heatmap_25pct.png` and `.pdf`",
            "- `figures/stage_a_policy_delta_heatmap_50pct.png` and `.pdf`",
            "- `figures/stage_a_policy_bootstrap_deltas.png` and `.pdf`",
            "- `tables/stage_a_policy_expansion_summary.md`",
            "- `tables/stage_a_policy_expansion_bootstrap.md`",
            "",
        ]
    )
    (NOTES / "Stage_A_Policy_Expansion_Result_Note_2026-06-02.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = discover_cases()
    if len(cases) != 180:
        print(f"WARNING: expected 180 usable cases, found {len(cases)}")

    records: list[dict] = []
    failures: list[dict] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_case_all, case): case for case in cases}
        for index, future in enumerate(as_completed(futures), start=1):
            case = futures[future]
            try:
                records.extend(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "dataset": case.dataset,
                        "service": case.service,
                        "fault": case.fault,
                        "instance": case.instance,
                        "error": str(exc),
                    }
                )
                print(f"FAILED {case.dataset} {case.service} {case.fault} {case.instance}: {exc}")
            if index % 10 == 0:
                print(f"completed_cases={index}/{len(cases)} records={len(records)} failures={len(failures)}")

    results = pd.DataFrame(records).sort_values(["dataset", "fault", "service", "instance", "budget", "policy", "seed"])
    results.to_csv(OUT_DIR / "case_results.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)

    summaries = summarize(results)
    summaries["case_policy"].to_csv(OUT_DIR / "case_policy_results.csv", index=False)
    summaries["overall"].to_csv(OUT_DIR / "summary_overall.csv", index=False)
    summaries["by_fault"].to_csv(OUT_DIR / "summary_by_fault.csv", index=False)
    summaries["comparison"].to_csv(OUT_DIR / "policy_vs_random_delta.csv", index=False)
    summaries["bootstrap"].to_csv(OUT_DIR / "policy_bootstrap_delta.csv", index=False)

    config = {
        "data_root": str(DATA_ROOT),
        "datasets": DATASETS,
        "budgets": BUDGETS,
        "policies": POLICIES,
        "random_seeds": RANDOM_SEEDS,
        "max_workers": MAX_WORKERS,
        "usable_cases": len(cases),
        "records": len(records),
        "failures": len(failures),
        "design_note": "Expanded Stage A trace-retention policies under fixed trace-only ranker. Random is averaged across seeds; deterministic policies are compared against case-level random means.",
    }
    (OUT_DIR / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    make_figures(summaries["comparison"], summaries["bootstrap"])
    make_tables_and_note(summaries, cases)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
