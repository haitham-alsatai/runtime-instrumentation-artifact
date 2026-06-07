from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "RE2"
OUT_DIR = ROOT / "results" / "stage_a" / "stage_a_dense_budget"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
NOTES = ROOT / "docs" / "generated"

DATASETS = ["RE2-OB", "RE2-TT"]
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00]
SEEDS = list(range(2026, 2036))
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


def keep_trace_ids(trace_end_times: pd.Series, inject_time_us: int, budget: float, seed: int) -> set[str]:
    normal_ids = trace_end_times[trace_end_times < inject_time_us].index.to_list()
    anomalous_ids = trace_end_times[trace_end_times >= inject_time_us].index.to_list()

    if not anomalous_ids:
        return set(normal_ids)
    if budget >= 1.0:
        return set(normal_ids) | set(anomalous_ids)

    kept_count = max(1, round(len(anomalous_ids) * budget))
    draw_seed = seed + int(round(budget * 10_000))
    kept_anomalous = (
        pd.Series(anomalous_ids)
        .sample(n=kept_count, replace=False, random_state=draw_seed)
        .tolist()
    )
    return set(normal_ids) | set(kept_anomalous)


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


def evaluate_case_all(case: CaseInfo) -> list[dict]:
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
    full_anom = int((trace_end_times >= inject_time_us).sum())

    records: list[dict] = []
    for seed in SEEDS:
        for budget in BUDGETS:
            kept_ids = keep_trace_ids(trace_end_times, inject_time_us, budget, seed)
            sampled = df[df["traceID"].isin(kept_ids)]
            ranked_services = trace_budget_rank_services(sampled, inject_time_us)
            kept_anom = int(sum(1 for trace_id in kept_ids if trace_end_times.loc[trace_id] >= inject_time_us))
            records.append(
                {
                    "dataset": case.dataset,
                    "service": case.service,
                    "fault": case.fault,
                    "instance": case.instance,
                    "seed": seed,
                    "budget": budget,
                    "top1": float(case.service in ranked_services[:1]),
                    "top3": float(case.service in ranked_services[:3]),
                    "top5": float(case.service in ranked_services[:5]),
                    "avg5": avg_at_5_from_services(ranked_services, case.service),
                    "full_trace_count": int(len(trace_end_times)),
                    "full_anomalous_trace_count": full_anom,
                    "kept_trace_count": int(len(kept_ids)),
                    "kept_anomalous_trace_count": kept_anom,
                    "total_reduction": 1.0 - (len(kept_ids) / len(trace_end_times)),
                    "post_injection_reduction": 1.0 - (kept_anom / full_anom) if full_anom else 0.0,
                    "top5_services": ranked_services[:5],
                }
            )
    return records


def write_md_table(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    seed_overall = (
        results.groupby(["dataset", "budget", "seed"], as_index=False)
        .agg(
            cases=("avg5", "size"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            avg5=("avg5", "mean"),
            total_reduction=("total_reduction", "mean"),
            post_injection_reduction=("post_injection_reduction", "mean"),
        )
    )

    overall = (
        seed_overall.groupby(["dataset", "budget"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            cases_per_seed=("cases", "mean"),
            top1_mean=("top1", "mean"),
            top3_mean=("top3", "mean"),
            top5_mean=("top5", "mean"),
            avg5_mean=("avg5", "mean"),
            avg5_std=("avg5", "std"),
            total_reduction_mean=("total_reduction", "mean"),
            post_injection_reduction_mean=("post_injection_reduction", "mean"),
        )
    )
    overall["avg5_ci95"] = 1.96 * overall["avg5_std"].fillna(0) / (overall["seeds"] ** 0.5)

    seed_by_fault = (
        results.groupby(["dataset", "fault", "budget", "seed"], as_index=False)
        .agg(
            cases=("avg5", "size"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            avg5=("avg5", "mean"),
            total_reduction=("total_reduction", "mean"),
            post_injection_reduction=("post_injection_reduction", "mean"),
        )
    )
    by_fault = (
        seed_by_fault.groupby(["dataset", "fault", "budget"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            cases_per_seed=("cases", "mean"),
            top1_mean=("top1", "mean"),
            top3_mean=("top3", "mean"),
            top5_mean=("top5", "mean"),
            avg5_mean=("avg5", "mean"),
            avg5_std=("avg5", "std"),
            total_reduction_mean=("total_reduction", "mean"),
            post_injection_reduction_mean=("post_injection_reduction", "mean"),
        )
    )
    by_fault["avg5_ci95"] = 1.96 * by_fault["avg5_std"].fillna(0) / (by_fault["seeds"] ** 0.5)

    min_rows: list[dict] = []
    for (dataset, fault), group in by_fault.groupby(["dataset", "fault"]):
        ordered = group.sort_values("budget")
        full_avg5 = float(ordered.loc[ordered["budget"] == 1.0, "avg5_mean"].iloc[0])
        target = 0.95 * full_avg5
        eligible = ordered[ordered["avg5_mean"] >= target]
        low = float(ordered.loc[ordered["budget"] == 0.05, "avg5_mean"].iloc[0])
        ten = float(ordered.loc[ordered["budget"] == 0.10, "avg5_mean"].iloc[0])
        min_rows.append(
            {
                "dataset": dataset,
                "fault": fault,
                "full_budget_avg5_mean": full_avg5,
                "target_95pct": target,
                "min_budget_for_95pct_full": float(eligible["budget"].iloc[0]) if not eligible.empty else None,
                "gain_05_to_full": full_avg5 - low,
                "gain_10_to_full": full_avg5 - ten,
            }
        )
    min_budget = pd.DataFrame(min_rows).sort_values(["dataset", "gain_05_to_full"], ascending=[True, False])

    return {
        "seed_overall": seed_overall,
        "overall": overall,
        "seed_by_fault": seed_by_fault,
        "by_fault": by_fault,
        "min_budget": min_budget,
    }


def make_figures(overall: pd.DataFrame, by_fault: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = by_fault[by_fault["dataset"] == dataset]
        for fault, group in subset.groupby("fault"):
            group = group.sort_values("budget")
            ax.plot(group["budget"] * 100, group["avg5_mean"], marker="o", linewidth=1.4, alpha=0.78, label=fault)

        ov = overall[overall["dataset"] == dataset].sort_values("budget")
        ax.errorbar(
            ov["budget"] * 100,
            ov["avg5_mean"],
            yerr=ov["avg5_ci95"],
            color="black",
            marker="s",
            linewidth=2.6,
            capsize=3,
            label="overall",
        )
        ax.set_title(dataset)
        ax.set_xlabel("Retained post-injection trace budget (%)")
        ax.set_xticks([5, 10, 15, 20, 25, 35, 50, 75, 100])
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_ylim(0.32, 1.04)

    axes[0].set_ylabel("Avg@5")
    axes[1].legend(loc="lower right", fontsize=8, frameon=True)
    fig.suptitle("Dense Stage A budget sweep across 10 random seeds", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage_a_dense_budget_curves.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / "stage_a_dense_budget_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def make_tables(overall: pd.DataFrame, min_budget: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    table = overall.copy()
    table = table[table["budget"].isin([0.05, 0.10, 0.25, 0.50, 1.00])].copy()
    table["budget"] = table["budget"].map(lambda value: f"{100 * value:.0f}%")
    for col in [
        "top1_mean",
        "top3_mean",
        "top5_mean",
        "avg5_mean",
        "avg5_ci95",
        "total_reduction_mean",
        "post_injection_reduction_mean",
    ]:
        table[col] = table[col].map(lambda value: f"{value:.3f}")
    write_md_table(
        table[
            [
                "dataset",
                "budget",
                "seeds",
                "cases_per_seed",
                "top1_mean",
                "top3_mean",
                "top5_mean",
                "avg5_mean",
                "avg5_ci95",
                "total_reduction_mean",
                "post_injection_reduction_mean",
            ]
        ],
        TABLES / "stage_a_dense_budget_summary.md",
    )

    min_table = min_budget.copy()
    min_table["min_budget_for_95pct_full"] = min_table["min_budget_for_95pct_full"].map(
        lambda value: "" if pd.isna(value) else f"{100 * value:.0f}%"
    )
    for col in ["full_budget_avg5_mean", "target_95pct", "gain_05_to_full", "gain_10_to_full"]:
        min_table[col] = min_table[col].map(lambda value: f"{value:.3f}")
    write_md_table(min_table, TABLES / "stage_a_dense_min_budget_by_fault.md")


def write_note(cases: list[CaseInfo], summaries: dict[str, pd.DataFrame]) -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    overall = summaries["overall"]
    min_budget = summaries["min_budget"]

    lines = [
        "# Dense Stage A Budget Sweep Result Note (2026-06-02)",
        "",
        "## Scope",
        "",
        f"- Datasets: {', '.join(DATASETS)}.",
        f"- Usable cases: {len(cases)} total; 90 per system.",
        f"- Budgets: {', '.join(f'{100 * b:.0f}%' for b in BUDGETS)} retained post-injection traces.",
        f"- Random seeds: {len(SEEDS)} seeds ({SEEDS[0]}-{SEEDS[-1]}).",
        "- Pre-injection traces are retained in full, matching the submitted Stage A design.",
        "- The fixed lightweight trace-only ranker is unchanged.",
        "",
        "## Headline overall Avg@5",
        "",
    ]
    for dataset in DATASETS:
        group = overall[overall["dataset"] == dataset].sort_values("budget")
        values = ", ".join(f"{100 * row.budget:.0f}%={row.avg5_mean:.3f}" for row in group.itertuples())
        lines.append(f"- {dataset}: {values}.")

    lines.extend(
        [
            "",
            "## Minimum budget for 95% of full-budget Avg@5",
            "",
        ]
    )
    for row in min_budget.itertuples(index=False):
        min_budget_value = "" if pd.isna(row.min_budget_for_95pct_full) else f"{100 * row.min_budget_for_95pct_full:.0f}%"
        lines.append(
            f"- {row.dataset} {row.fault}: {min_budget_value} "
            f"(full Avg@5={row.full_budget_avg5_mean:.3f}, gain 5% to full={row.gain_05_to_full:.3f})."
        )

    lines.extend(
        [
            "",
            "## Generated artifacts",
            "",
            "- `results/stage_a_dense_budget_rv2026/case_seed_results.csv`",
            "- `results/stage_a_dense_budget_rv2026/summary_overall.csv`",
            "- `results/stage_a_dense_budget_rv2026/summary_by_fault.csv`",
            "- `results/stage_a_dense_budget_rv2026/minimum_budget_by_fault.csv`",
            "- `figures/stage_a_dense_budget_curves.png` and `.pdf`",
            "- `tables/stage_a_dense_budget_summary.md`",
            "- `tables/stage_a_dense_min_budget_by_fault.md`",
            "",
        ]
    )
    (NOTES / "Stage_A_Dense_Budget_Result_Note_2026-06-02.md").write_text("\n".join(lines), encoding="utf-8")


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

    results = pd.DataFrame(records).sort_values(["dataset", "fault", "service", "instance", "seed", "budget"])
    results.to_csv(OUT_DIR / "case_seed_results.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)

    summaries = summarize(results)
    summaries["seed_overall"].to_csv(OUT_DIR / "summary_seed_overall.csv", index=False)
    summaries["overall"].to_csv(OUT_DIR / "summary_overall.csv", index=False)
    summaries["seed_by_fault"].to_csv(OUT_DIR / "summary_seed_by_fault.csv", index=False)
    summaries["by_fault"].to_csv(OUT_DIR / "summary_by_fault.csv", index=False)
    summaries["min_budget"].to_csv(OUT_DIR / "minimum_budget_by_fault.csv", index=False)

    config = {
        "data_root": str(DATA_ROOT),
        "datasets": DATASETS,
        "budgets": BUDGETS,
        "seeds": SEEDS,
        "max_workers": MAX_WORKERS,
        "usable_cases": len(cases),
        "design_note": "Dense Stage A random trace-retention expansion. Pre-injection traces retained in full; post-injection traces sampled by budget.",
    }
    (OUT_DIR / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    make_figures(summaries["overall"], summaries["by_fault"])
    make_tables(summaries["overall"], summaries["min_budget"])
    write_note(cases, summaries)


if __name__ == "__main__":
    main()
