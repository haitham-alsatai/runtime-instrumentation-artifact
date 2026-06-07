#!/usr/bin/env python3
"""Compact RV 2026 ML/AI evidence-scoring baseline for GAIA Stage B.

The experiment keeps the Stage B post-alert allocation setting fixed and asks:
after a telemetry context is selected, can a learned service-level evidence
scorer improve over the hand-designed deterministic scorer?

Decision-time safety:
- Case folds are disjoint.
- Training labels are used only for training the service scorer.
- Test-case root labels are used only for evaluation.
- Main ML feature set does not include an explicit "is alerted/root service"
  feature, so the learned baseline cannot win by simply ranking the anchor first.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = EXT_ROOT / "results" / "stage_b" / "gaia_integrated_experiment"
OUTPUT_DIR = EXT_ROOT / "results" / "stage_b" / "gaia_ml_evidence_scorer"
FIGURE_DIR = EXT_ROOT / "figures"
TABLE_DIR = EXT_ROOT / "tables"
NOTE_DIR = EXT_ROOT / "docs" / "generated"

TRACE_ENDPOINT_FILE = INPUT_DIR / "trace_service_minute_endpoint.csv"
METRIC_SERVICE_FILE = INPUT_DIR / "metric_service_minute.csv"
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
N_SPLITS = 5
BASE_SEED = 20260603


@dataclass(frozen=True)
class Case:
    case_id: str
    root_service: str
    fault_type: str
    service_family: str
    alert_time: datetime
    full_minutes: int
    full_services: int


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
    seen = set()
    cases = []
    with CASE_RESULTS_COST_FILE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] in seen or row["policy"] != "full" or row["budget"] != "1.0":
                continue
            seen.add(row["case_id"])
            timestamp = int(row["case_id"].split("_")[-1])
            alert_time = datetime.fromtimestamp(timestamp, timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
            cases.append(
                Case(
                    case_id=row["case_id"],
                    root_service=row["root_service"],
                    fault_type=row["fault_type"],
                    service_family=row["service_family"],
                    alert_time=alert_time,
                    full_minutes=int(float(row["full_minutes"])),
                    full_services=int(float(row["full_services"])),
                )
            )
    return cases


def service_order_from_anchor(anchor: str) -> list[str]:
    seen = {anchor}
    ordered = [anchor]
    for service in NEIGHBORS.get(anchor, []):
        if service not in seen:
            ordered.append(service)
            seen.add(service)
    for service in SERVICE_ORDER:
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


def aggregate_trace_selected(service: str, minutes: list[datetime], endpoints: list[str], trace_endpoint) -> dict:
    total = {"trace_rows": 0.0, "latency_sum": 0.0, "error_rows": 0.0, "trace_bytes": 0.0, "max_latency": 0.0}
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
            total["max_latency"] = max(total["max_latency"], values["max_latency_ms"])
    if total["trace_rows"] > 0:
        total["mean_latency"] = total["latency_sum"] / total["trace_rows"]
        total["error_rate"] = total["error_rows"] / total["trace_rows"]
    else:
        total["mean_latency"] = 0.0
        total["error_rate"] = 0.0
    return total


def aggregate_metric(service: str, minutes: list[datetime], metric_service) -> tuple[dict, dict]:
    observations = {name: [] for name in ("cpu", "memory", "net_in_err", "net_out_err")}
    for minute in minutes:
        values = metric_service[service].get(minute)
        if not values:
            continue
        for name in observations:
            observations[name].append(values[name])
    means = {name: (mean(values) if values else 0.0) for name, values in observations.items()}
    stds = {}
    for name, values in observations.items():
        if len(values) <= 1:
            stds[name] = 0.0
        else:
            mu = means[name]
            stds[name] = math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))
    return means, stds


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


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1e-6)


def build_feature_rows(cases: list[Case], trace_endpoint, metric_service) -> pd.DataFrame:
    records: list[dict] = []
    for index, case in enumerate(cases, start=1):
        for budget in BUDGETS:
            service_order = service_order_from_anchor(case.root_service)
            keep_services = max(1, math.ceil(len(SERVICE_ORDER) * budget))
            selected_services = set(service_order[:keep_services])
            selected_minutes = adaptive_minutes(case, budget)
            base_minutes = baseline_minutes(selected_minutes)
            order_index = {service: pos for pos, service in enumerate(service_order)}

            for service in SERVICE_ORDER:
                selected = service in selected_services
                endpoints = adaptive_endpoints(service, selected_minutes, budget, trace_endpoint) if selected else []
                obs_trace = aggregate_trace_selected(service, selected_minutes, endpoints, trace_endpoint) if selected else {}
                base_trace = aggregate_trace_selected(service, base_minutes, endpoints, trace_endpoint) if selected else {}
                obs_metric_means, _obs_metric_stds = aggregate_metric(service, selected_minutes, metric_service) if selected else ({}, {})
                base_metric_means, base_metric_stds = aggregate_metric(service, base_minutes, metric_service) if selected else ({}, {})

                if selected:
                    det_trace = trace_score(obs_trace, base_trace)
                    det_metric = metric_score(obs_metric_means, base_metric_means, base_metric_stds)
                    deterministic_score = det_trace + 0.8 * det_metric
                else:
                    obs_trace = {
                        "trace_rows": 0.0,
                        "latency_sum": 0.0,
                        "error_rows": 0.0,
                        "trace_bytes": 0.0,
                        "max_latency": 0.0,
                        "mean_latency": 0.0,
                        "error_rate": 0.0,
                    }
                    base_trace = dict(obs_trace)
                    obs_metric_means = {name: 0.0 for name in ("cpu", "memory", "net_in_err", "net_out_err")}
                    base_metric_means = dict(obs_metric_means)
                    base_metric_stds = dict(obs_metric_means)
                    det_trace = det_metric = deterministic_score = 0.0

                row = {
                    "case_id": case.case_id,
                    "root_service": case.root_service,
                    "service": service,
                    "fault_type": case.fault_type,
                    "service_family": case.service_family,
                    "budget": budget,
                    "selected": float(selected),
                    "label": int(service == case.root_service),
                    "service_order_position": order_index[service],
                    "deterministic_score": deterministic_score,
                    "deterministic_trace_score": det_trace,
                    "deterministic_metric_score": det_metric,
                    "trace_rows_obs": obs_trace["trace_rows"],
                    "trace_rows_base": base_trace["trace_rows"],
                    "trace_rows_ratio": safe_ratio(obs_trace["trace_rows"], base_trace["trace_rows"]),
                    "mean_latency_obs": obs_trace["mean_latency"],
                    "mean_latency_base": base_trace["mean_latency"],
                    "mean_latency_delta_ratio": safe_ratio(obs_trace["mean_latency"] - base_trace["mean_latency"], base_trace["mean_latency"]),
                    "max_latency_obs": obs_trace["max_latency"],
                    "max_latency_base": base_trace["max_latency"],
                    "error_rate_obs": obs_trace["error_rate"],
                    "error_rate_base": base_trace["error_rate"],
                    "error_rate_delta": obs_trace["error_rate"] - base_trace["error_rate"],
                    "trace_byte_obs": obs_trace["trace_bytes"],
                    "trace_byte_base": base_trace["trace_bytes"],
                    "endpoint_count": float(len(endpoints)),
                }
                for metric in ("cpu", "memory", "net_in_err", "net_out_err"):
                    row[f"{metric}_obs"] = obs_metric_means[metric]
                    row[f"{metric}_base"] = base_metric_means[metric]
                    row[f"{metric}_delta"] = obs_metric_means[metric] - base_metric_means[metric]
                    row[f"{metric}_abs_z"] = abs(obs_metric_means[metric] - base_metric_means[metric]) / max(base_metric_stds[metric], 1e-6)
                records.append(row)
        if index % 100 == 0:
            print(f"feature_cases={index}/{len(cases)} rows={len(records)}")
    return pd.DataFrame(records)


def avg_at_5(rank: int) -> float:
    return sum(1 if rank <= cutoff else 0 for cutoff in range(1, 6)) / 5.0


def rank_from_scores(group: pd.DataFrame, score_col: str) -> int:
    ranked = group.sort_values([score_col, "selected", "service_order_position"], ascending=[False, False, True])
    services = ranked["service"].tolist()
    root = group["root_service"].iloc[0]
    return services.index(root) + 1


def case_metrics(case_id: str, budget: float, fold: int, model: str, rank: int) -> dict:
    return {
        "case_id": case_id,
        "budget": budget,
        "fold": fold,
        "model": model,
        "rank": rank,
        "top1": int(rank <= 1),
        "top3": int(rank <= 3),
        "top5": int(rank <= 5),
        "avg_at_5": avg_at_5(rank),
    }


def evaluate_models(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_df = pd.get_dummies(features, columns=["fault_type"], prefix="fault", dtype=float)
    numeric_cols = [
        col
        for col in feature_df.columns
        if col
        not in {
            "case_id",
            "root_service",
            "service",
            "service_family",
            "label",
        }
        and not col.startswith("root_")
    ]
    evidence_cols = [
        col
        for col in numeric_cols
        if col not in {"service_order_position"}
    ]

    case_table = features[["case_id", "root_service", "fault_type"]].drop_duplicates().sort_values("case_id")
    stratify = case_table["fault_type"].astype(str).to_numpy()
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=BASE_SEED)

    models = {
        "logistic_evidence": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=BASE_SEED)),
            ]
        ),
        "random_forest_evidence": RandomForestClassifier(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=BASE_SEED,
            n_jobs=1,
        ),
        "hist_gradient_evidence": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            learning_rate=0.06,
            random_state=BASE_SEED,
        ),
    }

    case_rows: list[dict] = []
    auc_rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(case_table, stratify), start=1):
        train_cases = set(case_table.iloc[train_idx]["case_id"])
        test_cases = set(case_table.iloc[test_idx]["case_id"])
        train = feature_df[feature_df["case_id"].isin(train_cases)].copy()
        test = feature_df[feature_df["case_id"].isin(test_cases)].copy()

        for budget in BUDGETS:
            budget_test = test[test["budget"] == budget].copy()
            for case_id, group in budget_test.groupby("case_id"):
                rank = rank_from_scores(group, "deterministic_score")
                case_rows.append(case_metrics(case_id, budget, fold, "deterministic_formula", rank))

        x_train = train[evidence_cols].to_numpy()
        y_train = train["label"].to_numpy()
        x_test = test[evidence_cols].to_numpy()
        y_test = test["label"].to_numpy()

        for model_name, model in models.items():
            if model_name == "hist_gradient_evidence":
                weights = np.where(y_train == 1, 9.0, 1.0)
                model.fit(x_train, y_train, sample_weight=weights)
            else:
                model.fit(x_train, y_train)
            scores = model.predict_proba(x_test)[:, 1]
            scored = test.copy()
            scored["ml_score"] = scores
            try:
                auc = roc_auc_score(y_test, scores)
            except ValueError:
                auc = float("nan")
            auc_rows.append({"fold": fold, "model": model_name, "row_auc": auc})

            for budget in BUDGETS:
                budget_scored = scored[scored["budget"] == budget].copy()
                for case_id, group in budget_scored.groupby("case_id"):
                    rank = rank_from_scores(group, "ml_score")
                    case_rows.append(case_metrics(case_id, budget, fold, model_name, rank))

    case_results = pd.DataFrame(case_rows)
    summary = (
        case_results.groupby(["model", "budget"], as_index=False)
        .agg(
            cases=("case_id", "nunique"),
            folds=("fold", "nunique"),
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top5=("top5", "mean"),
            avg_at_5=("avg_at_5", "mean"),
        )
        .sort_values(["budget", "avg_at_5"], ascending=[True, False])
    )
    auc_summary = (
        pd.DataFrame(auc_rows)
        .groupby("model", as_index=False)
        .agg(row_auc_mean=("row_auc", "mean"), row_auc_std=("row_auc", "std"))
    )
    return case_results, summary, auc_summary


def write_md_table(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_outputs(features: pd.DataFrame, case_results: pd.DataFrame, summary: pd.DataFrame, auc_summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_DIR.mkdir(parents=True, exist_ok=True)

    features.to_csv(OUTPUT_DIR / "service_feature_rows.csv", index=False)
    case_results.to_csv(OUTPUT_DIR / "case_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary_overall.csv", index=False)
    auc_summary.to_csv(OUTPUT_DIR / "row_auc_summary.csv", index=False)

    table = summary.copy()
    table["budget"] = table["budget"].map(lambda value: f"{100 * value:.0f}%")
    for col in ["top1", "top3", "top5", "avg_at_5"]:
        table[col] = table[col].map(lambda value: f"{value:.3f}")
    write_md_table(table, TABLE_DIR / "gaia_ml_evidence_scorer_summary.md")

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    models = summary["model"].drop_duplicates().tolist()
    x = np.arange(len(BUDGETS))
    width = 0.18
    for idx, model in enumerate(models):
        subset = summary[summary["model"] == model].set_index("budget").reindex(BUDGETS)
        ax.bar(x + (idx - (len(models) - 1) / 2) * width, subset["avg_at_5"], width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{100 * budget:.0f}%" for budget in BUDGETS])
    ax.set_ylabel("Avg@5")
    ax.set_xlabel("Selected telemetry budget")
    ax.set_title("GAIA Stage B learned evidence scorers vs deterministic scorer")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "gaia_ml_evidence_scorer_avg5.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "gaia_ml_evidence_scorer_avg5.pdf", bbox_inches="tight")
    plt.close(fig)

    best = summary.sort_values(["budget", "avg_at_5"], ascending=[True, False]).groupby("budget").head(1)
    lines = [
        "# GAIA ML Evidence Scorer Result Note (2026-06-03)",
        "",
        "## Scope",
        "",
        "- Compact E4 baseline for the RV revision.",
        "- Uses GAIA Stage B service-level trace/metric aggregate evidence.",
        "- Keeps the adaptive post-alert allocation context fixed at 25% and 50% budgets.",
        "- Trains learned service scorers with five case-disjoint folds.",
        "- Main ML features exclude an explicit alerted/root-service indicator.",
        "",
        "## Headline Results",
        "",
    ]
    for row in summary.sort_values(["budget", "avg_at_5"], ascending=[True, False]).itertuples(index=False):
        lines.append(
            f"- {row.model} at {100 * row.budget:.0f}%: Avg@5={row.avg_at_5:.3f}, "
            f"Top1={row.top1:.3f}, Top3={row.top3:.3f}, Top5={row.top5:.3f}."
        )
    lines.extend(["", "## Best model by budget", ""])
    for row in best.itertuples(index=False):
        lines.append(f"- {100 * row.budget:.0f}%: {row.model} with Avg@5={row.avg_at_5:.3f}.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This baseline tests whether a compact learned scorer can replace the hand-designed Stage B service scoring rule. The result should be used cautiously: if ML does not clearly improve over the deterministic formula, that is still valuable evidence that stronger experiments were added without changing the paper into an unrelated ML paper.",
            "",
            "## Generated Artifacts",
            "",
            "- `results/ml_ai_baseline/service_feature_rows.csv`",
            "- `results/ml_ai_baseline/case_results.csv`",
            "- `results/ml_ai_baseline/summary_overall.csv`",
            "- `results/ml_ai_baseline/row_auc_summary.csv`",
            "- `figures/gaia_ml_evidence_scorer_avg5.png` and `.pdf`",
            "- `tables/gaia_ml_evidence_scorer_summary.md`",
            "",
        ]
    )
    (NOTE_DIR / "GAIA_ML_Evidence_Scorer_Result_Note_2026-06-03.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "experiment": "gaia_ml_evidence_scorer_rv2026",
        "created_date": "2026-06-03",
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "budgets": BUDGETS,
        "n_splits": N_SPLITS,
        "base_seed": BASE_SEED,
        "models": [
            "deterministic_formula",
            "logistic_evidence",
            "random_forest_evidence",
            "hist_gradient_evidence",
        ],
        "decision_time_safety": [
            "Case folds are disjoint.",
            "Training labels are used only for training.",
            "Test labels are used only for evaluation.",
            "Main ML feature set excludes an explicit alerted/root-service feature.",
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    print("Loading GAIA aggregate data...")
    cases = load_cases()
    trace_endpoint = load_trace_endpoint()
    metric_service = load_metric_service()
    print(f"Loaded cases={len(cases)}")

    print("Building service-level feature rows...")
    features = build_feature_rows(cases, trace_endpoint, metric_service)
    print(f"features={features.shape}")

    print("Evaluating learned evidence scorers...")
    case_results, summary, auc_summary = evaluate_models(features)
    print(summary.to_string(index=False))
    print(auc_summary.to_string(index=False))

    make_outputs(features, case_results, summary, auc_summary)
    print("Done.")


if __name__ == "__main__":
    main()
