#!/usr/bin/env python3
"""Build paper-facing RC-FAD tables from completed multi-seed runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


COMPARE_METHODS = [
    "fedavg",
    "fedprox",
    "moon",
    "fedsimsup",
    "confree",
    "fedavg_focal",
    "fedavg_cbloss",
    "fedprox_focal",
    "rcfad",
]
THRESHOLD_METHODS = [
    "fedavg_fixed",
    "fedavg_global_post",
    "fedavg_local_post",
    "fedprox_local_post",
    "confree_local_post",
    "rcfad_global_joint",
    "rcfad",
]
ABLATION_METHODS = [
    "base_no_modules",
    "wo_joint_threshold",
    "wo_fpr_constraint",
    "wo_dynamic_beta",
    "wo_risk_aggregation",
    "wo_update_reliability",
    "rcfad",
]

DISPLAY = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "moon": "MOON",
    "fedsimsup": "FedSimSup",
    "confree": "ConFREE",
    "fedavg_focal": "FedAvg+Focal",
    "fedavg_cbloss": "FedAvg+CBLoss",
    "fedprox_focal": "FedProx+Focal",
    "fedavg_fixed": "FedAvg-Fixed",
    "fedavg_global_post": "FedAvg-Global-Post",
    "fedavg_local_post": "FedAvg-Local-Post",
    "fedprox_local_post": "FedProx-Local-Post",
    "confree_local_post": "ConFREE-Local-Post",
    "rcfad_global_joint": "RC-FAD-Global-Joint",
    "base_no_modules": "FedAvg / No Modules",
    "wo_joint_threshold": "RC-FAD w/o Joint Threshold",
    "wo_fpr_constraint": "RC-FAD w/o Low-FPR Constraint",
    "wo_dynamic_beta": "RC-FAD w/o Dynamic Enhancement",
    "wo_risk_aggregation": "RC-FAD w/o Risk Aggregation",
    "wo_update_reliability": "RC-FAD w/o Update Reliability",
    "rcfad": "RC-FAD",
}

METRICS = [
    "auc",
    "auprc",
    "recall_at_fpr",
    "fnr_at_fpr_threshold",
    "fpr_at_fpr_threshold",
    "f1_at_fpr_threshold",
    "threshold_at_fpr",
]


def _float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _mean(vals: list[float]) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    return mean(vals) if vals else float("nan")


def _std(vals: list[float]) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    return stdev(vals) if len(vals) > 1 else 0.0


def _fmt(vals: list[float]) -> str:
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{mean(vals):.4f} ± {_std(vals):.4f}"


def read_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = data.get("run_config", {}) or {}
        selected = data.get("selected_by_best_metric", {}) or {}
        row: dict[str, Any] = {
            "path": str(path.parent),
            "dataset": str(cfg.get("dataset-name", "")),
            "method": str(cfg.get("method-name", "")),
            "seed": str(cfg.get("seed", "")),
            "rounds": str(cfg.get("num-server-rounds", "")),
        }
        for metric in METRICS:
            row[metric] = _float(selected.get(metric))
        rows.append(row)
    return rows


def group(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)
    return grouped


def stats(grouped: dict[tuple[str, str], list[dict[str, Any]]], dataset: str, method: str, metric: str) -> list[float]:
    return [_float(r.get(metric)) for r in grouped.get((dataset, method), [])]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_method_table(
    rows: list[dict[str, Any]],
    methods: list[str],
    datasets: list[str] | None,
) -> list[dict[str, Any]]:
    grouped = group(rows)
    datasets = datasets or sorted({r["dataset"] for r in rows})
    out: list[dict[str, Any]] = []
    for dataset in datasets:
        for method in methods:
            runs = grouped.get((dataset, method), [])
            if not runs:
                continue
            out.append(
                {
                    "Dataset": dataset,
                    "Method": DISPLAY.get(method, method),
                    "AUPRC": _fmt([r["auprc"] for r in runs]),
                    "AUROC": _fmt([r["auc"] for r in runs]),
                    "Recall@FPR": _fmt([r["recall_at_fpr"] for r in runs]),
                    "FNR@FPR": _fmt([r["fnr_at_fpr_threshold"] for r in runs]),
                    "FPR@Target": _fmt([r["fpr_at_fpr_threshold"] for r in runs]),
                    "F1@FPR": _fmt([r["f1_at_fpr_threshold"] for r in runs]),
                    "Seeds": len({r["seed"] for r in runs}),
                }
            )
    return out


def build_compare_ranking(rows: list[dict[str, Any]], datasets: list[str] | None) -> list[dict[str, Any]]:
    grouped = group(rows)
    datasets = datasets or sorted({r["dataset"] for r in rows})
    out: list[dict[str, Any]] = []
    for dataset in datasets:
        rcfad_auprc = _mean(stats(grouped, dataset, "rcfad", "auprc"))
        rcfad_recall = _mean(stats(grouped, dataset, "rcfad", "recall_at_fpr"))
        candidates: list[tuple[float, str]] = []
        for method in COMPARE_METHODS:
            if method == "rcfad":
                continue
            vals = stats(grouped, dataset, method, "auprc")
            if vals:
                candidates.append((_mean(vals), method))
        if not candidates or math.isnan(rcfad_auprc):
            continue
        best_auprc, best_method = max(candidates, key=lambda x: x[0])
        best_recall = _mean(stats(grouped, dataset, best_method, "recall_at_fpr"))
        out.append(
            {
                "Dataset": dataset,
                "RC-FAD AUPRC": f"{rcfad_auprc:.4f}",
                "Best Baseline": DISPLAY.get(best_method, best_method),
                "Best Baseline AUPRC": f"{best_auprc:.4f}",
                "Delta AUPRC": f"{rcfad_auprc - best_auprc:+.4f}",
                "RC-FAD Recall@FPR": f"{rcfad_recall:.4f}",
                "Best Baseline Recall@FPR": f"{best_recall:.4f}",
                "Delta Recall@FPR": f"{rcfad_recall - best_recall:+.4f}",
            }
        )
    return out


def build_ablation_delta(rows: list[dict[str, Any]], datasets: list[str] | None) -> list[dict[str, Any]]:
    grouped = group(rows)
    datasets = datasets or sorted({r["dataset"] for r in rows})
    out: list[dict[str, Any]] = []
    for dataset in datasets:
        full = _mean(stats(grouped, dataset, "rcfad", "auprc"))
        if math.isnan(full):
            continue
        row = {"Dataset": dataset, "RC-FAD AUPRC": f"{full:.4f}"}
        for method in ABLATION_METHODS:
            if method == "rcfad":
                continue
            val = _mean(stats(grouped, dataset, method, "auprc"))
            row[f"vs {DISPLAY.get(method, method)}"] = "" if math.isnan(val) else f"{full - val:+.4f}"
        out.append(row)
    return out


def build_duplicate_diagnostics(rows: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    grouped = group(rows)
    out: list[dict[str, Any]] = []
    for dataset in sorted({r["dataset"] for r in rows}):
        buckets: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for method in methods:
            runs = grouped.get((dataset, method), [])
            if not runs:
                continue
            key = (
                f"{_mean([r['auprc'] for r in runs]):.4f}",
                f"{_mean([r['recall_at_fpr'] for r in runs]):.4f}",
                f"{_mean([r['fpr_at_fpr_threshold'] for r in runs]):.4f}",
            )
            buckets[key].append(DISPLAY.get(method, method))
        for key, names in buckets.items():
            if len(names) > 1:
                out.append(
                    {
                        "Dataset": dataset,
                        "Rounded Metric Tuple": f"AUPRC={key[0]}, Recall@FPR={key[1]}, FPR@Target={key[2]}",
                        "Methods": "; ".join(names),
                        "Note": "Check whether methods are theoretically equivalent under this setting or need a stronger tuned run.",
                    }
                )
    return out


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-root")
    parser.add_argument("--threshold-root")
    parser.add_argument("--ablation-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--compare-datasets", default="")
    parser.add_argument("--threshold-datasets", default="")
    parser.add_argument("--ablation-datasets", default="")
    args = parser.parse_args()

    out = Path(args.output_root) / "_summary"
    out.mkdir(parents=True, exist_ok=True)

    if args.compare_root:
        rows = read_rows(Path(args.compare_root))
        datasets = parse_list(args.compare_datasets) or None
        table = build_method_table(rows, COMPARE_METHODS, datasets)
        ranking = build_compare_ranking(rows, datasets)
        dup = build_duplicate_diagnostics(rows, COMPARE_METHODS)
        write_csv(out / "table_main_comparison_3seed.csv", table)
        write_md(out / "table_main_comparison_3seed.md", table)
        write_csv(out / "table_main_comparison_ranking_3seed.csv", ranking)
        write_md(out / "table_main_comparison_ranking_3seed.md", ranking)
        write_csv(out / "duplicate_diagnostics_compare.csv", dup)
        write_md(out / "duplicate_diagnostics_compare.md", dup)

    if args.threshold_root:
        rows = read_rows(Path(args.threshold_root))
        datasets = parse_list(args.threshold_datasets) or None
        table = build_method_table(rows, THRESHOLD_METHODS, datasets)
        dup = build_duplicate_diagnostics(rows, THRESHOLD_METHODS)
        write_csv(out / "table_threshold_mechanism_3seed.csv", table)
        write_md(out / "table_threshold_mechanism_3seed.md", table)
        write_csv(out / "duplicate_diagnostics_threshold.csv", dup)
        write_md(out / "duplicate_diagnostics_threshold.md", dup)

    if args.ablation_root:
        rows = read_rows(Path(args.ablation_root))
        datasets = parse_list(args.ablation_datasets) or None
        table = build_method_table(rows, ABLATION_METHODS, datasets)
        delta = build_ablation_delta(rows, datasets)
        dup = build_duplicate_diagnostics(rows, ABLATION_METHODS)
        write_csv(out / "table_ablation_components_3seed.csv", table)
        write_md(out / "table_ablation_components_3seed.md", table)
        write_csv(out / "table_ablation_delta_3seed.csv", delta)
        write_md(out / "table_ablation_delta_3seed.md", delta)
        write_csv(out / "duplicate_diagnostics_ablation.csv", dup)
        write_md(out / "duplicate_diagnostics_ablation.md", dup)

    print(f"Wrote paper tables under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
