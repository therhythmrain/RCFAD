#!/usr/bin/env python3
"""Collect RC-FAD experiment summaries into CSV/Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

# Metric groups:
# - auc/auprc/recall_at_fpr are threshold-independent or low-FPR ranking metrics.
# - *_at_fpr metrics are computed with the calibrated threshold that controls FPR at epsilon.
# - accuracy/precision/recall/f1/fpr are fixed-threshold diagnostics (usually threshold=0.5 on server side).
METRICS = [
    "auc", "auprc", "recall_at_fpr",
    "accuracy_at_fpr", "precision_at_fpr", "recall_at_fpr_threshold",
    "f1_at_fpr_threshold", "fpr_at_fpr_threshold", "fnr_at_fpr_threshold",
    "threshold_at_fpr",
    "f1", "recall", "fpr", "precision", "accuracy",
    "fpr_violation", "fpr_violation_flag", "fpr_at_target_violation",
    "loss",
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
    "base_no_modules": "FedAvg / No Modules",
    "wo_joint_threshold": "RC-FAD w/o Joint Threshold",
    "wo_fpr_constraint": "RC-FAD w/o Low-FPR Constraint",
    "wo_dynamic_beta": "RC-FAD w/o Dynamic Enhancement",
    "wo_risk_aggregation": "RC-FAD w/o Risk Aggregation",
    "wo_update_reliability": "RC-FAD w/o Update Reliability",
    "rcfad": "RC-FAD",
}


def _float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def read_summaries(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = data.get("run_config", {})
        last = data.get("last_server", {}) or {}
        selected = data.get("selected_by_best_metric", {}) or {}
        best = data.get("best", {}) or {}
        row: dict[str, Any] = {
            "path": str(path.parent),
            "run_name": cfg.get("run-name", path.parent.name),
            "dataset": cfg.get("dataset-name", ""),
            "scenario": cfg.get("scenario-name", ""),
            "method": cfg.get("method-name", ""),
            "seed": cfg.get("seed", ""),
            "rounds": cfg.get("num-server-rounds", ""),
            "selected_round": selected.get("round", ""),
        }
        for m in METRICS:
            row[f"last_{m}"] = _float(last.get(m))
            row[f"selected_{m}"] = _float(selected.get(m))
            best_m = best.get(m, {}) if isinstance(best.get(m, {}), dict) else {}
            row[f"best_{m}"] = _float(best_m.get(m))
            row[f"best_{m}_round"] = best_m.get("round", "")
        rows.append(row)
    return rows


def read_client_threshold_rows(
    summary_rows: list[dict[str, Any]],
    row_mode: str = "selected",
) -> list[dict[str, Any]]:
    """Read client-side metrics at the server-selected round.

    This table is closer to a deployment diagnostic than the paper-facing
    target-FPR table: it uses each method's learned/fixed client threshold
    instead of recalibrating a threshold post hoc on the evaluation labels.
    """

    rows: list[dict[str, Any]] = []
    for summary in summary_rows:
        run_dir = Path(str(summary.get("path", "")))
        metrics_path = run_dir / "client_eval_metrics.csv"
        if not metrics_path.exists():
            continue
        try:
            with metrics_path.open(newline="", encoding="utf-8") as f:
                client_rows = list(csv.DictReader(f))
        except Exception:
            continue
        if not client_rows:
            continue

        selected_round = str(summary.get("selected_round", ""))
        if row_mode == "last":
            selected = client_rows[-1]
        else:
            selected = next((r for r in client_rows if str(r.get("round", "")) == selected_round), None)
            if selected is None:
                selected = client_rows[-1]

        row: dict[str, Any] = {
            "path": str(run_dir),
            "run_name": summary.get("run_name", run_dir.name),
            "dataset": summary.get("dataset", ""),
            "scenario": summary.get("scenario", ""),
            "method": summary.get("method", ""),
            "seed": summary.get("seed", ""),
            "rounds": summary.get("rounds", ""),
            "selected_round": selected.get("round", selected_round),
        }
        for m in [
            "accuracy", "precision", "recall", "fpr", "fnr", "f1",
            "auc", "auprc", "threshold", "target_fpr", "fpr_violation",
            "fpr_violation_flag", "eval_loss",
        ]:
            row[m] = _float(selected.get(m))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_mean_std(vals: list[float]) -> str:
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{mean(vals):.4f} ± {stdev(vals):.4f}"


def _fmt_percent(vals: list[float]) -> str:
    vals = [100.0 * v for v in vals if not math.isnan(v)]
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{mean(vals):.2f} ± {stdev(vals):.2f}"


def _std(vals: list[float]) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    return stdev(vals) if len(vals) > 1 else 0.0


def _mean_or(vals: list[float], fallback: float = float("nan")) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    return mean(vals) if vals else fallback


def _min_or(vals: list[float], fallback: float = float("nan")) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    return min(vals) if vals else fallback


def _select_detail_rows(path: Path, selected_round: Any, filename: str) -> list[dict[str, Any]]:
    metrics_path = path / filename
    if not metrics_path.exists():
        return []
    try:
        with metrics_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    if not rows:
        return []
    round_text = str(selected_round)
    selected = [r for r in rows if str(r.get("round", "")) == round_text]
    if selected:
        return selected
    last_round = rows[-1].get("round", "")
    return [r for r in rows if str(r.get("round", "")) == str(last_round)]


def _last_detail_rows(path: Path, filename: str) -> list[dict[str, Any]]:
    metrics_path = path / filename
    if not metrics_path.exists():
        return []
    try:
        with metrics_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    if not rows:
        return []
    last_round = rows[-1].get("round", "")
    return [r for r in rows if str(r.get("round", "")) == str(last_round)]


def read_ablation_chosen_runs(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for summary in summary_rows:
        method = str(summary.get("method", ""))
        if method not in ABLATION_METHODS:
            continue
        run_dir = Path(str(summary.get("path", "")))
        selected_round = summary.get("selected_round", "")
        eval_rows = _select_detail_rows(run_dir, selected_round, "client_eval_detail_metrics.csv")
        train_rows = _select_detail_rows(run_dir, selected_round, "client_train_detail_metrics.csv")

        recall_vals = [_float(r.get("recall")) for r in eval_rows]
        f1_vals = [_float(r.get("f1")) for r in eval_rows]
        fpr_vals = [_float(r.get("fpr")) for r in eval_rows]
        violation_vals = [_float(r.get("fpr_violation")) for r in eval_rows]
        flag_vals = [_float(r.get("fpr_violation_flag")) for r in eval_rows]
        threshold_vals = [_float(r.get("threshold")) for r in eval_rows]

        row = {
            "path": summary.get("path", ""),
            "run_name": summary.get("run_name", ""),
            "dataset": summary.get("dataset", ""),
            "scenario": summary.get("scenario", ""),
            "method": method,
            "seed": summary.get("seed", ""),
            "rounds": summary.get("rounds", ""),
            "selected_round": selected_round,
            "auprc": _float(summary.get("selected_auprc"), _float(summary.get("last_auprc"))),
            "recall": _mean_or(recall_vals, _float(summary.get("selected_recall"))),
            "f1": _mean_or(f1_vals, _float(summary.get("selected_f1"))),
            "fpr": _mean_or(fpr_vals, _float(summary.get("selected_fpr"))),
            "avg_fpr_violation": _mean_or(violation_vals, _float(summary.get("selected_fpr_violation"))),
            "violation_rate": _mean_or(flag_vals, _float(summary.get("selected_fpr_violation_flag"))),
            "worst_client_recall": _min_or(recall_vals, _float(summary.get("selected_recall"))),
            "client_fpr_std": _std(fpr_vals),
            "threshold_std": _std(threshold_vals),
        }

        beta_vals = [_float(r.get("train_beta", r.get("beta"))) for r in train_rows]
        risk_factor_vals = [_float(r.get("risk_aggregation_factor")) for r in train_rows]
        reliability_vals = [_float(r.get("update_reliability")) for r in train_rows]
        smooth_fpr_vals = [_float(r.get("smooth_train_fpr")) for r in train_rows]
        smooth_fnr_vals = [_float(r.get("smooth_train_fnr")) for r in train_rows]
        row.update({
            "train_beta": _mean_or(beta_vals),
            "risk_aggregation_factor_std": _std(risk_factor_vals),
            "update_reliability": _mean_or(reliability_vals),
            "smooth_train_fpr": _mean_or(smooth_fpr_vals),
            "smooth_train_fnr": _mean_or(smooth_fnr_vals),
        })
        out.append(row)
    return out


def aggregate_ablation_chosen_percent(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("dataset", "")), str(row.get("scenario", "")), str(row.get("method", "")))].append(row)
    out: list[dict[str, Any]] = []
    method_order = {m: i for i, m in enumerate(ABLATION_METHODS)}
    for (dataset, scenario, method), group in sorted(
        groups.items(), key=lambda x: (x[0][0], x[0][1], method_order.get(x[0][2], 999))
    ):
        out.append({
            "Dataset": dataset,
            "Scenario": scenario,
            "Method": DISPLAY.get(method, method),
            "AUPRC (%)": _fmt_percent([_float(g.get("auprc")) for g in group]),
            "Recall (%)": _fmt_percent([_float(g.get("recall")) for g in group]),
            "F1 (%)": _fmt_percent([_float(g.get("f1")) for g in group]),
            "FPR (%)": _fmt_percent([_float(g.get("fpr")) for g in group]),
            "Avg. FPR Violation (%)": _fmt_percent([_float(g.get("avg_fpr_violation")) for g in group]),
            "Violation Rate (%)": _fmt_percent([_float(g.get("violation_rate")) for g in group]),
            "Worst-client Recall (%)": _fmt_percent([_float(g.get("worst_client_recall")) for g in group]),
            "Client-FPR Std (%)": _fmt_percent([_float(g.get("client_fpr_std")) for g in group]),
            "Seeds": len({str(g.get("seed", "")) for g in group}),
        })
    return out


def aggregate_ablation_diagnostics_percent(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("dataset", "")), str(row.get("scenario", "")), str(row.get("method", "")))].append(row)
    out: list[dict[str, Any]] = []
    method_order = {m: i for i, m in enumerate(ABLATION_METHODS)}
    for (dataset, scenario, method), group in sorted(
        groups.items(), key=lambda x: (x[0][0], x[0][1], method_order.get(x[0][2], 999))
    ):
        out.append({
            "Dataset": dataset,
            "Scenario": scenario,
            "Method": DISPLAY.get(method, method),
            "Train Beta": _fmt_mean_std([_float(g.get("train_beta")) for g in group]),
            "Risk Factor Std": _fmt_mean_std([_float(g.get("risk_aggregation_factor_std")) for g in group]),
            "Update Reliability": _fmt_mean_std([_float(g.get("update_reliability")) for g in group]),
            "Smooth Train FPR (%)": _fmt_percent([_float(g.get("smooth_train_fpr")) for g in group]),
            "Smooth Train FNR (%)": _fmt_percent([_float(g.get("smooth_train_fnr")) for g in group]),
            "Threshold Std (%)": _fmt_percent([_float(g.get("threshold_std")) for g in group]),
            "Seeds": len({str(g.get("seed", "")) for g in group}),
        })
    return out


def aggregate(rows: list[dict[str, Any]], key_prefix: str = "selected") -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(
            str(r.get("dataset", "")),
            str(r.get("scenario", "")),
            str(r.get("method", "")),
        )].append(r)
    out: list[dict[str, Any]] = []
    for (dataset, scenario, method), group in sorted(groups.items()):
        row = {"dataset": dataset, "scenario": scenario, "method": method, "n_runs": len(group)}
        for m in METRICS:
            vals = [_float(g.get(f"{key_prefix}_{m}")) for g in group]
            row[m] = _fmt_mean_std(vals)
        out.append(row)
    return out


def aggregate_plain(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(
            str(r.get("dataset", "")),
            str(r.get("scenario", "")),
            str(r.get("method", "")),
        )].append(r)
    out: list[dict[str, Any]] = []
    for (dataset, scenario, method), group in sorted(groups.items()):
        row = {"dataset": dataset, "scenario": scenario, "method": method, "n_runs": len(group)}
        for m in metrics:
            vals = [_float(g.get(m)) for g in group]
            row[m] = _fmt_mean_std(vals)
        out.append(row)
    return out


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = [
        "dataset", "scenario", "method", "auc", "auprc", "recall_at_fpr",
        "accuracy_at_fpr",
        "f1_at_fpr_threshold", "precision_at_fpr", "fpr_at_fpr_threshold",
        "n_runs",
    ]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_generic_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fixed_threshold_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write diagnostics for the fixed 0.5 threshold.

    These metrics are useful for debugging, but they should not be the main
    paper table because all-normal predictions can have high accuracy in the
    one-vs-rest anomaly setup.
    """

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = [
        "dataset", "scenario", "method", "accuracy", "precision", "recall", "fpr",
        "f1", "loss", "n_runs",
    ]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_client_threshold_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write client learned/fixed threshold deployment diagnostics."""

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = [
        "dataset", "scenario", "method", "accuracy", "precision", "recall", "fpr",
        "fpr_violation", "fpr_violation_flag", "f1", "auprc", "threshold", "n_runs",
    ]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="outputs/experiments")
    parser.add_argument("--key-prefix", choices=["selected", "last", "best"], default="selected")
    args = parser.parse_args()

    root = Path(args.results_root)
    rows = read_summaries(root)
    out_dir = root / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "all_runs.csv", rows)
    agg = aggregate(rows, key_prefix=args.key_prefix)
    write_csv(out_dir / f"summary_{args.key_prefix}.csv", agg)
    # Paper-facing table: uses calibrated threshold metrics, including Accuracy@FPR.
    write_markdown(out_dir / f"summary_{args.key_prefix}.md", agg)
    # Diagnostic table: fixed threshold metrics, useful for explaining all-normal behavior.
    write_fixed_threshold_markdown(out_dir / f"fixed_threshold_diagnostics_{args.key_prefix}.md", agg)

    client_rows = read_client_threshold_rows(rows)
    write_csv(out_dir / f"client_threshold_runs_{args.key_prefix}.csv", client_rows)
    client_agg = aggregate_plain(
        client_rows,
        [
            "accuracy", "precision", "recall", "fpr", "fnr", "f1", "auc",
            "auprc", "threshold", "target_fpr", "fpr_violation",
            "fpr_violation_flag", "eval_loss",
        ],
    )
    write_csv(out_dir / f"client_threshold_diagnostics_{args.key_prefix}.csv", client_agg)
    write_client_threshold_markdown(
        out_dir / f"client_threshold_diagnostics_{args.key_prefix}.md",
        client_agg,
    )

    if args.key_prefix != "last":
        final_client_rows = read_client_threshold_rows(rows, row_mode="last")
        write_csv(out_dir / "client_threshold_runs_last.csv", final_client_rows)
        final_client_agg = aggregate_plain(
            final_client_rows,
            [
                "accuracy", "precision", "recall", "fpr", "fnr", "f1", "auc",
                "auprc", "threshold", "target_fpr", "fpr_violation",
                "fpr_violation_flag", "eval_loss",
            ],
        )
        write_csv(out_dir / "client_threshold_diagnostics_last.csv", final_client_agg)
        write_client_threshold_markdown(
            out_dir / "client_threshold_diagnostics_last.md",
            final_client_agg,
        )

    print(f"Found {len(rows)} run summaries")
    print(f"Wrote: {out_dir / 'all_runs.csv'}")
    print(f"Wrote: {out_dir / f'summary_{args.key_prefix}.csv'}")
    print(f"Wrote: {out_dir / f'summary_{args.key_prefix}.md'}")
    print(f"Wrote: {out_dir / f'fixed_threshold_diagnostics_{args.key_prefix}.md'}")
    print(f"Wrote: {out_dir / f'client_threshold_runs_{args.key_prefix}.csv'}")
    print(f"Wrote: {out_dir / f'client_threshold_diagnostics_{args.key_prefix}.csv'}")
    print(f"Wrote: {out_dir / f'client_threshold_diagnostics_{args.key_prefix}.md'}")
    if args.key_prefix != "last":
        print(f"Wrote: {out_dir / 'client_threshold_runs_last.csv'}")
        print(f"Wrote: {out_dir / 'client_threshold_diagnostics_last.csv'}")
        print(f"Wrote: {out_dir / 'client_threshold_diagnostics_last.md'}")

    ablation_runs = read_ablation_chosen_runs(rows)
    write_csv(out_dir / "ablation_chosen_runs.csv", ablation_runs)
    ablation_table = aggregate_ablation_chosen_percent(ablation_runs)
    write_csv(out_dir / "table_ablation_chosen_percent_3seed.csv", ablation_table)
    write_generic_markdown(out_dir / "table_ablation_chosen_percent_3seed.md", ablation_table)
    ablation_diag = aggregate_ablation_diagnostics_percent(ablation_runs)
    write_csv(out_dir / "table_ablation_diagnostics_percent_3seed.csv", ablation_diag)
    write_generic_markdown(out_dir / "table_ablation_diagnostics_percent_3seed.md", ablation_diag)
    print(f"Wrote: {out_dir / 'ablation_chosen_runs.csv'}")
    print(f"Wrote: {out_dir / 'table_ablation_chosen_percent_3seed.csv'}")
    print(f"Wrote: {out_dir / 'table_ablation_chosen_percent_3seed.md'}")
    print(f"Wrote: {out_dir / 'table_ablation_diagnostics_percent_3seed.csv'}")
    print(f"Wrote: {out_dir / 'table_ablation_diagnostics_percent_3seed.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
