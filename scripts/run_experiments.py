#!/usr/bin/env python3
"""Run experiments for the RC-FAD method described in Method.txt.

The script is intentionally organized around the paper claims:

* compare: representative FL baselines against the full RC-FAD method;
* threshold: joint threshold learning against post-hoc calibration;
* ablation: start from a no-module base, then remove one RC-FAD module at a time;
* hetero/sensitivity: expand the same methods across scenario grids.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import json
import signal
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _off() -> dict[str, Any]:
    """Disable threshold learning, low-FPR constraint, and dynamic beta."""

    return {
        "beta-power": 0.0,
        "beta-alpha": 0.0,
        "beta-zeta": 0.0,
        "beta-min": 1.0,
        "beta-max": 1.0,
        "mu-fnr": 0.0,
        "eta-lambda": 0.0,
        "lambda-init": 0.0,
        "fpr-penalty": 0.0,
        "lr-tau": 0.0,
    }


RCFAD_FULL: dict[str, Any] = {
    "method-name": "rcfad",
    "aggregation-key": "risk_weight",
    "c-fn": 2.0,
    "c-fp": 1.0,
    # Method.txt Sec. 3.4 coefficients:
    # beta_power=a1 for class scarcity, beta_alpha=a2 for FNR,
    # beta_zeta=a3 for FPR-constraint violation suppression.
    "beta-power": 1.0,
    "beta-alpha": 2.0,
    "beta-zeta": 4.0,
    "beta-min": 1.0,
    "beta-max": 1.5,
    "beta-reference-ratio": 0.01,
    "beta-gate-mode": "hard",
    "beta-round-decay": 20.0,
    # Method.txt Sec. 3.2-3.3 low-FPR Lagrangian and personalized threshold.
    "mu-fnr": 0.05,
    "eta-lambda": 1.0,
    "lambda-init": 0.5,
    "fpr-penalty": 1.0,
    "lr-tau": 0.0002,
    # Method.txt Sec. 3.5 risk-reliable aggregation.
    "risk-gamma-fnr": 1.0,
    "risk-gamma-fpr": 2.0,
    "update-reliability-rho": 0.1,
    "update-reliability-floor": 0.9,
    "update-reliability-delta": 1e-12,
    "risk-weight-min-factor": 0.5,
    "risk-weight-max-factor": 2.0,
}


METHODS: dict[str, dict[str, Any]] = {
    "base_no_modules": {
        "method-name": "base_no_modules",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        **_off(),
    },
    "fedavg": {
        "method-name": "fedavg",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        **_off(),
    },
    "fedavg_focal": {
        "method-name": "fedavg_focal",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        "loss-type": "focal",
        "focal-gamma": 2.0,
        "focal-alpha": 0.75,
        **_off(),
    },
    "fedprox": {
        "method-name": "fedprox",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        "fedprox-mu": 0.01,
        **_off(),
    },
    "moon": {
        "method-name": "moon",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        "moon-mu": 0.1,
        "moon-temperature": 0.5,
        **_off(),
    },
    "fedsimsup": {
        "method-name": "fedsimsup",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        "fedsimsup-mu": 0.2,
        "fedsimsup-temperature": 0.5,
        "fedsimsup-supervisor-ema": 0.5,
        **_off(),
    },
    "confree": {
        "method-name": "confree",
        "aggregation-key": "conflict_free",
        "c-fn": 1.0,
        "c-fp": 1.0,
        **_off(),
    },
    "cost_sensitive": {
        "method-name": "cost_sensitive",
        "aggregation-key": "num-examples",
        "c-fn": 3.0,
        "c-fp": 1.0,
        **_off(),
    },
    "posthoc_threshold": {
        "method-name": "posthoc_threshold",
        "aggregation-key": "num-examples",
        "c-fn": 3.0,
        "c-fp": 1.0,
        **_off(),
        # Static minority weight during training, threshold only calibrated
        # during evaluation through threshold_at_fpr metrics.
        "beta-power": 1.0,
        "beta-min": 1.0,
        "beta-max": 4.0,
    },
    "strong_baseline": {
        "method-name": "strong_baseline",
        "aggregation-key": "num-examples",
        "c-fn": 2.0,
        "c-fp": 1.0,
        "beta-power": 1.0,
        "beta-alpha": 0.0,
        "beta-zeta": 0.0,
        "beta-min": 1.0,
        "beta-max": 4.0,
        "mu-fnr": 0.05,
        "eta-lambda": 1.0,
        "lambda-init": 0.5,
        "fpr-penalty": 1.0,
        "lr-tau": 0.001,
    },
    "rcfad": dict(RCFAD_FULL),
    "fedavg_fixed": {
        "method-name": "fedavg_fixed",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        "eval-threshold-mode": "fixed",
        **_off(),
    },
    "fedavg_global_post": {
        "method-name": "fedavg_global_post",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        **_off(),
    },
    "fedavg_local_post": {
        "method-name": "fedavg_local_post",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        "eval-threshold-mode": "local_posthoc",
        **_off(),
    },
    "fedprox_local_post": {
        "method-name": "fedprox_local_post",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        "fedprox-mu": 0.01,
        "eval-threshold-mode": "local_posthoc",
        **_off(),
    },
    "confree_local_post": {
        "method-name": "confree_local_post",
        "aggregation-key": "conflict_free",
        "c-fn": 1.0,
        "c-fp": 1.0,
        "eval-threshold-mode": "local_posthoc",
        **_off(),
    },
    "rcfad_global_joint": {
        **RCFAD_FULL,
        "method-name": "rcfad_global_joint",
        "threshold-scope": "global",
    },
    "wo_joint_threshold": {
        **RCFAD_FULL,
        "method-name": "wo_joint_threshold",
        "lr-tau": 0.0,
        "eval-threshold-mode": "local_posthoc",
    },
    "wo_personal_threshold": {
        **RCFAD_FULL,
        "method-name": "wo_personal_threshold",
        "lr-tau": 0.0,
    },
    "wo_fpr_constraint": {
        **RCFAD_FULL,
        "method-name": "wo_fpr_constraint",
        "eta-lambda": 0.0,
        "lambda-init": 0.0,
        "fpr-penalty": 0.0,
    },
    "wo_dynamic_beta": {
        **RCFAD_FULL,
        "method-name": "wo_dynamic_beta",
        "beta-power": 0.0,
        "beta-alpha": 0.0,
        "beta-zeta": 0.0,
        "beta-min": 1.0,
        "beta-max": 1.0,
    },
    "wo_risk_aggregation": {
        **RCFAD_FULL,
        "method-name": "wo_risk_aggregation",
        "aggregation-key": "num-examples",
    },
    "wo_update_reliability": {
        **RCFAD_FULL,
        "method-name": "wo_update_reliability",
        "update-reliability-rho": 0.0,
    },
}


SUITES = {
    "quick": ["fedavg", "rcfad"],
    "compare": ["fedavg", "fedprox", "moon", "fedsimsup", "confree", "fedavg_focal", "rcfad"],
    "threshold": [
        "fedavg_fixed",
        "fedavg_global_post",
        "fedavg_local_post",
        "fedprox_local_post",
        "confree_local_post",
        "rcfad_global_joint",
        "rcfad",
    ],
    "ablation": [
        "base_no_modules",
        "rcfad",
        "wo_joint_threshold",
        "wo_fpr_constraint",
        "wo_dynamic_beta",
        "wo_risk_aggregation",
        "wo_update_reliability",
    ],
    "hetero": ["base_no_modules", "fedsimsup", "confree", "rcfad_global_joint", "rcfad"],
    "sensitivity": ["rcfad"],
    # Kept for backwards-compatible command lines.
    "main": ["fedavg", "cost_sensitive", "posthoc_threshold", "rcfad"],
}


HETERO_SCENARIOS: list[tuple[str, dict[str, Any]]] = [
    (
        "anomaly_ratio_hetero",
        {
            "partition-scheme": "risk_hetero",
            "min-anomaly-ratio": 0.005,
            "max-anomaly-ratio": 0.10,
            "normal-shift-max": 0.0,
            "epsilon-heterogeneous": False,
        },
    ),
    (
        "normal_score_shift",
        {
            "partition-scheme": "risk_hetero",
            "min-anomaly-ratio": 0.03,
            "max-anomaly-ratio": 0.03,
            "normal-shift-max": 0.25,
            "epsilon-heterogeneous": False,
        },
    ),
    (
        "epsilon_hetero",
        {
            "partition-scheme": "risk_hetero",
            "min-anomaly-ratio": 0.02,
            "max-anomaly-ratio": 0.08,
            "epsilon-heterogeneous": True,
            "epsilon-min": 0.01,
            "epsilon-max": 0.10,
        },
    ),
]


SENSITIVITY_SCENARIOS: list[tuple[str, dict[str, Any]]] = [
    ("epsilon_0_01", {"epsilon-fpr": 0.01}),
    ("epsilon_0_03", {"epsilon-fpr": 0.03}),
    ("epsilon_0_05", {"epsilon-fpr": 0.05}),
    ("epsilon_0_10", {"epsilon-fpr": 0.10}),
    ("dirichlet_alpha_0_1", {"partition-scheme": "risk_dirichlet", "dirichlet-alpha": 0.1}),
    ("dirichlet_alpha_0_3", {"partition-scheme": "risk_dirichlet", "dirichlet-alpha": 0.3}),
    ("dirichlet_alpha_0_5", {"partition-scheme": "risk_dirichlet", "dirichlet-alpha": 0.5}),
    ("dirichlet_alpha_1_0", {"partition-scheme": "risk_dirichlet", "dirichlet-alpha": 1.0}),
    ("beta_zeta_1", {"beta-zeta": 1.0}),
    ("beta_zeta_2", {"beta-zeta": 2.0}),
    ("beta_zeta_4", {"beta-zeta": 4.0}),
    ("beta_zeta_8", {"beta-zeta": 8.0}),
    ("lr_tau_0_0001", {"lr-tau": 0.0001}),
    ("lr_tau_0_0002", {"lr-tau": 0.0002}),
    ("lr_tau_0_0005", {"lr-tau": 0.0005}),
]


COMMON_CONFIG: dict[str, Any] = {
    "partition-scheme": "risk_hetero",
    "min-anomaly-ratio": 0.005,
    "max-anomaly-ratio": 0.10,
    "global-anomaly-ratio": 0.10,
    "epsilon-fpr": 0.05,
    "threshold-init": 0.5,
    "global-eval-threshold": 0.5,
    "temperature": 0.10,
    "batch-size": 64,
    "central-batch-size": 128,
    "fraction-evaluate": 0.5,
    "best-metric": "auprc",
    "risk-weight-warmup-rounds": 1,
}


DATASET_PRESETS: dict[str, dict[str, Any]] = {
    # Extreme financial fraud: very low anomaly prevalence and strict FPR.
    "creditcard": {
        "batch-size": 512,
        "central-batch-size": 2048,
        "min-anomaly-ratio": 0.0005,
        "max-anomaly-ratio": 0.01,
        "global-anomaly-ratio": 0.002,
        "epsilon-fpr": 0.01,
        "threshold-init": 0.05,
        "global-eval-threshold": 0.05,
    },
    # BAF base prevalence is close to the low-percent fraud regime.
    "baf": {
        "batch-size": 512,
        "central-batch-size": 2048,
        "min-anomaly-ratio": 0.002,
        "max-anomaly-ratio": 0.05,
        "global-anomaly-ratio": 0.01,
        "epsilon-fpr": 0.01,
        "threshold-init": 0.05,
        "global-eval-threshold": 0.05,
    },
    # AI4I machine failures are less rare than fraud, so use a slightly looser
    # FPR target while still evaluating low-false-alarm behavior.
    "ai4i": {
        "batch-size": 128,
        "central-batch-size": 512,
        "min-anomaly-ratio": 0.01,
        "max-anomaly-ratio": 0.08,
        "global-anomaly-ratio": 0.035,
        "epsilon-fpr": 0.03,
        "threshold-init": 0.05,
        "global-eval-threshold": 0.05,
    },
    # SECOM is small and high-dimensional; smaller batches keep client splits
    # from becoming too coarse.
    "secom": {
        "batch-size": 64,
        "central-batch-size": 256,
        "min-anomaly-ratio": 0.02,
        "max-anomaly-ratio": 0.12,
        "global-anomaly-ratio": 0.066,
        "epsilon-fpr": 0.03,
        "threshold-init": 0.05,
        "global-eval-threshold": 0.05,
    },
}


def dataset_key(name: str) -> str:
    normalized = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "creditcard": "creditcard",
        "creditcardfraud": "creditcard",
        "creditfraud": "creditcard",
        "baf": "baf",
        "bankaccountfraud": "baf",
        "bankfraud": "baf",
        "ai4i": "ai4i",
        "ai4i2020": "ai4i",
        "predictivemaintenance": "ai4i",
        "secom": "secom",
    }
    return aliases.get(normalized, normalized)


def _format_value(v: Any) -> str:
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def build_run_config(cfg: dict[str, Any]) -> str:
    return " ".join(f"{k}={_format_value(v)}" for k, v in cfg.items())


def flwr_executable() -> str:
    """Prefer the Flower executable from the active Python environment."""

    candidate = Path(sys.executable).with_name("flwr")
    return str(candidate) if candidate.exists() else "flwr"


def ray_executable() -> str:
    """Prefer the Ray executable from the active Python environment."""

    candidate = Path(sys.executable).with_name("ray")
    return str(candidate) if candidate.exists() else "ray"


def cleanup_runtime(env: dict[str, str], *, force: bool = False) -> None:
    """Clean up Flower/Ray simulation processes left behind by a run.

    Flower's local simulation can leave ``flwr-simulation`` actors alive after a
    failed run. Those actors keep CUDA contexts and large Ray object stores open,
    which can make every later experiment fail with CUDA OOM even though the
    failing run has already returned.
    """

    patterns = [
        "flwr-simulation",
        "flower-superexec",
        "flower-superlink",
        "flower-supernode",
    ]
    subprocess.run(
        [ray_executable(), "stop", "--force"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        timeout=60,
        check=False,
    )
    own_uid = os.getuid()
    own_pid = os.getpid()
    for sig, delay in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 0.0)):
        for proc_dir in Path("/proc").iterdir():
            if not proc_dir.name.isdigit():
                continue
            pid = int(proc_dir.name)
            if pid == own_pid:
                continue
            try:
                if proc_dir.stat().st_uid != own_uid:
                    continue
                cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                    "utf-8", errors="ignore"
                )
            except OSError:
                continue
            if any(pattern in cmdline for pattern in patterns):
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
        if delay:
            time.sleep(delay)
        if not force:
            break


def parse_extra_config(extra: str) -> dict[str, Any]:
    """Parse ``key=value`` tokens so extra config overrides existing keys."""

    parsed: dict[str, Any] = {}
    for token in shlex.split(extra.strip()):
        if "=" not in token:
            raise ValueError(f"Invalid --extra token {token!r}; expected key=value")
        key, value = token.split("=", 1)
        lower = value.lower()
        if lower in {"true", "false"}:
            parsed[key] = lower == "true"
            continue
        try:
            parsed[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            parsed[key] = float(value)
            continue
        except ValueError:
            pass
        parsed[key] = value
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(SUITES), default="quick")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "mnist"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--anomaly-class", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--num-supernodes", type=int, default=5)
    parser.add_argument("--results-root", default="outputs/experiments")
    parser.add_argument("--extra", default="", help="Extra run-config string appended to every run")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Limit hetero/sensitivity suites to the named scenarios",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--list-methods", action="store_true")
    args = parser.parse_args()
    extra_cfg = parse_extra_config(args.extra) if args.extra.strip() else {}

    if args.list_methods:
        print("Suites:")
        for suite, methods in SUITES.items():
            print(f"  {suite}: {', '.join(methods)}")
        print("\nMethods:")
        for name in sorted(METHODS):
            print(f"  {name}")
        return 0

    methods = args.methods if args.methods else SUITES[args.suite]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {unknown}. Available: {sorted(METHODS)}")

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_root = Path(args.results_root) / f"logs_{args.suite}_{timestamp}"
    logs_root.mkdir(parents=True, exist_ok=True)

    scenario_grid = [("", {})]
    if args.suite == "hetero":
        scenario_grid = HETERO_SCENARIOS
    elif args.suite == "sensitivity":
        scenario_grid = SENSITIVITY_SCENARIOS
    if args.scenarios is not None:
        wanted = set(args.scenarios)
        scenario_grid = [(name, cfg) for name, cfg in scenario_grid if name in wanted]
        missing = sorted(wanted - {name for name, _ in scenario_grid})
        if missing:
            raise SystemExit(f"Unknown scenario(s): {missing}")

    planned_runs = []
    for dataset in args.datasets:
        for seed in args.seeds:
            for scenario_name, scenario_cfg in scenario_grid:
                for method in methods:
                    scenario_part = f"_{scenario_name}" if scenario_name else ""
                    run_name = f"{args.suite}{scenario_part}_{dataset}_{method}_seed{seed}_{timestamp}"
                    cfg: dict[str, Any] = {
                        "run-name": run_name,
                        "results-root": args.results_root,
                        "dataset-name": dataset,
                        "num-server-rounds": args.rounds,
                        "seed": seed,
                        "anomaly-class": args.anomaly_class,
                        "learning-rate": args.learning_rate,
                        "local-epochs": args.local_epochs,
                        "scenario-name": scenario_name,
                    }
                    cfg.update(COMMON_CONFIG)
                    cfg.update(DATASET_PRESETS.get(dataset_key(dataset), {}))
                    cfg.update(METHODS[method])
                    cfg.update(scenario_cfg)
                    cfg.update(extra_cfg)
                    planned_runs.append((dataset, seed, scenario_name, method, run_name, cfg))

    manifest_path = logs_root / "manifest.json"
    manifest = []
    federation_config = f"num-supernodes={args.num_supernodes}"

    for i, (dataset, seed, scenario_name, method, run_name, cfg) in enumerate(planned_runs, start=1):
        run_config = build_run_config(cfg)
        cmd = [
            flwr_executable(),
            "run",
            ".",
            "--stream",
            "--run-config",
            run_config,
            "--federation-config",
            federation_config,
        ]
        log_path = logs_root / f"{run_name}.log"
        manifest.append({
            "dataset": dataset,
            "seed": seed,
            "scenario": scenario_name,
            "method": method,
            "run_name": run_name,
            "log_path": str(log_path),
            "run_config": cfg,
        })
        scenario_label = f" | {scenario_name}" if scenario_name else ""
        print(f"\n[{i}/{len(planned_runs)}] {dataset}{scenario_label} | {method} | seed={seed}")
        print("Command:", " ".join(shlex.quote(x) for x in cmd))
        if args.dry_run:
            continue

        with log_path.open("w", encoding="utf-8") as f:
            env = os.environ.copy()
            no_proxy_values = [
                item
                for item in env.get("NO_PROXY", env.get("no_proxy", "")).split(",")
                if item
            ]
            for local_host in ("127.0.0.1", "localhost"):
                if local_host not in no_proxy_values:
                    no_proxy_values.append(local_host)
            env["NO_PROXY"] = ",".join(no_proxy_values)
            env["no_proxy"] = env["NO_PROXY"]
            if str(cfg.get("device", "")).lower() == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
                env["RCFAD_DEVICE"] = "cpu"
            cleanup_runtime(env)
            proc = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,
            )
            cleanup_runtime(env, force=True)
        if proc.returncode != 0:
            print(f"Run failed with return code {proc.returncode}. Log: {log_path}", file=sys.stderr)
            if not args.continue_on_error:
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
                return proc.returncode

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.dry_run:
        print("\nCollecting results...")
        collect_cmd = [sys.executable, "scripts/collect_results.py", "--results-root", args.results_root]
        subprocess.run(collect_cmd, check=False)
        print(f"Done. Logs saved under: {logs_root}")
    else:
        print(f"Dry run complete. Manifest would be saved under: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
