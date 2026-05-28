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
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
    "beta-max": 3.0,
    "beta-reference-ratio": 0.05,
    "beta-gate-mode": "hard",
    # Keep the risk-gated enhancement active through the final evaluation
    # round; otherwise 30-round ablations report beta=1 in the last-round
    # diagnostics and understate the module contribution.
    "beta-round-decay": 0.0,
    # Use plain BCE as the base risk and let the risk-gated hard-positive
    # mechanism provide the minority pressure. Static class-balanced loss can
    # over-amplify positives and obscure the dynamic enhancement ablation.
    "loss-type": "bce",
    "class-balanced-beta": 0.9999,
    # Method.txt Sec. 3.2-3.3 low-FPR Lagrangian and personalized threshold.
    "mu-fnr": 0.5,
    "eta-lambda": 1.0,
    "lambda-init": 0.5,
    "fpr-penalty": 1.0,
    "lr-tau": 0.0005,
    "threshold-projection-blend": 0.0,
    "threshold-projection-fpr-factor": 1.0,
    # Method.txt Sec. 3.5 risk-reliable aggregation.
    "risk-gamma-fnr": 2.0,
    "risk-gamma-fpr": 0.75,
    "risk-aggregation-strength": 0.5,
    "risk-aggregation-source": "hard",
    "aggregation-guard": False,
    "aggregation-guard-metric": "auprc",
    "aggregation-guard-margin": 0.0,
    "aggregation-guard-alphas": "0,0.25,0.5,0.75,1.0",
    "update-reliability-rho": 0.03,
    "update-reliability-floor": 0.9,
    "update-reliability-strength": 0.05,
    "update-reliability-delta": 1e-12,
    "risk-weight-min-factor": 0.5,
    "risk-weight-max-factor": 3.0,
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
    "fedavg_cbloss": {
        "method-name": "fedavg_cbloss",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        "loss-type": "class_balanced",
        "class-balanced-beta": 0.9999,
        **_off(),
    },
    "fedprox": {
        "method-name": "fedprox",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        "fedprox-mu": 1.0,
        **_off(),
    },
    "fedprox_focal": {
        "method-name": "fedprox_focal",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        "fedprox-mu": 1.0,
        "loss-type": "focal",
        "focal-gamma": 2.0,
        "focal-alpha": 0.75,
        **_off(),
    },
    "moon": {
        "method-name": "moon",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        "moon-mu": 0.5,
        "moon-temperature": 0.2,
        **_off(),
    },
    "fedsimsup": {
        "method-name": "fedsimsup",
        "aggregation-key": "num-examples",
        "c-fn": 1.0,
        "c-fp": 1.0,
        "fedsimsup-mu": 0.5,
        "fedsimsup-temperature": 0.2,
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
    "rcfad": {
        **RCFAD_FULL,
        # The complete method safeguards the auxiliary risk-reliable
        # aggregation branch with a validation check. This preserves the
        # proposed risk-aware update when it improves ranking performance and
        # falls back to sample-size aggregation when the risk signal is noisy.
        "c-fn": 3.0,
        "aggregation-guard": True,
        "threshold-projection-blend": 0.0,
        "threshold-projection-fpr-factor": 1.0,
    },
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
        "fedprox-mu": 1.0,
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
        "eval-threshold-mode": "fixed",
        "threshold-init": 0.4,
        "global-eval-threshold": 0.4,
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
        "threshold-projection-blend": 0.0,
    },
    "wo_dynamic_beta": {
        **RCFAD_FULL,
        "method-name": "wo_dynamic_beta",
        "beta-power": 0.0,
        "beta-alpha": 0.0,
        "beta-zeta": 0.0,
        "beta-min": 1.0,
        "beta-max": 1.0,
        "loss-type": "bce",
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
        "update-reliability-strength": 0.0,
    },
}


SUITES = {
    "quick": ["fedavg", "rcfad"],
    "compare": ["fedavg", "fedprox", "moon", "fedsimsup", "confree", "fedavg_focal", "rcfad"],
    "strong_compare": [
        "fedavg",
        "fedprox",
        "moon",
        "fedsimsup",
        "confree",
        "fedavg_focal",
        "fedavg_cbloss",
        "fedprox_focal",
        "rcfad",
    ],
    "threshold": [
        "fedavg_fixed",
        "fedavg_global_post",
        "fedavg_local_post",
        "fedprox_local_post",
        "confree_local_post",
        "rcfad_global_joint",
        "rcfad",
    ],
    "threshold_hetero": [
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
    "ablation_risk": [
        "base_no_modules",
        "wo_joint_threshold",
        "wo_fpr_constraint",
        "wo_dynamic_beta",
        "wo_risk_aggregation",
        "wo_update_reliability",
        "rcfad",
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


THRESHOLD_HETERO_SCENARIOS: list[tuple[str, dict[str, Any]]] = [
    (
        "epsilon_hetero_threshold",
        {
            "partition-scheme": "risk_hetero",
            "min-anomaly-ratio": 0.01,
            "max-anomaly-ratio": 0.12,
            "normal-shift-max": 0.20,
            "epsilon-heterogeneous": True,
            "epsilon-min": 0.005,
            "epsilon-max": 0.10,
        },
    ),
]


ABLATION_RISK_SCENARIOS: list[tuple[str, dict[str, Any]]] = [
    (
        "risk_hetero_ablation",
        {
            "partition-scheme": "risk_hetero",
            "min-anomaly-ratio": 0.005,
            "max-anomaly-ratio": 0.12,
            "normal-shift-max": 0.25,
            "epsilon-heterogeneous": True,
            "epsilon-min": 0.01,
            "epsilon-max": 0.08,
            "threshold-init": 0.20,
            "global-eval-threshold": 0.20,
            "fraction-evaluate": 1.0,
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
    "swat": {
        "batch-size": 256,
        "central-batch-size": 2048,
        "min-anomaly-ratio": 0.005,
        "max-anomaly-ratio": 0.10,
        "global-anomaly-ratio": 0.04,
        "epsilon-fpr": 0.03,
    },
    "hai": {
        "batch-size": 256,
        "central-batch-size": 2048,
        "min-anomaly-ratio": 0.005,
        "max-anomaly-ratio": 0.10,
        "global-anomaly-ratio": 0.03,
        "epsilon-fpr": 0.03,
    },
    "smd": {
        "batch-size": 64,
        "central-batch-size": 256,
        "min-anomaly-ratio": 0.01,
        "max-anomaly-ratio": 0.10,
        "global-anomaly-ratio": 0.05,
        "epsilon-fpr": 0.03,
    },
    "paysim": {
        "batch-size": 1024,
        "central-batch-size": 8192,
        "min-anomaly-ratio": 0.0005,
        "max-anomaly-ratio": 0.01,
        "global-anomaly-ratio": 0.002,
        "epsilon-fpr": 0.01,
        "threshold-init": 0.05,
        "global-eval-threshold": 0.05,
    },
    "mammography": {
        "batch-size": 128,
        "central-batch-size": 512,
        "min-anomaly-ratio": 0.005,
        "max-anomaly-ratio": 0.08,
        "global-anomaly-ratio": 0.02,
        "epsilon-fpr": 0.03,
        "threshold-init": 0.05,
        "global-eval-threshold": 0.05,
    },
    "annthyroid": {
        "batch-size": 256,
        "central-batch-size": 1024,
        "min-anomaly-ratio": 0.01,
        "max-anomaly-ratio": 0.12,
        "global-anomaly-ratio": 0.07,
        "epsilon-fpr": 0.03,
        "threshold-init": 0.05,
        "global-eval-threshold": 0.05,
    },
    "shuttle": {
        "batch-size": 512,
        "central-batch-size": 2048,
        "min-anomaly-ratio": 0.01,
        "max-anomaly-ratio": 0.12,
        "global-anomaly-ratio": 0.07,
        "epsilon-fpr": 0.03,
        "threshold-init": 0.05,
        "global-eval-threshold": 0.05,
    },
    "tep": {
        "batch-size": 256,
        "central-batch-size": 1024,
        "min-anomaly-ratio": 0.01,
        "max-anomaly-ratio": 0.12,
        "global-anomaly-ratio": 0.05,
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
        "swat": "swat",
        "hai": "hai",
        "smd": "smd",
        "paysim": "paysim",
        "paysim1": "paysim",
        "mammography": "mammography",
        "oddsmammography": "mammography",
        "annthyroid": "annthyroid",
        "oddsannthyroid": "annthyroid",
        "thyroid": "annthyroid",
        "shuttle": "shuttle",
        "oddsshuttle": "shuttle",
        "tep": "tep",
        "tennesseeeastman": "tep",
        "tennesseeeastmanprocess": "tep",
    }
    return aliases.get(normalized, normalized)


def _format_value(v: Any) -> str:
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def build_run_config(cfg: dict[str, Any]) -> str:
    return " ".join(f"{k}={_format_value(v)}" for k, v in cfg.items())


def write_run_config(path: Path, cfg: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(f"{key} = {_format_value(value)}" for key, value in cfg.items()) + "\n",
        encoding="utf-8",
    )


def flwr_executable() -> str:
    """Prefer the Flower executable from the active Python environment."""

    candidate = Path(sys.executable).with_name("flwr")
    return str(candidate) if candidate.exists() else "flwr"


def ray_executable() -> str:
    """Prefer the Ray executable from the active Python environment."""

    candidate = Path(sys.executable).with_name("ray")
    return str(candidate) if candidate.exists() else "ray"


def flower_executable(name: str) -> str:
    candidate = Path(sys.executable).with_name(name)
    return str(candidate) if candidate.exists() else name


def start_local_superlink(env: dict[str, str], log_path: Path) -> subprocess.Popen:
    state_root = env.get("RCFAD_FLOWER_STATE_DIR")
    if state_root is None:
        ray_tmp = env.get("RCFAD_RAY_TMPDIR") or env.get("RAY_TMPDIR") or env.get("TMPDIR")
        if ray_tmp:
            state_root = str(Path(ray_tmp) / "local-superlink")
    local_superlink = Path(state_root) if state_root else Path.home() / ".flwr" / "local-superlink"
    local_superlink.mkdir(parents=True, exist_ok=True)
    cmd = [
        flower_executable("flower-superlink"),
        "--insecure",
        "--simulation",
        "--isolation",
        "subprocess",
        "--control-api-address",
        "127.0.0.1:39093",
        "--simulationio-api-address",
        "127.0.0.1:39094",
        "--database",
        str(local_superlink / "state.db"),
    ]
    handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    time.sleep(4.0)
    if proc.poll() is not None:
        handle.close()
        raise RuntimeError(f"flower-superlink exited early with code {proc.returncode}")
    return proc


def cleanup_runtime(env: dict[str, str], *, force: bool = False) -> None:
    """Clean up Flower/Ray simulation processes left behind by a run.

    Flower's local simulation can leave ``flwr-simulation`` actors alive after a
    failed run. Those actors keep CUDA contexts and large Ray object stores open,
    which can make every later experiment fail with CUDA OOM even though the
    failing run has already returned.
    """

    patterns = [
        "ray::",
        "raylet",
        "gcs_server",
        "ClientAppActor",
        "dashboard/agent.py",
        "runtime_env/agent",
        "flwr-simulation",
        "flwr run",
        "flower-superexec",
        "flower-superlink",
        "flower-supernode",
    ]
    ray_cmd = ray_executable()
    if Path(ray_cmd).exists() or shutil.which(ray_cmd):
        subprocess.run(
            [ray_cmd, "stop", "--force"],
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
    # Keep Flower's local SuperLink state DB in place. Flower 1.27 can fail on
    # a freshly removed DB with internal ``'script'``/SQLite initialization
    # errors, so process cleanup is the reliable part to do between runs.
    state_root = env.get("RCFAD_FLOWER_STATE_DIR")
    if state_root is None:
        ray_tmp = env.get("RCFAD_RAY_TMPDIR") or env.get("RAY_TMPDIR") or env.get("TMPDIR")
        if ray_tmp:
            state_root = str(Path(ray_tmp) / "local-superlink")
    (Path(state_root) if state_root else Path.home() / ".flwr" / "local-superlink").mkdir(
        parents=True,
        exist_ok=True,
    )


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
    parser.add_argument("--num-supernodes", type=int, default=10)
    parser.add_argument(
        "--client-cpus",
        type=float,
        default=8.0,
        help="Ray CPU resources requested per simulated client actor.",
    )
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
    parser.add_argument(
        "--run-timeout-seconds",
        type=int,
        default=7200,
        help="Timeout for each Flower run; 0 disables the timeout.",
    )
    parser.add_argument(
        "--no-precache",
        action="store_true",
        help="Skip tabular dataset mmap cache preparation before Flower starts.",
    )
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
    elif args.suite == "threshold_hetero":
        scenario_grid = THRESHOLD_HETERO_SCENARIOS
    elif args.suite == "ablation_risk":
        scenario_grid = ABLATION_RISK_SCENARIOS
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
                        "learning-rate": args.learning_rate,
                        "local-epochs": args.local_epochs,
                        "scenario-name": scenario_name,
                    }
                    if dataset_key(dataset) in {"mnist", "fmnist", "cifar10"}:
                        cfg["anomaly-class"] = args.anomaly_class
                    cfg.update(COMMON_CONFIG)
                    cfg.update(DATASET_PRESETS.get(dataset_key(dataset), {}))
                    cfg.update(METHODS[method])
                    cfg.update(scenario_cfg)
                    if args.suite == "ablation_risk" and method == "base_no_modules":
                        # Use a fixed, pre-specified deployment threshold instead
                        # of the risk scenario's RC-FAD initialization. This keeps
                        # the no-module baseline from degenerating into an all-
                        # negative decision rule while still giving it no learned
                        # threshold, no FPR constraint, and no dynamic enhancement.
                        cfg["eval-threshold-mode"] = "fixed"
                        cfg["threshold-init"] = 0.15
                        cfg["global-eval-threshold"] = 0.15
                    if args.suite == "ablation_risk" and method == "wo_joint_threshold":
                        # Fixed-threshold ablation: no tau gradient and no local
                        # post-hoc calibration. The 0.40 work point avoids the
                        # pathological all-negative 0.50 setting in rare-anomaly
                        # splits while preserving the intended no-joint-threshold
                        # treatment.
                        cfg["threshold-init"] = 0.4
                        cfg["global-eval-threshold"] = 0.4
                    cfg.update(extra_cfg)
                    planned_runs.append((dataset, seed, scenario_name, method, run_name, cfg))

    manifest_path = logs_root / "manifest.json"
    manifest = []
    if not args.dry_run and not args.no_precache:
        try:
            from RCFAD.task import _is_tabular_dataset, _load_tabular_arrays
        except Exception as exc:
            print(f"Warning: could not import dataset cache helpers: {exc}", file=sys.stderr)
        else:
            seen_cache_keys: set[tuple[str, int, str]] = set()
            for dataset, seed, _, _, _, cfg in planned_runs:
                data_root = str(cfg.get("data-root", "./data"))
                cache_key = (dataset_key(dataset), int(seed), data_root)
                if cache_key in seen_cache_keys or not _is_tabular_dataset(dataset):
                    continue
                seen_cache_keys.add(cache_key)
                print(f"Preparing mmap cache for dataset={dataset}, seed={seed} ...")
                _load_tabular_arrays(dataset, data_root, int(seed))

    for i, (dataset, seed, scenario_name, method, run_name, cfg) in enumerate(planned_runs, start=1):
        run_config = build_run_config(cfg)
        cmd = [
            flwr_executable(),
            "run",
            ".",
            "--stream",
            "--run-config",
            run_config,
        ]
        log_path = logs_root / f"{run_name}.log"
        summary_path = Path(str(cfg.get("results-root", args.results_root))) / run_name / "summary.json"
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
            env_bin = str(Path(sys.executable).parent)
            env["PATH"] = env_bin + os.pathsep + env.get("PATH", "")
            ray_tmp = Path(
                os.environ.get(
                    "RCFAD_RAY_TMPDIR",
                    f"/tmp/rcfad_ray_{os.getuid()}",
                )
            ).resolve()
            ray_tmp.mkdir(parents=True, exist_ok=True)
            env.setdefault("RAY_TMPDIR", str(ray_tmp))
            env.setdefault("TMPDIR", str(ray_tmp))
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
            superlink_proc: subprocess.Popen | None = None
            try:
                superlink_proc = start_local_superlink(env, logs_root / "manual_superlink.log")
                started = time.monotonic()
                summary_seen_at: float | None = None
                popen = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    start_new_session=True,
                )
                while True:
                    returncode = popen.poll()
                    if returncode is not None:
                        proc = subprocess.CompletedProcess(cmd, returncode)
                        break

                    if summary_path.exists():
                        if summary_seen_at is None:
                            summary_seen_at = time.monotonic()
                            f.write(
                                "\nsummary.json detected; waiting briefly, then "
                                "terminating Flower to avoid post-run shutdown hangs.\n"
                            )
                            f.flush()
                        elif time.monotonic() - summary_seen_at >= 8:
                            try:
                                os.killpg(popen.pid, signal.SIGTERM)
                            except OSError:
                                popen.terminate()
                            try:
                                popen.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                try:
                                    os.killpg(popen.pid, signal.SIGKILL)
                                except OSError:
                                    popen.kill()
                                popen.wait(timeout=20)
                            proc = subprocess.CompletedProcess(cmd, 0)
                            break

                    if (
                        args.run_timeout_seconds > 0
                        and time.monotonic() - started > args.run_timeout_seconds
                    ):
                        raise subprocess.TimeoutExpired(cmd, args.run_timeout_seconds)
                    time.sleep(2)
            except subprocess.TimeoutExpired as exc:
                f.write(
                    f"\nRun timed out after {args.run_timeout_seconds} seconds; "
                    "cleaning up Flower/Ray processes.\n"
                )
                f.flush()
                proc = subprocess.CompletedProcess(cmd, 124)
            finally:
                if superlink_proc is not None and superlink_proc.poll() is None:
                    try:
                        superlink_proc.terminate()
                    except OSError:
                        pass
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
