"""Flower ServerApp for RC-FAD."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from RCFAD.custom_strategy import CustomFedAdagrad
from RCFAD.task import Net, load_centralized_dataset, set_seed
from RCFAD.task import test as test_fn

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

app = ServerApp()
SERVER_RUN_CONFIG: dict[str, Any] = {}


def _cfg(config: Any, key: str, default: Any) -> Any:
    try:
        return config[key]
    except Exception:
        return default


def _device(config: Any) -> torch.device:
    requested = str(
        _cfg(config, "server-device", _cfg(config, "device", os.environ.get("RCFAD_DEVICE", "auto")))
    ).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    global SERVER_RUN_CONFIG
    SERVER_RUN_CONFIG = dict(context.run_config)

    fraction_evaluate = float(_cfg(context.run_config, "fraction-evaluate", 0.5))
    num_rounds = int(_cfg(context.run_config, "num-server-rounds", 10))
    lr = float(_cfg(context.run_config, "learning-rate", 0.01))
    aggregation_key = str(_cfg(context.run_config, "aggregation-key", "risk_weight"))
    seed = int(_cfg(context.run_config, "seed", 42))

    # Important for reproducibility: initialize the server model after seeding.
    set_seed(seed)
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = CustomFedAdagrad(
        fraction_evaluate=fraction_evaluate,
        weighted_by_key=aggregation_key,
    )

    current_time = datetime.now()
    run_name = str(_cfg(context.run_config, "run-name", "")).strip()
    results_root = Path(str(_cfg(context.run_config, "results-root", "outputs/experiments")))
    if run_name:
        save_path = Path.cwd() / results_root / run_name
        if save_path.exists():
            # Keep previous results by appending a timestamp if the run already exists.
            save_path = Path.cwd() / results_root / f"{run_name}_{current_time.strftime('%Y%m%d-%H%M%S')}"
    else:
        run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
        save_path = Path.cwd() / "outputs" / run_dir
    save_path.mkdir(parents=True, exist_ok=False)
    strategy.set_save_path(save_path)
    strategy.set_experiment_info(dict(context.run_config))
    with (save_path / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(dict(context.run_config), f, indent=2, ensure_ascii=False)
    print("Saving experiment artifacts to:", save_path)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on centralized test data."""

    seed = int(SERVER_RUN_CONFIG.get("seed", 42))
    set_seed(seed + 999_999 + int(server_round))
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = _device(SERVER_RUN_CONFIG)
    model.to(device)

    test_dataloader = load_centralized_dataset(
        dataset_name=str(SERVER_RUN_CONFIG.get("dataset-name", "mnist")),
        anomaly_class=int(SERVER_RUN_CONFIG.get("anomaly-class", 1)),
        batch_size=int(SERVER_RUN_CONFIG.get("central-batch-size", 128)),
        data_root=str(SERVER_RUN_CONFIG.get("data-root", "./data")),
        seed=seed,
    )

    eval_loss, metrics = test_fn(
        model,
        test_dataloader,
        device,
        threshold=float(SERVER_RUN_CONFIG.get("global-eval-threshold", SERVER_RUN_CONFIG.get("threshold-init", 0.5))),
        target_fpr=float(SERVER_RUN_CONFIG.get("epsilon-fpr", 0.01)),
    )

    # Keep fixed-threshold metrics for compatibility, and additionally return
    # target-FPR-calibrated metrics (e.g., f1_at_fpr_threshold) to avoid
    # confusing "all-normal at fixed threshold" with poor ranking quality.
    metric_names = [
        "accuracy", "precision", "recall", "fpr", "fnr", "f1",
        "auc", "auprc", "recall_at_fpr", "anomaly_ratio",
        "threshold", "target_fpr", "threshold_fixed", "threshold_at_fpr",
        "accuracy_at_fpr", "precision_at_fpr", "recall_at_fpr_threshold",
        "fpr_at_fpr_threshold", "fnr_at_fpr_threshold", "f1_at_fpr_threshold",
    ]
    out = {"loss": float(eval_loss)}
    for name in metric_names:
        out[name] = float(metrics.get(name, 0.0))
    target_fpr = float(out.get("target_fpr", SERVER_RUN_CONFIG.get("epsilon-fpr", 0.01)))
    out["fpr_violation"] = max(float(out.get("fpr", 0.0)) - target_fpr, 0.0)
    out["fpr_violation_flag"] = float(float(out.get("fpr", 0.0)) > target_fpr)
    out["fpr_at_target_violation"] = max(
        float(out.get("fpr_at_fpr_threshold", 0.0)) - target_fpr,
        0.0,
    )
    return MetricRecord(out)
