"""Flower ClientApp for RC-FAD."""

from __future__ import annotations

import os
import pickle
import time
from typing import Any

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from RCFAD.task import Net, TrainProcessMetadata, load_data, set_seed
from RCFAD.task import test as test_fn
from RCFAD.task import train as train_fn

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
print("Running client_app from:", os.path.abspath(__file__))

app = ClientApp()

# Client-side state. In Flower simulation this usually persists within the
# client process. It stores threshold and Lagrange multiplier for each client.
CLIENT_STATE: dict[int, dict[str, Any]] = {}


def _cfg(config: Any, key: str, default: Any) -> Any:
    try:
        return config[key]
    except Exception:
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _device(config: Any) -> torch.device:
    requested = str(
        _cfg(config, "client-device", _cfg(config, "device", os.environ.get("RCFAD_DEVICE", "auto")))
    ).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _state_id(partition_id: int, run_cfg: Any) -> int:
    # For the global-joint threshold diagnostic, all partitions hosted by the
    # same client process share one threshold state. In simulation this is an
    # approximation of a server-wide global threshold.
    if str(_cfg(run_cfg, "threshold-scope", "client")).lower() == "global":
        return -1
    return int(partition_id)


def _client_epsilon(run_cfg: Any, partition_id: int, num_partitions: int) -> float:
    if _as_bool(_cfg(run_cfg, "epsilon-heterogeneous", False)):
        eps_min = float(_cfg(run_cfg, "epsilon-min", _cfg(run_cfg, "epsilon-fpr", 0.01)))
        eps_max = float(_cfg(run_cfg, "epsilon-max", _cfg(run_cfg, "epsilon-fpr", 0.01)))
        if num_partitions <= 1:
            return eps_min
        return eps_min + (eps_max - eps_min) * (partition_id / max(1, num_partitions - 1))
    return float(_cfg(run_cfg, "epsilon-fpr", 0.01))


def _get_state(partition_id: int, threshold_init: float, lambda_init: float = 1.0) -> dict[str, Any]:
    if partition_id not in CLIENT_STATE:
        CLIENT_STATE[partition_id] = {
            "threshold": float(threshold_init),
            "lambda_fpr": float(lambda_init),
        }
    return CLIENT_STATE[partition_id]


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    start_time = time.time()

    run_cfg = context.run_config
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = _device(run_cfg)
    model.to(device)

    batch_size = int(_cfg(run_cfg, "batch-size", 32))
    dataset_name = str(_cfg(run_cfg, "dataset-name", "mnist"))
    anomaly_class = int(_cfg(run_cfg, "anomaly-class", 1))
    partition_scheme = str(_cfg(run_cfg, "partition-scheme", "risk_hetero"))
    seed = int(_cfg(run_cfg, "seed", 42))
    eval_config = msg.content["config"] if "config" in msg.content else {}
    server_round = int(_cfg(eval_config, "server-round", 1))
    set_seed(seed + partition_id * 1000 + server_round)
    min_anomaly_ratio = float(_cfg(run_cfg, "min-anomaly-ratio", 0.005))
    max_anomaly_ratio = float(_cfg(run_cfg, "max-anomaly-ratio", 0.10))
    dirichlet_alpha = float(_cfg(run_cfg, "dirichlet-alpha", 0.3))
    data_root = str(_cfg(run_cfg, "data-root", "./data"))

    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        dataset_name=dataset_name,
        anomaly_class=anomaly_class,
        partition_scheme=partition_scheme,
        seed=seed,
        min_anomaly_ratio=min_anomaly_ratio,
        max_anomaly_ratio=max_anomaly_ratio,
        dirichlet_alpha=dirichlet_alpha,
        data_root=data_root,
        dataloader_seed=seed + partition_id * 1000 + server_round,
        normal_shift_max=float(_cfg(run_cfg, "normal-shift-max", 0.0)),
    )

    state = _get_state(
        _state_id(partition_id, run_cfg),
        float(_cfg(run_cfg, "threshold-init", 0.5)),
        float(_cfg(run_cfg, "lambda-init", 1.0)),
    )

    train_config = msg.content["config"] if "config" in msg.content else {}
    lr = float(_cfg(train_config, "lr", _cfg(run_cfg, "learning-rate", 0.001)))
    global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    global_ref = None
    previous_ref = None
    supervisor_ref = None
    if float(_cfg(run_cfg, "moon-mu", 0.0)) > 0.0:
        global_ref = Net()
        global_ref.load_state_dict(global_state)
        prev_state = state.get("previous_model_state")
        if prev_state is not None:
            previous_ref = Net()
            previous_ref.load_state_dict(prev_state)
    if float(_cfg(run_cfg, "fedsimsup-mu", 0.0)) > 0.0:
        supervisor_state = state.get("fedsimsup_supervisor_state")
        if supervisor_state is not None:
            supervisor_ref = Net()
            supervisor_ref.load_state_dict(supervisor_state)
    train_loss, epoch_losses, risk_metrics, new_threshold, new_lambda_fpr = train_fn(
        model,
        trainloader,
        int(_cfg(run_cfg, "local-epochs", 1)),
        lr,
        device,
        threshold=state["threshold"],
        lambda_fpr=state["lambda_fpr"],
        epsilon_fpr=_client_epsilon(run_cfg, partition_id, num_partitions),
        temperature=float(_cfg(run_cfg, "temperature", 0.05)),
        c_fn=float(_cfg(run_cfg, "c-fn", 5.0)),
        c_fp=float(_cfg(run_cfg, "c-fp", 1.0)),
        global_anomaly_ratio=float(_cfg(run_cfg, "global-anomaly-ratio", 0.10)),
        beta_power=float(_cfg(run_cfg, "beta-power", 0.5)),
        beta_alpha=float(_cfg(run_cfg, "beta-alpha", 1.0)),
        beta_zeta=float(_cfg(run_cfg, "beta-zeta", 1.0)),
        beta_min=float(_cfg(run_cfg, "beta-min", 1.0)),
        beta_max=float(_cfg(run_cfg, "beta-max", 20.0)),
        beta_reference_ratio=float(_cfg(run_cfg, "beta-reference-ratio", 0.0)),
        beta_gate_mode=str(_cfg(run_cfg, "beta-gate-mode", "smooth")),
        beta_round_decay=float(_cfg(run_cfg, "beta-round-decay", 0.0)),
        mu_fnr=float(_cfg(run_cfg, "mu-fnr", 0.2)),
        eta_lambda=float(_cfg(run_cfg, "eta-lambda", 1.0)),
        lr_tau=float(_cfg(run_cfg, "lr-tau", 0.002)),
        fpr_penalty=float(_cfg(run_cfg, "fpr-penalty", 2.0)),
        server_round=server_round,
        risk_weight_warmup_rounds=int(_cfg(run_cfg, "risk-weight-warmup-rounds", 3)),
        risk_gamma_fnr=float(_cfg(run_cfg, "risk-gamma-fnr", 0.5)),
        risk_gamma_fpr=float(_cfg(run_cfg, "risk-gamma-fpr", 0.5)),
        risk_weight_min_factor=float(_cfg(run_cfg, "risk-weight-min-factor", 0.5)),
        risk_weight_max_factor=float(_cfg(run_cfg, "risk-weight-max-factor", 2.0)),
        loss_type=str(_cfg(run_cfg, "loss-type", "bce")),
        focal_gamma=float(_cfg(run_cfg, "focal-gamma", 2.0)),
        focal_alpha=float(_cfg(run_cfg, "focal-alpha", -1.0)),
        class_balanced_beta=float(_cfg(run_cfg, "class-balanced-beta", 0.9999)),
        fedprox_mu=float(_cfg(run_cfg, "fedprox-mu", 0.0)),
        global_params=global_state if float(_cfg(run_cfg, "fedprox-mu", 0.0)) > 0.0 else None,
        moon_mu=float(_cfg(run_cfg, "moon-mu", 0.0)),
        moon_temperature=float(_cfg(run_cfg, "moon-temperature", 0.5)),
        global_model=global_ref,
        previous_model=previous_ref,
        fedsimsup_mu=float(_cfg(run_cfg, "fedsimsup-mu", 0.0)),
        fedsimsup_temperature=float(_cfg(run_cfg, "fedsimsup-temperature", 0.5)),
        supervisor_model=supervisor_ref,
    )

    state["threshold"] = new_threshold
    state["lambda_fpr"] = new_lambda_fpr
    if float(_cfg(run_cfg, "moon-mu", 0.0)) > 0.0:
        state["previous_model_state"] = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    if float(_cfg(run_cfg, "fedsimsup-mu", 0.0)) > 0.0:
        new_supervisor_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        old_supervisor_state = state.get("fedsimsup_supervisor_state")
        ema = float(_cfg(run_cfg, "fedsimsup-supervisor-ema", 0.5))
        if old_supervisor_state is not None and 0.0 < ema < 1.0:
            new_supervisor_state = {
                k: ema * old_supervisor_state[k] + (1.0 - ema) * v
                for k, v in new_supervisor_state.items()
            }
        state["fedsimsup_supervisor_state"] = new_supervisor_state

    end_time = time.time()
    converged = abs(epoch_losses[-1] - epoch_losses[-2]) < 1e-3 if len(epoch_losses) >= 2 else False

    train_metadata = TrainProcessMetadata(
        training_time=end_time - start_time,
        converged=converged,
        training_losses={f"epoch_{i + 1}": loss for i, loss in enumerate(epoch_losses)},
        risk_metrics=risk_metrics,
    )
    meta_bytes = pickle.dumps(train_metadata)
    config_record = ConfigRecord({"meta": meta_bytes})

    model_record = ArrayRecord(model.state_dict())

    metrics = {
        # Required weight keys
        "num-examples": int(len(trainloader.dataset)),
        "risk_weight": float(risk_metrics.get("risk_weight", len(trainloader.dataset))),
        # Client identity is numeric so it can travel inside Flower MetricRecord.
        "client_id": float(partition_id),
        # Training metrics
        "train_loss": float(train_loss),
        "threshold": float(new_threshold),
        "lambda_fpr": float(new_lambda_fpr),
        "beta": float(risk_metrics.get("beta", 1.0)),
        "beta_logit": float(risk_metrics.get("beta_logit", 0.0)),
        "train_fpr": float(risk_metrics.get("fpr", 0.0)),
        "train_fnr": float(risk_metrics.get("fnr", 0.0)),
        # Smooth risk proxies implement the differentiable quantities in Method.txt.
        "smooth_train_fpr": float(risk_metrics.get("smooth_fpr", risk_metrics.get("fpr", 0.0))),
        "smooth_train_fnr": float(risk_metrics.get("smooth_fnr", risk_metrics.get("fnr", 0.0))),
        "constraint_violation": float(risk_metrics.get("constraint_violation", 0.0)),
        "local_anomaly_ratio": float(risk_metrics.get("anomaly_ratio", 0.0)),
        "train_recall": float(risk_metrics.get("recall", 0.0)),
        "train_auprc": float(risk_metrics.get("auprc", 0.0)),
        "train_recall_at_fpr": float(risk_metrics.get("recall_at_fpr", 0.0)),
        "target_fpr": float(_client_epsilon(run_cfg, partition_id, num_partitions)),
        "fpr_violation": max(
            float(risk_metrics.get("fpr", 0.0)) - float(_client_epsilon(run_cfg, partition_id, num_partitions)),
            0.0,
        ),
        "fpr_violation_flag": float(
            float(risk_metrics.get("fpr", 0.0)) > float(_client_epsilon(run_cfg, partition_id, num_partitions))
        ),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({
        "arrays": model_record,
        "metrics": metric_record,
        "train_metadata": config_record,
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local validation data."""

    run_cfg = context.run_config
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = _device(run_cfg)
    model.to(device)

    batch_size = int(_cfg(run_cfg, "batch-size", 32))
    dataset_name = str(_cfg(run_cfg, "dataset-name", "mnist"))
    anomaly_class = int(_cfg(run_cfg, "anomaly-class", 1))
    partition_scheme = str(_cfg(run_cfg, "partition-scheme", "risk_hetero"))
    seed = int(_cfg(run_cfg, "seed", 42))
    eval_config = msg.content["config"] if "config" in msg.content else {}
    server_round = int(_cfg(eval_config, "server-round", 1))
    set_seed(seed + partition_id * 1000 + server_round)
    min_anomaly_ratio = float(_cfg(run_cfg, "min-anomaly-ratio", 0.005))
    max_anomaly_ratio = float(_cfg(run_cfg, "max-anomaly-ratio", 0.10))
    dirichlet_alpha = float(_cfg(run_cfg, "dirichlet-alpha", 0.3))
    data_root = str(_cfg(run_cfg, "data-root", "./data"))

    _, valloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        dataset_name=dataset_name,
        anomaly_class=anomaly_class,
        partition_scheme=partition_scheme,
        seed=seed,
        min_anomaly_ratio=min_anomaly_ratio,
        max_anomaly_ratio=max_anomaly_ratio,
        dirichlet_alpha=dirichlet_alpha,
        data_root=data_root,
        dataloader_seed=seed + partition_id * 1000,
        normal_shift_max=float(_cfg(run_cfg, "normal-shift-max", 0.0)),
    )

    state = _get_state(
        _state_id(partition_id, run_cfg),
        float(_cfg(run_cfg, "threshold-init", 0.5)),
        float(_cfg(run_cfg, "lambda-init", 1.0)),
    )
    target_fpr = _client_epsilon(run_cfg, partition_id, num_partitions)
    threshold = float(state["threshold"])
    threshold_mode = str(_cfg(run_cfg, "eval-threshold-mode", "learned")).lower()
    if threshold_mode in {"fixed", "fixed_0_5", "fixed-0.5"}:
        threshold = float(_cfg(run_cfg, "threshold-init", 0.5))
    elif threshold_mode in {"local_posthoc", "local-posthoc", "posthoc"}:
        _, probe_metrics = test_fn(
            model,
            valloader,
            device,
            threshold=float(_cfg(run_cfg, "threshold-init", 0.5)),
            target_fpr=target_fpr,
        )
        threshold = float(probe_metrics.get("threshold_at_fpr", threshold))
    eval_loss, eval_metrics = test_fn(
        model,
        valloader,
        device,
        threshold=threshold,
        target_fpr=target_fpr,
    )

    metrics = {
        "num-examples": int(len(valloader.dataset)),
        # Include risk_weight so evaluate aggregation still works if weighted_by_key=risk_weight
        "risk_weight": float(len(valloader.dataset)),
        "conflict_free": float(len(valloader.dataset)),
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_metrics.get("accuracy", 0.0)),
        "accuracy": float(eval_metrics.get("accuracy", 0.0)),
        "precision": float(eval_metrics.get("precision", 0.0)),
        "recall": float(eval_metrics.get("recall", 0.0)),
        "fpr": float(eval_metrics.get("fpr", 0.0)),
        "fnr": float(eval_metrics.get("fnr", 0.0)),
        "f1": float(eval_metrics.get("f1", 0.0)),
        "auc": float(eval_metrics.get("auc", 0.0)),
        "auprc": float(eval_metrics.get("auprc", 0.0)),
        "recall_at_fpr": float(eval_metrics.get("recall_at_fpr", 0.0)),
        "threshold": float(threshold),
        "target_fpr": float(target_fpr),
        "fpr_violation": max(float(eval_metrics.get("fpr", 0.0)) - float(target_fpr), 0.0),
        "fpr_violation_flag": float(float(eval_metrics.get("fpr", 0.0)) > float(target_fpr)),
        # Metrics under the calibrated threshold that controls FPR at epsilon.
        # These are the recommended threshold-dependent metrics for paper tables.
        "threshold_at_fpr": float(eval_metrics.get("threshold_at_fpr", 0.0)),
        "accuracy_at_fpr": float(eval_metrics.get("accuracy_at_fpr", 0.0)),
        "precision_at_fpr": float(eval_metrics.get("precision_at_fpr", 0.0)),
        "recall_at_fpr_threshold": float(eval_metrics.get("recall_at_fpr_threshold", 0.0)),
        "fpr_at_fpr_threshold": float(eval_metrics.get("fpr_at_fpr_threshold", 0.0)),
        "fnr_at_fpr_threshold": float(eval_metrics.get("fnr_at_fpr_threshold", 0.0)),
        "f1_at_fpr_threshold": float(eval_metrics.get("f1_at_fpr_threshold", 0.0)),
    }

    metric_record = MetricRecord(metrics)
    return Message(content=RecordDict({"metrics": metric_record}), reply_to=msg)
