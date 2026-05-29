"""Local training and evaluation loops for RC-FAD."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from RCFAD.metrics import collect_probs_and_labels, evaluate_binary_metrics


def _smooth_fpr_fnr(
    probs: torch.Tensor,
    labels: torch.Tensor,
    tau: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    neg_mask = labels <= 0.5
    pos_mask = labels > 0.5

    if neg_mask.any():
        smooth_fpr = torch.sigmoid((probs[neg_mask] - tau) / temperature).mean()
    else:
        smooth_fpr = torch.zeros((), device=probs.device)

    if pos_mask.any():
        smooth_fnr = torch.sigmoid((tau - probs[pos_mask]) / temperature).mean()
    else:
        smooth_fnr = torch.zeros((), device=probs.device)

    return smooth_fpr, smooth_fnr


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    *,
    threshold: float = 0.5,
    lambda_fpr: float = 0.0,
    epsilon_fpr: float = 0.01,
    temperature: float = 0.05,
    c_fn: float = 5.0,
    c_fp: float = 1.0,
    global_anomaly_ratio: float = 0.10,
    beta_power: float = 0.5,
    beta_alpha: float = 1.0,
    beta_zeta: float = 1.0,
    beta_min: float = 1.0,
    beta_max: float = 20.0,
    beta_reference_ratio: float = 0.0,
    beta_gate_mode: str = "smooth",
    beta_round_decay: float = 0.0,
    mu_fnr: float = 0.2,
    eta_lambda: float = 1.0,
    lr_tau: float = 0.002,
    threshold_projection_blend: float = 0.0,
    threshold_projection_fpr_factor: float = 1.0,
    fpr_penalty: float = 2.0,
    server_round: int = 1,
    risk_weight_warmup_rounds: int = 3,
    risk_gamma_fnr: float = 0.5,
    risk_gamma_fpr: float = 0.5,
    risk_weight_min_factor: float = 0.5,
    risk_weight_max_factor: float = 2.0,
    loss_type: str = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: float = -1.0,
    class_balanced_beta: float = 0.9999,
    fedprox_mu: float = 0.0,
    global_params: Dict[str, torch.Tensor] | None = None,
    moon_mu: float = 0.0,
    moon_temperature: float = 0.5,
    global_model: nn.Module | None = None,
    previous_model: nn.Module | None = None,
    fedsimsup_mu: float = 0.0,
    fedsimsup_temperature: float = 0.5,
    supervisor_model: nn.Module | None = None,
) -> Tuple[float, List[float], Dict[str, float], float, float]:
    """Train local model with RC-FAD objective.

    Returns:
        final_loss, epoch_losses, risk_metrics, new_threshold, new_lambda_fpr
    """

    net.to(device)
    net.train()
    if global_model is not None:
        global_model.to(device)
        global_model.eval()
    if previous_model is not None:
        previous_model.to(device)
        previous_model.eval()
    if supervisor_model is not None:
        supervisor_model.to(device)
        supervisor_model.eval()
    if global_params is not None:
        global_params = {k: v.detach().to(device) for k, v in global_params.items()}

    # Pre-train risk state for the paper's dynamic minority enhancement.
    # The hard metrics remain useful diagnostics, but beta is driven by the
    # differentiable FPR/FNR proxies used in the theoretical method.
    pre_metrics = evaluate_binary_metrics(
        net, trainloader, device, threshold, epsilon_fpr, temperature
    )
    local_ratio = max(float(pre_metrics.get("anomaly_ratio", 0.0)), 1e-8)
    pre_hard_fnr = float(pre_metrics.get("fnr", 0.0))
    pre_hard_fpr = float(pre_metrics.get("fpr", 0.0))
    pre_smooth_fnr = float(pre_metrics.get("smooth_fnr", pre_hard_fnr))
    pre_smooth_fpr = float(pre_metrics.get("smooth_fpr", pre_hard_fpr))
    epsilon_safe = max(float(epsilon_fpr), 1e-8)
    global_ratio = max(float(global_anomaly_ratio), 1e-8)
    reference_ratio = max(global_ratio, float(beta_reference_ratio), 1e-8)
    scarcity = np.clip((reference_ratio - local_ratio) / reference_ratio, 0.0, 1.0)
    scarcity_gate = float(scarcity ** max(float(beta_power), 0.0))
    gate_mode = str(beta_gate_mode).lower()
    gate_fnr = pre_hard_fnr if gate_mode in {"hard", "hard_risk", "hard-risk"} else pre_smooth_fnr
    gate_fpr = pre_hard_fpr if gate_mode in {"hard", "hard_risk", "hard-risk"} else pre_smooth_fpr
    fnr_gate = 1.0 / (
        1.0 + np.exp(-float(beta_alpha) * (gate_fnr - epsilon_safe))
    )
    fpr_budget_margin = (epsilon_safe - gate_fpr) / epsilon_safe
    fpr_budget_gate = 1.0 / (
        1.0 + np.exp(-float(beta_zeta) * (fpr_budget_margin - 0.25))
    )
    if float(beta_round_decay) > 0.0:
        round_gate = max(
            0.0,
            1.0 - max(0, int(server_round) - 1) / max(float(beta_round_decay), 1.0),
        )
    else:
        round_gate = 1.0

    # Risk-gated hard-positive enhancement: increase minority pressure only
    # when positives are scarce, false negatives remain high, and FPR has room.
    beta_logit = scarcity_gate * fnr_gate * fpr_budget_gate * round_gate
    beta_k = float(beta_min + (beta_max - beta_min) * beta_logit)
    beta_k = float(np.clip(beta_k, beta_min, beta_max))

    dataset_size = max(1, int(len(trainloader.dataset)))
    pos_count = max(1.0, local_ratio * dataset_size)
    neg_count = max(1.0, (1.0 - local_ratio) * dataset_size)
    cb_beta = float(np.clip(class_balanced_beta, 0.0, 0.999999))
    if cb_beta > 0.0:
        pos_cb = (1.0 - cb_beta) / max(1.0 - cb_beta ** pos_count, 1e-12)
        neg_cb = (1.0 - cb_beta) / max(1.0 - cb_beta ** neg_count, 1e-12)
    else:
        pos_cb = neg_cb = 1.0
    cb_norm = max(pos_cb + neg_cb, 1e-12)
    pos_cb_weight = float(2.0 * pos_cb / cb_norm)
    neg_cb_weight = float(2.0 * neg_cb / cb_norm)

    tau = nn.Parameter(torch.tensor(float(threshold), dtype=torch.float32, device=device))
    optimizer = torch.optim.SGD(
        [
            {"params": net.parameters(), "lr": lr, "momentum": 0.9},
            {"params": [tau], "lr": lr_tau},
        ]
    )

    epoch_losses: List[float] = []

    for _ in range(epochs):
        running_loss = 0.0
        batches = 0

        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device).float()

            optimizer.zero_grad()
            logits = net(images)
            probs = torch.sigmoid(logits)

            # Stable weighted binary cross entropy:
            # y*softplus(-logit) = -y*log(sigmoid(logit))
            # (1-y)*softplus(logit) = -(1-y)*log(1-sigmoid(logit))
            with torch.no_grad():
                hard_positive_gate = torch.sigmoid((tau - probs) / max(float(temperature), 1e-6))
                positive_weight = float(c_fn) * (1.0 + (beta_k - 1.0) * hard_positive_gate)
            pos_loss = labels * F.softplus(-logits) * positive_weight
            neg_loss = (1.0 - labels) * F.softplus(logits) * c_fp
            if str(loss_type).lower() in {
                "class_balanced",
                "class-balanced",
                "cb",
                "cbloss",
                "class_balanced_loss",
            }:
                pos_loss = pos_loss * pos_cb_weight
                neg_loss = neg_loss * neg_cb_weight
            sample_loss = pos_loss + neg_loss
            if str(loss_type).lower() == "focal":
                p_t = labels * probs + (1.0 - labels) * (1.0 - probs)
                focal = torch.pow(torch.clamp(1.0 - p_t, min=0.0), float(focal_gamma))
                if float(focal_alpha) >= 0.0:
                    alpha_t = labels * float(focal_alpha) + (1.0 - labels) * (1.0 - float(focal_alpha))
                    focal = focal * alpha_t
                sample_loss = sample_loss * focal
            risk_loss = sample_loss.mean()

            smooth_fpr, smooth_fnr = _smooth_fpr_fnr(probs, labels, tau, temperature)
            # Use a hinge-style low-FPR constraint. This is more stable than
            # the raw Lagrangian term lambda*(FPR-epsilon) in early rounds.
            fpr_violation = torch.relu(smooth_fpr - epsilon_fpr)
            objective = risk_loss + (lambda_fpr + fpr_penalty) * fpr_violation + mu_fnr * smooth_fnr
            if float(fedprox_mu) > 0.0 and global_params is not None:
                prox = torch.zeros((), device=device)
                for name, param in net.named_parameters():
                    if name in global_params:
                        prox = prox + torch.sum((param - global_params[name]) ** 2)
                objective = objective + 0.5 * float(fedprox_mu) * prox
            if (
                float(moon_mu) > 0.0
                and global_model is not None
                and previous_model is not None
            ):
                current_features = F.normalize(net.forward_features(images), dim=1)
                with torch.no_grad():
                    global_features = F.normalize(global_model.forward_features(images), dim=1)
                    previous_features = F.normalize(previous_model.forward_features(images), dim=1)
                pos_sim = torch.sum(current_features * global_features, dim=1)
                neg_sim = torch.sum(current_features * previous_features, dim=1)
                contrast_logits = torch.stack([pos_sim, neg_sim], dim=1) / max(float(moon_temperature), 1e-6)
                contrast_labels = torch.zeros(images.size(0), dtype=torch.long, device=device)
                objective = objective + float(moon_mu) * F.cross_entropy(contrast_logits, contrast_labels)
            if (
                float(fedsimsup_mu) > 0.0
                and supervisor_model is not None
                and images.size(0) > 1
            ):
                current_features = F.normalize(net.forward_features(images), dim=1)
                with torch.no_grad():
                    supervisor_features = F.normalize(supervisor_model.forward_features(images), dim=1)
                    teacher_sim = torch.matmul(supervisor_features, supervisor_features.T)
                    teacher_dist = F.softmax(
                        teacher_sim / max(float(fedsimsup_temperature), 1e-6),
                        dim=1,
                    )
                student_sim = torch.matmul(current_features, current_features.T)
                student_log_dist = F.log_softmax(
                    student_sim / max(float(fedsimsup_temperature), 1e-6),
                    dim=1,
                )
                simsup_loss = F.kl_div(student_log_dist, teacher_dist, reduction="batchmean")
                objective = objective + float(fedsimsup_mu) * simsup_loss

            objective.backward()
            optimizer.step()

            with torch.no_grad():
                tau.clamp_(0.0, 1.0)

            running_loss += float(objective.item())
            batches += 1

        epoch_losses.append(running_loss / max(1, batches))

    new_threshold = float(tau.detach().cpu().item())

    # Personalized FPR-budget threshold projection.  The gradient-updated
    # threshold can be overly conservative on rare-anomaly clients; projecting
    # the client threshold back to the empirical low-FPR frontier recovers
    # recall while keeping the learned threshold as a state variable for the
    # next federated round.  This is part of joint training, not post-hoc
    # evaluation calibration, and is disabled when threshold learning is off.
    projection_blend = float(np.clip(threshold_projection_blend, 0.0, 1.0))
    projected_threshold = new_threshold
    if projection_blend > 0.0 and float(lr_tau) > 0.0:
        probs_np, labels_np = collect_probs_and_labels(net, trainloader, device)
        if len(labels_np) > 0 and np.any(labels_np <= 0.5):
            target = float(epsilon_fpr) * max(float(threshold_projection_fpr_factor), 0.0)
            target = float(np.clip(target, 0.0, 1.0))
            projected_threshold = _threshold_at_fpr(probs_np, labels_np, target)
            new_threshold = float(
                np.clip(
                    (1.0 - projection_blend) * new_threshold
                    + projection_blend * projected_threshold,
                    0.0,
                    1.0,
                )
            )

    post_metrics = evaluate_binary_metrics(
        net, trainloader, device, new_threshold, epsilon_fpr, temperature
    )
    smooth_fpr = float(post_metrics.get("smooth_fpr", post_metrics.get("fpr", 0.0)))
    smooth_fnr = float(post_metrics.get("smooth_fnr", post_metrics.get("fnr", 0.0)))
    constraint_violation = max(smooth_fpr - float(epsilon_fpr), 0.0)
    new_lambda_fpr = max(
        0.0,
        float(lambda_fpr) + float(eta_lambda) * constraint_violation,
    )

    # Client-side fallback for risk-aware aggregation. The exact paper weight is
    # finalized on the server, where update reliability q_k can be computed.
    risk_focus = (1.0 + risk_gamma_fnr * smooth_fnr) * np.exp(
        -risk_gamma_fpr * constraint_violation
    )
    risk_focus = float(
        np.clip(risk_focus, risk_weight_min_factor, risk_weight_max_factor)
    )
    if int(server_round) <= int(risk_weight_warmup_rounds):
        risk_focus = 1.0
    reliability = 1.0 / (
        1.0 + float(np.var(epoch_losses)) if len(epoch_losses) > 1 else 1.0
    )
    risk_weight = len(trainloader.dataset) * risk_focus * reliability

    post_metrics.update({
        "train_loss": epoch_losses[-1],
        "beta": beta_k,
        "beta_logit": float(beta_logit),
        "beta_scarcity_gate": float(scarcity_gate),
        "beta_fnr_gate": float(fnr_gate),
        "beta_fpr_budget_gate": float(fpr_budget_gate),
        "beta_round_gate": float(round_gate),
        "threshold_projected": float(projected_threshold),
        "threshold_projection_blend": float(projection_blend),
        "lambda_fpr": new_lambda_fpr,
        "smooth_fpr": smooth_fpr,
        "smooth_fnr": smooth_fnr,
        "constraint_violation": constraint_violation,
        "loss_type": 0.0,
        "fedsimsup_mu": float(fedsimsup_mu),
        "risk_weight": float(risk_weight),
        "num-examples": float(len(trainloader.dataset)),
    })

    return epoch_losses[-1], epoch_losses, post_metrics, new_threshold, new_lambda_fpr


def test(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
    target_fpr: float = 0.01,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate binary anomaly detection model."""

    net.to(device)
    net.eval()

    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device).float()
            logits = net(images)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += float(loss.item())
            batches += 1

    metrics = evaluate_binary_metrics(net, testloader, device, threshold, target_fpr)
    metrics["eval_loss"] = total_loss / max(1, batches)
    # Compatibility alias used by old logging/checkpoint code
    metrics["eval_acc"] = metrics.get("accuracy", 0.0)
    return metrics["eval_loss"], metrics
