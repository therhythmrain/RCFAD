"""Evaluation metrics for binary anomaly detection."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _compute_auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = float((labels == 1).sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    precision = tp / (np.arange(len(labels)) + 1)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def _recall_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    pos_total = float((labels == 1).sum())
    neg_total = float((labels == 0).sum())
    if pos_total == 0 or neg_total == 0:
        return 0.0
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / pos_total
    fpr = fp / neg_total
    valid = fpr <= target_fpr
    if not np.any(valid):
        return 0.0
    return float(np.max(recall[valid]))


def _threshold_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    """Return a score threshold whose negative pass rate is at most target_fpr."""
    neg_scores = scores[labels == 0]
    if len(neg_scores) == 0:
        return 0.5
    # Predict anomaly when score >= threshold. A plain quantile can collapse to
    # 0.0 when many scores are tied, which then predicts every negative sample
    # as anomalous. Step through score cutoffs and keep the best feasible one.
    target = float(np.clip(target_fpr, 0.0, 1.0))
    candidates = np.unique(neg_scores)
    best_threshold = _next_score_after_max(neg_scores)
    best_fpr = 0.0
    for threshold in candidates:
        fpr = float(np.mean(neg_scores >= threshold))
        if fpr <= target and fpr >= best_fpr:
            best_threshold = float(threshold)
            best_fpr = fpr
    return best_threshold


def _next_score_after_max(scores: np.ndarray) -> float:
    arr = np.asarray(scores)
    max_score = np.max(arr)
    if np.issubdtype(arr.dtype, np.floating):
        return float(
            np.nextafter(
                np.asarray(max_score, dtype=arr.dtype),
                np.asarray(np.inf, dtype=arr.dtype),
            )
        )
    return float(max_score) + 1.0


def _basic_hard_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    y = labels.astype(np.int64)

    tp = float(((preds == 1) & (y == 1)).sum())
    fp = float(((preds == 1) & (y == 0)).sum())
    tn = float(((preds == 0) & (y == 0)).sum())
    fn = float(((preds == 0) & (y == 1)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    acc = _safe_div(tp + tn, tp + fp + tn + fn)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
    }


def _hard_binary_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    target_fpr: float,
) -> Dict[str, float]:
    """Compute fixed-threshold and target-FPR-calibrated metrics."""
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    fixed = _basic_hard_metrics(probs, labels, threshold)
    auc = _compute_auc(probs, labels)
    auprc = _compute_auprc(probs, labels)
    recall_at_fpr = _recall_at_fpr(probs, labels, target_fpr)

    thr_at_fpr = _threshold_at_fpr(probs, labels, target_fpr)
    calibrated = _basic_hard_metrics(probs, labels, thr_at_fpr)
    if calibrated["fpr"] > float(target_fpr) + 1e-12:
        neg_scores = probs[labels == 0]
        if len(neg_scores) > 0:
            thr_at_fpr = _next_score_after_max(neg_scores)
            calibrated = _basic_hard_metrics(probs, labels, thr_at_fpr)

    out: Dict[str, float] = {
        **fixed,
        "auc": auc,
        "auprc": auprc,
        # Ranking metric: maximum recall achievable under target FPR.
        "recall_at_fpr": recall_at_fpr,
        # Explicit threshold-calibrated metrics for debugging and tables.
        "threshold_fixed": float(threshold),
        "threshold_at_fpr": float(thr_at_fpr),
        "accuracy_at_fpr": float(calibrated["accuracy"]),
        "precision_at_fpr": float(calibrated["precision"]),
        "recall_at_fpr_threshold": float(calibrated["recall"]),
        "fpr_at_fpr_threshold": float(calibrated["fpr"]),
        "fnr_at_fpr_threshold": float(calibrated["fnr"]),
        "f1_at_fpr_threshold": float(calibrated["f1"]),
    }
    return out


def _smooth_numpy_fpr_fnr(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    temperature: float,
) -> Tuple[float, float]:
    """Compute the paper's differentiable FPR/FNR proxies for risk state."""

    temp = max(float(temperature), 1e-6)
    smooth_pred = 1.0 / (1.0 + np.exp(-(probs - float(threshold)) / temp))
    neg = labels <= 0.5
    pos = labels > 0.5
    smooth_fpr = float(smooth_pred[neg].mean()) if np.any(neg) else 0.0
    smooth_fnr = float((1.0 - smooth_pred[pos]).mean()) if np.any(pos) else 0.0
    return smooth_fpr, smooth_fnr


def collect_probs_and_labels(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    net.eval()
    probs_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device).float()
            logits = net(images)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())

    if not probs_list:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return np.concatenate(probs_list).reshape(-1), np.concatenate(labels_list).reshape(-1)


def evaluate_binary_metrics(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    target_fpr: float,
    temperature: float | None = None,
) -> Dict[str, float]:
    probs, labels = collect_probs_and_labels(net, loader, device)
    if len(labels) == 0:
        return {}
    metrics = _hard_binary_metrics(probs, labels, threshold, target_fpr)
    metrics["anomaly_ratio"] = float(np.mean(labels))
    metrics["threshold"] = float(threshold)
    metrics["target_fpr"] = float(target_fpr)
    if temperature is not None:
        smooth_fpr, smooth_fnr = _smooth_numpy_fpr_fnr(
            probs, labels, threshold, temperature
        )
        metrics["smooth_fpr"] = smooth_fpr
        metrics["smooth_fnr"] = smooth_fnr
        metrics["smooth_fpr_violation"] = max(smooth_fpr - float(target_fpr), 0.0)
    return metrics


