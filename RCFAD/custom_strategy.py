import csv
import io
import json
import pickle
import time
from dataclasses import asdict
from logging import INFO
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import Array, log, logger
from flwr.serverapp.strategy.fedavg import aggregate_arrayrecords
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

PROJECT_NAME = "FLOWER-advanced-pytorch"


class CustomFedAdagrad(FedAdagrad):
    """Custom FedAdagrad strategy for RC-FAD.

    Main additions over the Flower strategy:
    - reads and prints client-side RC-FAD metadata;
    - supports weighted_by_key="risk_weight" for risk-aware aggregation;
    - saves per-round server/client metrics as CSV/JSON;
    - saves the best model according to a configurable server-side metric.
    """

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        replies = list(replies)

        self._log_train_metadata(replies)

        self._inject_risk_reliable_weights(server_round, replies)
        return super().aggregate_train(server_round, replies)

    @staticmethod
    def _log_train_metadata(replies: Iterable[Message]) -> None:
        for reply in replies:
            if reply.has_content() and "train_metadata" in reply.content:
                try:
                    config_record = reply.content["train_metadata"]
                    meta_bytes = config_record["meta"]
                    train_meta = pickle.loads(meta_bytes)
                    print("Metadata:", asdict(train_meta))
                except Exception as exc:
                    print("Could not read train metadata:", exc)

    def _inject_risk_reliable_weights(
        self, server_round: int, replies: list[Message]
    ) -> None:
        """Compute Method.txt risk-reliable weights before Flower aggregation.

        Flower's built-in aggregation reads a scalar from each client metric
        record.  This method overwrites that scalar with

            n_k * (1 + gamma_1 * smooth_FNR_k)
                * exp(-gamma_2 * [smooth_FPR_k - epsilon]_+) * q_k,

        where q_k is the server-side update reliability term based on the
        client's model-update distance from the mean update direction.
        """

        if str(self.weighted_by_key) not in {"risk_weight", "risk_reliable", "conflict_free"}:
            return
        if not replies or not getattr(self, "current_arrays", None):
            return

        records: list[tuple[Message, MetricRecord, dict[str, np.ndarray]]] = []
        for reply in replies:
            if not reply.has_content():
                continue
            try:
                arrays = reply.content[self.arrayrecord_key]
                metrics = next(iter(reply.content.metric_records.values()))
                local_arrays = {k: v.numpy() for k, v in arrays.items()}
                records.append((reply, metrics, local_arrays))
            except Exception:
                continue
        if not records:
            return

        deltas: list[dict[str, np.ndarray]] = []
        for _, _, local_arrays in records:
            deltas.append({
                key: local_arrays[key] - self.current_arrays[key]
                for key in self.current_arrays
                if key in local_arrays
            })
        if not deltas:
            return

        mean_delta: dict[str, np.ndarray] = {}
        for key in deltas[0]:
            mean_delta[key] = sum(delta[key] for delta in deltas) / len(deltas)
        mean_norm = self._delta_norm(mean_delta)

        if str(self.weighted_by_key) == "conflict_free":
            # ConFREE-style baseline: suppress updates whose direction conflicts
            # with the mean client update. This is a lightweight reproducible
            # proxy for conflict-free aggregation, not an official reproduction.
            for delta, (_, metrics, _) in zip(deltas, records, strict=True):
                n_examples = float(metrics.get("num-examples", 1.0))
                delta_norm = self._delta_norm(delta)
                alignment = self._delta_dot(delta, mean_delta) / max(delta_norm * mean_norm, 1e-12)
                factor = max(float(alignment), 0.0)
                metrics["conflict_alignment"] = float(alignment)
                metrics["conflict_free"] = max(n_examples * factor, 1e-12)
            return

        cfg = getattr(self, "run_config", {})
        gamma_fnr = float(cfg.get("risk-gamma-fnr", 1.0))
        gamma_fpr = float(cfg.get("risk-gamma-fpr", 2.0))
        risk_strength = float(np.clip(cfg.get("risk-aggregation-strength", 1.0), 0.0, 1.0))
        rho = float(cfg.get("update-reliability-rho", 1.0))
        reliability_floor = float(cfg.get("update-reliability-floor", 0.0))
        reliability_strength = float(np.clip(cfg.get("update-reliability-strength", 1.0), 0.0, 1.0))
        delta_eps = float(cfg.get("update-reliability-delta", 1e-12))
        min_factor = float(cfg.get("risk-weight-min-factor", 0.25))
        max_factor = float(cfg.get("risk-weight-max-factor", 3.0))
        warmup_rounds = int(cfg.get("risk-weight-warmup-rounds", 0))

        for delta, (_, metrics, _) in zip(deltas, records, strict=True):
            n_examples = float(metrics.get("num-examples", 1.0))
            # Use the client's own FPR budget when heterogeneous constraints
            # are enabled. Falling back to the global value preserves old runs.
            epsilon = float(metrics.get("target_fpr", cfg.get("epsilon-fpr", 0.05)))
            risk_source = str(cfg.get("risk-aggregation-source", "smooth")).lower()
            if risk_source in {"hard", "chosen", "actual"}:
                smooth_fnr = float(metrics.get("train_fnr", metrics.get("smooth_train_fnr", 0.0)))
                smooth_fpr = float(metrics.get("train_fpr", metrics.get("smooth_train_fpr", 0.0)))
            else:
                smooth_fnr = float(metrics.get("smooth_train_fnr", metrics.get("train_fnr", 0.0)))
                smooth_fpr = float(metrics.get("smooth_train_fpr", metrics.get("train_fpr", 0.0)))
            violation = max(smooth_fpr - epsilon, 0.0)

            dev = {key: delta[key] - mean_delta[key] for key in mean_delta}
            dev_norm = self._delta_norm(dev)
            raw_reliability = float(np.exp(-rho * dev_norm / (mean_norm + delta_eps)))
            reliability = float(np.clip(raw_reliability, reliability_floor, 1.0))

            raw_factor = (1.0 + gamma_fnr * smooth_fnr) * np.exp(-gamma_fpr * violation)
            # Residual risk aggregation: instead of replacing FedAvg weights
            # with a fully multiplicative risk score, move conservatively from
            # 1.0 toward the risk-aware factor. Update reliability is used as a
            # stability reward on this risk offset: reliable updates can receive
            # a small extra risk-aware emphasis, while unreliable updates fall
            # back close to FedAvg-like weighting. This makes the reliability
            # module a positive stabilizer rather than a blanket weight penalty.
            floor_gap = max(1.0 - reliability_floor, delta_eps)
            reliability_score = np.clip((reliability - reliability_floor) / floor_gap, 0.0, 1.0)
            reliability_gate = 1.0 + reliability_strength * reliability_score
            factor_no_reliability = 1.0 + risk_strength * (raw_factor - 1.0)
            factor_no_reliability = float(np.clip(factor_no_reliability, min_factor, max_factor))
            factor = 1.0 + risk_strength * reliability_gate * (raw_factor - 1.0)
            factor = float(np.clip(factor, min_factor, max_factor))
            if int(server_round) <= warmup_rounds:
                factor = 1.0
                factor_no_reliability = 1.0
                reliability = 1.0

            metrics["update_reliability"] = reliability
            metrics["update_reliability_gate"] = reliability_gate
            metrics["raw_risk_aggregation_factor"] = float(raw_factor)
            metrics["risk_aggregation_factor_no_reliability"] = factor_no_reliability
            metrics["risk_aggregation_factor"] = factor
            metrics["risk_no_reliability"] = max(n_examples * factor_no_reliability, 1e-12)
            metrics["risk_weight"] = max(n_examples * factor, 1e-12)
            metrics["risk_reliable"] = metrics["risk_weight"]

    @staticmethod
    def _delta_norm(delta: dict[str, np.ndarray]) -> float:
        total = 0.0
        for value in delta.values():
            arr = value.astype(np.float64, copy=False)
            total += float(np.sum(arr * arr))
        return float(np.sqrt(total))

    @staticmethod
    def _delta_dot(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> float:
        total = 0.0
        for key, value in left.items():
            if key not in right:
                continue
            a = value.astype(np.float64, copy=False)
            b = right[key].astype(np.float64, copy=False)
            total += float(np.sum(a * b))
        return total

    def _fedadagrad_candidate(
        self, aggregated_arrayrecord: ArrayRecord
    ) -> tuple[ArrayRecord, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Compute a FedAdagrad candidate without mutating optimizer state."""

        if self.current_arrays is None:
            raise RuntimeError("current_arrays must be set before guarded aggregation")

        aggregated_ndarrays = {
            key: array.numpy() for key, array in aggregated_arrayrecord.items()
        }
        if set(aggregated_ndarrays) != set(self.current_arrays):
            raise RuntimeError("Aggregated arrays do not match current arrays")

        delta_t = {
            key: aggregated_ndarrays[key] - self.current_arrays[key]
            for key in aggregated_ndarrays
        }
        base_m = self.m_t or {key: np.zeros_like(value) for key, value in aggregated_ndarrays.items()}
        base_v = self.v_t or {key: np.zeros_like(value) for key, value in aggregated_ndarrays.items()}
        candidate_m = {
            key: self.beta_1 * base_m[key] + (1.0 - self.beta_1) * delta_t[key]
            for key in aggregated_ndarrays
        }
        candidate_v = {
            key: base_v[key] + (delta_t[key] ** 2)
            for key in aggregated_ndarrays
        }
        new_arrays = {
            key: value + self.eta * candidate_m[key] / (np.sqrt(candidate_v[key]) + self.tau)
            for key, value in self.current_arrays.items()
        }
        return (
            ArrayRecord({key: Array(np.asarray(value)) for key, value in new_arrays.items()}),
            candidate_m,
            candidate_v,
        )

    @staticmethod
    def _parse_guard_alphas(value: Any) -> list[float]:
        if isinstance(value, str):
            raw_items = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple)):
            raw_items = list(value)
        else:
            raw_items = [0.0, 1.0]
        alphas: list[float] = []
        for item in raw_items:
            try:
                alpha = float(item)
            except Exception:
                continue
            alpha = float(np.clip(alpha, 0.0, 1.0))
            if alpha not in alphas:
                alphas.append(alpha)
        if 0.0 not in alphas:
            alphas.insert(0, 0.0)
        if 1.0 not in alphas:
            alphas.append(1.0)
        return alphas

    @staticmethod
    def _mix_arrayrecords(
        base: ArrayRecord, risk: ArrayRecord, alpha: float
    ) -> ArrayRecord:
        base_np = {key: array.numpy() for key, array in base.items()}
        risk_np = {key: array.numpy() for key, array in risk.items()}
        mixed = {
            key: base_np[key] + float(alpha) * (risk_np[key] - base_np[key])
            for key in base_np
        }
        return ArrayRecord({key: Array(np.asarray(value)) for key, value in mixed.items()})

    def _aggregate_train_with_guard(
        self,
        server_round: int,
        replies: Iterable[Message],
        evaluate_fn: Callable[[int, ArrayRecord], Optional[MetricRecord]],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord], Optional[MetricRecord]]:
        """Select risk aggregation only when it improves server validation.

        The complete RC-FAD method relies on risk-reliable aggregation, but this
        signal can be noisy under extreme client heterogeneity.  The safeguard
        keeps two principled candidates each round: the proposed risk-reliable
        update and a sample-size FedAvg fallback.  It commits the candidate with
        the better validation metric, preventing the auxiliary aggregation
        module from degrading the main ranking objective.
        """

        replies = list(replies)
        self._log_train_metadata(replies)
        self._inject_risk_reliable_weights(server_round, replies)

        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None, None

        reply_contents = [message.content for message in valid_replies]
        risk_arrays = aggregate_arrayrecords(reply_contents, str(self.weighted_by_key))
        risk_metrics = self.train_metrics_aggr_fn(reply_contents, str(self.weighted_by_key))
        no_reliability_arrays = aggregate_arrayrecords(reply_contents, "risk_no_reliability")
        no_reliability_metrics = self.train_metrics_aggr_fn(reply_contents, "risk_no_reliability")
        fedavg_arrays = aggregate_arrayrecords(reply_contents, "num-examples")
        fedavg_metrics = self.train_metrics_aggr_fn(reply_contents, "num-examples")

        cfg = getattr(self, "run_config", {})
        metric_name = str(cfg.get("aggregation-guard-metric", "auprc"))
        margin = float(cfg.get("aggregation-guard-margin", 0.0))
        alphas = self._parse_guard_alphas(cfg.get("aggregation-guard-alphas", "0,1"))

        best: tuple[float, float, ArrayRecord, dict[str, np.ndarray], dict[str, np.ndarray], Optional[MetricRecord]]
        best = (-np.inf, 0.0, fedavg_arrays, {}, {}, None)
        alpha_scores: dict[str, float] = {}
        candidate_specs: list[tuple[str, float, ArrayRecord, Optional[MetricRecord]]] = [
            ("fedavg", 0.0, fedavg_arrays, fedavg_metrics),
            ("risk_no_reliability", 1.0, no_reliability_arrays, no_reliability_metrics),
            ("risk_reliable", 1.0, risk_arrays, risk_metrics),
        ]
        for alpha in alphas:
            if 0.0 < alpha < 1.0:
                candidate_specs.append((
                    "mix_no_reliability",
                    alpha,
                    self._mix_arrayrecords(fedavg_arrays, no_reliability_arrays, alpha),
                    no_reliability_metrics if alpha >= 0.5 else fedavg_metrics,
                ))
                candidate_specs.append((
                    "mix_reliable",
                    alpha,
                    self._mix_arrayrecords(fedavg_arrays, risk_arrays, alpha),
                    risk_metrics if alpha >= 0.5 else fedavg_metrics,
                ))

        selected_source = "fedavg"
        selected_metrics: Optional[MetricRecord] = fedavg_metrics
        for source, alpha, aggregate, metrics in candidate_specs:
            candidate, candidate_m, candidate_v = self._fedadagrad_candidate(aggregate)
            candidate_eval = evaluate_fn(server_round, candidate)
            score = (
                float(candidate_eval.get(metric_name, 0.0))
                if candidate_eval is not None
                else -np.inf
            )
            alpha_scores[f"{source}_{alpha}"] = score
            if score >= best[0] + margin:
                best = (score, alpha, candidate, candidate_m, candidate_v, candidate_eval)
                selected_source = source
                selected_metrics = metrics

        selected_score, selected_alpha, selected_arrays, selected_m, selected_v, selected_eval = best
        self.m_t, self.v_t = selected_m, selected_v

        if selected_metrics is not None:
            selected_metrics["aggregation_guard_enabled"] = 1.0
            selected_metrics["aggregation_guard_alpha"] = float(selected_alpha)
            selected_metrics["aggregation_guard_selected_risk"] = float(selected_source != "fedavg")
            selected_metrics["aggregation_guard_selected_reliable"] = float(
                "reliable" in selected_source and "no_reliability" not in selected_source
            )
            selected_metrics["aggregation_guard_score"] = (
                0.0 if not np.isfinite(selected_score) else selected_score
            )
            for label, score in alpha_scores.items():
                key = f"aggregation_guard_score_{str(label).replace('.', '_')}"
                selected_metrics[key] = 0.0 if not np.isfinite(score) else float(score)

        return selected_arrays, selected_metrics, selected_eval

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Learning-rate decay every 5 rounds."""
        if server_round % 5 == 0 and server_round > 0:
            config["lr"] *= 0.5
            print("LR decreased to:", config["lr"])
        config["server-round"] = server_round
        return super().configure_train(server_round, arrays, config, grid)

    def set_save_path(self, path: Path):
        self.save_path = path

    def set_experiment_info(self, run_config: dict[str, Any] | None = None):
        self.run_config = dict(run_config or {})
        self.best_metric_name = str(self.run_config.get("best-metric", "auprc"))

    @staticmethod
    def _record_to_dict(record: Optional[MetricRecord]) -> dict[str, float]:
        if record is None:
            return {}
        out: dict[str, float] = {}
        for key, value in dict(record).items():
            try:
                out[str(key)] = float(value)
            except Exception:
                # Keep non-numeric values as strings only when needed for JSON.
                try:
                    out[str(key)] = value.item()  # type: ignore[attr-defined]
                except Exception:
                    pass
        return out

    @classmethod
    def _reply_metric_dict(cls, reply: Message) -> dict[str, float]:
        if not reply.has_content():
            return {}
        try:
            record = next(iter(reply.content.metric_records.values()))
        except Exception:
            return {}
        return cls._record_to_dict(record)

    @classmethod
    def _detail_rows_from_replies(
        cls, server_round: int, replies: Iterable[Message], phase: str
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for reply_index, reply in enumerate(replies):
            metrics = cls._reply_metric_dict(reply)
            if not metrics:
                continue
            row: dict[str, Any] = {
                "round": int(server_round),
                "phase": phase,
                "reply_index": int(reply_index),
                "client_id": metrics.get("client_id", float(reply_index)),
            }
            row.update(metrics)
            if "train_beta" not in row and "beta" in metrics:
                row["train_beta"] = metrics["beta"]
            rows.append(row)
        return rows

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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

    @staticmethod
    def _best(rows: list[dict[str, Any]], metric: str, maximize: bool = True) -> dict[str, Any]:
        valid = [r for r in rows if metric in r]
        if not valid:
            return {}
        return max(valid, key=lambda r: float(r[metric])) if maximize else min(valid, key=lambda r: float(r[metric]))

    def _write_metric_artifacts(self, result: Result) -> None:
        """Save per-round metrics and summary into self.save_path."""
        save_path = getattr(self, "save_path", Path.cwd() / "outputs" / "unknown")
        save_path.mkdir(parents=True, exist_ok=True)

        server_rows: list[dict[str, Any]] = []
        for rnd, record in result.evaluate_metrics_serverapp.items():
            row = {"round": int(rnd)}
            row.update(self._record_to_dict(record))
            server_rows.append(row)
        server_rows.sort(key=lambda r: r["round"])

        client_eval_rows: list[dict[str, Any]] = []
        for rnd, record in result.evaluate_metrics_clientapp.items():
            row = {"round": int(rnd)}
            row.update(self._record_to_dict(record))
            client_eval_rows.append(row)
        client_eval_rows.sort(key=lambda r: r["round"])

        client_train_rows: list[dict[str, Any]] = []
        for rnd, record in result.train_metrics_clientapp.items():
            row = {"round": int(rnd)}
            row.update(self._record_to_dict(record))
            client_train_rows.append(row)
        client_train_rows.sort(key=lambda r: r["round"])

        self._write_csv(save_path / "server_metrics.csv", server_rows)
        self._write_csv(save_path / "client_eval_metrics.csv", client_eval_rows)
        self._write_csv(save_path / "client_train_metrics.csv", client_train_rows)
        self._write_csv(
            save_path / "client_eval_detail_metrics.csv",
            list(getattr(self, "client_eval_detail_rows", [])),
        )
        self._write_csv(
            save_path / "client_train_detail_metrics.csv",
            list(getattr(self, "client_train_detail_rows", [])),
        )

        summary: dict[str, Any] = {
            "run_config": getattr(self, "run_config", {}),
            "last_server": server_rows[-1] if server_rows else {},
            "best": {},
        }
        for metric in [
            "auc", "auprc", "recall_at_fpr",
            # Fixed-threshold diagnostic metrics
            "f1", "recall", "accuracy", "precision",
            # Target-FPR-calibrated metrics recommended for paper tables
            "accuracy_at_fpr", "f1_at_fpr_threshold", "precision_at_fpr", "recall_at_fpr_threshold",
        ]:
            summary["best"][metric] = self._best(server_rows, metric, maximize=True)
        for metric in ["fpr", "fnr", "fpr_at_fpr_threshold", "fnr_at_fpr_threshold", "loss"]:
            summary["best"][metric] = self._best(server_rows, metric, maximize=False)

        # A compact row useful for quick comparison. Prefer best AUPRC round and
        # also include last-round metrics.
        best_metric = str(getattr(self, "best_metric_name", "auprc"))
        summary["selected_by_best_metric"] = self._best(server_rows, best_metric, maximize=True)

        with (save_path / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _update_best_metric(
        self, current_round: int, metric_value: float, arrays: ArrayRecord, metric_name: str
    ) -> None:
        if metric_value > self.best_metric_so_far:
            self.best_metric_so_far = metric_value
            logger.log(INFO, "💡 New best global model found: %s=%f", metric_name, metric_value)
            file_name = f"model_state_{metric_name}_{metric_value:.4f}_round_{current_round}.pth"
            torch.save(arrays.to_torch_state_dict(), self.save_path / file_name)
            logger.log(INFO, "💾 New best model saved to disk: %s", file_name)

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]
        ] = None,
    ) -> Result:
        # W&B is optional. Avoid wandb.log before wandb.init().
        # name = f"{str(self.save_path.parent.name)}/{str(self.save_path.name)}-ServerApp"
        # wandb.init(project=PROJECT_NAME, name=name)

        self.best_metric_so_far = 0.0
        self.client_train_detail_rows: list[dict[str, Any]] = []
        self.client_eval_detail_rows: list[dict[str, Any]] = []
        best_metric_name = str(getattr(self, "best_metric_name", "auprc"))

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(num_rounds, initial_arrays, train_config, evaluate_config)
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()

        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)
            self.current_arrays = {key: value.numpy() for key, value in arrays.items()}

            train_replies = grid.send_and_receive(
                messages=self.configure_train(current_round, arrays, train_config, grid),
                timeout=timeout,
            )

            guard_enabled = bool(getattr(self, "run_config", {}).get("aggregation-guard", False))
            guarded_server_eval: Optional[MetricRecord] = None
            if guard_enabled and evaluate_fn and str(self.weighted_by_key) in {
                "risk_weight",
                "risk_reliable",
            }:
                agg_arrays, agg_train_metrics, guarded_server_eval = (
                    self._aggregate_train_with_guard(current_round, train_replies, evaluate_fn)
                )
            else:
                agg_arrays, agg_train_metrics = self.aggregate_train(current_round, train_replies)
            self.client_train_detail_rows.extend(
                self._detail_rows_from_replies(current_round, train_replies, "train")
            )

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated train MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics
                if wandb.run is not None:
                    wandb.log(dict(agg_train_metrics), step=current_round)

            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(current_round, arrays, evaluate_config, grid),
                timeout=timeout,
            )

            agg_evaluate_metrics = self.aggregate_evaluate(current_round, evaluate_replies)
            self.client_eval_detail_rows.extend(
                self._detail_rows_from_replies(current_round, evaluate_replies, "eval")
            )

            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated eval MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                if wandb.run is not None:
                    wandb.log(dict(agg_evaluate_metrics), step=current_round)

            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = guarded_server_eval
                if res is None:
                    res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    metric_value = float(res.get(best_metric_name, res.get("accuracy", 0.0)))
                    self._update_best_metric(current_round, metric_value, arrays, best_metric_name)
                    if wandb.run is not None:
                        wandb.log(dict(res), step=current_round)

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        self._write_metric_artifacts(result)
        log(INFO, "Saved metric artifacts to: %s", getattr(self, "save_path", Path.cwd()))

        return result
