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
from flwr.common import log, logger
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

        for reply in replies:
            if reply.has_content() and "train_metadata" in reply.content:
                try:
                    config_record = reply.content["train_metadata"]
                    meta_bytes = config_record["meta"]
                    train_meta = pickle.loads(meta_bytes)
                    print("Metadata:", asdict(train_meta))
                except Exception as exc:
                    print("Could not read train metadata:", exc)

        self._inject_risk_reliable_weights(server_round, replies)
        return super().aggregate_train(server_round, replies)

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
        epsilon = float(cfg.get("epsilon-fpr", 0.05))
        gamma_fnr = float(cfg.get("risk-gamma-fnr", 1.0))
        gamma_fpr = float(cfg.get("risk-gamma-fpr", 2.0))
        rho = float(cfg.get("update-reliability-rho", 1.0))
        reliability_floor = float(cfg.get("update-reliability-floor", 0.0))
        delta_eps = float(cfg.get("update-reliability-delta", 1e-12))
        min_factor = float(cfg.get("risk-weight-min-factor", 0.25))
        max_factor = float(cfg.get("risk-weight-max-factor", 3.0))
        warmup_rounds = int(cfg.get("risk-weight-warmup-rounds", 0))

        for delta, (_, metrics, _) in zip(deltas, records, strict=True):
            n_examples = float(metrics.get("num-examples", 1.0))
            smooth_fnr = float(metrics.get("smooth_train_fnr", metrics.get("train_fnr", 0.0)))
            smooth_fpr = float(metrics.get("smooth_train_fpr", metrics.get("train_fpr", 0.0)))
            violation = max(smooth_fpr - epsilon, 0.0)

            dev = {key: delta[key] - mean_delta[key] for key in mean_delta}
            dev_norm = self._delta_norm(dev)
            raw_reliability = float(np.exp(-rho * dev_norm / (mean_norm + delta_eps)))
            reliability = float(np.clip(raw_reliability, reliability_floor, 1.0))

            factor = (1.0 + gamma_fnr * smooth_fnr) * np.exp(-gamma_fpr * violation)
            factor *= reliability
            factor = float(np.clip(factor, min_factor, max_factor))
            if int(server_round) <= warmup_rounds:
                factor = 1.0
                reliability = 1.0

            metrics["update_reliability"] = reliability
            metrics["risk_aggregation_factor"] = factor
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

            agg_arrays, agg_train_metrics = self.aggregate_train(current_round, train_replies)

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

            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated eval MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                if wandb.run is not None:
                    wandb.log(dict(agg_evaluate_metrics), step=current_round)

            if evaluate_fn:
                log(INFO, "Global evaluation")
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
