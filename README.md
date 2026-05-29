# RC-FAD: Risk-Constrained Federated Anomaly Detection

RC-FAD is a Flower/PyTorch implementation of federated anomaly detection under
client-level low false-positive-rate constraints.  The code supports
reproducible comparison, threshold-mechanism, ablation, heterogeneity, and
sensitivity experiments for the paper workflow.

## Core Idea

RC-FAD focuses on risk-sensitive anomaly detection where each client can have a
different normal-score distribution, anomaly ratio, and false-positive
tolerance.  The implementation combines:

- client-specific trainable thresholds `tau_k`;
- differentiable FPR/FNR proxies for low-FPR constrained optimization;
- risk-gated hard-positive enhancement for rare anomaly learning;
- risk/reliability-aware server aggregation.

The most important experimental distinction is between ranking metrics
(`AUPRC`, `Recall@FPR<=tau`) and deployment-threshold metrics
(`Recall@chosen`, `F1@chosen`, `FPR@chosen`, client violation diagnostics).

## Repository Layout

```text
RCFAD/
  client_app.py        Flower ClientApp
  server_app.py        Flower ServerApp
  custom_strategy.py   aggregation, logging, checkpointing, diagnostics
  task.py              backward-compatible facade for old imports
  constants.py         shared constants
  model.py             binary image/tabular anomaly detector
  data.py              dataset loading, caching, partitioning
  metrics.py           AUPRC/AUROC/FPR/FNR and threshold diagnostics
  training.py          local training and evaluation loops
  utils.py             seeding and metadata helpers
scripts/
  run_experiments.py   experiment launcher
  collect_results.py   result aggregation
  build_paper_tables.py paper-ready tables
```

## Installation

```bash
pip install -e .
```

The project has been used with Python 3.10, Flower 1.27, PyTorch 2.8, and
Torchvision 0.23.  GPU runs can be selected through run-config extras such as
`device=cuda:0 client-device=cuda:0 server-device=cuda:0`.

## Datasets

Supported dataset names include:

```text
cifar10, mnist, fmnist,
creditcard, baf, ai4i, secom,
swat, hai, smd, paysim,
mammography, annthyroid, shuttle, tep
```

For tabular and time-series datasets, put files under `data/<dataset>/` or pass
`--extra "data-root=/path/to/data"`.  UCI/ADBench datasets can often be
downloaded automatically; Kaggle-style datasets such as BAF, PaySim, and SWaT
usually require manual download.

## Quick Smoke Test

Dry-run the commands without launching Flower:

```bash
python scripts/run_experiments.py \
  --suite quick \
  --datasets ai4i \
  --seeds 42 \
  --rounds 1 \
  --dry-run
```

Run a tiny CPU smoke test:

```bash
python scripts/run_experiments.py \
  --suite quick \
  --datasets ai4i \
  --seeds 42 \
  --rounds 1 \
  --num-supernodes 1 \
  --client-cpus 2 \
  --results-root outputs/smoke_quick
```

GPU example:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py \
  --suite quick \
  --datasets ai4i \
  --seeds 42 \
  --rounds 3 \
  --num-supernodes 1 \
  --client-cpus 8 \
  --results-root outputs/smoke_quick_gpu \
  --extra "device=cuda:0 client-device=cuda:0 server-device=cuda:0"
```

## Main Experiments

List available suites and methods:

```bash
python scripts/run_experiments.py --list-methods
```

Representative comparison:

```bash
python scripts/run_experiments.py \
  --suite compare \
  --datasets ai4i secom annthyroid mammography tep shuttle \
  --seeds 42 1234 3407 \
  --rounds 30 \
  --results-root outputs/stage_compare
```

Risk-heterogeneous ablation:

```bash
python scripts/run_experiments.py \
  --suite ablation_risk \
  --datasets ai4i secom annthyroid mammography tep shuttle \
  --seeds 42 1234 3407 \
  --rounds 30 \
  --results-root outputs/stage_ablation_risk
```

Threshold-mechanism experiment without `FedAvg-Local-Post`:

```bash
python scripts/run_experiments.py \
  --suite threshold_hetero \
  --datasets ai4i secom annthyroid mammography tep shuttle \
  --seeds 42 1234 3407 \
  --rounds 30 \
  --methods fedavg_fixed fedavg_global_post fedprox_local_post confree_local_post rcfad_global_joint rcfad \
  --results-root outputs/stage_threshold
```

GPU threshold run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py \
  --suite threshold_hetero \
  --datasets ai4i secom annthyroid mammography tep shuttle \
  --seeds 42 1234 3407 \
  --rounds 30 \
  --num-supernodes 1 \
  --client-cpus 32 \
  --methods fedavg_fixed fedavg_global_post fedprox_local_post confree_local_post rcfad_global_joint rcfad \
  --results-root outputs/stage_threshold_gpu \
  --extra "device=cuda:0 client-device=cuda:0 server-device=cuda:0"
```

## Result Collection

```bash
python scripts/collect_results.py --results-root outputs/stage_threshold
python scripts/build_paper_tables.py \
  --threshold-root outputs/stage_threshold \
  --output-root outputs/paper_tables
```

Important summary files:

```text
<results-root>/_summary/all_runs.csv
<results-root>/_summary/summary_selected.csv
<results-root>/_summary/client_threshold_runs_selected.csv
<results-root>/_summary/client_threshold_diagnostics_selected.csv
<results-root>/_summary/table_threshold_core_percent_3seed_all6.csv
<results-root>/_summary/table_threshold_deployment_diagnostics_percent_3seed_all6.csv
```

For paper tables, report percentages with two decimals, for example
`33.24 ± 1.02`, and put the unit in the table header (`AUPRC (%)`,
`F1@chosen (%)`, `FPR@chosen (%)`).

## Notes

- `AUPRC` and `AUROC` evaluate score ranking and are threshold-independent.
- `Recall@FPR<=tau` and `F1@FPR<=tau` are useful low-FPR operating-point
  ranking diagnostics.
- Threshold-mechanism claims should use chosen/deployment-threshold metrics and
  client-level diagnostics, not only post-hoc target-FPR metrics.
- Large Flower/Ray GPU runs are more stable with low simulation concurrency,
  for example `--num-supernodes 1 --client-cpus 32`.
