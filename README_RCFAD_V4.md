# RC-FAD v4 Reproducible Experiment Version

This version focuses on reproducibility and experiment automation.

## Key changes from v3

1. **Fixed random seeds**
   - Server-side model initialization is seeded before `Net()` is created.
   - Client-side data shuffling is seeded with `seed + partition_id * 1000 + server_round`.
   - Validation/test loaders are deterministic.
   - PyTorch/CUDA deterministic flags are enabled where possible.

2. **Explicit experiment configuration**
   - `scripts/run_experiments.py` now explicitly passes important config values such as
     `epsilon-fpr`, `threshold-init`, `temperature`, `partition-scheme`, `global-anomaly-ratio`,
     and risk-weight caps.
   - This avoids silent differences caused by changing `pyproject.toml` defaults.

3. **Better server-side metrics**
   - In addition to fixed-threshold metrics (`f1`, `recall`, `fpr`), server evaluation now outputs
     target-FPR-calibrated metrics:
     - `threshold_at_fpr`
     - `precision_at_fpr`
     - `recall_at_fpr_threshold`
     - `fpr_at_fpr_threshold`
     - `f1_at_fpr_threshold`
   - `recall_at_fpr` remains the ranking metric used for low-FPR anomaly detection.

4. **Formatted results**
   - `server_metrics.csv`, `client_eval_metrics.csv`, `client_train_metrics.csv`, and `summary.json`
     are saved for every run.
   - `scripts/collect_results.py` summarizes all runs into CSV/Markdown files.

## Recommended first test

Run the same command twice and compare Round 0 and Round 30 in `server_metrics.csv`:

```bash
python scripts/run_experiments.py --suite quick --datasets cifar10 --seeds 42 --rounds 30
python scripts/run_experiments.py --suite quick --datasets cifar10 --seeds 42 --rounds 30
```

Round 0 should now be the same for the same seed. Small GPU/Ray differences can still exist, but
large jumps caused by random server initialization should disappear.

## Main experiments

```bash
python scripts/run_experiments.py --suite main --datasets cifar10 fmnist --seeds 42 123 2025 --rounds 30
```

## Ablation experiments

```bash
python scripts/run_experiments.py --suite ablation --datasets cifar10 fmnist --seeds 42 123 2025 --rounds 30
```

## Collect results

```bash
python scripts/collect_results.py --results-root outputs/experiments
```

Main files:

```text
outputs/experiments/_summary/all_runs.csv
outputs/experiments/_summary/summary_selected.csv
outputs/experiments/_summary/summary_selected.md
```

Use `AUPRC`, `Recall@FPR`, and target-FPR-calibrated metrics as primary anomaly detection evidence.
Do not rely on `accuracy` alone, because all-normal prediction can produce high accuracy in one-vs-rest anomaly settings.
