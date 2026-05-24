# RC-FAD v5: Accuracy@FPR Results Update

This version keeps the v4 reproducibility changes and adds/exports Accuracy@FPR for paper tables.

## New/updated outputs

Each run still writes:

- `server_metrics.csv`
- `client_eval_metrics.csv`
- `client_train_metrics.csv`
- `summary.json`

The ServerApp-side and ClientApp-side metrics now include:

- `accuracy_at_fpr`
- `precision_at_fpr`
- `recall_at_fpr_threshold`
- `f1_at_fpr_threshold`
- `fpr_at_fpr_threshold`
- `threshold_at_fpr`

`accuracy`, `precision`, `recall`, `f1`, and `fpr` remain fixed-threshold diagnostic metrics.

## Summary files

Run:

```bash
python scripts/collect_results.py --results-root outputs/experiments
```

It generates:

- `outputs/experiments/_summary/summary_selected.csv`
- `outputs/experiments/_summary/summary_selected.md`
- `outputs/experiments/_summary/fixed_threshold_diagnostics_selected.md`

Use `summary_selected.md`/CSV as the main paper table because it contains Accuracy@FPR and other calibrated low-FPR metrics. Use `fixed_threshold_diagnostics_selected.md` only as a diagnostic table if reviewers ask about default-threshold accuracy.
