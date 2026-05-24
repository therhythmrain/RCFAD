# RC-FAD 实验执行指南

本目录说明如何用当前代码验证 `Method.txt` 中的理论算法。实验脚本已收敛为三类：

| Suite | 目的 | 默认方法 |
|---|---|---|
| `quick` | 小轮数冒烟测试 | `fedavg`, `rcfad` |
| `compare` | 3-4 个主要对比方法 | `fedavg`, `cost_sensitive`, `posthoc_threshold`, `rcfad` |
| `ablation` | 逐个移除 RC-FAD 模块 | `rcfad`, `wo_personal_threshold`, `wo_fpr_constraint`, `wo_dynamic_beta`, `wo_risk_aggregation`, `wo_update_reliability` |

## 1. 冒烟测试

```bash
python scripts/run_experiments.py --suite quick --datasets cifar10 mnist --seeds 42 --rounds 5
```

如果只是检查命令是否正确，不实际运行：

```bash
python scripts/run_experiments.py --suite quick --datasets cifar10 mnist --seeds 42 --rounds 5 --dry-run
```

## 2. 主要对比实验

```bash
python scripts/run_experiments.py --suite compare --datasets cifar10 mnist --seeds 42 123 2025 --rounds 30
```

建议表格比较：

- `AUPRC`
- `Recall@FPR`
- `Accuracy@FPR`
- `F1@FPR`
- `FPR@threshold`

这些指标在 `summary_selected.md` 和 `summary_selected.csv` 中。

## 3. 逐模块消融实验

```bash
python scripts/run_experiments.py --suite ablation --datasets cifar10 mnist --seeds 42 123 2025 --rounds 30
```

每个消融只关闭一个关键模块：

| 方法 | 关闭内容 | 对应论文模块 |
|---|---|---|
| `wo_personal_threshold` | 固定客户端阈值，不学习 `tau_k` | 3.3 |
| `wo_fpr_constraint` | 移除低误报约束和拉格朗日乘子 | 3.2, 3.3 |
| `wo_dynamic_beta` | 固定异常类增强因子 | 3.4 |
| `wo_risk_aggregation` | 使用样本量聚合 | 3.5 |
| `wo_update_reliability` | 令更新可靠性 `q_k=1` | 3.5.1 |

## 4. 自定义运行

只跑某几个方法：

```bash
python scripts/run_experiments.py --methods rcfad wo_dynamic_beta --datasets cifar10 --seeds 42 --rounds 30
```

追加 Flower run-config：

```bash
python scripts/run_experiments.py --suite compare --datasets mnist --extra "epsilon-fpr=0.01 learning-rate=0.003"
```

列出可用方法：

```bash
python scripts/run_experiments.py --list-methods
```

## 5. 汇总结果

```bash
python scripts/collect_results.py --results-root outputs/experiments
```

输出：

```text
outputs/experiments/_summary/all_runs.csv
outputs/experiments/_summary/summary_selected.csv
outputs/experiments/_summary/summary_selected.md
outputs/experiments/_summary/fixed_threshold_diagnostics_selected.md
```
