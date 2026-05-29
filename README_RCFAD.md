# RC-FAD 最终实验说明

RC-FAD 是面向风险敏感异常检测的联邦学习实验代码。本文方法的核心不是单纯提高平均准确率，而是在异构客户端下同时优化异常排序能力、低误报约束可行性和客户端个性化阈值。

## 方法模块

| 模块 | 代码位置 | 作用 |
|---|---|---|
| 个性化阈值联合学习 | `RCFAD/client_app.py`, `RCFAD/training.py` | 每个客户端维护 `tau_k`，并与模型参数联合优化 |
| 可导低 FPR 约束 | `RCFAD/training.py::_smooth_fpr_fnr` | 用平滑 FPR/FNR 代理嵌入训练目标 |
| 风险门控 hard-positive 增强 | `RCFAD/training.py::train` | 对低于当前阈值的异常样本加权，缓解漏报 |
| 风险可靠性聚合 | `RCFAD/custom_strategy.py` | 基于风险状态和更新可靠性调整聚合权重 |
| 客户端级诊断 | `RCFAD/custom_strategy.py`, `scripts/collect_results.py` | 输出 violation rate、client-FPR std、worst-client recall 等指标 |

## 代码结构

`RCFAD/task.py` 现在只保留兼容导入层，避免所有逻辑挤在一个文件里：

```text
RCFAD/constants.py   常量
RCFAD/model.py       模型
RCFAD/data.py        数据集读取、缓存、客户端划分
RCFAD/metrics.py     AUPRC/AUROC/FPR/FNR 和阈值指标
RCFAD/training.py    训练与测试循环
RCFAD/utils.py       随机种子和元信息
```

## 推荐数据集

正文建议优先使用已经验证效果较好的工业/医学异常检测数据集：

```text
AI4I, SECOM, Annthyroid, Mammography, TEP, Shuttle
```

金融或工控扩展数据集可以作为补充或附录：

```text
Credit Card, PaySim, BAF, SWaT, HAI, SMD
```

## 快速检查

```bash
python scripts/run_experiments.py \
  --suite quick \
  --datasets ai4i \
  --seeds 42 \
  --rounds 1 \
  --dry-run
```

## 主对比实验

```bash
python scripts/run_experiments.py \
  --suite compare \
  --datasets ai4i secom annthyroid mammography tep shuttle \
  --seeds 42 1234 3407 \
  --rounds 30 \
  --results-root outputs/stage_compare
```

正文主表建议指标：

```text
AUPRC (%), Recall@FPR<=tau (%), F1@FPR<=tau (%), FPR@chosen (%)
```

## 消融实验

```bash
python scripts/run_experiments.py \
  --suite ablation_risk \
  --datasets ai4i secom annthyroid mammography tep shuttle \
  --seeds 42 1234 3407 \
  --rounds 30 \
  --results-root outputs/stage_ablation_risk
```

正文消融表建议指标：

```text
AUPRC (%), F1@chosen (%), FPR@chosen (%),
Recall@FPR<=tau (%), F1@FPR<=tau (%)
```

如果某个模块只带来很小变化，不要强行写成显著提升；可以表述为稳定性辅助模块，并把更细的客户端诊断放附录。

## 阈值机制实验

当前最终阈值实验取消了 `FedAvg-Local-Post`，方法包括：

```text
FedAvg-Fixed
FedAvg-Global-Post
FedProx-Local-Post
ConFREE-Local-Post
RC-FAD-Global-Joint
RC-FAD
```

运行命令：

```bash
python scripts/run_experiments.py \
  --suite threshold_hetero \
  --datasets ai4i secom annthyroid mammography tep shuttle \
  --seeds 42 1234 3407 \
  --rounds 30 \
  --methods fedavg_fixed fedavg_global_post fedprox_local_post confree_local_post rcfad_global_joint rcfad \
  --results-root outputs/stage_threshold
```

GPU 版本：

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

阈值表不要只看 `AUPRC` 和 `Recall@FPR<=tau`，还要报告部署阈值下的：

```text
Recall@chosen (%), F1@chosen (%), FPR@chosen (%),
Violation Rate (%), Client-FPR Std (%), Worst-client Recall (%)
```

## 汇总表

```bash
python scripts/collect_results.py --results-root outputs/stage_threshold
python scripts/build_paper_tables.py \
  --threshold-root outputs/stage_threshold \
  --output-root outputs/paper_tables
```

重点查看：

```text
outputs/stage_threshold/_summary/summary_selected.csv
outputs/stage_threshold/_summary/client_threshold_diagnostics_selected.csv
outputs/stage_threshold/_summary/table_threshold_core_percent_3seed_all6.csv
outputs/stage_threshold/_summary/table_threshold_deployment_diagnostics_percent_3seed_all6.csv
```

论文表格统一使用百分数，两位小数：

```text
33.24 ± 1.02
```

不要在单元格中写 `%`，单位放在表头。
