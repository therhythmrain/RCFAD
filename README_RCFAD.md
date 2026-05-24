# RC-FAD: 可导低误报约束下的联邦个性化风险优化

本项目把 Flower/PyTorch quickstart 改造成论文 `Method.txt` 对应的联邦异常检测实验代码。数据集仍使用公开视觉数据集做可复现实验：选定一个类别作为异常类 `label=1`，其他类别作为正常类 `label=0`。

当前推荐数据集：

- `cifar10`
- `mnist`
- `creditcard`（Credit Card Fraud Detection 表格数据；本地存在 `data/creditcard.csv` 时直接读取，否则尝试从公开 CSV 镜像自动下载）
- `baf`（Bank Account Fraud / BAF；需要手动下载后放到 `data/baf/Base.csv`）
- `ai4i`（AI4I 2020 Predictive Maintenance；本地不存在时自动从 UCI 下载）
- `secom`（SECOM 半导体制造异常；本地不存在时自动从 UCI 下载）

默认异常类为 `anomaly-class=1`，默认客户端划分为 `risk_hetero`，即不同客户端拥有不同异常比例，用来模拟联邦风险异构。

表格数据集说明：

| 数据集名 | 标签列 | 数据准备 |
|---|---|---|
| `creditcard` | `Class` | 可自动下载；也可手动放置 `data/creditcard.csv` |
| `baf` | `fraud_bool` | 手动下载 BAF，并放置 `data/baf/Base.csv` |
| `ai4i` | `Machine failure` | 自动下载 UCI zip；会丢弃 `TWF/HDF/PWF/OSF/RNF` 避免标签泄漏 |
| `secom` | `secom_labels.data` 中 `1` 为异常 | 自动下载 UCI zip；缺失传感器值用训练前全表中位数填充 |

所有表格数据会自动做分类变量 one-hot、缺失值填充、分层 train/test 切分、训练集统计量标准化，并 pad/truncate 到统一的 tabular 输入维度。

## 方法实现对应关系

| 论文模块 | 代码位置 | 说明 |
|---|---|---|
| 可导 FPR/FNR 代理 | `RCFAD/task.py::_smooth_fpr_fnr`、`_smooth_numpy_fpr_fnr` | 用 Sigmoid 平滑替代不可导指示函数 |
| 个性化阈值联合学习 | `RCFAD/task.py::train`、`RCFAD/client_app.py::CLIENT_STATE` | 每个客户端维护自己的 `threshold/tau_k` 和 `lambda_fpr` |
| 低误报拉格朗日约束 | `RCFAD/task.py::train` | 优化 `risk_loss + mu*FNR + lambda*[FPR-epsilon]_+` |
| 动态少数类增强 | `RCFAD/task.py::train` | 按异常稀缺度、FNR 风险和 FPR 余量生成 `beta_k`，并只增强低于当前阈值的 hard-positive 样本 |
| 风险可靠性聚合 | `RCFAD/custom_strategy.py::_inject_risk_reliable_weights` | 服务端按 FNR、FPR 违背项和更新可靠性 `q_k` 生成聚合权重 |
| 实验与消融 | `scripts/run_experiments.py` | 提供 `compare`、`threshold`、`ablation`、`hetero`、`sensitivity`、`quick` 实验 |

## 推荐实验

先做快速检查：

```bash
python scripts/run_experiments.py --suite quick --datasets cifar10 mnist --seeds 42 --rounds 5 --dry-run
python scripts/run_experiments.py --suite quick --datasets cifar10 mnist --seeds 42 --rounds 5 --learning-rate 0.001
```

查看当前全部实验组、方法名：

```bash
python scripts/run_experiments.py --list-methods
```

正式主方法对比实验：

```bash
python scripts/run_experiments.py --suite compare --datasets cifar10 mnist --seeds 42 123 2025 --rounds 30 --learning-rate 0.001
```

表格异常检测主对比实验：

```bash
python scripts/run_experiments.py --suite compare --datasets creditcard ai4i secom --seeds 42 --rounds 30 --results-root outputs/experiments_tabular_compare
```

BAF 需要先准备数据：

```bash
mkdir -p data/baf
# 将下载得到的 Base.csv 放到 data/baf/Base.csv
python scripts/run_experiments.py --suite compare --datasets baf --seeds 42 --rounds 30 --results-root outputs/experiments_baf_compare
```

对比方法：

- `fedavg`：普通 FedAvg + BCE。
- `fedprox`：FedProx 风格 proximal 正则。
- `moon`：MOON 风格表征对比正则。
- `fedsimsup`：FedSimSup 风格本地监督器 + 特征相似度蒸馏。
- `confree`：ConFREE 风格冲突抑制聚合。
- `fedavg_focal`：FedAvg + Focal Loss。
- `rcfad`：完整方法。

说明：`moon`、`fedsimsup` 和 `confree` 是为了本实验框架实现的轻量近似基线，适合作为工程对比；若论文需要严格声称“复现某篇方法”，建议另外对齐其官方代码和超参数。

阈值机制专项实验：

```bash
python scripts/run_experiments.py --suite threshold --datasets cifar10 mnist --seeds 42 123 2025 --rounds 30 --learning-rate 0.001
```

阈值对比方法：

- `fedavg_fixed`：FedAvg，固定阈值。
- `fedavg_global_post`：FedAvg，训练后全局目标 FPR 阈值诊断。
- `fedavg_local_post`：FedAvg，客户端训练后本地阈值校准。
- `fedprox_local_post`：FedProx，客户端训练后本地阈值校准。
- `confree_local_post`：ConFREE 风格聚合，客户端训练后本地阈值校准。
- `rcfad_global_joint`：RC-FAD，但使用全局共享阈值状态。
- `rcfad`：完整 RC-FAD，客户端个性化阈值联合学习。

逐模块消融实验：

```bash
python scripts/run_experiments.py --suite ablation --datasets cifar10 mnist --seeds 42 123 2025 --rounds 30 --learning-rate 0.001
```

消融项：

- `base_no_modules`：无任何 RC-FAD 模块的底座，普通 BCE + 样本量聚合。
- `wo_joint_threshold`：移除联合阈值学习，改为训练后本地阈值校准。
- `wo_fpr_constraint`：移除低误报约束项和拉格朗日乘子。
- `wo_dynamic_beta`：移除风险状态驱动的动态少数类增强。
- `wo_risk_aggregation`：移除风险可靠性聚合，退化为样本量聚合。
- `wo_update_reliability`：保留风险聚合，但移除模型更新可靠性 `q_k`。

异构场景实验：

```bash
python scripts/run_experiments.py --suite hetero --datasets cifar10 mnist --seeds 42 123 2025 --rounds 30 --learning-rate 0.001
```

默认自动展开 3 个场景：

- `anomaly_ratio_hetero`：客户端异常比例不同。
- `normal_score_shift`：客户端正常样本分数分布偏移。
- `epsilon_hetero`：客户端误报约束 `epsilon_k` 不同。

默认方法包括 `base_no_modules`、`fedsimsup`、`confree`、`rcfad_global_joint`、`rcfad`，用于同时验证异构场景下完整方法、全局阈值、代表性联邦基线和无模块底座的差异。

可只跑指定场景：

```bash
python scripts/run_experiments.py --suite hetero --scenarios normal_score_shift --datasets mnist --seeds 42 --rounds 30
```

敏感性实验：

```bash
python scripts/run_experiments.py --suite sensitivity --datasets cifar10 mnist --seeds 42 123 2025 --rounds 30 --learning-rate 0.001
```

默认包括：

- `epsilon_0_01`、`epsilon_0_03`、`epsilon_0_05`、`epsilon_0_10`
- `dirichlet_alpha_0_1`、`dirichlet_alpha_0_3`、`dirichlet_alpha_0_5`、`dirichlet_alpha_1_0`
- `beta_zeta_1`、`beta_zeta_2`、`beta_zeta_4`、`beta_zeta_8`
- `lr_tau_0_0001`、`lr_tau_0_0002`、`lr_tau_0_0005`

可只跑一个小网格：

```bash
python scripts/run_experiments.py --suite sensitivity --scenarios epsilon_0_01 epsilon_0_03 epsilon_0_05 epsilon_0_10 --datasets mnist --seeds 42 --rounds 30
```

## 结果文件

每个 run 会写入：

```text
outputs/experiments/<run-name>/
  run_config.json
  server_metrics.csv
  client_eval_metrics.csv
  client_train_metrics.csv
  summary.json
  model_state_auprc_*.pth
```

汇总表：

```bash
python scripts/collect_results.py --results-root outputs/experiments
```

主要看：

- `outputs/experiments/_summary/summary_selected.md`
- `outputs/experiments/_summary/summary_selected.csv`
- `outputs/experiments/_summary/fixed_threshold_diagnostics_selected.md`
- `outputs/experiments/_summary/client_threshold_diagnostics_selected.md`

论文表格建议优先使用 `AUPRC`、`Recall@FPR`、`Accuracy@FPR`、`F1@FPR`。固定阈值 `accuracy/recall/f1` 只作为诊断，因为异常检测中全判正常也可能得到较高 accuracy。
如果需要展示真实部署阈值效果，使用 `client_threshold_diagnostics_selected.md`，它按客户端实际 learned/fixed threshold 计算指标，不使用测试集后验 FPR 阈值校准。
异构和敏感性实验的汇总表包含 `scenario` 列，避免把不同场景的结果混在一起平均。
