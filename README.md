# TRAILS

TRAILS 是 **Trajectory and Risk-informed Analysis of Irregular Longitudinal
Subtypes** 的缩写。项目目标是构建面向非同步多变量医学纵向数据的深度生成式生存轨迹聚类模型，用于在真实世界随访不规则、变量缺失、检查频率不一致的条件下识别具有不同动态轨迹和事件风险的患者亚型。

## 当前范围

当前版本是阶段一基础版：变长访问序列上的 GRU-D Surv-VaDER/VaDE 原型。

- 模拟器采用 VaDeSC-EHR 风格的数据生成主线：cluster-specific latent profile -> 随机非线性轨迹生成 -> pseudo attention -> 观测序列 -> Weibull 生存结局。
- 输入是非同步采样的多变量医学检查序列，包括血液检查、肝肾功能、炎症指标、肿瘤标志物和肿瘤负荷等连续变量。
- 数据保留 `mask` 和 `delta_time`，用于表达变量级缺失和距离上次观测的时间间隔。
- 编码器使用 GRU-D，显式建模缺失模式和时间间隔。
- 解码器使用 GRU，从患者级 latent representation 重构纵向轨迹。
- 聚类模块使用 VaDE 风格的可学习 Gaussian mixture latent prior。
- 生存模块使用 cluster-specific Weibull mixture survival head，混合权重来自 VaDE posterior。

本阶段不实现 mTAN、mixed-type likelihood、competing risks 或 recurrent events。

## 数据结构

模拟数据保存为 `torch.save` payload，加载后是 `ClinicalTimeSeriesDataset`。生成机制参考 VaDeSC-EHR 的模拟流程，但将 ICD softmax/argmax 分类 decoder 替换为连续临床变量 decoder。每个患者样本包含：

- `times`: 该患者的检查时间序列。
- `x`: 检查值矩阵，形状为 `(n_visits, n_features)`。
- `mask`: 对应变量是否被观测。
- `delta_time`: 每个变量距离上次观测的时间间隔。
- `survival_time`: 随访或事件时间。
- `event`: 事件指示。
- `cluster_label`: 模拟数据中的真实潜在亚型；真实数据可不提供该字段。

Dataset 的 `metadata` 会保留 `latent_z`、`cluster_means`、`cluster_covariances`、`survival_coefficients` 和生成参数，方便后续仿真实验评估聚类恢复、风险区分和敏感性分析。

## 命令

生成模拟数据：

```bash
uv run main.py simulate --out data/simulated/demo.pt --patients 128 --clusters 3 --hidden-size 100 --latent-dim 5 --attention-layers 3 --seed 2026
```

一次性生成训练、验证和测试数据：

```bash
uv run main.py simulate --out data/simulated/realistic --split-patients 3000 1000 1000 --clusters 4 --min-visits 3 --max-visits 16 --followup-days 1095 --hidden-size 128 --latent-dim 8 --attention-layers 4 --attention-heads 4 --censoring-rate 0.45 --seed 2026
```

训练基础模型：

```bash
uv run main.py train --data data/simulated/demo.pt --epochs 1 --batch-size 16
```

训练命令默认会把本次实验保存到 `runs/<YYYYmmdd-HHMMSS>/`，包括
`config.json`、`history.json`、`history.csv`、`test_metrics.json`、`model.pt`
和 `history.png`。可以用空格列表控制保存内容：

```bash
uv run main.py train --data data/simulated/demo.pt --save-artifacts config history test plot
```

如需独立测试集，可以传入 `--test-data`；如不想保存本次运行，可传入
`--save-artifacts none`。

调试训练过程时可以开启 SwanLab 实时记录每个 epoch 的训练、验证和最终测试指标：

```bash
uv run main.py train --data data/simulated/demo.pt --val-data data/simulated/demo.pt --epochs 5 --batch-size 16 --swanlab --swanlab-project TRAILS --swanlab-experiment debug-demo
```

训练时加入验证集并监控模拟数据的聚类恢复效果：

```bash
uv run main.py train --data data/simulated/demo.pt --val-data data/simulated/demo.pt --epochs 1 --warmup-epochs 1 --batch-size 16
```

目前所有命令行都集中在根目录 `main.py`。`trails` 主包只包含核心方法代码；`trails_simulate` 和 `trails_case` 只能作为下游包引用 `trails`。

## Roadmap

阶段一：GRU-D Surv-VaDER 基础版

- 完成 VaDeSC-EHR 风格的连续型多变量非同步采样模拟器。
- 使用 GRU-D encoder 处理 `x/mask/delta_time`。
- 使用 GRU decoder 重构纵向轨迹。
- 加入 VaDE Gaussian mixture latent prior、warmup 后 deterministic k-means 初始化，以及 Weibull survival head。
- 比较是否加入 survival loss 对聚类风险区分度的影响。

阶段二：mTAN-Surv-VaDER 主模型

- 将输入从固定访问序列扩展为 observation-level irregular events。
- 使用 mTAN-style reference time attention 将非同步观测映射到 reference grid。
- 与固定时间窗、插值、GRU-D、Latent ODE/ODE-RNN 等方法比较。
- 系统评估 reference time points 数量和设置方式。

阶段三：复杂医学数据与事件扩展

- 支持连续、二分类、计数、有序变量等 mixed-type longitudinal variables。
- 扩展到 competing risks、recurrent events 和 treatment-specific outcomes。
- 增加真实队列案例验证、KM 曲线、prototype trajectory 和临床变量解释。

## 实验计划

- 模拟实验：构造不同类别具有不同轨迹、不同事件风险、二者同时不同、以及 informative observation process 的场景。
- 真实数据实验：优先考虑肿瘤新辅助治疗、慢病管理、ICU 风险分型或真实世界疾病进展队列。
- 消融实验：去掉 survival decoder、替换 GRU-D/mTAN 前端、改变 survival loss 权重、去掉 mask 或 delta time。
- 基线方法：GBTM、LCMM、GMM、VaDER、RNN/Transformer autoencoder、Cox、RSF、DeepSurv、DeepHit、Deep Survival Machines、GRU-D、Latent ODE 和 mTAN without clustering。

## 开发检查

```bash
uv run ruff format
uv run ruff check --fix
UV_CACHE_DIR=/tmp/uv-cache uv run pyright
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```
