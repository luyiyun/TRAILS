# TRAILS

TRAILS 是 **Trajectory and Risk-informed Analysis of Irregular Longitudinal
Subtypes** 的缩写。项目目标是构建面向非同步多变量医学纵向数据的深度生成式生存轨迹聚类模型，用于在真实世界随访不规则、变量缺失、检查频率不一致的条件下识别具有不同动态轨迹和事件风险的患者亚型。

## 当前范围

当前版本是阶段一基础版：变长访问序列上的模块化 Surv-VaDER/VaDE 原型。

- 模拟器采用 VaDeSC-EHR 风格的数据生成主线：cluster-specific latent profile -> 随机非线性轨迹生成 -> pseudo attention -> 观测序列 -> Weibull 生存结局。
- 输入是非同步采样的多变量医学检查序列，包括血液检查、肝肾功能、炎症指标、肿瘤标志物和肿瘤负荷等连续变量。
- 数据保留 `mask` 和 `delta_time`，用于表达变量级缺失和距离上次观测的时间间隔。
- 编码器支持 GRU-D 或标准 mTAN-style 多时间注意力输入层，显式建模缺失模式和时间间隔。
- 解码器支持 GRU/LSTM/Transformer，从患者级 latent representation 重构纵向轨迹。
- 聚类模块使用 VaDE 风格的可学习 Gaussian mixture latent prior。
- 生存模块使用 cluster-specific Weibull mixture survival head，混合权重来自 VaDE posterior。

本阶段不实现 mixed-type likelihood、competing risks 或 recurrent events。

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

实验入口使用 Hydra 配置。默认命令是生成模拟 train/test split：

```bash
uv run main.py command=simulate simulation=quick paths.data_root=data/simulated
```

模拟场景放在 `configs/simulation/`：

```bash
uv run main.py command=simulate simulation=quick paths.data_root=data/simulated
uv run main.py command=simulate simulation=base paths.data_root=data/simulated
uv run main.py command=simulate simulation=imbalance paths.data_root=data/simulated
uv run main.py command=simulate simulation=censored paths.data_root=data/simulated
uv run main.py command=simulate simulation=high_dimension paths.data_root=data/simulated
```

所有参数都可以通过 Hydra 覆盖，例如：

```bash
uv run main.py command=simulate simulation=quick 'simulation.train_size=[128]' 'simulation.test_size=[32]'
uv run main.py command=simulate simulation=base 'simulation.generator.n_clusters=[2,3,4]'
```

`simulation.generator` 控制数据生成机制；`simulation.mechanism_seed` 不设置时默认使用
`simulation.seed`。`simulation.train_size` 和 `simulation.test_size` 是按位置配对
的列表，每个样本量层级再与 `simulation.generator.n_clusters` 列表做组合，并在
`simulation.repeats` 内重复。相同 `simulation.name × K` 固定生成机制，不同样本量层级
和 repeat 使用独立 sample seed。

单独生成模拟数据：

```bash
uv run main.py command=simulate simulation=base paths.data_root=data/simulated
```

输出路径为
`<data_root>/<simulation.name>/train_<train>_test_<test>/k<K>/<repeat>/train.pt`
和对应 `test.pt`，并在场景目录下写出 `simulation_manifest.csv` 与
`simulation_summary.json`。

训练已有的 train/test split：

```bash
uv run main.py command=train training=base paths.data_root=data/simulated/base
uv run main.py command=train training=mtan paths.data_root=data/simulated/high_dimension
```

训练显式指定的 train/test `.pt` 数据：

```bash
uv run main.py command=train training=small paths.data=data/simulated/base/train_500_test_300/k2/0/train.pt paths.test_data=data/simulated/base/train_500_test_300/k2/0/test.pt
```

对已有 train/test split 运行轻量基线方法：

```bash
uv run main.py command=baseline paths.data_root=data/simulated/base
```

对已有 train/test split 运行 Optuna 搜索：

```bash
uv run main.py command=optim paths.data_root=data/simulated/base optim.n_trials=20
uv run main.py command=optim paths.data_root=data/simulated/base optim.run_id=base/train_500_test_300/k2/0 optim.n_trials=20
```

合并训练与基线结果并生成图表：

```bash
uv run main.py command=summary summary.train_root=outputs/train-... summary.baseline_root=outputs/baseline-...
```

`command=train`、`command=baseline` 和 `command=optim` 都会递归扫描
`paths.data_root` 下所有 sibling `train.pt`/`test.pt` 目录；也可以用
`paths.data + paths.test_data` 显式指定单个 split。输出会镜像数据相对路径，例如
`outputs/.../train_500_test_300/k2/0/trails.pt`。训练和基线会优先从 dataset metadata
中的 `generation_params.n_clusters` 推断 K，metadata 缺失时才回退到 YAML 默认值。

`command=train` 会在 Hydra run 目录下保存 `train_summary.json`、`train_metrics.csv`
和 `<run_id>/trails.pt`。`command=baseline` 会保存
`baseline_summary.json`、`baseline_metrics.csv` 和
`<run_id>/<method>.pt`，用于比较 summary-feature KMeans、risk-stratified
summary-feature KMeans 和 FPCA-KMeans。`command=optim` 一次只对一个数据 split
创建 study；当 `paths.data_root` 下有多个 split 且未提供 `optim.run_id` 时，会在终端
列出编号让用户选择一个。`command=summary` 读取显式 train/baseline run 目录下的
metrics CSV，保存合并 CSV、聚合 CSV、summary JSON 和 `figures/*.png`。

命令结束时 stdout 会打印精简的可读 summary；完整机器可读结果保存在上述
JSON/CSV artifacts 中。

可以用 `training.artifacts.names` 控制训练保存内容：

```bash
uv run main.py command=train training=base paths.data_root=data/simulated/base 'training.artifacts.names=[config,history,test,plot]'
uv run main.py command=train training=base paths.data_root=data/simulated/base 'training.artifacts.names=[none]'
```

SwanLab 由配置控制，多组 split 训练会自动在实验名后追加 run id：

```bash
uv run main.py command=train training=base paths.data_root=data/simulated/base training.swanlab.mode=disabled
uv run main.py command=train training=mtan paths.data_root=data/simulated/censored training.swanlab.mode=disabled
```

目前所有命令行都集中在根目录 `main.py`。`trails` 主包只包含核心方法代码；
`trails_simulate` 和 `trails_case` 只能作为下游包引用 `trails`。

## Roadmap

阶段一：模块化 Surv-VaDER 基础版

- 完成 VaDeSC-EHR 风格的连续型多变量非同步采样模拟器。
- 使用 GRU-D 或 mTAN-style encoder 处理 `x/mask/delta_time`。
- 使用 GRU/LSTM/Transformer decoder 重构纵向轨迹。
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

- 模拟实验：构造基础、小样本不均衡、高缺测高删失、高维 biomarker 四类场景；
  每类场景展开 5 个样本量层级、4 个真实聚类数量和 5 次重复的网格。
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
