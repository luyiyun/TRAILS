# TRAILS

TRAILS 是 **Trajectory and Risk-informed Analysis of Irregular Longitudinal
Subtypes** 的缩写。项目目标是构建面向非同步多变量医学纵向数据的深度生成式生存轨迹聚类模型，用于在真实世界随访不规则、变量缺失、检查频率不一致的条件下识别具有不同动态轨迹和事件风险的患者亚型。

## 当前范围

当前版本是阶段一基础版：变长访问序列上的模块化 Surv-VaDER/VaDE 原型。

- 模拟器采用 VaDeSC-EHR 风格的数据生成主线：cluster-specific latent profile -> 随机非线性轨迹生成 -> pseudo attention -> 观测序列 -> Weibull 生存结局。
- 输入是非同步采样的多变量医学检查序列，包括血液检查、肝肾功能、炎症指标、肿瘤标志物和肿瘤负荷等连续变量。
- 数据保留 `mask` 和 `delta_time`，用于表达变量级缺失和距离上次观测的时间间隔。
- 编码器支持 GRU-D、原始 aligned mTAN 多时间注意力输入层，并保留旧的 compact
  per-feature `mtan2` 输入层用于对照。
- 解码器支持 GRU/LSTM/Transformer，从患者级 latent representation 重构纵向轨迹。
- 聚类模块使用 VaDE 风格的可学习 Gaussian mixture latent prior。
- 生存模块由患者 latent mean 输出一组 Weibull shape/scale，不跨簇混合生存分布。

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

MIMIC 固定划分的比较流程为 `06_split` → `07_run` → `08_baselines` → `09_evaluation`：

```bash
uv run python -m scripts.mimic.08_baselines input_dir=outputs/mimic_case/<trails-run> paths.dir=outputs/mimic_case/<baseline-run>
uv run python -m scripts.mimic.09_evaluation input_dir=outputs/mimic_case/<trails-run> 'baseline_dirs=[outputs/mimic_case/<baseline-run>]'
```

08 复用07保存的三划分和训练配置，不重新划分患者；所有方法只使用train拟合，
validation可用于早停，test仅用于冻结预测后的评价。`methods`可选择子集或配置多个seed。
R方法需要远端 `lcmm`、`JMbayes2`、`data.table`、`jsonlite`、`R.utils`、`nlme`和`survival`。
09支持多个已完成的基线目录，输出方法×seed×split结果及统一比较表；
聚类方法不通过簇KM构造预测C-index、AUC、IBS或校准。真实数据与患者级产物留在远端。

实验入口使用 Hydra 配置。每个任务都有独立脚本；生成模拟 train/test split：

```bash
uv run python scripts/simulate.py simulation=quick
```

模拟场景放在 `configs/simulation/`：

```bash
uv run python scripts/simulate.py simulation=quick
uv run python scripts/simulate.py simulation=base
uv run python scripts/simulate.py simulation=imbalance
uv run python scripts/simulate.py simulation=censored
uv run python scripts/simulate.py simulation=high_dimension
```

所有参数都可以通过 Hydra 覆盖，例如：

```bash
uv run python scripts/simulate.py simulation=quick 'train_size=[128]' 'test_size=[32]'
uv run python scripts/simulate.py simulation=base 'generator.n_clusters=[2,3,4]'
```

`generator` 控制数据生成机制；`mechanism_seed` 不设置时默认使用
`seed`。`train_size` 和 `test_size` 是按位置配对
的列表，每个样本量层级再与 `generator.n_clusters` 列表做组合，并在
`repeats` 内重复。相同 `name × K` 固定生成机制，不同样本量层级
和 repeat 使用独立 sample seed。

单独生成模拟数据：

```bash
uv run python scripts/simulate.py simulation=base
```

输出路径为
`paths.dir/train_<train>_test_<test>/k<K>/<repeat>/train.pt` 和对应 `test.pt`。
默认 `paths.root=outputs/simulate`、`paths.prefix=<simulation>`、`paths.suffix=<timestamp>`，
并组合成 `paths.dir`。同一个目录下还会写出 `simulation_manifest.csv` 与
`simulation_summary.json`。

训练已有的 train/test split：

```bash
uv run python scripts/train.py training=base paths.data_root=data/simulated/base
uv run python scripts/train.py training=mtan paths.data_root=data/simulated/high_dimension
```

训练显式指定的 train/test `.pt` 数据：

```bash
uv run python scripts/train.py training=small paths.explicit_split.enabled=true paths.explicit_split.train_data=data/simulated/base/train_500_test_300/k2/0/train.pt paths.explicit_split.test_data=data/simulated/base/train_500_test_300/k2/0/test.pt
```

真实数据建模使用 `scripts/case.py`。先把真实队列预处理成两个 CSV：

- `patients.csv`：必需列为 `patient_id`、`survival_time`、`event`，可选列为
  `cluster_label`。`event` 必须是 `0/1`，`survival_time` 与纵向观测时间使用同一单位。
- `observations.csv`：必需列为 `patient_id`、`time`、`feature`、`value`。每行是一位患者
  在某个时间点的某个变量观测；同一个 `patient_id + time + feature` 不能重复。

运行示例：

```bash
uv run python scripts/case.py observations_csv=data/case/observations.csv patients_csv=data/case/patients.csv
```

`scripts/case.py` 默认加载 `training=case`，默认开启 SwanLab、保存完整训练 artifacts，
并开启 latent embedding diagnostics。命令使用全部患者训练，`trainer.valid_size`
只作为内部 early stopping validation。输出默认保存在 `paths.dir`，默认形如
`outputs/case/case-<timestamp>/`，包括 `case_dataset.pt`、`case_dataset_summary.json`、
`config.json`、`history.json`、`history.csv`、`history.png`、`model.pt`、
`predictions.pt`、`patient_clusters.csv`、`cluster_summary.csv`、
`cluster_feature_summary.csv` 和 `case_summary.json`。`patient_clusters.csv`
包含患者 ID、预测 cluster、风险分数、cluster posterior probabilities、生存结局和观测摘要，
便于后续制作 KM 曲线、cluster composition 和变量分布图。

对已有 train/test split 运行轻量基线方法：

```bash
uv run python scripts/baseline.py paths.data_root=data/simulated/base
```

对已有 train/test split 运行 Optuna 搜索：

```bash
uv run python scripts/optim.py paths.data_root=data/simulated/base optim.n_trials=20
uv run python scripts/optim.py paths.data_root=data/simulated/base 'optim.run_ids=[base/train_500_test_300/k2/0]' optim.n_trials=20
uv run python scripts/optim.py paths.data_root=data/simulated/base paths.dir=outputs/optim/base-round1 optim.n_trials=20
uv run python scripts/optim.py paths.data_root=data/simulated/base paths.dir=outputs/optim/base-round1 optim.resume=true optim.n_trials=20
```

合并训练与基线结果并生成图表：

```bash
uv run python scripts/summary.py 'train_roots=[outputs/train/base-...,outputs/train/mtan-...]' 'baseline_roots=[outputs/baseline/base-...]' 'train_labels=[base,mtan]' 'baseline_labels=[kmeans]'
```

单次运行也使用列表形式：

```bash
uv run python scripts/summary.py 'train_roots=[outputs/train/base-...]' 'baseline_roots=[outputs/baseline/base-...]'
```

`scripts/train.py`、`scripts/baseline.py` 和 `scripts/optim.py` 都会递归扫描
`paths.data_root` 下所有 sibling `train.pt`/`test.pt` 目录；也可以用
`paths.explicit_split.enabled=true` 加 `paths.explicit_split.train_data/test_data`
显式指定单个 split。通用命令根配置在 `configs/<command>.yaml`，MIMIC 命令配置集中在
`configs/mimic/`；Hydra 元数据和命令输出保存在
同一个 `paths.dir`，默认由 `paths.root`、`paths.prefix` 和 `paths.suffix` 组合，也可以手动
覆盖，例如 `paths.dir=outputs/train/my-run`。`paths.data_root` 和 explicit split 默认值由
train、baseline、optim 各自的根配置直接声明。输入路径（如 `paths.data_root`、summary roots、
case CSV）相对启动命令时的当前目录解析；输出路径统一相对 `paths.dir` 解析。train、baseline
和 optim 会镜像数据相对路径，例如 `outputs/train/base-.../train_500_test_300/k2/0/trails.pt`。
训练和基线会优先从 dataset metadata 中的 `generation_params.n_clusters` 推断 K，metadata
缺失时才回退到 YAML 默认值。

`scripts/train.py` 会在每个 split 开始和结束时打印已耗时与剩余时间估计，并在
`paths.dir` 下保存 `train_summary.json`、`train_metrics.csv`
和 `<run_id>/trails.pt`。`scripts/baseline.py` 会保存
`baseline_summary.json`、`baseline_metrics.csv` 和
`<run_id>/<method>.pt`，用于比较 summary-feature KMeans、risk-stratified
summary-feature KMeans 和 FPCA-KMeans。`scripts/optim.py` 会对所有选中 split
共享同一组超参数 trial，并以平均 C-index 与平均 ARI 作为多目标；`optim.run_ids`
为空表示使用全部 split，非空时只选择指定 split。`optim.parallel` 提供共享进程池，
`optim.resume=true paths.dir=<已有运行目录>` 会在数据 fingerprint 一致时继续追加 trials。
每次 optim 会写出 `trials.csv`、`pareto_trials.json`、`top_trials.csv` 和 `figures/`
下的 Pareto、目标历史、split heatmap 等图表。`scripts/summary.py` 读取显式 train/baseline run 目录下的
metrics CSV，保存合并 CSV、聚合 CSV、summary JSON，并为每个 simulation scenario
生成一张按 `metrics × K` 排布的带误差条总图 PNG/PDF。

命令结束时 logging 会打印精简的可读 summary；完整机器可读结果保存在上述
JSON/CSV artifacts 中。

训练 split 默认串行执行；需要多进程时显式设置 `parallel.workers`：

```bash
uv run python scripts/train.py training=base paths.data_root=data/simulated/base parallel.workers=4
```

未配置 `parallel.devices` 时，每个 worker 都沿用
`trainer.device`，因此也允许多个进程同时使用同一张 GPU，例如默认的
`cuda:0`；请按显存情况控制 `workers`。如果有多张 GPU，可以轮转分配设备：

```bash
uv run python scripts/train.py training=base paths.data_root=data/simulated/base parallel.workers=4 'parallel.devices=[cuda:0,cuda:1]'
```

终端输出统一走 logging，并与 tqdm 进度条兼容；train 会显示总 split 进度条以及
当前活跃 worker 的训练进度条。

可以用 `artifacts.names` 控制训练保存内容：

```bash
uv run python scripts/train.py training=base paths.data_root=data/simulated/base 'artifacts.names=[config,history,test,plot]'
uv run python scripts/train.py training=base paths.data_root=data/simulated/base 'artifacts.names=[none]'
```

SwanLab 由配置控制，多组 split 训练会自动在实验名后追加 run id：

```bash
uv run python scripts/train.py training=base paths.data_root=data/simulated/base swanlab.mode=disabled
uv run python scripts/train.py training=mtan paths.data_root=data/simulated/censored swanlab.mode=disabled
```

命令入口位于 `scripts/`，每个命令脚本拥有独立 Hydra root config。
`trails` 主包只包含核心方法代码；`trails_simulate` 和 `trails_case`
保留模拟、训练、case 分析所需的可复用 helper，并只能作为下游包引用 `trails`。

## Roadmap

阶段一：模块化 Surv-VaDER 基础版

- 完成 VaDeSC-EHR 风格的连续型多变量非同步采样模拟器。
- 使用 GRU-D、原始 aligned mTAN 或旧 compact `mtan2` encoder 处理 `x/mask/delta_time`。
- 使用 GRU/LSTM/Transformer decoder 重构纵向轨迹。
- 加入 VaDE Gaussian mixture latent prior、warmup 后 deterministic k-means 初始化，以及 Weibull survival head。
- 比较是否加入 survival loss 对聚类风险区分度的影响。

阶段二：mTAN-Surv-VaDER 主模型

- 将输入从固定访问序列扩展为 observation-level irregular events。
- 使用原始 mTAN reference time attention 将 aligned 非同步观测映射到 reference grid。
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
