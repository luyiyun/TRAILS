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

已有中格式观测表（每个患者—时间点一行、每个特征一列）时，可直接构建数据集：

```python
import pandas as pd
from trails import ClinicalTimeSeriesDataset

dataset = ClinicalTimeSeriesDataset.from_dataframes(
    patients=pd.DataFrame({"patient_id": ["p1"], "survival_time": [500.0], "event": [1]}),
    observations=pd.DataFrame({
        "patient_id": ["p1", "p1"],
        "time_days": [0.0, 30.0],
        "CEA": [2.0, 3.0],
        "CA242": [float("nan"), 12.0],
    }),
    time_col="time_days",
    use_features=["CEA", "CA242"],
)
```

`from_dataframes()` 保持患者表顺序，按时间排序各患者的访视，将 NaN 转为零值和
缺失 mask，并计算 `delta_time`；不会聚合重复时间点或拟合预处理。省略
`use_features` 时使用观测表中除 ID、时间外的列，顺序保持不变。默认返回 aligned
样本，也可设置 `return_kind="compact"`。患者 ID 和观测摘要自动写入 `metadata`。
已有长格式 CSV 时使用 `load_from_csv()`：内部先透视为中格式，再调用
`from_dataframes()`；CSV 入口保留观测文件中的患者首次出现顺序和列映射元数据。

## 命令

MIMIC 比较流程以 `06_split` 生成的冻结划分为共同输入：`07_run` 和
`08_baselines` 分别读取该划分，`09_evaluation` 再联合读取两类预测结果。
06 会拒绝覆盖已经存在的 seed 目录，需要新划分时应使用新的输出目录或 seed。

```bash
# 默认n_clusters=null：按配置的候选K和seeds在train/validation上选择K
uv run python -m scripts.mimic.07_run
# 显式指定K时跳过选择
uv run python -m scripts.mimic.07_run n_clusters=3
# 扩展seeds列表后启用跨seed稳定性汇总；trainer.seed必须包含在列表中
uv run python -m scripts.mimic.07_run 'k_selection.seeds=[20260517,20260518,20260519]'
uv run python -m scripts.mimic.08_baselines split_dir=data/real/mimic-iv-3.1/derived/trails_splits/seed-20260517 n_clusters=3 paths.dir=outputs/mimic_case/<baseline-run>
uv run python -m scripts.mimic.09_evaluation 'trails_dirs=[outputs/mimic_case/<trails-run>]' 'baseline_dirs=[outputs/mimic_case/<baseline-run>]'
# 在一次09运行中比较多个TRAILS实验，并直接计算模型间ARI/NMI
uv run python -m scripts.mimic.09_evaluation 'trails_dirs=[outputs/mimic_case/<primary-trails-run>,outputs/mimic_case/<other-trails-run>]'
```

`trails_dirs`不能为空；其中第一个目录仅作为主TRAILS方法。09会从所有07/08 manifest
推导共同的06冻结划分；任一结果引用不同划分时会直接报错。

07自动选择K时不会使用test；只有K及`trainer.seed`对应的最终模型锁定后才加载test。
选择结果保存在运行目录的`k_selection/`，包括逐运行指标、跨seed稳定性、K汇总和候选模型。
08 通过`split_dir`直接读取06保存的三划分，并通过`n_clusters`明确指定K；它与07之间没有
输入依赖。所有方法只使用train拟合，validation可用于早停，test仅用于冻结预测后的评价。
`methods`可选择子集或配置多个seed。
R方法需要远端 `lcmm`、`JMbayes2`、`data.table`、`jsonlite`、`R.utils`、`nlme`和`survival`。
09支持一个主TRAILS目录、任意多个同冻结划分的额外TRAILS目录和多个已完成的基线目录；
每个07/08 manifest都必须直接记录其06划分引用，旧08 manifest不再兼容。09确认这些引用
一致后，从共同的06目录加载数据集和预处理参数，
输出方法×seed×split结果、统一比较表及所有有聚类能力方法之间的ARI/NMI；
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


## CRC 云南六阶段流程

新产物使用 `v2`，旧目录不覆盖。观测表统一为中格式：
`patient_id,time_days,特征1,特征2,...`；缺失观测保留 NaN。

```bash
uv run python -m scripts.crc_yunnan.01_cohort
uv run python -m scripts.crc_yunnan.02_eda
uv run python -m scripts.crc_yunnan.03_split
uv run python -m scripts.crc_yunnan.04_preproc
uv run python -m scripts.crc_yunnan.05_run
uv run python -m scripts.crc_yunnan.06_evaluate inputs.run_dir=outputs/crc_yunnan/<运行名>/modeling
```

01 完成基础清洗与纳排，另存特征分组和中文排除汇总。02 输出中文
`eda_summary.json`、`eda_tables.xlsx` 和七类图。03 只保存患者划分与原始尺度
CSV；04 选择结局、窗口及固定面板，仅在训练观测上拟合变换，再保存中格式和
建模张量。05 读取 04 的 `preproc_manifest.json`，只对存在的数据集输出结果。
02 的 `outcome` 控制 landmark 人群、覆盖率分母和时间切点比较，默认与本轮
建模一致使用 OS；完整队列生存曲线仍同时展示 DFS 和 OS。
各命令的 YAML 位于 `configs/crc_yunnan/`，旧长格式和旧编号入口不再使用。

默认 03 同时生成 random 和 temporal-2017，默认 04/05 使用 random。
预处理时间划分时，同时设置输入和独立输出目录：

```bash
uv run python -m scripts.crc_yunnan.04_preproc \
  inputs.split_dir=data/derived/crc_yunnan/splits/v2/temporal-2017/seed-20260908 \
  paths.dir=data/derived/crc_yunnan/preproc/v2/os/landmark-24m/E60/auto-log1p-robust/temporal-2017/seed-20260908
uv run python -m scripts.crc_yunnan.05_run split.strategy=temporal-2017
```

### 全样本固定 K 训练

以下命令先仅保存完整基础队列，再按默认 OS、24 个月和 E60 规则形成全体合格
患者。若同时需要普通划分，去掉第一条命令中的 `split.strategies=[]`。

```bash
uv run python -m scripts.crc_yunnan.03_split \
  'split.strategies=[]' split.save_full_dataset=true
uv run python -m scripts.crc_yunnan.04_preproc \
  inputs.train_dir=data/derived/crc_yunnan/splits/v2/full \
  'inputs.test_dirs=[]' paths.dir=data/derived/crc_yunnan/preproc/v2/full-e60
uv run python -m scripts.crc_yunnan.05_run \
  split.dir=data/derived/crc_yunnan/preproc/v2/full-e60 n_clusters=3
```

04 的 `inputs.test_dirs` 可包含一个或多个目录，例如 validation 与 test；省略该
字段或设置空列表时只处理训练集。其他数据集沿用输入目录末级名称，`validation`
专用于早停和模型选择。每个目录必须是 03 的产物，且输入患者互斥。
无独立 validation 时，05 要求固定 K，关闭内部划分，并按训练指标早停；其结果
是训练数据上的描述，不是独立验证。更换预处理方案时选择新的 `paths.dir`，
随后将该目录传给 05 的 `split.dir`。

探索入口顺延为 `05_explore.sh` 和 `05_explore_preprocessing.sh`；后者先运行一次
03，再对同一划分运行 36 套预处理和各两个训练 seed，测试预测不进入探索汇总。

### OS 分阶段探索（测试集参与选优）

`05_explore_os.sh` 复用现有 random 划分，通过 04/05/06 的 Hydra 参数探索，
不改默认配置。OS 固定，比较 12/24 个月 landmark、六套面板、log 与缩放方式，
随后比较训练参数、少量结构及组合，最后对四个入围方案补齐三个 seed。
阶段 A/B/C/D/E 最多新增 48/8/26/6/8 次完整训练；已有相同配置复用。
每次训练保持主训练最多 300 epoch、最少 100 epoch、patience=30，验证集用于早停。

```bash
bash scripts/crc_yunnan/05_explore_os.sh outputs/crc_yunnan/<探索名> --dry-run
bash scripts/crc_yunnan/05_explore_os.sh outputs/crc_yunnan/<探索名>
# 中断后使用完全相同的脚本、配置、路径和预算继续：
bash scripts/crc_yunnan/05_explore_os.sh outputs/crc_yunnan/<探索名> --resume
```

默认输入 `data/derived/crc_yunnan/splits/v2/random/seed-20260908`，可用
`--split-dir` 指定另一份既有 random 划分。默认预算 15 小时，可用
`--budget-hours` 缩短；达到重复训练和评价的预留边界后，不再启动新的探索组合，
保留完整单次训练预算。实际时长会受 GPU、面板及早停影响，当前训练不会被强行终止。
`--dry-run` 只验证全部参数模板，不创建结果目录；自适应阶段的入围名单由实际指标决定。

本轮明确以测试集参与选优，结果属于探索证据。预测榜按**术后**3/5年平均 AUC 排序，
并列时比较 C-index、较低 Brier；综合榜在每个 landmark 内将 AUC 与
`−log10(log-rank p)` 百分位等权评分。12个月窗口预测未来24/48个月，24个月窗口
预测未来12/36个月。缺失指标保留原因并排末位；空簇、小簇只标记，不排除预测结果。
同一模型同时评价两个时间点，不因预测时点重复训练。

输出目录包含 `preproc/`、`runs/`、`evaluation/`、`logs/` 和 `summary/`。
`progress.csv`、`current_run.txt` 显示任务、命令、状态和当前日志；`plan.json`
冻结各阶段实际入围名单。失败组合保留 traceback，其他组合继续；续跑复用完成产物，
未完成尝试另存目录。中文 `summary/exploration_tables.xlsx` 汇总全部指标、校准、
簇人数、排行榜、跨 seed ARI、重复结果及运行记录，摘要说明缺失和预算跳过项。
四个最终方案各选择综合表现最好的 seed 运行完整 06，三个 seed 的统计全部保留。
06 原有图的月份仍从 landmark 起算，探索汇总另列对应术后月份；06 中的“独立评价集”
仅描述输入来源，本轮 test 已参与选优。取回仅限聚合表、图、摘要和安全日志，
模型、患者表和预测留在远端。

### OS 三因素并行探索

`05_explore_os_training.sh` 固定 OS、12个月 landmark、D20、auto-log1p/robust，
复用同一份 random 划分，只预处理一次。batch size 固定128，learning rate继承
默认配置（当前0.001），主训练保持100–300轮、验证损失早停、patience=30。
原有 `05_explore_os.sh` 保留。

```bash
bash scripts/crc_yunnan/05_explore_os_training.sh outputs/crc_yunnan/<新探索名> --dry-run
bash scripts/crc_yunnan/05_explore_os_training.sh outputs/crc_yunnan/<新探索名> --workers 3
# 续跑可调整并发数；实验参数、脚本、目录和预算必须保持一致。
bash scripts/crc_yunnan/05_explore_os_training.sh outputs/crc_yunnan/<新探索名> --resume --workers 2
```

默认 `--split-dir` 为 `data/derived/crc_yunnan/splits/v2/random/seed-20260908`，
`--workers` 支持1/2/3，默认3，同一张 `cuda:0` 上运行独立进程。
默认 `--budget-hours 15`，可缩短；按并发下实测耗时预留最终K选择及评价时间。
已开始的任务不因预算缩短训练。并行提速取决于GPU利用率和资源竞争，
内存不足时可保留结果并以较低并发续跑，不会自动改batch size。

探索分为四阶段，每个阶段完成后统一排序并冻结下一阶段，完成顺序不影响选优：

| 阶段 | 探索内容 | 新增训练上限 |
| --- | --- | ---: |
| A | `uncertainty`/`fixed`各比较9个权重比例；重建=1，生存=0.05/0.2/1，聚类=0.02/0.1/0.5 | 18 |
| B | 每种方式选出最佳比例，分别比较warmup=5/10/30、初始化迭代=5/20/50；基准复用 | 8 |
| C | 每种方式组合较优warmup与初始化设置；相同配置复用 | 2 |
| D | 每种方式选一个方案，以K=2–10和三个seed执行现有05选择 | 54 |

正常完成最多82次训练，不含失败或中断后的重新尝试。A–C固定K=3、seed=20260909，
A阶段warmup=10、初始化迭代=5。`uncertainty` 的比例只是初始值，实际权重继续学习；
`gmm_init_iters` 指一次K-means初始化的迭代轮数，不是独立重启次数。
D使用seed 20260909/20260908/20260910，最多同时运行两个K选择进程，各自内部串行，
沿用one-standard-error及三个seed均无空簇、最小簇比例至少5%的门槛。
计算验证集跨seed ARI但不增加ARI门槛。没有合格K时保留科学结论，不放宽门槛。

A–C沿用test参与选优的综合榜：术后3/5年平均AUC和`−log10(log-rank p)`百分位等权，
并列比较AUC、C-index、较低Brier及稳定方案ID，缺失百分位为0。
K本身只用train/validation选择，代表seed沿用05的验证指标排序。
入选K的三个已保存模型用于重复统计，无额外训练；每个成功方案的代表模型运行完整06。
全部结果属于探索证据，不是独立测试评价。

`plan.json`冻结阶段名单并保存续跑状态，`progress.csv`记录每次尝试，
`current_run.txt`以JSON列出当前所有任务及PID、日志。普通任务错误停止新任务派发，
已运行任务收尾，原始traceback保留。续跑复用完整产物，未完成尝试另开目录。
`summary/`提供中文Excel、JSON摘要及PNG/PDF图，包含预测、簇占用、K选择、
跨seed ARI、逐轮原始损失与实际权重，以及相对同加权方式默认K3基准的差异。
单seed筛选与三seed结果分开报告，预算跳过阶段明确标记。核心汇总先于完整06保存，
即使06遇到Cox收敛等错误，已有探索结果仍保留，原始错误继续上抛。
模型、预测及患者表留在执行服务器。
`--dry-run`只解析候选配置，不读取真实数据、创建结果目录或启动训练。

### 评价与可视化

06 用 `CRCEvaluationConfig` 解析 Hydra 配置，只需指定 `inputs.run_dir`。它从 05
运行清单追溯 04 数据、01 特征分组及候选模型，沿用已锁定的 K、seed、结局、
面板和 landmark。默认评价 train、validation、test；单集使用 `'datasets=[train]'`。
默认输出建模目录同级的 `evaluation/`，日志位于 `logs/06_evaluate/`，拒绝覆盖旧结果。

产物包括中文 `evaluation_summary.json`、一个 `evaluation_tables.xlsx`，以及 PNG/PDF
图：簇规模、后验置信度/熵、UMAP、KM及风险人数、校正Cox、全部基线变量、全部面板
特征的轨迹和覆盖率、生存预测/校准、训练过程、K选择和跨seed ARI。单张长绘图封装
于 `evaluation_plots.py`；各数据集复用的统计位于 `evaluation.py`。

- UMAP仅在训练embedding上拟合，其他集调用同一投影器的transform；二维形状不作为
  聚类质量证明。所有图沿用05簇编号，按簇着色时保持一致。
- Cox调整年龄（每10岁）、性别、AJCC分期及原发部位；参考簇由训练平均预测风险
  最低者确定。结果属于调整后的描述性关联，不自动删协变量或增加惩罚。
- 临床变量报告缺失人数、连续变量中位数/IQR和分类人数/比例；总体检验使用
  Kruskal–Wallis、χ²或稀疏表Monte Carlo Fisher，每个数据集内作BH校正。
- 轨迹按04保存的参数逆变换到原始尺度，术后至landmark每90天分箱，末箱包含
  landmark。先取患者箱内中位数再取簇中位数/IQR；覆盖率分母为该集该簇全部患者。
- 生存时间从landmark起算，每月30天。报告05同口径Harrell C-index、60个月截断
  IPCW C-index、1–60个月动态AUC/Brier/IBS，以及12/36/60个月的分位组KM校准。
  删失分布仅来自训练集；不支持的时间点保留并标记原因，不自动缩短积分区间。
- ARI在同一冻结验证集上逐个加载全部候选模型推理，不重训、不重新选择K或seed。
  固定K无候选结果时跳过该部分；ARI结合空簇和最小簇比例解释。

输出仅含聚合结果，不导出患者级embedding坐标。真实数据评价在远端运行；模型、
预测和患者表留在远端，结果取回范围遵守该轮分析的授权。
