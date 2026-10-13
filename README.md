# TRAILS

TRAILS 是 **Trajectory and Risk-informed Analysis of Irregular Longitudinal
Subtypes** 的缩写。项目目标是构建面向非同步多变量医学纵向数据的深度生成式生存轨迹聚类模型，用于在真实世界随访不规则、变量缺失、检查频率不一致的条件下识别具有不同动态轨迹和事件风险的患者亚型。

## 当前范围

当前版本是阶段一基础版：变长访问序列上的簇特异生存变分 TRAILS 模型。

- 模拟器采用 VaDeSC-EHR 风格的数据生成主线：cluster-specific latent profile -> 随机非线性轨迹生成 -> pseudo attention -> 观测序列 -> Weibull 生存结局。
- 输入是非同步采样的多变量医学检查序列，包括血液检查、肝肾功能、炎症指标、肿瘤标志物和肿瘤负荷等连续变量。
- 数据保留 `mask` 和 `delta_time`，用于表达变量级缺失和距离上次观测的时间间隔。
- 编码器支持 GRU-D、原始 aligned mTAN 多时间注意力输入层，并保留旧的 compact
  per-feature `mtan2` 输入层用于对照。
- 解码器支持 GRU/LSTM/Transformer，从患者级 latent representation 重构纵向轨迹。
- 聚类模块使用 VaDE 风格的可学习 Gaussian mixture latent prior。
- 生存模块支持多输出 scale 加共享 shape，或 latent 与簇 one-hot 拼接的双参数 Weibull 头。

本阶段不实现 mixed-type likelihood、competing risks 或 recurrent events。

## 生存损失与评价（TorchSurv）

`TrailsModel` 的生成分布为
\[
p(x,T,z,c)=p(c)p(z\mid c)p(x\mid z)p(T\mid z,c).
\]
默认训练时三个分支共用一次重参数采样的 \(z\sim q_\phi(z\mid x)\)；验证和预测使用
\(z=m_\phi(x)\)。`model.survival_head_architecture=multi_output`（默认）使用共享隐藏层加 K 维输出：
\(\log\lambda_k(z)=[f_\theta(z)]_k\)，输出分量对应簇身份，实现
\(p(T\mid z,c)\)。隐藏层宽度保持 latent_dim，激活保持 ReLU；CRC 默认一个隐藏层。
`survival_head_hidden_layers=0` 表示各簇独立的 z 线性效应和截距。
此模式的标量 `log(shape)` 全局共享，初值为 0。

设置 `model.survival_head_architecture=one_hot` 时，对每个患者枚举 K 个簇，
将同一 latent 与各簇 one-hot 拼接后输入共享网络：
\[
(\log\lambda_{ik},\log\rho_{ik})=f_\theta([z_i,\operatorname{onehot}(k)]).
\]
输入宽度为 `latent_dim + K`，输出宽度为 2；scale 和 shape 均可随患者和簇变化，
不再另设共享 shape。隐藏层仍由 `survival_head_hidden_layers` 控制，宽度为 latent_dim，
激活为 ReLU；零隐藏层是 latent 效应与簇截距相加，非线性交互需至少一个隐藏层。
两种头均直接输出 log 参数，不使用 softplus，指数变换得到正值 scale/shape。
两种头均计算全部 K 个分量，并用 `p(c|z)` 混合生存曲线，不先选定一个硬簇。

`model.survival_latent_source` 可设为 `sample`（默认）或 `mean`：前者复用同一次
采样，后者仅将生存头输入改为可反向传播的 latent mean，重建和聚类仍使用采样 z。
两种设置在验证和预测时均使用 latent mean。MIMIC、CRC 及通用训练命令均可追加
`model.survival_latent_source=mean`；Python API 使用
`ModelConfig(survival_latent_source="mean")`。该参数随检查点保存，旧检查点缺省为
`sample`。它与 `survival_head_architecture` 独立组合；`mean` 仅改变生存输入，
保留所选头的结构和结局条件责任度；下述生存似然中的 z 相应替换为均值。例如：

```yaml
model:
  survival_head_architecture: one_hot
  survival_latent_source: mean
  survival_head_hidden_layers: 1
```

Python API 对应 `ModelConfig(survival_head_architecture="one_hot", survival_latent_source="mean")`。
通用和 CRC 默认均为
`model.survival_loss=weibull`。`cox` 枚举保留，但构造模型立即抛出
`NotImplementedError`（簇特异 Cox 尚未实现），不再估计 TRAILS Cox 基线；独立基线模型不受影响。

令 \(a_{ik}=\delta_i\log h_k(t_i\mid z_i)-H_k(t_i\mid z_i)\)。适配模块使用
TorchSurv 的 `log_hazard` 和 `_cumulative_hazard` 向量化得到逐患者、逐簇似然，
时间下限为 `1e-4`；保留库的极值截断，全删失批次也可计算。
变分分布取 \(q_\phi(z\mid x)q(c\mid z,t,\delta)\)，其中
\[
\log\gamma_{ik}=\operatorname{logsoftmax}_k[\log\pi_k+\log p(z_i\mid k)+a_{ik}].
\]
生存损失和联合 KL 使用同一责任度，梯度贯穿后验：
\[
\mathcal S=-\frac1B\sum_{i,k}\gamma_{ik}a_{ik},\qquad
\mathcal K=\frac1B\sum_i\left[\log q_\phi(z_i\mid x_i)
+\sum_k\gamma_{ik}(\log\gamma_{ik}-\log\pi_k-\log p(z_i\mid k))\right].
\]
原 masked 重建损失及 `fixed/uncertainty` 组合方式不变，因此总目标是**加权变分目标**。
责任度始终使用未经任务权重缩放的似然；不声称任意权重下都是标准 ELBO。
单次采样 KL 的数值可为负，不额外截断。Warmup 训练重建和生存、关闭联合 KL，
随后用训练潜均值执行原 K-means 初始化。生存权重为 0 时，责任度退回 \(p(c\mid z)\)，
不通过 KL 间接读取结局。训练轮数、早停和 K 选择门槛/排序规则保持原协议。

常规预测始终使用不依赖结局的 \(r_{ik}=p(c=k\mid z_i)\)，返回混合曲线
\(S_i(t)=\sum_k r_{ik}S_k(t\mid z_i)\)，不平均 Weibull 参数。
`cluster_probabilities`、`predict()`、`predict_proba()`、占簇及 ARI 均使用 r。
验证损失使用结局后验，但预测指标只使用 r 和混合曲线。

```python
prediction = estimator.predict(data)
risk = prediction.risk_score(48 * 30)  # landmark 后48个月的混合事件概率
curves = prediction.survival([24 * 30, 48 * 30])  # (N, 时间点数)
retrospective = estimator.predict(data, include_survival_posterior=True)
gamma = retrospective.survival_conditioned_cluster_probabilities  # (N,K)，仅供回顾分析
```

显式请求回顾后验不会改变普通簇标签、混合权重或生存曲线；关闭生存任务时请求它会报错。
`survival_log_params` 为 `(N,K,2)`，末维为 log(scale)/log(shape)，派生 scale/shape 均为
`(N,K)`。`risk_score(t)` 返回 `1-S(t)`；`median_survival_time()` 在分量中位数构成的
log 时间区间内向量化二分，`risk_score(method="median_survival")` 保留负 log 中位时间排序。

`estimator.fit()` 完成内部划分后，在实际训练/验证集入口检查至少一个事件；`test()`
也保留入口检查。模型和损失不重复判断事件是否存在，普通加载器允许全删失批次。
纯 `predict()` 不要求事件；生存任务关闭也保留既有数据集入口约定。

评价的 IPCW 始终由训练集估计并显式传给 TorchSurv；AUC 使用 cumulative/dynamic
定义，Brier 输入生存概率。无病例/对照、删失权重不受支持等评价点保留缺失和原因，IBS
不缩短预设积分区间。CRC 术后3/5年 AUC 均值仍是两点算术平均；综合动态 AUC 则使用
库的 `Auc.integral()`。C-index 在整轮汇总后计算，不平均批次 C-index。

生存指标入口接收同一设备上的 Tensor，并返回该设备上的 Tensor；不在计算前搬到
CPU 或统一转成 float64。训练和 K 选择保留模型设备，仅在写入最终标量结果时转换。
CRC/MIMIC 的 pandas 报表在调用端转换 NumPy/Tensor。`Cindex` 只承担跨批次累积；
`concordance_index` 保留无有效比较对时的既有返回约定；`SurvivalMetrics` 保存训练集
删失参考，负责传入 IPCW 和检查支持范围，所有指标公式仍由 TorchSurv 计算。
TorchSurv 0.2 的 AUC 部分临时张量没有指定设备，调用时使用局部设备上下文；
多时间点 `Auc.integral()` 内部 KM 默认使用 CPU，GPU 路径显式创建
同设备的库 KM 估计器，再调用库的 `_integrate_cumulative`，不复制积分公式。

**格式变更：** checkpoint 和预测文件使用版本 **3**，保存生存类型、簇分量 log 参数，
预测文件可附带回顾性后验。模型 checkpoint 另保存结构标识
`survival_head_architecture`（`multi_output` 或 `one_hot`），重载时检查它与配置一致。
原始版本 3 多输出头未保存该标识和配置字段，也可直接加载；
显式标识为 `conditioned_one_hot` 的 checkpoint 拒绝读取，需使用该结构的原代码。
预测参数语义和形状不变，已有版本 3
预测文件仍可读取。版本 1/2 一律拒绝读取。探索脚本中的版本 3 标记本身不代表模型
结构相同，历史运行应使用原代码续跑。TorchSurv 由 `uv.lock`
锁定；私有累积风险接口只在生存适配模块使用，升级依赖时需复核。

历史 OS24/E60 的 `conditioned_one_hot` 头与本次双参数 `one_hot` 头不同；该次对照
未显示占簇改善，不代表本次新头已经经过真实队列验证。默认仍为多输出头。
初始化、warmup、早停和 K 选择协议保持现有设置。

并列风险容差显式保持 `1e-8`，但 TorchSurv 0.2 的 C-index 内部使用 float32 判断
风险并列。例如 float64 的 `1` 与 `1+2e-8` 会被视为并列，与旧 scikit-survival 的
双精度比较有差异。常规合成数据的指标对照通过；近浮点精度边界以所锁定库的实现为准，
不在项目中复制一套指标算法纠正库的舍入行为。

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

MIMIC 流程为 `01–05 → 06_split → 07_run / 08_baselines → 09_figure / 10_supplementary`。
队列、23项特征、0–48小时观察窗、48小时landmark后28天结局及冻结划分规则保持不变。
06以Pydantic解析配置，复用已转换的矩阵直接调用`from_dataframes()`；预处理仍仅在train
拟合1%/99%截尾与标准化，不通过临时CSV重建张量。已有seed目录拒绝覆盖。

07默认逐个训练K=2–5、seed=20260517，保存全部候选；TRAILS不再自动选K、排名或因占簇
门槛失败而终止。整数K和列表K均可，seeds必须明确给出。模型与训练默认参数保持不变。
train和独立validation用于训练/早停，所有模型保存后才加载test。
08的全部基线、消融、各自选K规则和预算不变，与07没有运行依赖。

下面是后续真实重跑的命令示例，本次代码更新没有执行这些命令：

```bash
uv run python -m scripts.mimic.06_split
uv run python -m scripts.mimic.07_run 'n_clusters=[2,3,4,5]' 'seeds=[20260517]' paths.dir=outputs/mimic_case/new-candidates
uv run python -m scripts.mimic.08_baselines split_dir=data/real/mimic-iv-3.1/derived/trails_splits/seed-20260517 paths.dir=outputs/mimic_case/new-baselines
uv run python -m scripts.mimic.09_figure inputs.run_dir=outputs/mimic_case/new-candidates n_clusters=3 model_seed=20260517 selection_basis=validation
uv run python -m scripts.mimic.10_supplementary inputs.run_dir=outputs/mimic_case/new-candidates n_clusters=3 model_seed=20260517 selection_basis=validation
```

09/10独立读取冻结结果。`inputs.run_dir`、`n_clusters`、`model_seed`、`selection_basis`
均须明确指定；后者为`prespecified`、`validation`或`test_exploratory`。
如果人工选择参考了test指标，必须使用`test_exploratory`，图和摘要会标注探索分析。
可用`comparison_runs=[{run_dir: ..., n_clusters: 4, model_seed: 20260517, label: alternative}]`
加入额外TRAILS候选，用`baseline_dirs=[...]`加入已完成的08目录。09还须给每个基线名称
设置展示seed，例如`+baseline_seeds.summary_kmeans=20260517`；10保留所请求K的全部
TRAILS seed及08全部重复。顺序为主TRAILS、配置中的额外模型、08清单顺序，绝不按test性能排序。
所有07/08输入必须引用同一06冻结划分。旧MIMIC结果保留，不能自动转换或混作新模型结果。

09输出240×330 mm英文A–J综合图，包含方法性能、校准、占簇、ARI标签一致性、23项阶段
画像、六项代表轨迹、validation/test KM及风险人数、test调整Cox和临床/七项治疗暴露。
默认比较test，画像和轨迹使用train；四阶段为[0,6)、[6,12)、[12,24)、[24,48]小时。
轨迹按4小时先取每位患者中位数，再汇总簇中位数/IQR，末箱包含48小时。逆标准化值标为
“临床单位下的截尾后数值”。默认少于10名患者的画像单元显示灰色、轨迹点标记×。
输出PDF、SVG、600 dpi PNG及聚合`SourceData.xlsx`、配置、摘要。

10输出各集完整评价、23项轨迹、临床特征、治疗连续指标、校准/Brier曲线、全部候选历史
和K曲线、独立validation上的跨seed ARI、方法间ARI/NMI，以及仅在train拟合、其余集
transform的UMAP。SourceData仅含聚合数据；原始数据、模型和患者级预测留在数据执行环境。
R方法仍需要原有`lcmm`、`JMbayes2`等依赖。

C-index使用模型记录的风险定义，支持`median_survival`；7/14/21天动态AUC分别使用该日
`1-S(t)`。IBS使用完整1–27天网格，删失分布只由train估计；不外推冻结曲线、不缩短积分
区间，不支持的指标保留缺失和原因。仅聚类的方法不产生患者级生存预测指标。Cox仍调整
年龄、性别、分组种族和SOFA，并采用train中28天KM死亡风险最低的簇作为参考。

公共候选接口也供CRC 05使用；CRC模型路径、既有指标及06/07后处理兼容性不变：

```python
from trails import ClusterNumberEvaluator

evaluator = ClusterNumberEvaluator([2, 3, 4, 5], seeds=[20260517], estimator_config=config)
result = evaluator.fit(train, validation_data=validation,
                       evaluation_data={"test": "data/frozen/test/dataset.pt"},
                       result_dir="outputs/candidates")
# 无validation时训练完整train；从不隐式拆分，所有候选valid_size=0。
evaluator = ClusterNumberEvaluator.load("outputs/candidates")
evaluator.config, evaluator.candidates
evaluator.run_metrics, evaluator.k_summary, evaluator.stability_pairs
figure = evaluator.plot_metrics(dataset="validation")
history = evaluator.load_history(seed=20260517, n_clusters=3)
model = evaluator.load_estimator(seed=20260517, n_clusters=3, device="cpu")
prediction = evaluator.load_prediction(seed=20260517, n_clusters=3, dataset="test")
```

`ClusterNumberEvaluatorConfig`和`ClusterNumberEvaluationResult`替代原选择器类型。
`evaluator.json`格式版本1只记录配置、候选及相对路径；`run_metrics.csv`、`k_summary.csv`、
`stability_pairs.csv`保存聚合表。目录整体迁移可直接重载；只下载这四个文件也可浏览和绘图，
缺少模型/预测/历史时仅对应的按需加载方法报错。旧选K产物不自动迁移，历史CRC由原有
后处理兼容分支读取。`selection_score`列名为历史兼容保留，含义仅为描述性综合分：
`sqrt(Cindex² + (1 − BIC_normalized)²)`，在每个dataset/seed内独立归一化，不生成排名。
BIC仍用潜在高斯混合似然和`K*(2*latent_dim)+K−1`参数数目；占簇门槛仅作诊断。
单seed或无独立validation时，跨seed稳定性为空并记录原因。

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

其他真实数据可使用公开的`ClinicalTimeSeriesDataset.from_dataframes()`或`load_from_csv()`
准备数据，再通过`TrailsEstimator.fit()`或上述公共候选接口建模。旧`case.py`已在忽略的
`deprecated`归档中，依赖旧选择器的归档脚本需使用原代码版本，不属于当前工作流或静态检查范围。

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


## CRC 云南建模与绘图流程

新产物使用 `v2`，旧目录不覆盖。观测表统一为中格式：
`patient_id,time_days,特征1,特征2,...`；缺失观测保留 NaN。

```bash
uv run python -m scripts.crc_yunnan.01_cohort
uv run python -m scripts.crc_yunnan.02_eda
uv run python -m scripts.crc_yunnan.03_split
uv run python -m scripts.crc_yunnan.04_preproc
uv run python -m scripts.crc_yunnan.05_run
uv run python -m scripts.crc_yunnan.06_figure inputs.run_dir=outputs/crc_yunnan/<运行名>/modeling
uv run python -m scripts.crc_yunnan.07_supplementary inputs.run_dir=outputs/crc_yunnan/<运行名>/modeling
```

01 完成基础清洗与纳排，另存特征分组和中文排除汇总。02 输出中文
`eda_summary.json`、`eda_tables.xlsx` 和七类图。03 只保存患者划分与原始尺度
CSV；04 选择结局、窗口及固定面板，仅在训练观测上拟合变换，再保存中格式和
建模张量。05 读取 04 的 `preproc_manifest.json`，只对存在的数据集输出结果。
02 的 `outcome` 控制 landmark 人群、覆盖率分母和时间切点比较，默认与本轮
建模一致使用 OS；完整队列生存曲线仍同时展示 DFS 和 OS。
各命令的 YAML 位于 `configs/crc_yunnan/`，旧长格式和旧编号入口不再使用。

当前普通YAML统一采用G14/N2-K6；完整配置说明见 `configs/crc_yunnan/README.md`。
默认03仅生成random并复用原v2主划分，04/05使用清洗后的G14输入。
以下非默认示例需要先显式生成相应划分。
预处理时间划分时，同时设置输入和独立输出目录：

```bash
uv run python -m scripts.crc_yunnan.04_preproc \
  inputs.split_dir=data/derived/crc_yunnan/splits/v2-g14-clean-20261008/temporal-2017/seed-20260908 \
  paths.dir=data/derived/crc_yunnan/preproc/v2-g14-clean-20261008/os/landmark-24m/G14/auto-log1p-robust/temporal-2017/seed-20260908
uv run python -m scripts.crc_yunnan.05_run split.strategy=temporal-2017
```

### 全样本固定 K 训练

以下非默认示例先仅保存完整基础队列，再按默认 OS、24 个月和 G14 规则形成全体合格
患者。若同时需要普通划分，去掉第一条命令中的 `split.strategies=[]`。

```bash
uv run python -m scripts.crc_yunnan.03_split \
  'split.strategies=[]' split.save_full_dataset=true
uv run python -m scripts.crc_yunnan.04_preproc \
  inputs.train_dir=data/derived/crc_yunnan/splits/v2-g14-clean-20261008/full \
  'inputs.test_dirs=[]' paths.dir=data/derived/crc_yunnan/preproc/v2-g14-clean-20261008/full-g14
uv run python -m scripts.crc_yunnan.05_run \
  split.dir=data/derived/crc_yunnan/preproc/v2-g14-clean-20261008/full-g14 n_clusters=3
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

### 候选指标、正文图与补充图

05 使用唯一 `run.yaml`，默认训练G14/N2-K6、seed20260908。
已完成的K2–10诊断复用既有产物，不自动重复训练。
`n_clusters=5` 或 `'n_clusters=[2,5]'` 控制候选K，`'seeds=[20260908]'` 控制模型seed。
所有模型训练完成后才读取test，报告三集 C-index、潜空间混合 BIC/NLL、簇占用、
人数和事件数。无自动入选K、最佳seed或综合排名。空簇和最小占比5%只作为诊断，
不阻止明确指定的候选评价。test不参与训练/早停；人工参考test指标选择K属于探索。

模型与历史位于 `models/seed-<seed>/k<K>/`，各数据集预测也在该候选目录下。
根目录 `run_metrics.csv` 保存逐seed/K/数据集指标，`k_summary.csv` 保存跨seed均值、
标准差及占簇诊断，`run_manifest.json`（schema 3）引用全部候选。

06与07独立读取05，默认K6、model_seed20260908，必须显式提供输入目录。
07的跨K组图默认读取215上的九个冻结N2聚合目录；若05输入自身包含所需网格，
显式设 `k_scan.run_dirs=[]`。默认各自输出到同级 `figure/`、
`supplementary/` 下的 `k<K>/`，日志分别在 `logs/06_figure/`、
`logs/07_supplementary/`。其他模型seed须覆盖 `model_seed`；显式置为null时只在
仅一个可用seed的情况下推断。图件/统计随机seed由 `seed` 和 `umap.seed` 分别控制。
历史schema-2模型及候选也可以读取；历史未保存预测的候选在冻结数据上推理，不训练。

```bash
uv run python -m scripts.crc_yunnan.05_run \
  split.dir=data/derived/crc_yunnan/<04产物目录> \
  'n_clusters=[2,5]' 'seeds=[20260908]' \
  paths.dir=outputs/crc_yunnan/<运行名>/modeling

uv run python -m scripts.crc_yunnan.06_figure \
  inputs.run_dir=outputs/crc_yunnan/<运行名>/modeling \
  'n_clusters=[2,5]' model_seed=20260908

uv run python -m scripts.crc_yunnan.07_supplementary \
  inputs.run_dir=outputs/crc_yunnan/<运行名>/modeling \
  'n_clusters=[2,5]' model_seed=20260908 'k_scan.run_dirs=[]'
```

以上是未执行的示例；实际运行前将确切命令追加至 `history.sh`。

06 的 `figure.yaml` 默认复现认可的六面板结构：稀疏临床设置、阶段画像、
代表轨迹、三集UMAP、三集KM，以及测试集调整Cox。
每个K输出 `Figure_K<K>.pdf/svg/png`、`SourceData.xlsx`、
`figure_summary.json` 和 `resolved_config.yaml`；PNG默认600 dpi，文字可编辑。
当前为190×255 mm，图内全部为英文。
`figure.atlas` / `figure.trajectories` 明确配置 feature、英文label、原始unit和
线性/log轴；当前为G14的14项画像及训练集选定的6项轨迹。更换面板时须修改这些列表，
不推测单位。`figure.phase_days` 默认 [0,90,360,720]，必须覆盖模型窗口；
各面板数据集由 profile_dataset、display_datasets、cox_dataset 明确控制。

07 的 `supplementary.yaml` 保留完整临床检验、全指标轨迹/覆盖、占簇/置信度、
UMAP、KM/Cox、预测/校准、训练历史、候选指标和跨seed验证集ARI。
输出中文 `evaluation_tables.xlsx` / `evaluation_summary.json`、英文PNG/PDF/SVG，
以及根目录 `k_comparison.csv`。单seed不计算稳定性，摘要明确说明原因。
候选指标图分别展示各数据集，不标记自动入选K。

- UMAP仅拟合训练latent mean，再投影其他集；同一模型两入口使用同样参数、颜色及簇编号，
  不导出坐标。二维分离不证明聚类有效性或稳定性。
- 阶段画像及时间箱轨迹均先计算患者中位数，再计算簇中位数/IQR；不插值，
  不平均已有箱中位数。画像沿用04训练尺度，轨迹按既有参数逆变换。
  正文 n<10 灰显并断线，完整统计与覆盖率保留在Source Data。
- KM及95%CI来自完整随访；显示0–60个月不改变log-rank或Cox。风险人数按
  survival_time >= month*30计算。Cox沿用年龄、性别、分期、原发部位及完整病例，
  参考簇由训练平均预测风险确定，不能固定为历史C3，也不自动加惩罚。
- 临床总体检验及BH校正、生存预测时间点和train-only IPCW沿用原评价口径。
  不支持的指标保留null和原因；真正的计算错误保留原始traceback。
- 明确指定的低占簇K可以评价，但摘要保留诊断；已查看的test及test-informed人工K选择
  不构成新的独立验证。患者数据、模型和预测留远端，仅取回授权聚合产物。
- 共用计算位于 `evaluation.py`，单图位于 `evaluation_plots.py`，
  正文组图位于 `figure_plots.py`；旧06_evaluate入口及evaluate.yaml已移除。
