# CRC r089 复现配置

`scripts/crc_yunnan/01_cohort.py` 至 `07_supplementary.py` 使用本目录的 Hydra 配置。
01、03、04 的科学参数已与原实验保存的配置核对；02 仅做描述性分析，不改变建模输入。
`run.yaml` 的模型和训练参数与 r089/seed20260908 的 K2、K5 原模型配置一致。

| 阶段 | 配置 | 固定设置 |
| --- | --- | --- |
| 01 队列 | `cohort.yaml` | v2 原始数据、研究ID seed20260908、原日期修正和排除规则、同日中位数 |
| 02 EDA | `eda.yaml` | OS 描述性分析 |
| 03 划分 | `split.yaml` | seed20260908，DFS/OS 联合分层，random 64%/16%/20% |
| 04 预处理 | `preproc.yaml` | OS、720天 landmark、E60 固定顺序、至少2个观测日期、train 拟合 auto-log1p/robust |
| 05 候选训练 | `run.yaml` | r089，seed20260908，K2–10；三集指标汇总，不自动选择 |
| 06 正文图 | `figure.yaml` | 显式指定K；六面板、Source Data、可编辑PDF/SVG及600 dpi PNG |
| 07 补充图 | `supplementary.yaml` | 显式指定K；完整评价、补充图及跨K比较 |

r089 使用 GRU-D + 单层 LSTM 编码器/解码器，宽度64、latent32、dropout0、生存头1层。
先做30轮仅重建预训练，再执行多启动 K-means++（10次、最多200轮），
各分量共享全局对角方差，初始化方差下界1e-4；经验混合比例继续训练。
预训练后重置生存头、保留 Adam；固定重建/生存/KL 系数为1/0.05/0.1。
batch64、学习率0.001且恒定；正式训练100–300轮，验证 loss 早停，patience30。
占簇使用 p(c|z)，C-index 使用负log混合中位生存时间。

`parallel.torch_threads=4` 同时控制 Torch intra-op 和 BLAS/OpenMP；
`runtime.torch_interop_threads=4` 控制 Torch inter-op。
`runtime` 显式保持原实验的 cuDNN/TF32 设置，确定性算法关闭。
因此相同参数是复现的必要条件，不能保证不同硬件/依赖环境下逐位相同。

## 运行方式

01 支持通过 `processing.observation_value_bounds` 显式配置检测值的 `min/max`；
界限可单边指定，端点保留，默认空字典不筛选检测值。有限数值中的超界原始记录
在同日聚合前剔除，不换算单位、不改小数点，也不直接删除患者。队列摘要分别报告
原始记录和基础队列内有效日期观测的剔除数；原有无效数值/日期规则仍独立执行。
探索规则应随实际命令记录，不能将正常参考区间直接作为删除范围。
中文字典键可通过 OmegaConf 内置 `oc.create` 从引号保护的 JSON 插值加载；
G14 的完整可复现命令保存在根目录 `history.sh`。

04 增加可选 `features.panel=G14`，依次使用 CEA、CA242、CA125、血清白蛋白、
总蛋白、白球比、前白蛋白、胆碱酯酶、红细胞、红细胞平均体积、平均血小板体积、
中性粒细胞绝对值、乳酸脱氢酶、尿素。默认仍为 E60；面板与清洗同时改变时，
与历史结果的差异只能作联合改变后的探索解释。

原实验在99服务器 `/home/rongzw/projects/TRAILS` 运行：Python3.12.12、
PyTorch2.12.0+cu130、CUDA13.0、NVIDIA L20。依赖使用 `uv run --locked`。
源码中的数据处理、损失和训练逻辑也须保持对应版本；更改源码后应重新验证。

在尚未生成 v2 派生产物的目录中，执行：

```bash
uv run --locked --python 3.12.12 python -m scripts.crc_yunnan.01_cohort
uv run --locked --python 3.12.12 python -m scripts.crc_yunnan.02_eda
uv run --locked --python 3.12.12 python -m scripts.crc_yunnan.03_split
uv run --locked --python 3.12.12 python -m scripts.crc_yunnan.04_preproc
uv run --locked --python 3.12.12 python -m scripts.crc_yunnan.05_run
```

05默认 `n_clusters=[2,3,4,5,6,7,8,9,10]`、`seeds=[20260908]`。
整数K也支持；无validation时沿用训练监控，不内部划分。每个seed/K保存于
`models/seed-<seed>/k<K>/`。全部训练结束后生成各集预测、`run_metrics.csv`、
`k_summary.csv` 和引用全部模型的运行清单，取消自动K/seed选择。
无空簇及最小占比5%保留为诊断标记，不阻止明确请求的评价。
05也报告test，人工参考这些指标选择K属于探索评价。

06/07必须提供 `n_clusters`；多seed时还须提供 `model_seed`。
默认输出同级figure/supplementary下的k<K>目录，两者独立运行。
历史05清单也支持，不覆盖旧结果。正文默认E60指标；换面板或landmark时须明确调整
figure.yaml的指标/单位/阶段配置，不自动删除指标。

```bash
uv run python -m scripts.crc_yunnan.06_figure \
  inputs.run_dir=outputs/crc_yunnan/<运行名>/modeling 'n_clusters=[2,5]' model_seed=20260908
uv run python -m scripts.crc_yunnan.07_supplementary \
  inputs.run_dir=outputs/crc_yunnan/<运行名>/modeling 'n_clusters=[2,5]' model_seed=20260908
```

以上为未执行示例。06输出可编辑正文图及Source Data；07输出完整统计工作簿、
英文补充图及比较表。统计解释和可配置字段见根目录README的“候选指标、正文图与补充图”。
旧06_evaluate及evaluate.yaml已移除，历史命令不改写。

01–04 默认拒绝覆盖已有产物。在当前服务器可直接复用已核验的 v2 冻结输入，
从05开始；如需从原始数据重新生成，使用新的输出版本，例如：

```bash
uv run --locked --python 3.12.12 python -m scripts.crc_yunnan.01_cohort version=v2-r089-repro
uv run --locked --python 3.12.12 python -m scripts.crc_yunnan.02_eda version=v2-r089-repro
uv run --locked --python 3.12.12 python -m scripts.crc_yunnan.03_split version=v2-r089-repro split_version=v2-r089-repro
uv run --locked --python 3.12.12 python -m scripts.crc_yunnan.04_preproc version=v2-r089-repro
uv run --locked --python 3.12.12 python -m scripts.crc_yunnan.05_run split.dir=data/derived/crc_yunnan/preproc/v2-r089-repro/os/landmark-24m/E60/auto-log1p-robust/random/seed-20260908
```

执行前将所选的确切命令追加至根目录 `history.sh`；以上示例尚未执行。
患者数据、模型和预测留在远端，仅取回聚合结果。

## 数据与结果参照

原始数据入口 `data/real/crc_yunnan` 在99指向 `/mnt/data/CRC_Yunnan_Data/v2/no_name`。
截至2026-10-07，已只读核对原始文件与现有冻结输入：

| 文件 | SHA256 |
| --- | --- |
| `surv_all_no_name.csv` | `269ace392a5a678437715337c6044722babd796f6ec80ae142129d3f792c72b8` |
| `long_combine_no_name.csv` | `d3c692b1f1859b2431181110193940a7d5f1b379017cab022cf4aa4aab292c55` |
| `uv.lock` | `678d0c5f9979db8778cbb9437ec712d4813c4ab870d58bab9057eaae971e3e0f` |
| 原 v2 `train/dataset.pt` | `122e3f6e77141a514ca3aee54815978359fa460f27db0ddb53afe811a2daf405` |
| 原 v2 `validation/dataset.pt` | `60690dea8508b345cc4c843eb240f424e433061b0fabcc7f74fe49035cfa668f` |
| 原 v2 `test/dataset.pt` | `d855c29c50c3be0d03497581089ad225f22c6bb39e3a7aede293a58c8c4466b9` |

重新生成时应保留原始文件内容及行顺序、患者划分和 E60 特征顺序。
预处理后应有 train1854/328事件、validation444/75事件、test586/94事件。
新目录会改变张量文件内的路径元信息，重新序列化的文件 SHA256 可能不同，
应比较患者顺序、张量及预处理参数本身，不能仅凭文件哈希差异判定数据不一致。

| K | 验证 C-index | 测试 C-index |
| --- | --- | --- |
| 2 | 0.625197 | 0.652538 |
| 5 | 0.682320 | 0.618713 |

这些是现有原模型的参照值，本轮仅核对配置和流程，尚未用新入口重新训练真实数据。
K5 是补充探索；K 与参数的选择应继续依据训练/验证，测试表现只作报告。
