# CRC 云南：当前 G14 / N2-K6 配置

本目录只保留一组正式 YAML，每个命令对应一个文件，不再提供 `_g14_n2` 覆盖文件。
当前默认即已选定的 G14 / N2-K6。E60/r089 的历史配置和结果保存在原结果快照及
`history.sh` 中，不是当前默认方案；历史命令不改写。

| 阶段 | 配置 | 当前设置 |
| --- | --- | --- |
| 01 队列 | `cohort.yaml` | G14探索性界限、原日期修正和排除规则、同日中位数 |
| 02 EDA | `eda.yaml` | 清洗后队列的OS描述性分析 |
| 03 划分 | `split.yaml` | 复用原v2主划分，仅random、seed20260908、64%/16%/20% |
| 04 预处理 | `preproc.yaml` | G14、OS、720天landmark、至少2个日期、train拟合auto-log1p/robust |
| 05 训练 | `run.yaml` | N2、K6、seed20260908、单worker、cuda:0、4线程 |
| 06 正文图 | `figure.yaml` | K6、seed20260908、CPU、14项画像及6项选定轨迹、三集UMAP/KM |
| 07 补充图 | `supplementary.yaml` | K6、seed20260908、CPU、完整评价及冻结K2–K10指标组图 |

## 哪个配置对应当前结果

`run.yaml` 保留r089网络与训练设置，只采用已确认的N2变更：G14冻结输入、
学习率5e-4、uniform混合先验、固定混合权重。默认只训练K6。
模型是GRU-D＋单层LSTM编码器/解码器，宽度64、latent32、dropout0、生存头1层；
重建/生存/聚类权重为1/0.05/0.1，batch64，30轮仅重建预热，正式训练100–300轮，
验证loss早停、patience30，C-index按负log混合中位生存时间排序。

当前主图对应215上的 `outputs/crc_yunnan/g14-hp-20261009/trials/N2-k6`。
`figure.yaml` 的科学和绘图参数与已交付主图一致；输入及新输出目录在命令行明确指定：

```bash
# 215 /home/rongzw/projects/TRAILS；示例尚未执行，复用模型，不训练。
uv run --locked python -m scripts.crc_yunnan.06_figure \
  inputs.run_dir=outputs/crc_yunnan/g14-hp-20261009/trials/N2-k6 \
  paths.dir=outputs/crc_yunnan/n2-replot/figure \
  paths.run_dir=outputs/crc_yunnan/n2-replot/logs/figure
uv run --locked python -m scripts.crc_yunnan.07_supplementary \
  inputs.run_dir=outputs/crc_yunnan/g14-hp-20261009/trials/N2-k6 \
  paths.dir=outputs/crc_yunnan/n2-replot/supplementary \
  paths.run_dir=outputs/crc_yunnan/n2-replot/logs/supplementary
```

两条命令直接使用普通配置名，无须 `--config-name`。06/07仍要求提供
`inputs.run_dir`，避免误用模型。07默认使用215上已核验的九个聚合目录
`outputs/crc_yunnan/g14-n2-complete-20261011/scan_inputs/N2-k2` 至 `N2-k10`。
它们只含同参数05的manifest、配置和聚合指标，不包含模型或患者文件。
若将来明确使用一个已经包含完整K网格的新05目录，可覆盖 `k_scan.run_dirs=[]`，
使07从该输入目录读取K指标；缺失K保留断线，不自动训练。

当前配置表示复现实验所需的设置，不保证不同硬件或依赖版本下训练逐位相同。
已交付结果以各输出目录的 `resolved_config.yaml`、训练历史和SHA核验记录为准。
原环境为Python3.12.12、PyTorch2.12.0+cu130、NVIDIA L20；使用uv锁定依赖。

## 清洗与冻结输入

01/02/04版本和03输出版本统一为 `v2-g14-clean-20261008`。
03的 `split_version=v2` 专用于复用原主划分，不能随输出版本一起更改。
01的14项界限与本轮已批准规则一致，区间端点保留，在同日聚合前删除单条超界观测，
不修正小数点/单位，也不直接删除患者。这些界限不是公认生理极限。
实际剔除107条；基础4100人不变。04得到train1853/327事件、validation444/75、test586/94。

G14顺序固定为CEA、CA242、CA125、白蛋白、总蛋白、白球比、前白蛋白、胆碱酯酶、
红细胞、MCV、MPV、中性粒细胞、LDH和尿素。配置保留历史面板字典作为可选定义，
当前仅选择G14，不会自动训练其他面板。

01–04拒绝覆盖已有产物；当前应直接复用已经核验的冻结输入。
如需重新生成，必须另选输出版本并保留原v2主划分；如需再次训练，须另行明确授权。
此前60次搜索加4次补训已完成，本次配置合并不启动任何计算。
实际执行分析或训练前，将完整命令追加到根目录 `history.sh`；患者数据、模型、预测
及嵌入坐标留在远端，仅取回聚合图表。

## 图形与指标解释

图c的六项为CEA、CA242、白蛋白、白球比、胆碱酯酶、MCV。
选择使用训练集90天箱中位数，经原train拟合参数变换；每簇每箱n≥10且每簇对至少
4个共同箱，覆盖率几何均值作权重。水平距离和扣除首箱后的变化距离经d/(1+d)限幅，
分别除以固定箱数，两部分各占50%。枚举六项组合，对每簇对分别取组合内最大的水平/
趋势分，再对簇对等权平均；并列时按单项平均分与原指标顺序。
06复算并核验固定六项，导出 `Trajectory_selection`、`Trajectory_pairs`。
此分数不是模型特征重要性；各箱患者可能不同，不能解释为个体变化或机制证明。

主图d/e含三集；KM短竖线表示真实删失时点，阴影为95%CI；f为HR实心点和95%CI。
07组图包含综合分 `sqrt(C-index² + (1 - minmax(BIC))²)`、C-index、潜空间混合BIC、
最小簇占比、空簇数、归一化占用熵。BIC归一化在各集合/seed内按完整可用K范围计算；
历史BIC仍含固定混合权重的K−1项，不是完整联合模型BIC。竖线标记手动固定K6，
不依据本组图自动改选K。

三集已参与探索；单seed不支持稳定性结论，N2-K6仍未通过此前全部生存分离门槛。
本轮完整结果说明见 `remote-results/crc-g14-n2-complete-20261011/RESULTS.md`。
