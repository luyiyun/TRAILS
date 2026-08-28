# 结对编程记录

## 当前约定

- 角色：Codex 是 driver，用户是 navigator。
- 批次：每批不超过 200 行手写代码、测试和配置的新增与修改；纯删除不限行数。
- 交接：每批完成后等待 navigator 明确审查。

## 项目初始化

- 项目目标：用 TRAILS 对非同步多变量医学纵向数据进行生存轨迹聚类，识别兼具轨迹差异与事件风险差异的临床亚型。
- 当前阶段：阶段一原型已具备真实数据 case 入口、K 选择、聚类汇总和归因能力；论文级 MIMIC 实例验证尚未设计和执行。
- 关键入口：`scripts/case.py`、`scripts/cluster_attribution.py`、`configs/case.yaml`、`src/trails_case/`。
- 验证命令：依次运行 `uv run ruff format`、`uv run ruff check --fix`、`UV_CACHE_DIR=/tmp/uv-cache uv run pyright`、`UV_CACHE_DIR=/tmp/uv-cache uv run pytest`。
- 工作树基线：分支 `main`，HEAD `5468ad2fb365ec7c6aea709b0dfb7d6b4a3e103e`；启用前 `.gitignore` 有用户已暂存的 4 行新增，无未暂存或未跟踪文件。
- 初始化时间：2026-08-25（Asia/Shanghai）。

## 项目计划

由 driver 与 navigator 共同讨论创建。已确认成人首次 ICU 脓毒症队列、0–48 小时轨迹、48 小时 landmark 和之后 28 天死亡；队列细则、方法比较与结果图表仍需逐批确认。

## 进行中任务

- [ ] MIMIC 真实数据实例验证：已保存检查点，当前暂停
- [x] 整理 `src/trails` 文档注释：已完成
- [x] 重构 K 选择架构：已完成

### MIMIC 真实数据实例验证

- 目标：验证 TRAILS 相对合理基线的优势，并发现稳定、可解释且有临床意义的轨迹亚型。
- 当前状态：正式初始 K 选择已完成、聚合结果已验证取回并分析；没有候选 K 通过预设门槛，等待 navigator 决定是否仍按机械扩展规则增加至5 seeds，当前暂停主动推进。
- 已确认决定：共同讨论研究计划；每批上限 200 行；真实数据和患者级衍生物默认留在远端，仅取回获批聚合结果与图表；主分析采用成人首次 ICU 脓毒症队列、入 ICU 后 0–48 小时轨迹、48 小时 landmark、landmark 后 28 天全因死亡结局；早期 Sepsis-3 使用相对 ICU 入科 `[-6 h, +24 h]`，同时输出完整 Sepsis-3；主分析纳入 48 小时前已转出 ICU 但仍存活者，另以 ICU LOS≥48 小时做敏感性分析；MIMIC 预处理/EDA 脚本默认不写单独测试、不提供 CLI 参数，除非 navigator 另行指定。
- 当前检查点：94,458 个 ICU stay 中 41,296 个符合 Sepsis-3，37,628 个符合 early Sepsis-3；每名成人取最早 early stay 后为 29,418 人，存活至 48 小时的主队列为 28,133 人，ICU LOS≥48 小时敏感性队列为 18,180 人。
- 队列规则：复现官方 mimic-code 的疑似感染与滚动 SOFA；基线 SOFA 未知时按 0；同时记录 Sepsis-3 与其早期子集；每人取最早符合早期 Sepsis-3 的 ICU stay；轨迹窗为 ICU 入科后 0–48 小时；仅纳入 landmark 时存活者；主要结局为其后 28 天全因死亡。
- 下一小批：navigator 审查初选结果后，决定是否实现并运行5-seed扩展；在此之前不运行 sealed test，也不提交更多模型。
- 风险或待决定事项：脚本机械给出 `expansion_scope=all_candidates`，但现有最差占用记录会保留，K=2和K=4因空簇、K=5因最小簇1.44%已不可能通过当前门槛；K=3虽无占用违规，但3-seed平均ARI仅0.325，新增7个seed-pair平均ARI需约0.932才可能达到0.75。需决定继续全部扩展的科研价值，或在不查看sealed test的前提下重新审查稳定性策略。随机test只能称为内部留出验证，不能称为外部验证。
- 涉及文件：本批新增 `scripts/mimic_select_k.py` 184行和 `configs/mimic_select_k.yaml` 9行；`mimic_case.py` 新增3行并纯删除K选择实现/分支；`AGENTS.md` 新增或修改3行。手写新增/修改共199行，低于200行，`PAIR.md` 不计入。
- 验证记录：Ruff format/check、Pyright 均通过；完整 pytest 为 `144 passed, 3 warnings`。Hydra配置组合确认独立命令默认启用K=2–5；12模型合成smoke得到12条指标、12条稳定性记录、4条K汇总和12个模型；`mimic_case.py` 会拒绝K选择开关并指向新命令；`git diff --check` 通过。首次smoke仅因验证命令使用了错误的生成器import路径而在产品代码执行前失败，修正验证import后通过。
- 远端记录：资源运行 `mimic-case-resource-scipy-20260827-202635` 已完成。`mimic-feature-raw-20260828-110002` 在快照 `fbc1bd82` 上exit code 0。正式 `mimic-k-initial-20260828-110305` 同快照运行10.97小时后exit code 0，完成3 seeds×K=2–5共12模型；seed winner为4/5/4，所有K均未通过门槛，`preliminary_selected_k=null`、`expansion_scope=all_candidates`，sealed test未评估。仅取回5个聚合文件与有界记录至 `remote-results/mimic-k-initial-20260828-110305/`，患者级输入、模型、标签和分配均留远端。
- 最近交接：初始K选择终态、聚合完整性、隐私边界和本地复核均已完成；K=3平均C-index 0.659、平均ARI 0.325且无小簇，K=4平均C-index最高为0.672但存在空簇且ARI仅0.140。等待navigator审查是否继续5-seed扩展。

#### 正式建模方案（已确认）

- 队列与结局：主队列保持 28,133 名成人首次 early Sepsis-3 index ICU stay；0–48 小时异步轨迹，48 小时 landmark，之后 28 天全因死亡。
- 数据划分：按 `28天事件 × 48小时前离开ICU` 分层，固定 `64%/16%/20%` train/validation/sealed test；`split_seed=20260517` 固定，模型 seed 单独变化。
- 预处理：1/99% 截尾、均值和标准差只在 train 拟合，再原样应用到 validation/test；缺失仍由 mask 表示，不插补。
- 模型配置：保持已验证架构、batch 256、learning rate 5e-4、warmup 20、GMM 初始化 20、min/max epoch 100/300、patience 30、gradient clip 1.0，不在 MIMIC 上扩张架构搜索。
- K 与稳定性：候选 K=2–5；先用 3 个预先指定 seed 完整训练。若各 seed 的 K 结论一致、无空簇、最小簇均≥5%、validation 上 seed 间 ARI≥0.75，则只为入选 K 增补至 5 seeds；否则对全部 K 增补至 5 seeds。
- K 决策：BIC+C-index 复合分数沿用现有定义；稳定性和最小簇作为门槛，不再加入任意权重。通过门槛后采用 one-standard-error 规则选择最简单的 K。
- 锁模与测试：在入选 K 的 5 个模型中，以 validation 上平均 seed 间 ARI 最高的 medoid 模型作为代表模型；锁定后只评估一次 sealed test。其他 seeds 用于报告稳定性分布，不根据 test 结果换模型。
- 评价：test 报告 C-index、IPCW C-index、7/14/21天动态 AUC、1–27天 IBS/校准；聚类报告占比、entropy、seed 间 ARI、Kaplan–Meier/log-rank、调整后 Cox HR、轨迹差异和归因。
- 方法优势：在完全相同划分和 K 下比较 summary-KMeans、risk-stratified KMeans、FPCA-KMeans，并加入 TRAILS 去除 survival loss 的关键消融；预测性能另与生存模型基线比较。
- 临床解释：按簇汇总年龄、性别、种族、SOFA、ICU 类型、观察时长/密度、器官支持和结局；治疗变量只作描述，不作亚型定义，避免把治疗反应误当基线表型。
- 敏感性：至少重复 ICU LOS≥48 小时队列；主分析同时显式报告早出 ICU 比例，避免把恢复/转出造成的观测终止误解为生理亚型。
- 隐私：患者划分、模型、预测、簇分配和 bootstrap 明细全部留远端；只取回聚合选择表、置信区间、簇级表和图。

## 已完成记忆

- 2026-08-28：完成 `src/trails` 全部模块和定义的中文 docstring 整理；公共 API 使用详细说明，内部辅助对象使用一句话说明，未改变运行逻辑或公开签名。
- 2026-08-28：完成 K 选择架构重构；`ClusterNumberSelector`、配置和 DataFrame 结果对象统一承载单/多 seed 选择、稳定性、门槛、one-SE 与保存，`TrailsEstimator` 不再包含选择逻辑，MIMIC `06_select_k.py` 已使用新 API。通用 case/SwanLab 的旧字段迁移属于后续独立整理范围。
