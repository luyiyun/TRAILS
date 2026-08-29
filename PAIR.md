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

- [ ] MIMIC 真实数据实例验证：纵向轨迹第十七批待审查
- [x] 整理 `src/trails` 文档注释：已完成
- [x] 重构 K 选择架构：已完成

### MIMIC 真实数据实例验证

- 目标：验证 TRAILS 相对合理基线的优势，并发现稳定、可解释且有临床意义的轨迹亚型。
- 当前状态：初始 K 选择未产生有效结论；navigator 已决定暂缓 K 选择，先固定 K 跑通数据划分、训练/完整推理和后续正式评价流水线。
- 已确认决定：共同讨论研究计划；每批上限 200 行；真实数据和患者级衍生物默认留在远端，仅取回获批聚合结果与图表；主分析采用成人首次 ICU 脓毒症队列、入 ICU 后 0–48 小时轨迹、48 小时 landmark 和之后 28 天全因死亡；按 `28天事件 × 48小时前离开ICU` 分层生成默认 64%/16%/20% 的 train/validation/test ID，split seeds、比例和路径均可通过 Hydra 调整，ID 文件只记录 `patient_id`；MIMIC Hydra 配置集中到 `configs/mimic/`；统一 `07_run.py` 使用 train 训练、validation 早停并直接保存 train/validation/test 的完整预测与模型产物，不区分 pilot/final；正式评价由后续脚本完成；K 选择脚本移出编号序列并标记未完成，待完整分析后再插入并统一重编号；保留现有数字编号文件名，将 `scripts/mimic` 作为工作流包通过 `python -m` 执行，单次使用逻辑留在入口脚本、MIMIC内部复用逻辑留在同目录普通模块、跨工作流复用才保留在`src/`。
- 当前检查点：94,458 个 ICU stay 中 41,296 个符合 Sepsis-3，37,628 个符合 early Sepsis-3；每名成人取最早 early stay 后为 29,418 人，存活至 48 小时的主队列为 28,133 人，ICU LOS≥48 小时敏感性队列为 18,180 人。
- 队列规则：复现官方 mimic-code 的疑似感染与滚动 SOFA；基线 SOFA 未知时按 0；同时记录 Sepsis-3 与其早期子集；每人取最早符合早期 Sepsis-3 的 ICU stay；轨迹窗为 ICU 入科后 0–48 小时；仅纳入 landmark 时存活者；主要结局为其后 28 天全因死亡。
- 下一小批：navigator确认第十七批后冻结`08_evaluate.py`，随后按已定方向转入固定K、固定seed的`07_run.py`训练效率与可接受结果探索，不继续扩张08范围。
- 风险或待决定事项：MIMIC编号命令现在统一要求从项目根目录使用`python -m scripts.mimic.<编号命令>`，不能再把带相对导入的`07_run.py`作为文件路径直接执行；`configs/mimic/run.yaml`仍沿用原有1 epoch流程检查默认值。多个split seeds与model seeds必须保持独立，随机test只能称为内部留出验证。
- 暂缓整理：`CaseResultTables`当前只在`scripts/case.py`中直接使用，负责生成患者簇、簇汇总和簇特征汇总三类表；其未使用包装函数和`CasePredictionPayload`依赖等待后续专门整理`scripts/case.py`时一并处理，本批不改动该类。
- 涉及文件：本批新增`src/trails/prediction.py`，修改`src/trails/{__init__,estimator,trainer,selection}.py`、`src/trails_simulate/training.py`、`scripts/{case,mimic/07_run}.py`及两个公共API测试文件，共173行新增/修改低于200行上限；`AGENTS.md`与`PAIR.md`记录不计入。
- 涉及文件：本批修改`src/trails_case/evaluation.py`、`scripts/case.py`、`scripts/mimic/07_run.py`和`scripts/mimic/08_evaluate.py`，共58行新增/修改及对无人调用包装函数的纯删除，低于200行上限；`AGENTS.md`与`PAIR.md`记录不计入。
- 验证记录：Ruff format/check、Pyright均通过；完整pytest为`41 passed, 1 warning`；60例train/30例test三簇合成端到端smoke仅保存`dataset.pt + model_prediction.pt`即完成Harrell/IPCW C-index、7/14/21天动AUC、log-rank和簇汇总，确认不需要`predictions.pt`；`git diff --check`通过。
- 涉及文件：本批修改`scripts/mimic/08_evaluate.py`、`scripts/mimic/config.py`和`configs/mimic/evaluate.yaml`，共92行新增/修改低于200行上限；`AGENTS.md`与`PAIR.md`记录不计入。
- 验证记录：Ruff format/check、Pyright均通过；完整pytest为`41 passed, 1 warning`；120例train/60例test合成删失数据smoke确认动态AUC仅为7/14/21天，Brier、IBS和加权校准误差完整覆盖1–27天，`calibration.csv`每天均覆盖全60人；完全相同预测的单组回退亦已验证；`git diff --check`通过。初次Pyright遇到NumPy/pandas/KM第三方存根歧义，规范输入类型后已从Ruff开始重跑全套检查；字段拆分修订后再次重跑全套。
- 涉及文件：本批修改`scripts/mimic/08_evaluate.py`，共95行新增/修改低于200行上限；`AGENTS.md`与`PAIR.md`记录不计入。评价流程同时读取train/validation/test，validation和test均以train估计删失分布并分别输出；当前分位数组KM校准保留基于`scikit-survival`的实现，未引入需要lifelines拟合器且方法定义不同的平滑校准接口。
- 验证记录：Ruff format/check、Pyright均通过；完整pytest为`41 passed, 1 warning`；72例train、48例validation、54例test的三簇合成smoke确认两套评价目录、标签、train删失参考、1–27天Brier与校准产物均完整生成；`git diff --check`通过。
- 涉及文件：本批仅修改`scripts/mimic/08_evaluate.py`，共136行新增/修改低于200行上限；新增脚本内部`SurvivalCalibration`类，`calculate()`缓存完整1–27天分组KM表和加权绝对误差，`plot()`复用缓存并为validation/test分别保存7/14/21天PNG/PDF校准面板。
- 验证记录：Ruff format/check、Pyright均通过；完整pytest为`41 passed, 1 warning`；72例train、48例validation、54例test端到端smoke确认两套PNG/PDF及JSON路径均生成，validation PNG已完成视觉检查；`git diff --check`通过。
- 涉及文件：本批仅修改`scripts/mimic/08_evaluate.py`，新增76行低于200行上限；validation/test分别新增预测簇KM图及动态AUC、逐日Brier、逐日分组校准误差三面板图，PNG/PDF路径均写入评价JSON，未改变既有指标计算。
- 验证记录：使用`MPLCONFIGDIR=/tmp/mpl`完成Ruff format/check、Pyright和完整pytest，结果为`41 passed, 1 warning`；72例train、48例validation、54例test端到端smoke确认两套新增PNG/PDF及JSON路径，validation的KM图和时间指标图已完成视觉检查；`git diff --check`通过。
- 涉及文件：本批修改`scripts/mimic/04_extract_features.py`、`scripts/mimic/data.py`和`scripts/mimic/07_run.py`，共28行新增/修改低于200行上限；age、gender、race和脓毒症识别时sofa_score进入patients.csv、split dataset元数据及patient_outputs.csv，但不进入聚类特征。
- 验证记录：使用`MPLCONFIGDIR=/tmp/mpl`完成Ruff format/check、Pyright和完整pytest，结果为`41 passed, 1 warning`；14例患者的临时CSV→三split dataset→07患者输出smoke确认四项协变量的字段、患者顺序和值完整保持；`git diff --check`通过。
- 涉及文件：本批修改`scripts/mimic/08_evaluate.py`和`pyproject.toml`，共192行手写新增/修改低于200行上限；`uv.lock`为工具生成。新增lifelines 0.30依赖、`AdjustedCoxAnalysis`、train低风险统一参考簇、validation/test独立调整Cox表/JSON及森林图；数值缺失采用透明complete-case计数，种族合并为五组。
- 验证记录：使用`MPLCONFIGDIR=/tmp/mpl`完成Ruff format/check、Pyright和完整pytest，最终结果为`41 passed, 1 warning`；Pyright首次暴露pandas存根歧义后改用显式Series/NumPy并全量重跑。600例独立Cox探针及360例train、300例validation、300例test端到端smoke通过；第一次周期性smoke人工制造完全共线后改为独立随机协变量，最终两套参考簇、HR/CI表和PNG/PDF均验证，森林图已视觉检查；`git diff --check`通过。
- 涉及文件：本批仅修改`scripts/mimic/08_evaluate.py`，共135行手写新增/修改低于200行上限；新增`ClusterClinicalCharacteristics`，为validation/test分别保存Overall及每个配置簇的13行临床特征宽表；年龄为mean (SD)，SOFA为median [IQR]，性别、五类种族和数值缺失为n (%)，保留空簇列且不做p值检验。
- 验证记录：仅对本批脚本执行Ruff format/check以避开未审查的K-selection工作树变更，Pyright和完整pytest均按全项目运行，结果为`41 passed, 1 warning`；360例train、300例validation、300例test端到端smoke确认两套13行描述表和JSON索引，独立空簇探针确认空簇列及单患者SD边界；`git diff --check`通过。
- 涉及文件：本批修改`scripts/mimic/08_evaluate.py`、`scripts/mimic/config.py`和`configs/mimic/evaluate.yaml`，共175行手写新增/修改低于200行上限，另有5行纯删除；新增`ClusterTrajectoryAnalysis`和可调4小时时间箱，validation/test分别保存完整变量-时间箱-配置簇网格的患者层median/IQR、患者数、观测数及PNG/PDF面板，数值按训练预处理参数还原到截尾后的临床单位。
- 验证记录：仅对本批Python文件执行Ruff format/check，Pyright和完整pytest按全项目运行，最终为`41 passed, 1 warning`；Pyright首次发现pandas quantile存根歧义，改用NumPy分位点后从格式化起全量重跑。360例train、300例validation、300例test端到端smoke确认两套CSV/PNG/PDF和JSON索引，独立核算确认临床单位逆变换、患者层聚合及人数/观测数，23变量合成探针确认828行完整网格并完成图形视觉检查；`git diff --check`通过。
- 最近交接：第十六批已获navigator审查通过并进入暂存区；第十七批纵向轨迹已完成。driver判断`08_evaluate.py`已达到当前固定模型评价范围的完成标准，请navigator审查并确认。
- 远端记录：资源运行 `mimic-case-resource-scipy-20260827-202635` 已完成。`mimic-feature-raw-20260828-110002` 在快照 `fbc1bd82` 上exit code 0。正式 `mimic-k-initial-20260828-110305` 同快照运行10.97小时后exit code 0，完成3 seeds×K=2–5共12模型；seed winner为4/5/4，所有K均未通过门槛，`preliminary_selected_k=null`、`expansion_scope=all_candidates`，sealed test未评估。仅取回5个聚合文件与有界记录至 `remote-results/mimic-k-initial-20260828-110305/`，患者级输入、模型、标签和分配均留远端。

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
