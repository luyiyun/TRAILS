# Project Guide

## Purpose

TRAILS studies deep survival trajectory clustering for asynchronous multivariate
medical longitudinal data. The method targets patient subtypes that differ in
both clinical trajectories and time-to-event risk.

The current scope is phase one: a modular Surv-VaDER/VaDE prototype for
variable-length clinical visit sequences with `x`, `mask`, and `delta_time`.
Mixed-type likelihoods, competing risks, and recurrent events are later roadmap
items.

## Layout

- `scripts/`: one Hydra CLI entrypoint per command. Top-level commands use
  `uv run python scripts/<command>.py ...`; numbered MIMIC commands use
  `uv run python -m scripts.mimic.<numbered_command> ...`.
- `scripts/configs.py`: shared frozen-split TRAILS command configuration, reused
  by MIMIC and CRC Yunnan; dataset defaults remain in their YAML or thin subclasses.
- `scripts/utils/`: reusable command-layer methods and artifact contracts shared
  by MIMIC and future workflow packages such as `scripts/simulation/`.
- `configs/`: Hydra command roots plus shared simulation, training, baseline, optimization, summary, and case configuration.
- `src/trails/`: reusable core code: data, model, trainer, estimator, metrics.
- `src/trails_simulate/`: synthetic clinical data generation; imports `trails`.
- `src/trails_case/`: shared generic case-study utilities; imports `trails` plus
  shared command config models from `trails_simulate.config`.
- `tests/`: tests for the user-facing API exported from `src/trails/__init__.py`.

## Import Boundaries

Allowed dependencies:

- `scripts/* -> trails`
- `scripts/* -> trails_simulate`
- `scripts/* -> trails_case`
- `scripts.mimic command modules -> scripts.mimic` non-command support modules
- workflow packages under `scripts/ -> scripts.utils`
- workflow packages under `scripts/ -> scripts.configs`
- `scripts.configs -> trails_case.config` for the existing command config inheritance
- `scripts.utils -> trails`
- `trails_simulate -> trails`
- `trails_case -> trails`
- `trails_case -> trails_simulate.config` for shared command config models only

Forbidden dependencies:

- `trails -> trails_simulate`
- `trails -> trails_case`
- `trails -> scripts`
- `trails_case -> trails_simulate` runtime modules outside `trails_simulate.config`
- `scripts.utils -> scripts.mimic` or another dataset-specific workflow package
- MIMIC command modules importing another numbered command module

Keep command orchestration out of `src/trails`; the main package should remain a
clean reusable method library.

## Commands

- Quick simulation split generation: `uv run python scripts/simulate.py simulation=quick`
- Paper simulation grid generation for one scene: `uv run python scripts/simulate.py simulation=base`
- Train existing splits: `uv run python scripts/train.py training=base paths.data_root=data/simulated/base`
- Train with mTAN-style input: `uv run python scripts/train.py training=mtan paths.data_root=data/simulated/base`
- Run real-data case modeling: `uv run python scripts/case.py observations_csv=data/case/observations.csv patients_csv=data/case/patients.csv`
- Generate MIMIC patient splits and tensor datasets: `uv run python -m scripts.mimic.06_split`
- Split CRC Yunnan patients and middle-format observations: `uv run python -m scripts.crc_yunnan.03_split`
- Prepare CRC Yunnan model datasets: `uv run python -m scripts.crc_yunnan.04_preproc`
- Run CRC Yunnan K selection and TRAILS on one frozen split: `uv run python -m scripts.crc_yunnan.05_run`
- Evaluate a completed CRC Yunnan run: `uv run python -m scripts.crc_yunnan.06_evaluate inputs.run_dir=outputs/crc_yunnan/<run>/modeling`
- Run MIMIC modeling with configured or automatically selected K: `uv run python -m scripts.mimic.07_run`
- Run MIMIC baselines on frozen splits: `uv run python -m scripts.mimic.08_baselines split_dir=data/real/mimic-iv-3.1/derived/trails_splits/seed-20260517`
- Evaluate frozen MIMIC predictions: `uv run python -m scripts.mimic.09_evaluation 'trails_dirs=[outputs/mimic_case/<primary-run>,outputs/mimic_case/<other-runs>]' 'baseline_dirs=[outputs/mimic_case/<baselines>]'`
- Run lightweight baselines on existing splits: `uv run python scripts/baseline.py paths.data_root=data/simulated/base`
- Run Optuna tuning on existing splits: `uv run python scripts/optim.py paths.data_root=data/simulated/base`
- Summarize train and baseline results: `uv run python scripts/summary.py 'train_roots=[outputs/train/base-...,outputs/train/mtan-...]' 'baseline_roots=[outputs/baseline/base-...]' 'train_labels=[base,mtan]' 'baseline_labels=[kmeans]'`
- Compute cluster attribution lines for a saved model: `uv run python scripts/cluster_attribution.py --model-path outputs/case/case-.../model.pt --data-path outputs/case/case-.../case_dataset.pt --plot-features 6`
- Format: `uv run ruff format`
- Lint: `uv run ruff check --fix`
- Type check: `UV_CACHE_DIR=/tmp/uv-cache uv run pyright`
- Tests: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`

## Coding Rules（代码风格规范）

以下规范适用于本项目，除非用户另有明确要求。

### 通用约定

- 使用 Python 和 uv。
- 优先使用成熟的现有包提供的函数，避免重复实现已有功能。
- 数值计算和表格处理优先使用 NumPy、SciPy、pandas 的函数与向量化操作，
  尽量减少原生循环。CSV 的读取、导出和后续处理优先使用 pandas，
  避免用 stdlib `csv` 或逐行循环重复实现已有表格操作；张量和模型计算使用 torch。
- 未经用户明确要求，不额外添加 audit、manifest、test 等代码。
- 在复杂研究逻辑处使用简洁中文注释，尤其是模拟机制、模型结构和损失函数；
  不要给显而易见的赋值或样板代码写流水账式注释。
- 对于较为复杂的基础设施逻辑（例如多进程进度条），优先使用内聚的面向对象
  封装，避免将状态、上下文和协调逻辑分散在多个松散 helper 中。
- 默认让异常直接向上抛出并保留原始 traceback；不要用 `try/except` 仅为改写错误、
  记录状态或继续批处理。只有调用方能够实质恢复时才捕获异常。

### 配置管理

- 需要单独管理配置时，使用 Hydra + Pydantic：Hydra 负责读取、组合配置和提供
  基础命令行功能；Pydantic 负责配置的解析、类型转换和验证。
- YAML 提供默认值。Hydra 完成配置组合和插值解析后，统一构造 Pydantic 配置模型，
  业务代码直接读取模型中的值，不重复进行配置类型转换和校验。
- 通过合适的 Pydantic 字段类型、约束和类内 validator 完成配置解析与验证，
  不额外编写独立的配置验证函数。
- 使用 OmegaConf 保存 Pydantic 配置为 YAML 时，直接使用
  `OmegaConf.save(config.model_dump(mode="json"), path)`；一般的字典配置也直接传入
  `OmegaConf.save`，不额外包装 `OmegaConf.create(...)`。
- 模拟参数使用 Hydra YAML 和命令行覆盖，不重新引入 Makefile 实验包装层。
- 除非用户要求，MIMIC 预处理和 EDA 脚本沿用固定项目路径，不增加命令行参数。

### 静态类型与 pandas

- 所有新增 Python 代码应有静态类型注释，并通过 `standard` 模式的 Pyright 检查。
- 优先通过合适的类型注释解决类型问题；确实无法解决时，再考虑 `cast` 等静态
  类型转换方式。
- pandas 代码不要为了调用 Series 或 DataFrame 的方法而添加
  `cast(pd.Series, ...)` 或 `cast(pd.DataFrame, ...)`，直接调用相应方法即可。
- 若 pandas 类型推断仍导致误报，可在必要的局部行使用带具体错误码的
  `# type: ignore[错误码]`，但应慎用，不以忽略注释掩盖真实类型错误。

### `src/`：面向对象与模型 API

- 主要采用面向对象风格。
- 深度学习方法重点实现 `model`、`trainer` 和 `estimator`，明确各自职责：
  - `model` 定义网络架构和 `forward` 前向传播；通过控制参数（如 `mode`）选择
    执行阶段，例如计算损失或仅生成预测。
  - `trainer` 封装训练过程，独立支持不同模型的训练循环，例如 GAN 或 diffusion model。
  - `estimator` 在内部调用 `model` 和 `trainer`，提供真正面向用户的公开 API。
- `model` 和 `trainer` 实例化时接收 Pydantic 配置模型，由该模型统一管理参数、
  完成类型转换和验证。
- `estimator` 同时支持传入 Pydantic 配置模型和普通参数两种实例化方式，
  底层均使用 Pydantic 配置模型统一管理。
- `src/trails` 内部的项目模块优先使用相对导入；`src/trails_simulate` 和
  `src/trails_case` 使用从 `trails` 开始的绝对导入。
- `src/trails_case` 仅可从 `trails_simulate.config` 导入共享命令配置模型，
  不依赖模拟数据生成、训练或评价的运行时模块。

### `scripts/`：线性分析流程

- 主要采用线性、过程式的分析风格，按执行顺序组织读取、处理、分析和输出。
- 分析脚本使用上游已保证的数据或结构时，直接依赖上游的数据约定，不重复验证
  字段、唯一键、取值范围和表间对应关系等已保证的内容。仅验证本步骤新增或
  上游尚未保证的要求；读取时必要的类型恢复和分析所需的描述性统计照常保留。
- 代码按功能分块，用中文注释标题说明每个大块的主要功能；大块内部可继续划分
  小块，用简短注释说明功能。不同块之间至少留一个空行。
- 大块标题统一使用以下格式：

  ```python
  ##################################################
  # 一、读取和整理输入数据
  ##################################################
  ```

- 普通处理逻辑仅在被多次使用时抽象为工具函数；只使用一次的逻辑保留在相应流程中，
  必要时添加简洁注释。下述绘图代码是单次使用逻辑的封装例外。
- 绘制单张图的较长代码应封装为独立函数，即使仅调用一次；尤其是 Matplotlib、
  Seaborn 的建图、绘制、标注和布局代码，避免大量内联代码淹没分析流程。
- 绘图函数通过参数接收所需数据、统计结果和绘图选项，封装该图的绘制及必要的
  保存、关闭操作；优先复用主流程已计算的统计结果。`run()` 保留分析步骤和清晰的
  绘图函数调用，不展开逐轴绘制细节。

### 测试与版本管理

- 运行现有检查与新增测试代码是两回事；代码修改后按下方 Verification 执行检查，
  不据此自动增加测试代码。
- 用户要求新增测试时，项目测试范围仍限于 `src/trails/__init__.py` 中 `__all__`
  导出的用户 API 及其公开方法。不为辅助函数、内部实现、`src/` 下其他包或
  `scripts/` 下的命令脚本添加测试；发现这类既有测试时删除，而不是继续维护。
- 未经用户明确要求，不执行 `git add`、`git commit` 或其他会改变 Git 索引、分支、
  提交历史的命令；代码的暂存和版本管理由用户手动完成。Git 状态与 diff 等只读检查不受此限。
- 项目架构决策变化时，同步更新本文件。

## Current Decisions

- Package name: `trails`.
- Reusable baseline implementations and command-layer artifact contracts live in
  `scripts/utils`; dataset-specific workflow packages own only configuration,
  orchestration, and adapters. `src/trails_simulate` and `src/trails_case` are
  legacy workflow modules to migrate into `scripts/` or remove incrementally.
- Shared custom preprocessing steps live in `scripts/utils/preprocessing.py`.
  They accept multi-feature matrices, fit each column while ignoring NaN, and
  preserve missing positions. Split commands pivot patient-time observations to
  feature columns and directly fit one sklearn Pipeline on train, then restore
  the original observed rows. Standard/minmax use native sklearn scalers;
  robust scaling computes the population-SD fallback only when IQR is zero.
  Custom steps retain only fitted attributes needed for transformation and the
  sklearn interface. Disabled log uses Pipeline passthrough. MIMIC retains its
  five-column parameter CSV; CRC exports only feature, transform, center and scale.
  MIMIC dataset preparation lives in `06_split.py`; shared baseline covariate
  names live in `scripts/mimic/config.py`.
- CLI lives in command-specific Hydra scripts under `scripts/`; no package console script is configured.
- Non-MIMIC command scripts use root configs at `configs/<command>.yaml`; Hydra
  configs for `scripts/mimic/` live under `configs/mimic/`. `scripts/simulate.py`
  generates train/test split simulation data, while training and baseline
  comparison are separate scripts.
- Simulation scenarios live under `configs/simulation/`: `quick`, `base`,
  `imbalance`, `censored`, and `high_dimension`. `quick` is for smoke tests;
  the other four are paper simulation scenes.
- Training model presets live under `configs/training/`: `small`, `base`,
  `large`, and `mtan`. Training scene selection comes from `paths.data_root` or
  `paths.explicit_split`, not from simulation config.
- `trainer.batch_size: null` means the trainer resolves batch size from
  the loaded training split size using a conservative automatic rule; explicit
  integer overrides keep their exact value.
- Command-level root configs compose only the fields needed by their script and
  follow the config locations above. Simulation, baseline, summary, case, and training
  fields are flattened into the command root after preset composition; `paths`
  remains the shared path namespace, and `optim` intentionally keeps its
  `optim.*` namespace. Generator parameters live under `generator`, while TRAILS
  model/trainer/artifacts/diagnostics/SwanLab parameters live at the command root.
- Case-study config reuses shared training, diagnostics, artifacts, SwanLab, and
  output-path config models from `trails_simulate.config`; case paths use only
  output directory fields and do not carry simulation data split fields.
- `train_size` and `test_size` are equal-length lists that
  are paired by position. Each paired sample-size level is crossed with
  `generator.n_clusters`, and each combination is repeated
  `repeats` times. Formal simulation scenes use train sizes
  `[500, 1000, 2000, 3000, 5000]` and fixed test size `300`; `quick` remains a
  small smoke-test configuration.
- `repeats` means paired split repeats within each sample-size and K
  combination: each repeat generates one source simulation dataset and splits it
  into train/test.
- Generator instantiation fixes DGP mechanism parameters using
  `mechanism_seed` when set, otherwise `seed`. The same
  `name × K` combination uses a fixed mechanism seed; sample seeds vary
  across sample-size levels and repeats for patient draws, train/test split
  shuffling. Train and optim command seeds come from `trainer.seed`
  plus the discovered split index; baseline seeds come from `seed`
  plus the discovered split index.
- Validation data is cut internally from `train.pt` by `trainer.valid_size`, is
  not saved as a separate `val.pt`, and is used for early stopping; if no validation
  split is requested, early stopping monitors the training metric instead. The
  monitor can be total loss, survival loss, or C-index.
- Hydra metadata and command outputs go under the single user-visible `paths.dir`
  directory. Each command root config sets `hydra.run.dir: ${paths.dir}` and
  defines `paths.root`, `paths.prefix`, and `paths.suffix`, which compose the
  default `paths.dir`; users may override `paths.dir` directly, for example
  `paths.dir=outputs/train/my-run`. Train, baseline, and optim root configs
  declare their own `paths.data_root` and `paths.explicit_split` defaults directly.
  Input paths such as `paths.data_root`, explicit split paths, summary roots, and
  case CSVs are resolved relative to the directory where the command is launched.
  Output paths are resolved relative to
  `paths.dir`; train, baseline, and optim outputs mirror the relative data split
  path discovered under `paths.data_root`.
- Simulation uses a VaDeSC-EHR-style latent-cluster generator adapted to
  continuous asynchronous clinical measurements.
- Simulation outputs a `ClinicalTimeSeriesDataset` saved via `torch.save`.
- Canonical saved patient samples are aligned samples containing `times`, `x`, `mask`,
  `delta_time`, `survival_time`, `event`, and optional `cluster_label`. At runtime,
  `AlignedClinicalSample` and `CompactClinicalSample` convert between views through
  dataclass methods, and `ClinicalTimeSeriesDataset(return_kind=...)` stores samples
  in the requested view at initialization rather than converting in `__getitem__`.
  Compact samples left-align per-feature observations and carry per-feature `times`,
  `mask`, and `feature_lengths`.
- `ClinicalTimeSeriesDataset.from_dataframes()` accepts a patient table and a
  middle-format observation table (one patient-time row, one column per feature).
  It preserves patient-table order, sorts visits, derives masks and time gaps,
  and records patient IDs and observation summaries. CRC `04_preproc.py` uses
  this public API after train-fitted preprocessing instead of building samples.
  `load_from_csv()` reads long-format CSVs, pivots once, and delegates sample
  construction to the same API while retaining CSV metadata and patient order.
- Dataset metadata preserves latent profiles, cluster parameters, survival
  coefficients, and generation parameters for simulation-study evaluation.
- The phase-one model uses a modular encoder: an asynchronous input layer (`grud`,
  original aligned `mtan`, or compact per-feature `mtan2`) followed by a nonlinear
  mapping layer (`gru`, `lstm`, or `transformer`) and SeqPool.
- The original `mtan` input path consumes aligned `(B, T, D)` batches, concatenates
  `x` and `mask` as the attention value, and maps observations onto a training-set
  global reference-time grid before the mapping layer.
- The legacy `mtan2` input path consumes compact `(B, T, D)` batches by attending
  separately over each sample-feature observation stream, then concatenates
  per-feature embeddings into `(B, reference_points, D * feature_embedding_dim)`.
- The reconstruction decoder is configurable as `gru`, `lstm`, or
  `transformer`; recurrent decoders support latent-initialized hidden state or
  repeated latent plus visit-time input, while transformer decoding only uses
  repeated latent plus visit-time input.
- Clustering uses a VaDE-style learnable Gaussian mixture latent prior,
  initialized by deterministic k-means after warmup.
- The survival head maps each patient's latent mean to one Weibull shape/scale
  pair, with a configurable number of latent-width hidden layers before the
  Weibull output; survival likelihood and curves are not mixed across clusters.
- Validation and test metrics include ACC/ARI/NMI only when true cluster labels are
  available; test metrics also report predicted-cluster occupancy diagnostics.
- `TrailsEstimator.predict()` performs one forward pass and returns a
  `TrailsPrediction`. Its ordinary methods `predict()`, `predict_proba()`,
  `risk_score(horizon)`, and `survival()` derive cluster labels, posterior probabilities,
  fixed-horizon event risks, and survival curves from the saved latent and
  patient-specific Weibull parameters. `trainer.risk_horizon` supplies the
  configurable horizon when `trainer.cindex_risk_score=event_probability` (the default).
  `risk_score(method="median_survival")` returns negative log predicted median survival
  time without a horizon. The trainer and K selector honor the same configured risk method.
- Train and baseline commands recursively discover all sibling `train.pt`/`test.pt`
  directories under `paths.data_root`, infer K from dataset metadata when present,
  and save unified prediction payloads under mirrored run directories plus
  command-level metrics CSV and summary JSON.
- Train command progress reports include split index, elapsed time, per-split
  duration, and estimated remaining time.
- Train command split execution is configurable with `parallel.workers`.
  The default is serial (`1`); `workers > 1` uses spawn-based process parallelism
  across discovered train/test splits. If `parallel.devices` is empty,
  every worker keeps `trainer.device`, including same-GPU concurrency;
  otherwise worker slots rotate through the configured device list.
- CLI terminal messages use logging with tqdm-compatible output. Train progress
  keeps a total split bar plus per-worker training bars with fixed positions.
- `scripts/baseline.py` runs lightweight simulation
  comparators on existing train/test splits: summary-feature k-means and
  risk-stratified summary-feature k-means, plus FPCA-KMeans via `scikit-fda`. It
  writes baseline summary JSON and metrics CSV under `paths.dir`.
- `scripts/optim.py` recursively discovers existing train/test splits and optimizes
  one shared hyperparameter trial over all selected splits by averaging C-index
  and ARI. Use `optim.run_ids` to select a subset; an empty list means all splits.
  `optim.parallel` controls the shared process pool across trial/split jobs.
  Use `optim.resume=true paths.dir=<existing-run-dir>` to append trials to an
  existing study; dataset fingerprints are checked before resume.
- `scripts/summary.py` accepts any number of train and baseline run directories
  via `train_roots` and `baseline_roots`, adds source-aware
  method labels when repeated method names appear across roots, aggregates by
  scenario/sample size/K/method label, and writes CSV/JSON plus one publication-facing
  metrics-by-K PNG/PDF grid per scenario.
- `scripts/cluster_attribution.py` is the lightweight attribution entrypoint. It
  loads a saved `TrailsEstimator` checkpoint and matching `ClinicalTimeSeriesDataset`,
  computes Captum integrated-gradient contributions of observed values to VaDE cluster
  logits, aggregates them into fixed time-bin by feature tables with SEM, and writes
  one multi-cluster line plot using `--plot-features` as either Top-N or explicit
  feature names.
- `scripts/case.py` reads real-data CSV inputs:
  `patients.csv` with `patient_id`, `survival_time`, `event`, and optional
  `cluster_label`; `observations.csv` with `patient_id`, `time`, `feature`,
  and `value`. It trains on all patients, uses `trainer.valid_size`
  only for internal early stopping, and saves the converted dataset, model,
  history, predictions, patient-level clusters, cluster summaries, feature
  summaries, and `case_summary.json` under `paths.dir`.
- CRC Yunnan uses `01_cohort -> 02_eda -> 03_split -> 04_preproc -> 05_run -> 06_evaluate`,
  with Hydra configs `cohort`, `eda`, `split`, `preproc`, `run`, and `evaluate` under
  `configs/crc_yunnan`. New cohort/split/preproc artifacts use v2; historical v1
  artifacts remain unchanged and are not accepted by the new workflow.
  Downstream commands trust upstream schemas, keys, outcome values and dataset
  names. The shared `read_tables` only restores CSV types and preserves upstream
  row order. Each command checks only its newly introduced requirements, such as
  temporal split years, cross-input patient overlap, panel/window eligibility,
  transformed values and modeling output-name conflicts.
- `01_cohort.py` resolves Hydra config and validates it with `CRCCohortConfig` from
  `scripts/crc_yunnan/config.py`, then passes typed values to `run`. Configuration
  models own path/type conversion and option validation; YAML owns defaults and
  the script retains source-data checks and cleaning. It assigns deterministic research
  IDs, applies date corrections and same-day feature aggregation, and performs
  base exclusions before splitting (reversed follow-up and either outcome over
  7305 days by default). It preserves the supplied DFS/OS definitions. Observations
  are middle-format CSV: one row per `patient_id,time_days`, one numeric column per
  feature, and NaN for unobserved values. Feature groups live in
  `feature_metadata.csv`; baseline covariates and outcomes remain in patients.csv.
  A Chinese cohort summary reports overlapping rule counts and sequential exclusions.
- `02_eda.py` resolves Hydra config through `scripts/crc_yunnan/config.py::CRCEDAConfig`
  before calling its typed `run`; the model owns path/type conversion, baseline column
  validation, positive time/plot parameters, and landmark ordering. It consumes the
  base cohort directly in middle format, restoring CSV types without repeating 01's
  schema, key, or feature-metadata checks. Scattered cohort,
  quality and outcome statistics go to Chinese `eda_summary.json`; all aggregate
  tables share `eda_tables.xlsx`, with one Chinese sheet per table. Seven PNG figures
  have Chinese labels. Observation counts mean nonmissing feature values, distinct
  from patient-timepoint rows; landmark and time-bin coverage retain explicit denominators.
  Its `outcome` selects the landmark population, coverage denominators and temporal
  cutoff summaries (OS by default); overall survival curves still describe both outcomes.
- `03_split.py` only partitions patients and saves original-scale middle CSVs, IDs
  and manifests. Random 64/16/20 and temporal-2017 splits use DFS/OS joint event
  stratification and seed 20260908. Master assignments are reused across outcomes,
  windows and panels, checking rules and base-cohort IDs without source-file hashes.
  `split.save_full_dataset=true` also saves one `full/` directory at the split root;
  `split.strategies=[]` is allowed only with that flag. It performs no feature
  transformations, landmark/panel selection or tensor construction.
- `04_preproc.py` validates `CRCPreprocConfig` in the local `config.py` after Hydra
  resolution. `inputs.train_dir` selects the fitting data and `inputs.test_dirs`
  lists transform-only inputs (default validation/test; omitted or empty means none).
  Output train is always named `train`; other names come from input directory names.
  Duplicate names/paths and overlapping patient IDs fail. Default analysis uses
  OS, 720-day landmark, E60, at least two observed dates over the entire fixed panel.
  Outcomes must exceed landmark; observation windows include both endpoints and
  survival starts at landmark. Missing/constant train features fail rather than being
  silently dropped. Only final train observations fit auto-log1p and robust scaling
  (or configured none/standard/minmax), using shared sklearn preprocessing. NaN masks
  are preserved; log-selected features reject negative values and outputs must fit
  finite float32. Each set saves patients, IDs, transformed middle CSV and dataset.pt;
  the root saves four-column preprocessing parameters, resolved config and
  preproc_manifest.json. It builds aligned samples directly from middle matrices.
- `05_run.py` uses `split.dir` to reference the 04 bundle. It reads train and optional
  `validation` first, with `trainer.valid_size=0`. Other configured evaluation sets
  are read only after the estimator is locked. With validation, automatic K selection
  retains occupancy gates, one-standard-error selection and final seed ranking by
  selection score, C-index, BIC, then seed. Without validation, fixed K is mandatory
  and early stopping monitors training data. Full-cohort preprocessing/training uses
  all eligible patients and reports no independent validation/test when absent.
  Current defaults use OS/24 months, E60, auto-log1p/robust, K=2–10 across seeds
  20260908/09/10 and no SwanLab.
  Model, history, available-set predictions/metrics and a short provenance manifest
  are saved without copying datasets. Patient-level artifacts stay remote.
- `06_evaluate.py` resolves `CRCEvaluationConfig`, follows 05/04/03 manifests back to
  the frozen data and cohort metadata, and writes Chinese aggregate JSON, one Excel
  workbook, and PNG/PDF figures to a separate evaluation directory. It preserves
  cluster labels and colors, fits UMAP only on train, and computes clinical summaries,
  KM/log-rank, unpenalized adjusted Cox, original-scale patient-weighted trajectories,
  confidence, prediction/calibration metrics, training and K-selection diagnostics.
  Survival times use 30-day months after landmark; trajectory times start at surgery.
  Censoring weights come only from train. Unsupported metrics retain their configured
  time points with null values and reasons; genuine computation errors propagate.
  Cross-seed ARI uses all saved candidate models on frozen validation, without fitting
  or changing model selection. Fixed-K runs without candidates skip ARI explicitly.
  No patient-level embedding coordinates or new manifests are exported. Shared CRC
  statistics and single-figure functions live in `evaluation.py` and `evaluation_plots.py`.
- MIMIC concept preprocessing is executed directly from the pinned external
  mimic-code SQL files by `01_build_sepsis.py`; downstream extraction reads the
  resulting official tables and adds study-specific cohort/window aggregation.
  Do not copy or reimplement available official concept SQL in project scripts.
  `04_extract_features.py` uses official total `norepinephrine_equivalent_dose`
  intervals for vasopressor presence, merged duration, observation fraction,
  peak and positive-dose duration-weighted mean; it does not export per-drug
  indicators or recalculate NED conversion factors. Total NED retains official
  values without applying the historical per-drug outlier threshold.
- MIMIC `06_split.py` saves the ID-only external split plus train-fitted
  train/validation/test tensor datasets and preprocessing parameters. Later commands
  consume these frozen datasets; command modules import only non-command support
  modules from the workflow package and do not import one another. It refuses to
  overwrite any existing target seed directory and preflights every configured seed
  before writing, so each seed directory is an immutable split bundle.
- MIMIC `06_split.py` supports `random` and `temporal` strategies. Temporal splitting
  assigns `anchor_year_group` values starting at the configured cutoff to test, then
  randomly stratifies validation from the earlier development cohort.
- MIMIC patient inputs and each frozen split preserve baseline `age`, `gender`,
  `race`, and sepsis-onset `sofa_score` covariates for adjusted descriptive Cox
  analysis; these variables are not added to the longitudinal clustering inputs.
- `scripts/mimic/01_build_sepsis.py` through `07_run.py` form the current ordered
  MIMIC analysis workflow. `06_split.py` saves ID-only train/validation/test
  partitions and their tensor datasets. In `07_run.py`, a non-null top-level
  `n_clusters` runs that fixed K, while `n_clusters: null` selects K from the
  configured candidates and seeds using train plus validation only. The configured
  `trainer.seed` must be among the selection seeds and identifies the predeclared
  candidate estimator used for downstream predictions; test is loaded only after K
  and that final estimator are locked. Single-seed selection leaves stability pairs
  empty, while multiple seeds enable pairwise ARI summaries and the optional
  `min_mean_pairwise_ari` gate. Selection artifacts use the public
  `ClusterNumberSelectionResult` contract under `k_selection.result_dir`.
  `07_run.py` then exports per-split datasets, complete `TrailsPrediction` objects,
  patient-level tables, metrics, the model, preprocessing parameters, history, and
  a run manifest. Their Hydra command configs live under `configs/mimic/`; `split.yaml` declares
  `feature_order: []`, which preserves the observed CSV order by default and can
  be overridden directly through Hydra.
- `scripts/mimic/08_baselines.py` fits each method/seed only on frozen train
  (validation may control early stopping), saves models and three-split predictions,
  and records its direct 06 split reference plus artifact SHA256 hashes in an atomic
  manifest. `split_dir` is its sole dataset input; every clustering method owns
  its `n_clusters`, where an integer or one-element list means fixed K and a longer
  list requests method-specific K selection; methods without clustering do not expose it.
  The registry declares each method's capabilities, K-selection rule, model suffix,
  and prediction format; manifests distinguish `not_applicable`, `fixed`, and
  `automatic` K modes explicitly.
  Shared model/trainer presets are resolved into the no-survival method config,
  which constructs its `TrailsEstimator` without depending on a 07 run. Its
  multi-K mode minimizes train latent
  Gaussian-mixture BIC and never uses survival outcomes. Shared baselines
  write `BaselinePrediction` NPZ; `ufpca_kmeans` concatenates per-variable UFPCA
  scores, while `mfpca_kmeans` uses FDApy joint MFPCA and DCM/VaDeSC remain
  UFPCA-based. R exchange files and checkpoints stay remote.
  Any method error propagates immediately with its original traceback; a partial run
  is not accepted as a complete comparison. New run directories are required.
- `09_evaluation.py` accepts any number of completed `baseline_dirs`, branches
  between `TrailsPrediction` and baseline NPZ readers,
  and evaluates each method/seed/split once according to its cluster and survival
  capabilities. `evaluation.py` retains shared plotting/calculation classes. Cluster evaluation
  reports occupancy, entropy, KM/log-rank, adjusted Cox, clinical characteristics,
  trajectories, organ support/treatment differences and label agreement. Treatment
  evaluation reads the 04-generated `interventions.csv`, compares seven binary
  exposures in all patients and twelve continuous endpoints among corresponding
  exposed patients, and saves descriptive/test tables plus PNG/PDF panels per split;
  it never generates cluster-only survival
  predictions or predictive C-index/AUC/IBS/calibration from cluster KM curves.
  Survival evaluation reports Harrell/IPCW C-index, cumulative/dynamic AUC, daily
  Brier/IBS and quantile-group KM calibration, always estimating censoring from train.
  One unified comparison table retains unavailable capability fields as missing values.
  Degenerate clusters and unestimable Cox
  effects remain explicit diagnostics, not silently relabeled successes.
  Its non-empty `trails_dirs` places the primary TRAILS method first. Every 07/08
  manifest must directly reference the same 06 split; 09 rejects old manifests or
  mismatched references, then loads datasets and preprocessing from that common split.
  Each run receives a unique comparison key and participates in cross-run ARI/NMI.
- `AdjustedCoxAnalysis` uses the train split's lowest observed KM mortality cluster as the
  shared validation/test reference and reports cluster hazard ratios adjusted for
  age, gender, grouped race, and sepsis-onset SOFA with lifelines' default Efron
  ties handling; forest plots show only the adjusted cluster effects. Each split
  also saves a descriptive clinical-characteristics table with Overall and every
  configured cluster column; it reports age as mean (SD), SOFA as median [IQR],
  and gender, grouped race, and numeric missingness as n (%). Overall group
  differences use distribution-aware continuous tests and expected-count-aware
  categorical tests; violin/box and categorical proportion panels accompany the table.
  `ClusterTrajectoryAnalysis` restores longitudinal values to clinical
  units using the train-fitted preprocessing parameters, bins the 0–48 hour
  window at configurable four-hour intervals, first takes each patient's median
  within a feature-bin, and then saves cluster median/IQR tables and PNG/PDF
  trajectory panels for validation and test.
- In generic `scripts/case.py`, `k_selection.enabled=true` runs estimator-level holdout K selection before
  final case training. Empty `k_selection.candidate_clusters` means
  `2..model.n_clusters`; candidates are scored by validation C-index and
  latent MoG BIC using `sqrt(CI^2 + (1 - BIC_norm)^2)`. Candidate training and
  selection metrics share the same holdout validation split. Empty
  `k_selection.seeds` uses `trainer.seed`; an explicit list supports multi-seed
  stability and must include `trainer.seed`, which identifies the downstream
  estimator. Case runs inherit the selected candidate estimator instead of
  retraining on all patients, and
  candidate models, histories, metrics, configs, and aggregate selection tables
  are saved under `k_selection.result_dir`.
- `configs/case.yaml` composes the `training=case` preset, which defaults to SwanLab
  enabled, complete artifacts, and latent embedding diagnostics.

## Verification

Run these commands in order after code changes:

1. `uv run ruff format`
2. `uv run ruff check --fix`
3. `UV_CACHE_DIR=/tmp/uv-cache uv run pyright`
4. `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`

If any step fails, fix it before considering the change complete.
