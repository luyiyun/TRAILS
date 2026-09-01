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
- Run fixed-K MIMIC modeling: `uv run python -m scripts.mimic.07_run`
- Run MIMIC baselines on frozen splits: `uv run python -m scripts.mimic.08_baselines input_dir=outputs/mimic_case/<run>`
- Evaluate frozen MIMIC clusters: `uv run python -m scripts.mimic.09_eval_cluster input_dir=outputs/mimic_case/<run> 'baseline_dirs=[outputs/mimic_case/<baselines>]'`
- Evaluate frozen MIMIC survival predictions: `uv run python -m scripts.mimic.09_eval_survival input_dir=outputs/mimic_case/<run> 'baseline_dirs=[outputs/mimic_case/<baselines>]'`
- Run lightweight baselines on existing splits: `uv run python scripts/baseline.py paths.data_root=data/simulated/base`
- Run Optuna tuning on existing splits: `uv run python scripts/optim.py paths.data_root=data/simulated/base`
- Summarize train and baseline results: `uv run python scripts/summary.py 'train_roots=[outputs/train/base-...,outputs/train/mtan-...]' 'baseline_roots=[outputs/baseline/base-...]' 'train_labels=[base,mtan]' 'baseline_labels=[kmeans]'`
- Compute cluster attribution lines for a saved model: `uv run python scripts/cluster_attribution.py --model-path outputs/case/case-.../model.pt --data-path outputs/case/case-.../case_dataset.pt --plot-features 6`
- Format: `uv run ruff format`
- Lint: `uv run ruff check --fix`
- Type check: `UV_CACHE_DIR=/tmp/uv-cache uv run pyright`
- Tests: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`

## Coding Rules

- Use Python and uv.
- Use Hydra YAML plus CLI overrides for simulation parameters; do not reintroduce
  Makefile experiment wrappers.
- All new Python code must have type annotations.
- `pyright` uses `standard` mode.
- CSV/tabular ingestion, export, and downstream tabular manipulation should
  prefer pandas DataFrames and numpy vectorized operations over stdlib `csv` and
  ad hoc row loops; torch remains the right tool for tensor/model code.
- Inside `src/trails`, prefer relative imports for project modules.
- Inside `src/trails_simulate` and `src/trails_case`, use absolute imports from
  `trails`.
- `src/trails_case` may import only shared command config models from
  `trails_simulate.config`; do not depend on simulation generation, training, or
  evaluation runtime modules.
- 在复杂研究逻辑处使用简洁中文注释，尤其是模拟机制、模型结构和损失函数；
  不要给显而易见的赋值或样板代码写流水账式注释。
- 如果一段代码只在局部流程中使用一次，不需要为了形式拆成额外函数；在相关
  内容附近添加简洁中文注释说明意图即可。
- 对于较为复杂的基础设施逻辑（例如多进程进度条），优先使用内聚的面向对象
  封装，避免将状态、上下文和协调逻辑分散在多个松散 helper 中。
- Avoid `try/except` unless the code can recover or add useful diagnostics.
- 仅测试 `src/trails/__init__.py` 中 `__all__` 导出的用户 API 及其公开方法。
  不为辅助函数、内部实现、`src/` 下其他包或 `scripts/` 下的命令脚本添加测试；
  发现这类既有测试时删除，而不是继续维护。
- MIMIC preprocessing and EDA scripts use fixed project paths without CLI arguments unless requested.
- Update this file when architecture decisions change.

## Current Decisions

- Package name: `trails`.
- Reusable baseline implementations and command-layer artifact contracts live in
  `scripts/utils`; dataset-specific workflow packages own only configuration,
  orchestration, and adapters. `src/trails_simulate` and `src/trails_case` are
  legacy workflow modules to migrate into `scripts/` or remove incrementally.
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
  configurable horizon used for training and evaluation C-index calculations.
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
- MIMIC `06_split.py` saves the ID-only external split plus train-fitted
  train/validation/test tensor datasets and preprocessing parameters. Later commands
  consume these frozen datasets; command modules import only non-command support
  modules from the workflow package and do not import one another.
- MIMIC patient inputs and each frozen split preserve baseline `age`, `gender`,
  `race`, and sepsis-onset `sofa_score` covariates for adjusted descriptive Cox
  analysis; these variables are not added to the longitudinal clustering inputs.
- `scripts/mimic/01_build_sepsis.py` through `07_run.py` form the current ordered
  MIMIC analysis workflow. `06_split.py` saves ID-only train/validation/test
  partitions and their tensor datasets, and `07_run.py` trains on one fixed-K split and exports per-split
  datasets, complete `TrailsPrediction` objects, patient-level tables, metrics,
  the model, preprocessing parameters, history, and a run manifest. Their Hydra
  command configs live under `configs/mimic/`; `split.yaml` declares
  `feature_order: []`, which preserves the observed CSV order by default and can
  be overridden directly through Hydra.
  K selection is currently excluded as unfinished.
- `scripts/mimic/08_baselines.py` fits each method/seed only on frozen train
  (validation may control early stopping), saves models and three-split predictions,
  and records source/artifact SHA256 hashes in an atomic manifest. It directly uses
  `TrailsEstimator` for no-survival ablation, copying 07's complete training config
  and changing only survival loss weight and the configured seed. Shared baselines
  write `BaselinePrediction` NPZ; R exchange files and checkpoints stay remote.
  Failed methods are recorded, other methods may finish, but the batch exits nonzero
  and is not accepted as a complete comparison. New run directories are required.
- `09_eval_cluster.py` and `09_eval_survival.py` replace the old `08_evaluate.py`.
  They accept any number of completed `baseline_dirs`, verify frozen-source hashes,
  branch between `TrailsPrediction` and baseline NPZ readers, and write method/seed/
  split artifacts below `evaluation/cluster` and `evaluation/survival` respectively.
  `evaluation.py` retains shared plotting/calculation classes. Cluster evaluation
  reports occupancy, entropy, KM/log-rank, adjusted Cox, clinical characteristics,
  trajectories and label agreement; it never generates cluster-only survival
  predictions or predictive C-index/AUC/IBS/calibration from cluster KM curves.
  Survival evaluation reports Harrell/IPCW C-index, cumulative/dynamic AUC, daily
  Brier/IBS and quantile-group KM calibration, always estimating censoring from train.
  Both write unified comparison tables. Degenerate clusters and unestimable Cox
  effects remain explicit diagnostics, not silently relabeled successes.
- `AdjustedCoxAnalysis` uses the train split's lowest observed KM mortality cluster as the
  shared validation/test reference and reports cluster hazard ratios adjusted for
  age, gender, grouped race, and sepsis-onset SOFA with lifelines' default Efron
  ties handling; forest plots show only the adjusted cluster effects. Each split
  also saves a descriptive clinical-characteristics table with Overall and every
  configured cluster column; it reports age as mean (SD), SOFA as median [IQR],
  and gender, grouped race, and numeric missingness as n (%) without hypothesis
  tests. `ClusterTrajectoryAnalysis` restores longitudinal values to clinical
  units using the train-fitted preprocessing parameters, bins the 0–48 hour
  window at configurable four-hour intervals, first takes each patient's median
  within a feature-bin, and then saves cluster median/IQR tables and PNG/PDF
  trajectory panels for validation and test.
- In generic `scripts/case.py`, `k_selection.enabled=true` runs estimator-level holdout K selection before
  final case training. Empty `k_selection.candidate_clusters` means
  `2..model.n_clusters`; candidates are scored by validation C-index and
  latent MoG BIC using `sqrt(CI^2 + (1 - BIC_norm)^2)`. Candidate training and
  selection metrics share the same holdout validation split. Case runs inherit
  the best candidate estimator instead of retraining on all patients, and
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

<!-- pair-programming:active:start -->
## 结对编程（已启用）

本项目默认使用已安装的全局 `$pair-programming` skill。除非用户在当前请求中明确指定不用，否则所有编码任务开始前都必须自动调用该 skill，并读取和维护 `.pair/PAIR.md` 及其当前任务文件。Codex 担任 driver，用户担任 navigator。

- `.pair/` 默认由项目根 `.gitignore` 排除；`.pair/PAIR.md` 只保存任务索引，每个任务的计划树和过程保存在 `.pair/tasks/` 的独立 Markdown 文件中。
- 每批手写代码、测试和配置的新增与修改不得超过 navigator 确认的行数（默认 200 行），而且必须形成语法完整、逻辑完整、可审查的最小单元；纯删除不限行数，同一重构中尽量先删除已确认废弃的内容，再添加替代代码。
- 每批完成后进行必要的最小验证，向 navigator 汇报改动和验证结果，然后停止并等待审查；未经通过不得开始下一批。
- 优先采用简单务实的实现、中文解释性注释和静态类型；不添加当前需求之外的抽象、工具函数、测试或校验。
- navigator 明确说明、暂存或提交的代码改动视为已确认决定，后续不得回退；未说明且未暂存的意外差异应先询问确认。
- 每次编码前同步索引和当前任务树；并行任务、暂停点、风险、决定、交接和完成记忆均按 skill 约定记录。
<!-- pair-programming:active:end -->
