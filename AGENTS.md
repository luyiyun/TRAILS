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

- `main.py`: all CLI commands. Use `uv run main.py ...`.
- `configs/`: Hydra simulation, training, baseline, and optimization configuration.
- `src/trails/`: reusable core code: data, model, trainer, estimator, metrics.
- `src/trails_simulate/`: synthetic clinical data generation; imports `trails`.
- `src/trails_case/`: future real-data/case-study utilities; imports `trails`.
- `tests/`: smoke, data, model, estimator, CLI, and architecture tests.

## Import Boundaries

Allowed dependencies:

- `main.py -> trails`
- `main.py -> trails_simulate`
- `trails_simulate -> trails`
- `trails_case -> trails`

Forbidden dependencies:

- `trails -> trails_simulate`
- `trails -> trails_case`
- `trails -> main`

Keep command orchestration out of `src/trails`; the main package should remain a
clean reusable method library.

## Commands

- Quick simulation split generation: `uv run main.py command=simulate simulation=quick paths.data_root=data/simulated`
- Paper simulation grid generation for one scene: `uv run main.py command=simulate simulation=base paths.data_root=data/simulated`
- Train existing splits: `uv run main.py command=train training=base paths.data_root=data/simulated/base`
- Train with mTAN-style input: `uv run main.py command=train training=mtan paths.data_root=data/simulated/base`
- Run lightweight baselines on existing splits: `uv run main.py command=baseline paths.data_root=data/simulated/base`
- Run Optuna tuning on existing splits: `uv run main.py command=optim paths.data_root=data/simulated/base`
- Summarize train and baseline results: `uv run main.py command=summary summary.train_root=outputs/train-... summary.baseline_root=outputs/baseline-...`
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
- Inside `src/trails`, prefer relative imports for project modules.
- Inside `src/trails_simulate` and `src/trails_case`, use absolute imports from
  `trails`.
- 在复杂研究逻辑处使用简洁中文注释，尤其是模拟机制、模型结构和损失函数；
  不要给显而易见的赋值或样板代码写流水账式注释。
- 如果一段代码只在局部流程中使用一次，不需要为了形式拆成额外函数；在相关
  内容附近添加简洁中文注释说明意图即可。
- Avoid `try/except` unless the code can recover or add useful diagnostics.
- Add or update tests for every substantial behavior change.
- Update this file when architecture decisions change.

## Current Decisions

- Package name: `trails`.
- CLI lives only in root `main.py`; no package console script is configured.
- `main.py` is a Hydra app. Default `command=simulate` generates train/test
  split simulation data. Training and baseline comparison are separate commands.
- Simulation scenarios live under `configs/simulation/`: `quick`, `base`,
  `imbalance`, `censored`, and `high_dimension`. `quick` is for smoke tests;
  the other four are paper simulation scenes.
- Training model presets live under `configs/training/`: `small`, `base`,
  `large`, and `mtan`. Training scene selection comes from `paths.data_root` or
  explicit `paths.data` plus `paths.test_data`, not from simulation config.
- `training.trainer.batch_size: null` means the trainer resolves batch size from
  the loaded training split size using a conservative automatic rule; explicit
  integer overrides keep their exact value.
- Command-level config namespaces are `simulation`, `training`, `baseline`,
  `optim`, `summary`, and shared `paths`; generator parameters live under
  `simulation.generator`, while TRAILS model/trainer/artifacts/diagnostics/SwanLab
  parameters live under `training`.
- `simulation.train_size` and `simulation.test_size` are equal-length lists that
  are paired by position. Each paired sample-size level is crossed with
  `simulation.generator.n_clusters`, and each combination is repeated
  `simulation.repeats` times. Formal simulation scenes use train sizes
  `[500, 1000, 2000, 3000, 5000]` and fixed test size `300`; `quick` remains a
  small smoke-test configuration.
- `simulation.repeats` means paired split repeats within each sample-size and K
  combination: each repeat generates one source simulation dataset and splits it
  into train/test.
- Generator instantiation fixes DGP mechanism parameters using
  `simulation.mechanism_seed` when set, otherwise `simulation.seed`. The same
  `simulation.name × K` combination uses a fixed mechanism seed; sample seeds vary
  across sample-size levels and repeats for patient draws, train/test split
  shuffling. Train, baseline, and optim command seeds come from
  `training.trainer.seed` plus the discovered split index.
- Validation data is cut internally from `train.pt` by `training.trainer.valid_size`, is
  not saved as a separate `val.pt`, and is used for early stopping; if no validation
  split is requested, early stopping monitors the training metric instead.
- Hydra outputs go under `outputs/` by default and are ignored by git. Command run
  directories are named `command-<timestamp>`. Train, baseline, and optim outputs
  mirror the relative data split path discovered under `paths.data_root`.
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
- The phase-one model uses a modular encoder: an asynchronous input layer (`grud` or
  standard mTAN-style multi-time attention) followed by a nonlinear mapping layer
  (`gru`, `lstm`, or `transformer`) and SeqPool.
- The reconstruction decoder is configurable as `gru`, `lstm`, or
  `transformer`; recurrent decoders support latent-initialized hidden state or
  repeated latent plus visit-time input, while transformer decoding only uses
  repeated latent plus visit-time input.
- Clustering uses a VaDE-style learnable Gaussian mixture latent prior,
  initialized by deterministic k-means after warmup.
- The survival head remains a cluster-specific Weibull mixture whose mixture
  weights are the VaDE posterior cluster probabilities, with a configurable
  number of latent-width hidden layers before the Weibull output.
- Validation and test metrics include ACC/ARI/NMI only when true cluster labels are
  available; test metrics also report predicted-cluster occupancy diagnostics.
- Train and baseline commands recursively discover all sibling `train.pt`/`test.pt`
  directories under `paths.data_root`, infer K from dataset metadata when present,
  and save unified prediction payloads under mirrored run directories plus
  command-level metrics CSV and summary JSON.
- `command=baseline` lives in `trails_simulate` and runs lightweight simulation
  comparators on existing train/test splits: summary-feature k-means and
  risk-stratified summary-feature k-means, plus FPCA-KMeans via `scikit-fda`. It
  writes baseline summary JSON and metrics CSV under the Hydra run directory.
- `command=optim` recursively discovers existing train/test splits but runs only
  one split per invocation. Use `optim.run_id` for reproducible batch execution;
  otherwise the command interactively lists discovered split numbers for selection.
- `command=summary` reads explicit train and baseline Hydra run directories,
  combines `train_metrics.csv` and `baseline_metrics.csv`, aggregates metrics by
  scenario/sample size/K/method, and writes CSV/JSON plus publication-facing PNG
  figures under the summary Hydra run directory.

## Verification

Run these commands in order after code changes:

1. `uv run ruff format`
2. `uv run ruff check --fix`
3. `UV_CACHE_DIR=/tmp/uv-cache uv run pyright`
4. `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`

If any step fails, fix it before considering the change complete.
