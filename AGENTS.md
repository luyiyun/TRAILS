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
- `configs/`: Hydra simulation/training configuration and reusable scenarios.
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

- Quick simulation split generation: `uv run main.py scenario=quick`
- Normal SwanLab tuning run: `uv run main.py command=train scenario=debug paths.data_root=data/simulated/debug training.trainer.max_epochs=5 training.swanlab.mode=disabled`
- Formal repeated simulation: `uv run main.py scenario=formal_5x`
- Simulate only: `uv run main.py command=simulate scenario=quick paths.data_root=data/simulated/quick`
- Train existing splits: `uv run main.py command=train scenario=quick paths.data_root=data/simulated/quick`
- Run lightweight baselines on existing splits: `uv run main.py command=baseline scenario=quick paths.data_root=data/simulated/quick`
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
- Common scenarios live under `configs/scenario/`: `quick`,
  `debug`, and `formal_5x`.
- Command-level config namespaces are `simulation`, `training`, `baseline`,
  `optim`, and shared `paths`; generator parameters live under
  `simulation.generator`, while TRAILS model/trainer/artifacts/diagnostics/SwanLab
  parameters live under `training`.
- `simulation.repeats` means paired split repeats: each repeat generates one
  source simulation dataset and splits it into train/test.
- Generator instantiation fixes DGP mechanism parameters using
  `simulation.mechanism_seed` when set, otherwise `simulation.seed`; repeat sample
  seeds use `simulation.seed + repeat_index` for patient draws, train/test split
  shuffling, and downstream model training.
- Validation data is cut internally from `train.pt` by `training.trainer.valid_size`, is
  not saved as a separate `val.pt`, and is used for early stopping; if no validation
  split is requested, early stopping monitors the training metric instead.
- Hydra outputs go under `outputs/` by default and are ignored by git. Command run
  directories are named `command-<timestamp>`, with repeat outputs flattened under
  numeric directories such as `0/`, `1/`, and `2/`.
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
- Train and baseline commands save unified prediction payloads directly under each
  numeric repeat directory, plus command-level metrics CSV and summary JSON.
- `command=baseline` lives in `trails_simulate` and runs lightweight simulation
  comparators on existing train/test splits: summary-feature k-means and
  risk-stratified summary-feature k-means. It writes baseline summary JSON and
  metrics CSV under the Hydra run directory.

## Verification

Run these commands in order after code changes:

1. `uv run ruff format`
2. `uv run ruff check --fix`
3. `UV_CACHE_DIR=/tmp/uv-cache uv run pyright`
4. `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`

If any step fails, fix it before considering the change complete.
