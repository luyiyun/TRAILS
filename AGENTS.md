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
- `configs/`: Hydra experiment configuration and reusable scenarios.
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

- Quick end-to-end experiment: `uv run main.py scenario=quick`
- Normal SwanLab tuning run: `uv run main.py scenario=debug trainer.max_epochs=5 swanlab.mode=disabled`
- Formal repeated experiment: `uv run main.py scenario=formal_5x`
- Simulate only: `uv run main.py command=simulate scenario=quick paths.data_root=data/simulated/quick`
- Train existing splits: `uv run main.py command=train scenario=quick paths.data_root=data/simulated/quick`
- Format: `uv run ruff format`
- Lint: `uv run ruff check --fix`
- Type check: `UV_CACHE_DIR=/tmp/uv-cache uv run pyright`
- Tests: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`

## Coding Rules

- Use Python and uv.
- Use Hydra YAML plus CLI overrides for experiment parameters; do not reintroduce
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
- `main.py` is a Hydra app. Default `command=experiment` generates split
  simulation data and trains once per repeat.
- Common scenarios live under `configs/scenario/`: `quick`,
  `debug`, and `formal_5x`.
- `experiment.repeats` means paired repeats: each repeat generates one source
  simulation dataset, splits it into train/test, and trains on that pair.
- Repeat seeds use `experiment.seed + repeat_index`; the same repeat seed drives
  source simulation, train/test split shuffling, and model training.
- Validation data is cut internally from `train.pt` by `trainer.valid_size` and
  is not saved as a separate `val.pt` by `command=experiment`.
- Hydra outputs go under `outputs/` by default and are ignored by git.
- Simulation uses a VaDeSC-EHR-style latent-cluster generator adapted to
  continuous asynchronous clinical measurements.
- Simulation outputs a `ClinicalTimeSeriesDataset` saved via `torch.save`.
- Each patient sample contains `times`, `x`, `mask`, `delta_time`,
  `survival_time`, `event`, and optional `cluster_label`.
- Dataset metadata preserves latent profiles, cluster parameters, survival
  coefficients, and generation parameters for simulation-study evaluation.
- The phase-one model uses a modular encoder: an asynchronous input layer
  (`grud` or lightweight `mtan`) followed by a nonlinear mapping layer (`gru`,
  `lstm`, or `transformer`) and SeqPool.
- The reconstruction decoder is configurable as `gru`, `lstm`, or
  `transformer`; recurrent decoders support latent-initialized hidden state or
  repeated latent plus visit-time input, while transformer decoding only uses
  repeated latent plus visit-time input.
- Clustering uses a VaDE-style learnable Gaussian mixture latent prior,
  initialized by deterministic k-means after warmup.
- The survival head remains a cluster-specific Weibull mixture whose mixture
  weights are the VaDE posterior cluster probabilities, with a configurable
  number of latent-width hidden layers before the Weibull output.
- Validation and test metrics include ARI/NMI only when true cluster labels are
  available; test metrics also report predicted-cluster occupancy diagnostics.

## Verification

Run these commands in order after code changes:

1. `uv run ruff format`
2. `uv run ruff check --fix`
3. `UV_CACHE_DIR=/tmp/uv-cache uv run pyright`
4. `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`

If any step fails, fix it before considering the change complete.
