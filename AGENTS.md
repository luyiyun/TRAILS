# Project Guide

## Purpose

TRAILS studies deep survival trajectory clustering for asynchronous multivariate
medical longitudinal data. The method targets patient subtypes that differ in
both clinical trajectories and time-to-event risk.

The current scope is phase one: a GRU-D Surv-VaDER/VaDE prototype for
variable-length clinical visit sequences with `x`, `mask`, and `delta_time`.
mTAN, mixed-type likelihoods, competing risks, and recurrent events are later
roadmap items.

## Layout

- `main.py`: all CLI commands. Use `uv run main.py ...`.
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

- Simulate: `uv run main.py simulate --out data/simulated/demo.pt --patients 128 --clusters 3 --seed 2026`
- Simulate train/validation/test splits: `uv run main.py simulate --out data/simulated/demo --split-patients 128 32 32 --clusters 3 --seed 2026`
- Train: `uv run main.py train --data data/simulated/demo.pt --epochs 1 --batch-size 16`
- Train with validation metrics: `uv run main.py train --data data/simulated/demo.pt --val-data data/simulated/demo.pt --epochs 1 --warmup-epochs 1 --batch-size 16`
- Train with SwanLab logging: add `--swanlab --swanlab-project TRAILS --swanlab-experiment debug-demo`
- Format: `uv run ruff format`
- Lint: `uv run ruff check --fix`
- Type check: `UV_CACHE_DIR=/tmp/uv-cache uv run pyright`
- Tests: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`

## Coding Rules

- Use Python and uv.
- All new Python code must have type annotations.
- `pyright` uses `standard` mode.
- Inside `src/trails`, prefer relative imports for project modules.
- Inside `src/trails_simulate` and `src/trails_case`, use absolute imports from
  `trails`.
- 在复杂研究逻辑处使用简洁中文注释，尤其是模拟机制、模型结构和损失函数；
  不要给显而易见的赋值或样板代码写流水账式注释。
- Avoid `try/except` unless the code can recover or add useful diagnostics.
- Add or update tests for every substantial behavior change.
- Update this file when architecture decisions change.

## Current Decisions

- Package name: `trails`.
- CLI lives only in root `main.py`; no package console script is configured.
- Simulation uses a VaDeSC-EHR-style latent-cluster generator adapted to
  continuous asynchronous clinical measurements.
- Simulation outputs a `ClinicalTimeSeriesDataset` saved via `torch.save`.
- Each patient sample contains `times`, `x`, `mask`, `delta_time`,
  `survival_time`, `event`, and optional `cluster_label`.
- Dataset metadata preserves latent profiles, cluster parameters, survival
  coefficients, and generation parameters for simulation-study evaluation.
- The phase-one model uses GRU-D as encoder and GRU as decoder.
- Clustering uses a VaDE-style learnable Gaussian mixture latent prior,
  initialized by deterministic k-means after warmup.
- The survival head remains a cluster-specific Weibull mixture whose mixture
  weights are the VaDE posterior cluster probabilities.
- Validation and test metrics include ARI/NMI only when true cluster labels are
  available.

## Verification

Run these commands in order after code changes:

1. `uv run ruff format`
2. `uv run ruff check --fix`
3. `UV_CACHE_DIR=/tmp/uv-cache uv run pyright`
4. `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`

If any step fails, fix it before considering the change complete.
