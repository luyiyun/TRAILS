"""冻结CRC云南数据上的分片搜索；模型、标签和患者数据只留在执行服务器。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, BinaryIO

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import adjusted_rand_score

from trails import (
    ClinicalTimeSeriesDataset,
    ClusterNumberSelector,
    TrailsConfig,
    TrailsEstimator,
)
from trails.trainer import HistoryEntry

from .config import CRCEvaluationConfig, CRCSearchConfig, CRCSearchJob, CRCSearchRecipe
from .evaluation import cluster_statistics, cluster_survival, survival_evaluation

LOGGER = logging.getLogger(__name__)
NUMERICAL_EXIT = 42


##################################################
# 一、生成固定配方并解析核心模型参数
##################################################

FACTORS: dict[str, list[Any]] = {
    "warmup": [10, 0, 30, 100],
    "objective": ["reconstruction_survival", "reconstruction"],
    "initialization": ["legacy5", "legacy50", "kmeans", "kmeans_global", "gmm"],
    "variance_floor": [0.0001, 0.01],
    "prior": ["empirical", "uniform"],
    "train_prior": [False, True],
    "reset_head": [False, True],
    "reset_optimizer": [False, True],
    "weighting": ["uncertainty_continue", "fixed", "uncertainty_reset", "uncertainty_freeze"],
    "survival_weight": [0.2, 0.05, 1.0],
    "kl_weight": [0.1, 0.01, 1.0],
    "learning_rate": [0.001, 0.0001, 0.0003, 0.003],
    "scheduler": ["constant", "cosine", "plateau"],
    "architecture": ["transformer", "gru", "lstm", "mtan"],
    "width": [64, 32],
    "latent": [32, 8, 16],
    "dropout": [0.0, 0.1],
    "survival_layers": [1, 0, 2],
    "batch": [128, 64, 256],
    "monitor": ["loss", "cindex"],
}


def fingerprint(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def recipes(config: CRCSearchConfig) -> dict[str, CRCSearchRecipe]:
    baseline = CRCSearchRecipe()
    values = [baseline]
    for factor, candidates in FACTORS.items():
        for value in candidates[1:]:
            values.append(CRCSearchRecipe.model_validate({**baseline.model_dump(), factor: value}))
    seen = {fingerprint(value.model_dump()) for value in values}
    if len(values) != 38 or len(seen) != 38:
        raise RuntimeError("基准和单因素配方必须恰为38套且不重复")
    rng = np.random.default_rng(config.search.sampling_seed)
    while len(values) < 96:
        candidate = CRCSearchRecipe.model_validate(
            {name: options[int(rng.integers(len(options)))] for name, options in FACTORS.items()}
        )
        key = fingerprint(candidate.model_dump())
        if key not in seen:
            values.append(candidate)
            seen.add(key)
    return {f"r{index:03d}": value for index, value in enumerate(values[: config.search.n_recipes])}


def estimator_config(
    config: CRCSearchConfig, recipe: CRCSearchRecipe, job: CRCSearchJob
) -> TrailsConfig:
    model = config.model.model_dump()
    model.update(
        n_clusters=job.n_clusters,
        latent_dim=recipe.latent,
        dropout=recipe.dropout,
        survival_head_hidden_layers=recipe.survival_layers,
        mixture_logits_trained=recipe.train_prior,
    )
    model["loss"].update(
        weighting="fixed" if recipe.weighting == "fixed" else "uncertainty",
        reconstruction_weight=1.0,
        survival_weight=recipe.survival_weight,
        cluster_weight=recipe.kl_weight,
    )
    recurrent = recipe.architecture != "transformer"
    mapping = "gru" if recipe.architecture == "mtan" else recipe.architecture
    model["encoder"]["input"].update(
        kind="mtan" if recipe.architecture == "mtan" else "grud",
        hidden_dim=recipe.width,
        n_heads=2,
        num_ref_points=16,
        time_embedding_kind="mtan" if recipe.architecture == "mtan" else "projection",
    )
    model["encoder"]["mapping"].update(
        kind=mapping,
        hidden_dim=recipe.width,
        n_layers=1 if recurrent else 2,
        n_heads=2,
    )
    model["decoder"].update(
        kind=mapping,
        hidden_dim=recipe.width,
        n_layers=1 if recurrent else 2,
        n_heads=2,
        conditioning="initial_state" if recurrent else "concat_time",
    )
    trainer = config.trainer.model_dump()
    trainer.update(
        seed=job.seed,
        valid_size=0.0,
        warmup_epochs=recipe.warmup,
        warmup_objective=recipe.objective,
        batch_size=recipe.batch,
        learning_rate=recipe.learning_rate,
        early_stopping_monitor=recipe.monitor,
        gmm_init_iters=50 if recipe.initialization == "legacy50" else 5,
        mixture_init="legacy"
        if recipe.initialization.startswith("legacy")
        else recipe.initialization,
        mixture_init_n_init=5 if recipe.initialization == "gmm" else 10,
        mixture_init_max_iter=200,
        mixture_variance_floor=recipe.variance_floor,
        mixture_prior=recipe.prior,
        initialize_without_warmup=True,
        reset_survival_after_warmup=recipe.reset_head,
        reset_optimizer_after_warmup=recipe.reset_optimizer,
        uncertainty_after_warmup=recipe.weighting.removeprefix("uncertainty_")
        if recipe.weighting != "fixed"
        else "continue",
        lr_scheduler=recipe.scheduler,
        scheduler_min_lr_ratio=0.01,
        scheduler_factor=0.5,
        scheduler_patience=10,
        cindex_risk_score="median_survival",
        epoch_diagnostics=True,
    )
    return TrailsConfig.model_validate(
        {
            "seed": job.seed,
            "data": {"n_features": config.search.n_features},
            "model": model,
            "trainer": trainer,
        }
    )


def protocol(config: CRCSearchConfig) -> dict[str, Any]:
    search = config.search.model_dump(exclude={"recipe_ids"})
    return {
        "version": 1,
        "search": search,
        "split_seed": config.split.seed,
        "split_strategy": config.split.strategy,
        "survival": config.survival.model_dump(),
        "shards": config.shard.count,
        "recipes": {key: value.model_dump() for key, value in recipes(config).items()},
        "core": estimator_config(
            config,
            CRCSearchRecipe(),
            CRCSearchJob(
                recipe_id="r000",
                n_clusters=config.search.screen_k,
                seed=config.search.screen_seeds[0],
            ),
        ).model_dump(mode="json"),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    serialized = frame.to_json(orient="records")
    assert serialized is not None
    return json.loads(serialized)


def data_fingerprints(config: CRCSearchConfig) -> dict[str, str]:
    root = config.split.dir.resolve()
    files = [root / name / "dataset.pt" for name in ("train", "validation")]
    files += [root / "preprocessing_parameters.csv", root / "preproc_manifest.json"]
    hashes: dict[str, str] = {}
    for path in files:
        with path.open("rb") as handle:
            hashes[str(path.relative_to(root))] = hashlib.file_digest(handle, "sha256").hexdigest()
    return hashes


def selector(
    config: CRCSearchConfig,
    seeds: list[int],
    candidates: list[int],
    recipe: CRCSearchRecipe | None = None,
) -> ClusterNumberSelector:
    return ClusterNumberSelector(
        candidates,
        seeds=seeds,
        split_seed=config.split.seed,
        require_non_empty=True,
        min_cluster_fraction=config.search.min_cluster_fraction,
        selection_rule="one_standard_error",
        compute_stability=True,
        estimator_config=estimator_config(
            config,
            recipe or CRCSearchRecipe(),
            CRCSearchJob(
                recipe_id="r000",
                n_clusters=config.search.screen_k,
                seed=seeds[0],
            ),
        ),
    )


##################################################
# 二、独立进程训练及非有限数值的有限恢复
##################################################


def run_worker(config: CRCSearchConfig) -> None:
    root = config.paths.dir.resolve()
    job = config.job
    recipe = recipes(config)[job.recipe_id]
    core = estimator_config(config, recipe, job)
    identity = {
        "protocol_hash": fingerprint(protocol(config)),
        "data_hashes": data_fingerprints(config),
        "recipe_id": job.recipe_id,
        "seed": job.seed,
        "n_clusters": job.n_clusters,
        "config": core.model_dump(mode="json"),
    }
    write_json(root / "identity.json", identity)
    OmegaConf.save(core.model_dump(mode="json"), root / "resolved_config.yaml")
    train = ClinicalTimeSeriesDataset.load(config.split.dir / "train/dataset.pt")
    validation = ClinicalTimeSeriesDataset.load(config.split.dir / "validation/dataset.pt")
    if len(train) != config.search.train_count or len(validation) != config.search.validation_count:
        raise ValueError("冻结train/validation人数不符合实验协议")
    torch.set_num_threads(config.shard.threads)
    torch.set_num_interop_threads(config.shard.threads)
    started = time.monotonic()
    if core.trainer.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(core.trainer.device))
        torch.cuda.reset_peak_memory_stats(core.trainer.device)
    history: list[HistoryEntry] = []

    def save_epoch(entry: HistoryEntry) -> None:
        history.append(entry)
        frame = pd.json_normalize([dict(item) for item in history])
        temporary = root / "history.csv.tmp"
        frame.to_csv(temporary, index=False)
        temporary.replace(root / "history.csv")
        write_json(
            root / "progress.json",
            {
                "stage": entry["stage"],
                "epoch": entry["epoch"],
                "global_epoch": entry["global_epoch"],
                "elapsed_seconds": time.monotonic() - started,
            },
        )

    estimator = TrailsEstimator(core)
    try:
        estimator.fit(train, validation_data=validation, history_callback=save_epoch)
        metrics = ClusterNumberSelector.evaluate_fitted(estimator, validation)
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Non-finite fitted validation metrics.")
    except (FloatingPointError, RuntimeError) as error:
        # 仅明确的非有限损失/梯度可恢复；其他错误保留traceback并停止机器队列。
        numerical = isinstance(error, FloatingPointError) or (
            "non-finite" in str(error).lower() and "norm" in str(error).lower()
        )
        if not numerical:
            raise
        traceback.print_exc()
        pd.DataFrame(estimator.trainer.diagnostic_history).to_csv(
            root / "diagnostics.csv", index=False
        )
        write_json(
            root / "result.json",
            {
                **identity,
                "status": "numerical_failure",
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise SystemExit(NUMERICAL_EXIT) from error
    estimator.save(root / "model.pt")
    with (root / "model.pt").open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    write_json(root / "history.json", history)
    write_json(root / "initialization.json", estimator.trainer.initialization_metrics)
    pd.DataFrame(estimator.trainer.diagnostic_history).to_csv(root / "diagnostics.csv", index=False)
    torch.save(estimator.predict(validation).predict(), root / "validation_labels.pt")
    peak = (
        float(torch.cuda.max_memory_allocated(core.trainer.device))
        if core.trainer.device.startswith("cuda")
        else 0.0
    )
    write_json(
        root / "result.json",
        {
            **identity,
            **metrics,
            "status": "completed",
            "model_path": str(root / "model.pt"),
            "checkpoint_sha256": checkpoint_hash,
            "labels_path": str(root / "validation_labels.pt"),
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_bytes": peak,
            "epochs": len(history),
            "best_global_epoch": history[-1].get("best_global_epoch"),
        },
    )


@dataclass
class ActiveTrial:
    job: CRCSearchJob
    root: Path
    process: subprocess.Popen[bytes]
    lock: BinaryIO
    log: BinaryIO


class SearchQueue:
    """有限并发的独立进程队列；锁随子进程继承，断线续跑不会重叠训练。"""

    def __init__(self, config: CRCSearchConfig) -> None:
        self.config = config
        self.root = config.paths.dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.protocol_hash = fingerprint(protocol(config))
        self.hashes = data_fingerprints(config)
        self.recipes = recipes(config)
        self.active: list[ActiveTrial] = []
        self.started = time.monotonic()

    def trial_root(self, job: CRCSearchJob) -> Path:
        return self.root / "trials" / job.recipe_id / f"k{job.n_clusters}" / f"seed-{job.seed}"

    def check_result(self, path: Path, job: CRCSearchJob) -> dict[str, Any] | None:
        if not (path / "result.json").exists():
            return None
        result = read_json(path / "result.json")
        expected = estimator_config(self.config, self.recipes[job.recipe_id], job).model_dump(
            mode="json"
        )
        if (
            result["protocol_hash"] != self.protocol_hash
            or result["data_hashes"] != self.hashes
            or result["config"] != expected
        ):
            raise ValueError(f"已存在试验的数据/参数与当前协议不同：{path}")
        if result["status"] == "completed" and not all(
            (path / name).is_file()
            for name in ("model.pt", "validation_labels.pt", "history.csv", "diagnostics.csv")
        ):
            raise FileNotFoundError(f"已完成试验缺少必要产物：{path}")
        return result

    def start(self, job: CRCSearchJob) -> ActiveTrial:
        root = self.trial_root(job)
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / ".trial.lock").open("a+b")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker = self.config.model_dump(mode="json")
        worker.update(mode="worker", job=job.model_dump())
        worker["paths"]["dir"] = str(root)
        worker["split"]["dir"] = str(self.config.split.dir.resolve())
        worker["hydra"] = {"run": {"dir": str(root)}, "job": {"chdir": False}}
        OmegaConf.save(worker, root / "worker.yaml")
        environment = os.environ.copy()
        environment.update(
            {
                name: str(self.config.shard.threads)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            }
        )
        environment["HYDRA_FULL_ERROR"] = "1"
        log = (root / "trial.log").open("ab")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.crc_yunnan.07_search",
                "--config-path",
                str(root),
                "--config-name",
                "worker",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            pass_fds=(lock.fileno(),),
        )
        write_json(root / "process.json", {"pid": process.pid, "job": job.model_dump()})
        LOGGER.info(
            "启动 %s K=%d seed=%d PID=%d", job.recipe_id, job.n_clusters, job.seed, process.pid
        )
        return ActiveTrial(job, root, process, lock, log)

    def run(self, jobs: list[CRCSearchJob], reuse: dict[tuple[str, int, int], Path]) -> None:
        queue = deque(jobs)
        with (self.root / ".queue.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            payload = protocol(self.config)
            existing = self.root / "protocol.json"
            if existing.exists() and read_json(existing) != payload:
                raise ValueError("输出目录已有不同协议，须使用新目录")
            write_json(existing, payload)
            stopped: list[dict[str, Any]] = []
            while queue or self.active:
                while queue and len(self.active) < self.config.shard.workers and not stopped:
                    job = queue.popleft()
                    source = reuse.get(
                        (job.recipe_id, job.n_clusters, job.seed), self.trial_root(job)
                    )
                    result = self.check_result(source, job)
                    if result is not None:
                        if not self.config.resume:
                            raise FileExistsError(f"试验已存在且resume=false：{source}")
                        continue
                    if (source / "exit.json").exists():
                        raise RuntimeError(f"先前基础设施/未知错误须先处理，保留日志：{source}")
                    if source != self.trial_root(job):
                        raise FileNotFoundError(f"冻结的复用试验不存在：{source}")
                    self.active.append(self.start(job))
                for trial in self.active.copy():
                    code = trial.process.poll()
                    if code is None:
                        continue
                    trial.log.close()
                    trial.lock.close()
                    self.active.remove(trial)
                    write_json(trial.root / "exit.json", {"exit_code": code})
                    result = self.check_result(trial.root, trial.job)
                    if code not in (0, NUMERICAL_EXIT) or result is None:
                        stopped.append(
                            {
                                "job": trial.job.model_dump(),
                                "exit_code": code,
                                "log": str(trial.root / "trial.log"),
                            }
                        )
                    elif (code == 0) != (result["status"] == "completed"):
                        raise RuntimeError(f"进程退出码与试验终态冲突：{trial.root}")
                self.summarize(jobs, reuse, stopped)
                if stopped and not self.active:
                    raise RuntimeError(f"机器队列停止，保留原始日志：{stopped}")
                if self.active:
                    time.sleep(2)
            self.export_diagnostics(jobs, reuse)

    def export_diagnostics(
        self, jobs: list[CRCSearchJob], reuse: dict[tuple[str, int, int], Path]
    ) -> None:
        stages: list[pd.DataFrame] = []
        epochs: list[pd.DataFrame] = []
        initializations: list[dict[str, Any]] = []
        for job in jobs:
            source = reuse.get((job.recipe_id, job.n_clusters, job.seed), self.trial_root(job))
            for name, frames in (("history.csv", epochs), ("diagnostics.csv", stages)):
                path = source / name
                if path.exists() and path.stat().st_size > 1:
                    frames.append(pd.read_csv(path).assign(**job.model_dump()))
            if (source / "initialization.json").exists():
                initializations.append(
                    {**job.model_dump(), **read_json(source / "initialization.json")}
                )
        for name, frames in (("epoch_metrics.csv", epochs), ("stage_diagnostics.csv", stages)):
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(self.root / name, index=False)
        pd.DataFrame(initializations).to_csv(self.root / "initialization_metrics.csv", index=False)

    def summarize(
        self,
        jobs: list[CRCSearchJob],
        reuse: dict[tuple[str, int, int], Path],
        stopped: list[dict[str, Any]],
    ) -> None:
        rows: list[dict[str, Any]] = []
        labels: dict[tuple[str, int, int], np.ndarray] = {}
        for job in jobs:
            key = (job.recipe_id, job.n_clusters, job.seed)
            root = reuse.get(key, self.trial_root(job))
            result = self.check_result(root, job)
            row: dict[str, Any] = {
                **job.model_dump(),
                "status": "pending",
                "owner": self.config.shard.index,
                "trial_dir": str(root),
                "protocol_hash": self.protocol_hash,
                "data_hash": fingerprint(self.hashes),
            }
            if result is not None:
                row.update(
                    {
                        name: value
                        for name, value in result.items()
                        if name not in ("config", "data_hashes")
                    }
                )
                if result["status"] == "completed":
                    labels[key] = torch.load(result["labels_path"], weights_only=True).numpy()
            elif (root / "exit.json").exists():
                row.update(status="failed", exit_code=read_json(root / "exit.json")["exit_code"])
            rows.append(row)
        metrics = pd.DataFrame(rows)
        pairs: list[dict[str, Any]] = []
        for _, group in metrics.groupby(["recipe_id", "n_clusters"]):
            recipe_id = group["recipe_id"].iloc[0]
            k = group["n_clusters"].iloc[0]
            for a, b in combinations(sorted(group["seed"].tolist()), 2):
                first, second = (str(recipe_id), int(k), a), (str(recipe_id), int(k), b)
                if first in labels and second in labels:
                    pairs.append(
                        {
                            "recipe_id": recipe_id,
                            "n_clusters": k,
                            "seed_a": a,
                            "seed_b": b,
                            "ari": adjusted_rand_score(labels[first], labels[second]),
                        }
                    )
        stability = pd.DataFrame(
            pairs, columns=["recipe_id", "n_clusters", "seed_a", "seed_b", "ari"]
        )
        metrics.to_csv(self.root / "run_metrics.csv", index=False)
        stability.to_csv(self.root / "stability_pairs.csv", index=False)
        complete = metrics["status"].ne("pending")
        durations = (
            metrics.loc[metrics["status"].eq("completed"), "elapsed_seconds"]
            if "elapsed_seconds" in metrics
            else pd.Series(dtype=float)
        )
        eta = (
            float(durations.mean() * (~complete).sum() / self.config.shard.workers)
            if len(durations)
            else None
        )
        running: list[dict[str, Any]] = []
        for trial in self.active:
            progress = trial.root / "progress.json"
            running.append(
                {
                    **trial.job.model_dump(),
                    "pid": trial.process.pid,
                    **(read_json(progress) if progress.exists() else {}),
                }
            )
        write_json(
            self.root / "progress.json",
            {
                "total": len(jobs),
                "completed": int(np.asarray(metrics["status"].eq("completed")).sum()),
                "numerical_failure": int(
                    np.asarray(metrics["status"].eq("numerical_failure")).sum()
                ),
                "pending": int(np.asarray(~complete).sum()),
                "running": running,
                "stopped": stopped,
                "elapsed_seconds": time.monotonic() - self.started,
                "eta_seconds": eta,
            },
        )
        write_json(
            self.root / "summary.json",
            {
                "protocol_hash": self.protocol_hash,
                "data_hash": fingerprint(self.hashes),
                "run_metrics": records(metrics),
                "stability_pairs": records(stability),
            },
        )


##################################################
# 三、冻结分片、汇总初筛并比较各配方内的K
##################################################


def screen_jobs(config: CRCSearchConfig) -> list[CRCSearchJob]:
    all_recipes = recipes(config)
    if set(config.search.recipe_ids) - all_recipes.keys():
        raise ValueError("recipe_ids含未知配方")
    return [
        CRCSearchJob(recipe_id=key, n_clusters=config.search.screen_k, seed=seed)
        for index, key in enumerate(all_recipes)
        if index % config.shard.count == config.shard.index
        and (not config.search.recipe_ids or key in config.search.recipe_ids)
        for seed in config.search.screen_seeds
    ]


def run_plan(config: CRCSearchConfig) -> None:
    root = config.paths.dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "protocol.json", protocol(config))
    pd.DataFrame(
        [
            {
                "recipe_id": key,
                "owner": index % config.shard.count,
                "kind": "baseline"
                if index == 0
                else "single_factor"
                if index < 38
                else "combination",
                **value.model_dump(),
            }
            for index, (key, value) in enumerate(recipes(config).items())
        ]
    ).to_csv(root / "recipes.csv", index=False)
    LOGGER.info(
        "配方%d套；当前分片%d/%d，初筛%d次；总初筛%d次",
        config.search.n_recipes,
        config.shard.index,
        config.shard.count,
        len(screen_jobs(config)),
        config.search.n_recipes * len(config.search.screen_seeds),
    )


def merge(config: CRCSearchConfig) -> None:
    root = config.paths.dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    payloads = [read_json(path) for path in config.inputs.summaries]
    if not payloads or any(
        value["protocol_hash"] != fingerprint(protocol(config)) for value in payloads
    ):
        raise ValueError("汇总输入为空或与当前协议不同")
    if len({value["data_hash"] for value in payloads}) != 1:
        raise ValueError("两机数据指纹不同")
    metrics = pd.DataFrame([row for value in payloads for row in value["run_metrics"]])
    pairs = pd.DataFrame(
        [row for value in payloads for row in value["stability_pairs"]],
        columns=["recipe_id", "n_clusters", "seed_a", "seed_b", "ari"],
    )
    if metrics.duplicated(["recipe_id", "n_clusters", "seed"]).any():
        raise ValueError("分片重复或重复提供汇总文件")
    if pairs.duplicated(["recipe_id", "n_clusters", "seed_a", "seed_b"]).any():
        raise ValueError("稳定性分片重复")
    if not bool(np.asarray(metrics["status"].isin(["completed", "numerical_failure"])).all()):
        raise ValueError("还有未完成或基础设施失败单元，不能进行下一阶段")
    expected_ids = config.search.recipe_ids or list(recipes(config))
    if config.merge_stage == "screen":
        expected = {
            (key, config.search.screen_k, seed)
            for key in expected_ids
            for seed in config.search.screen_seeds
        }
        actual = set(
            metrics[["recipe_id", "n_clusters", "seed"]].itertuples(index=False, name=None)
        )
        if actual != expected:
            raise ValueError("初筛分片有遗漏或额外单元")
        owners = {key: index % config.shard.count for index, key in enumerate(recipes(config))}
        if any(row["owner"] != owners[row["recipe_id"]] for row in records(metrics)):
            raise ValueError("初筛owner不符合冻结分片")
        summaries: list[dict[str, Any]] = []
        for key, raw_group in metrics.groupby("recipe_id", sort=True):
            group: pd.DataFrame = raw_group
            success = bool(np.asarray(group["status"].eq("completed")).all())
            minimum = (
                float(np.asarray(group["cluster_min_fraction"], dtype=float).min())
                if "cluster_min_fraction" in group and success
                else 0.0
            )
            empty = (
                int(np.asarray(group["cluster_empty_count"], dtype=float).max())
                if "cluster_empty_count" in group and success
                else config.search.screen_k
            )
            ari = pairs.loc[pairs["recipe_id"].eq(key), "ari"]
            summaries.append(
                {
                    "recipe_id": key,
                    "passes_gate": bool(
                        success
                        and minimum >= config.search.min_cluster_fraction
                        and empty == 0
                        and len(ari) == 1
                    ),
                    "mean_pairwise_ari": float(ari.mean()),
                    "mean_cindex": float(np.asarray(group["cindex"], dtype=float).mean())
                    if "cindex" in group
                    else np.nan,
                    "min_cluster_fraction": minimum,
                    "max_empty_clusters": empty,
                }
            )
        table = pd.DataFrame(summaries).sort_values(
            [
                "passes_gate",
                "mean_pairwise_ari",
                "mean_cindex",
                "min_cluster_fraction",
                "recipe_id",
            ],
            ascending=[False, False, False, False, True],
        )
        finalists = table.loc[table["passes_gate"], "recipe_id"].head(config.search.top_n).tolist()
        loads = [0] * config.shard.count
        assignments: list[dict[str, Any]] = []
        for key in finalists:
            cells = metrics.loc[metrics["recipe_id"].eq(key)]
            owner = int(cells["owner"].iloc[0])
            if cells["owner"].nunique() != 1:
                raise ValueError("同配方seed分属不同机器")
            reuse = {str(int(row["seed"])): row["trial_dir"] for row in records(cells)}
            assignments.append(
                {
                    "recipe_id": key,
                    "n_clusters": config.search.screen_k,
                    "owner": owner,
                    "reuse": reuse,
                }
            )
            loads[owner] += len(config.search.validation_seeds) - len(reuse)
        for key in finalists:
            for k in config.search.candidates:
                if k == config.search.screen_k:
                    continue
                owner = min(range(config.shard.count), key=lambda index: (loads[index], index))
                assignments.append({"recipe_id": key, "n_clusters": k, "owner": owner, "reuse": {}})
                loads[owner] += len(config.search.validation_seeds)
        write_json(
            root / "validation_assignment.json",
            {
                "protocol_hash": fingerprint(protocol(config)),
                "data_hash": payloads[0]["data_hash"],
                "finalists": finalists,
                "groups": assignments,
                "new_training_counts": loads,
            },
        )
        table.to_csv(root / "recipe_summary.csv", index=False)
        write_json(
            root / "screen_selection.json",
            {"finalists": finalists, "recipes": records(table), "test_read": False},
        )
        if not finalists:
            write_json(
                root / "final_selection.json",
                {"selected": None, "ranked_recipes": [], "test_read": False},
            )
        LOGGER.info(
            "初筛完成%d次，占簇合格%d套；入围%s；新增训练分配%s",
            len(metrics),
            int(np.asarray(table["passes_gate"]).sum()),
            finalists,
            loads,
        )
    else:
        if config.inputs.assignment is None:
            raise ValueError("K汇总需要冻结的assignment")
        assignment = read_json(config.inputs.assignment)
        if (
            assignment["protocol_hash"] != fingerprint(protocol(config))
            or assignment["data_hash"] != payloads[0]["data_hash"]
        ):
            raise ValueError("K汇总的冻结分配与协议/数据不同")
        expected = {
            (item["recipe_id"], item["n_clusters"], seed)
            for item in assignment["groups"]
            for seed in config.search.validation_seeds
        }
        if (
            set(metrics[["recipe_id", "n_clusters", "seed"]].itertuples(index=False, name=None))
            != expected
        ):
            raise ValueError("K验证分片有遗漏或额外单元")
        owners = {
            (item["recipe_id"], item["n_clusters"]): item["owner"] for item in assignment["groups"]
        }
        if any(
            row["owner"] != owners[(row["recipe_id"], row["n_clusters"])]
            for row in records(metrics)
        ):
            raise ValueError("K验证owner不符合冻结分配")
        finalists: list[dict[str, Any]] = []
        for key in assignment["finalists"]:
            result = selector(
                config,
                config.search.validation_seeds,
                config.search.candidates,
                recipes(config)[key],
            ).summarize_completed(
                metrics.loc[metrics["recipe_id"].eq(key)],
                pairs.loc[pairs["recipe_id"].eq(key)],
            )
            result.save(root / key)
            if result.selected_k is None:
                continue
            row = (
                result.k_summary.loc[result.k_summary["n_clusters"].eq(result.selected_k)]
                .iloc[0]
                .to_dict()
            )
            finalists.append({"recipe_id": key, **row})
        ranked = pd.DataFrame(finalists)
        selected: dict[str, Any] | None = None
        if not ranked.empty:
            ranked = ranked.sort_values(
                ["mean_pairwise_ari", "mean_cindex", "min_cluster_fraction", "recipe_id"],
                ascending=[False, False, False, True],
            )
            best = ranked.iloc[0]
            key, k = str(best["recipe_id"]), int(best["n_clusters"])
            cells = pd.read_csv(root / key / "run_metrics.csv")
            cell = (
                cells.loc[cells["n_clusters"].eq(k)]
                .sort_values(
                    ["selection_score", "cindex", "latent_mixture_bic", "seed"],
                    ascending=[False, False, True, True],
                )
                .iloc[0]
            )
            selected = {
                "recipe_id": key,
                "n_clusters": k,
                "seed": int(cell["seed"]),
                "owner": int(cell["owner"]),
                "model_path": str(cell["model_path"]),
                "checkpoint_sha256": str(cell["checkpoint_sha256"]),
                "protocol_hash": fingerprint(protocol(config)),
                "data_hash": payloads[0]["data_hash"],
            }
            core = estimator_config(
                config,
                recipes(config)[key],
                CRCSearchJob(recipe_id=key, n_clusters=k, seed=selected["seed"]),
            )
            OmegaConf.save(core.model_dump(mode="json"), root / "recommended_recipe.yaml")
        ranked.to_csv(root / "validated_recipes.csv", index=False)
        write_json(
            root / "final_selection.json",
            {"selected": selected, "ranked_recipes": records(ranked), "test_read": False},
        )
        LOGGER.info("K验证完成%d单元；锁定%s；test尚未读取", len(metrics), selected)
    metrics.to_csv(root / "run_metrics.csv", index=False)
    pairs.to_csv(root / "stability_pairs.csv", index=False)
    plot_search(metrics, root / "collapse.png")


def run_validation(config: CRCSearchConfig) -> None:
    if config.inputs.assignment is None:
        raise ValueError("validate需要初筛产生的assignment")
    assignment = read_json(config.inputs.assignment)
    if assignment["protocol_hash"] != fingerprint(protocol(config)) or assignment[
        "data_hash"
    ] != fingerprint(data_fingerprints(config)):
        raise ValueError("冻结分配与协议/数据不一致")
    jobs: list[CRCSearchJob] = []
    reuse: dict[tuple[str, int, int], Path] = {}
    for item in assignment["groups"]:
        if item["owner"] != config.shard.index:
            continue
        for seed in config.search.validation_seeds:
            jobs.append(
                CRCSearchJob(recipe_id=item["recipe_id"], n_clusters=item["n_clusters"], seed=seed)
            )
            if str(seed) in item["reuse"]:
                reuse[(item["recipe_id"], item["n_clusters"], seed)] = Path(
                    item["reuse"][str(seed)]
                )
    if jobs:
        SearchQueue(config).run(jobs, reuse)


##################################################
# 四、锁定唯一模型后的单次测试评价与聚合图
##################################################


def plot_search(metrics: pd.DataFrame, path: Path) -> None:
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    if "cluster_min_fraction" in metrics:
        for seed, group in metrics.groupby("seed"):
            x = np.arange(len(group))
            axes[0].scatter(x, group["cluster_min_fraction"], s=15, label=str(seed))
            axes[1].scatter(x, group["cindex"], s=15, label=str(seed))
    axes[0].axhline(0.05, color="red", linestyle="--")
    axes[0].set(ylabel="Minimum cluster fraction", xlabel="Trial", ylim=(0, 1))
    axes[1].set(ylabel="Validation C-index", xlabel="Trial", ylim=(0, 1))
    for axis in axes:
        if axis.collections:
            axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_diagnostics(root: Path) -> None:
    from matplotlib import pyplot as plt

    files = sorted(root.glob("trials/*/k*/seed-*/history.csv"))
    if not files:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for path in files:
        frame = pd.read_csv(path)
        for axis, columns in zip(
            axes,
            (
                ["valid.cluster_min_fraction"],
                [
                    "train.reconstruction_loss_weight",
                    "train.survival_loss_weight",
                    "train.vade_kl_loss_weight",
                ],
                ["train.learning_rate"],
            ),
            strict=True,
        ):
            for column in columns:
                if column in frame:
                    axis.plot(frame["global_epoch"], frame[column], alpha=0.25, linewidth=0.7)
    for axis, label in zip(
        axes, ("Validation minimum fraction", "Actual loss weights", "Learning rate"), strict=True
    ):
        axis.set(xlabel="Global epoch (including warmup)", ylabel=label)
    axes[0].axhline(0.05, color="red", linestyle="--")
    figure.savefig(root / "training_diagnostics.png", dpi=180)
    plt.close(figure)


def finalize(config: CRCSearchConfig) -> None:
    if config.inputs.selection is None or config.shard.index != 0:
        raise ValueError("finalize须在99分片提供final_selection.json")
    choice = read_json(config.inputs.selection)["selected"]
    root = config.paths.dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if choice is None:
        write_json(
            root / "final_evaluation.json",
            {"selected": None, "test_read": False, "reason": "没有合格模型"},
        )
        return
    if choice["protocol_hash"] != fingerprint(protocol(config)) or choice[
        "data_hash"
    ] != fingerprint(data_fingerprints(config)):
        raise ValueError("锁定模型的协议/冻结数据不一致")
    with (root / ".finalize.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "final_evaluation.json").exists():
            if read_json(root / "final_evaluation.json")["selected"] != choice:
                raise ValueError("此目录已终评其他模型")
            LOGGER.info("终评已完成，直接复用")
            return
        checkpoint = config.inputs.checkpoint or Path(choice["model_path"])
        with checkpoint.open("rb") as handle:
            model_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        if model_hash != choice["checkpoint_sha256"]:
            raise ValueError("checkpoint内容与验证阶段锁定的文件不同")
        locked = {"selected": choice, "checkpoint_sha256": model_hash}
        if (root / "locked_model.json").exists() and read_json(
            root / "locked_model.json"
        ) != locked:
            raise ValueError("锁模后不得更换checkpoint")
        write_json(root / "locked_model.json", locked)
        estimator = TrailsEstimator.load(checkpoint, device=config.trainer.device)
        expected = estimator_config(
            config,
            recipes(config)[choice["recipe_id"]],
            CRCSearchJob(
                recipe_id=choice["recipe_id"],
                n_clusters=choice["n_clusters"],
                seed=choice["seed"],
            ),
        )
        if estimator.config != expected:
            raise ValueError("checkpoint不匹配锁定配方/K/seed")
        if (root / "test_started.json").exists():
            raise RuntimeError("此前test评价未正常完成，须先诊断；禁止自动重复读取test")
        write_json(root / "test_started.json", locked)
        # 在唯一模型和配置锁定之后才读取test；完成文件阻止续跑重复终评。
        train = ClinicalTimeSeriesDataset.load(config.split.dir / "train/dataset.pt")
        test = ClinicalTimeSeriesDataset.load(config.split.dir / "test/dataset.pt")
        prediction = estimator.predict(test)
        probabilities = prediction.cluster_probabilities.numpy()
        frame = pd.DataFrame(
            {
                "survival_time": [float(sample.survival_time) for sample in test.samples],
                "event": [bool(sample.event) for sample in test.samples],
                "pred_cluster": prediction.predict().numpy(),
                "risk_score": prediction.risk_score(method="median_survival").numpy(),
                "confidence": probabilities.max(axis=1),
                "entropy": -(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum(
                    axis=1
                )
                / np.log(choice["n_clusters"]),
            }
        )
        train_frame = pd.DataFrame(
            {
                "survival_time": [float(sample.survival_time) for sample in train.samples],
                "event": [bool(sample.event) for sample in train.samples],
            }
        )
        evaluation = CRCEvaluationConfig.model_validate(
            {
                "inputs": {"run_dir": str(checkpoint.parent)},
                "datasets": ["test"],
                "columns": {"numeric_baseline": [], "categorical_baseline": []},
                "umap": {
                    "n_neighbors": 15,
                    "min_dist": 0.1,
                    "metric": "euclidean",
                    "seed": choice["seed"],
                },
                "survival": config.survival.model_dump(),
                "trajectory_bin_days": 90,
                "trajectory_panels_per_page": 6,
                "fisher_resamples": 1000,
                "seed": choice["seed"],
                "device": config.trainer.device,
                "plot": {"dpi": 180, "font_families": ["DejaVu Sans"]},
                "paths": {"dir": str(root), "run_dir": str(root.parent / "finalize_logs")},
            }
        )
        metrics, curves, calibration, overall = survival_evaluation(
            frame, train_frame, prediction, evaluation
        )
        occupancy = cluster_statistics(frame, choice["n_clusters"])
        km, fits, logrank = cluster_survival(frame, choice["n_clusters"], config.survival.months)
        occupancy.to_csv(root / "test_occupancy.csv", index=False)
        km.to_csv(root / "test_km.csv", index=False)
        curves.to_csv(root / "test_survival_metrics.csv", index=False)
        calibration.to_csv(root / "test_calibration.csv", index=False)
        overall.to_csv(root / "test_overall_survival.csv", index=False)
        plot_final_km(fits, root / "test_km.png")
        write_json(
            root / "final_evaluation.json",
            {
                "selected": choice,
                "test_read": True,
                "survival": metrics,
                "logrank": logrank,
                "occupancy": records(occupancy),
                "checkpoint_sha256": model_hash,
                "test_used_for_selection": False,
            },
        )


def plot_final_km(fits: dict[int, Any], path: Path) -> None:
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5), layout="constrained")
    for label, km in fits.items():
        km.plot_survival_function(ax=axis, label=f"Cluster {label}")
    axis.set(xlabel="Months after landmark", ylabel="OS survival probability", ylim=(0, 1.03))
    figure.savefig(path, dpi=180)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="crc_yunnan/search")
def main(raw: DictConfig) -> None:
    config = CRCSearchConfig.model_validate(OmegaConf.to_container(raw, resolve=True))
    if config.mode == "plan":
        run_plan(config)
    elif config.mode == "worker":
        run_worker(config)
    elif config.mode == "screen":
        jobs = screen_jobs(config)
        if jobs:
            SearchQueue(config).run(jobs, {})
        plot_diagnostics(config.paths.dir.resolve())
    elif config.mode == "validate":
        run_validation(config)
        plot_diagnostics(config.paths.dir.resolve())
    elif config.mode == "merge":
        merge(config)
    else:
        finalize(config)


if __name__ == "__main__":
    main()
