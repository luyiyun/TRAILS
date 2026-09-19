#!/usr/bin/env bash
# OS三因素探索：固定12月/D20；同卡并行，患者级产物留在执行服务器。
set -euo pipefail
script_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
cd -- "$(dirname -- "$script_path")/../.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/trails-os-training-matplotlib}"
export HYDRA_FULL_ERROR=1
exec uv run python - "$script_path" "$@" <<'PY'
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from itertools import combinations, product
from pathlib import Path
from typing import Any, Literal, TextIO

import matplotlib
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from scipy.special import xlogy
from scripts.configs import TrailsApplicationConfig
from scripts.crc_yunnan.config import CRCEvaluationConfig, CRCPreprocConfig
from scripts.crc_yunnan.evaluation import (
    SPLIT_LABELS,
    cluster_statistics,
    cluster_survival,
    survival_evaluation,
)
from scripts.crc_yunnan.evaluation_plots import save_figure
from sklearn.metrics import adjusted_rand_score

from trails import ClinicalTimeSeriesDataset, TrailsEstimator, TrailsPrediction
from trails.artifacts import flatten_history

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

##################################################
# 一、固定数据与候选空间；学习率继承现有Hydra默认值。
##################################################
SEEDS = (20260909, 20260908, 20260910)
MODES = ("uncertainty", "fixed")
DATASETS = ("train", "validation", "test")
K_VALUES = tuple(range(2, 11))
SURVIVAL_WEIGHTS = (0.05, 0.2, 1.0)
CLUSTER_WEIGHTS = (0.02, 0.1, 0.5)
WARMUPS = (5, 10, 30)
INITIALIZATIONS = (5, 20, 50)
STAGE_LIMITS = {"A": 18, "B": 8, "C": 2, "D": 54}
MODULES = {"preproc": "04_preproc", "run": "05_run", "evaluate": "06_evaluate"}
BASE_OVERRIDES: dict[str, Any] = {
    "model": "base",
    "trainer": "full",
    "split.strategy": "random",
    "split.seed": 20260908,
    "model.loss.reconstruction_weight": 1.0,
    "trainer.batch_size": 128,
    "trainer.seed": SEEDS[0],
    "trainer.valid_size": 0.0,
    "trainer.min_epochs": 100,
    "trainer.max_epochs": 300,
    "trainer.early_stop": True,
    "trainer.early_stopping_patience": 30,
    "trainer.early_stopping_monitor": "loss",
    "trainer.cindex_risk_score": "median_survival",
    "trainer.device": "cuda:0",
    "swanlab.enabled": False,
}


class Options(BaseModel):
    output: Path
    split_dir: Path
    budget_hours: float = Field(default=15, gt=0, le=15)
    workers: int = Field(default=3, ge=1, le=3)
    dry_run: bool = False
    resume: bool = False


@dataclass(frozen=True)
class Recipe:
    weighting: str
    survival_weight: float = 0.2
    cluster_weight: float = 0.1
    warmup: int = 10
    initialization: int = 5

    @property
    def key(self) -> str:
        return (
            f"{self.weighting}-s{self.survival_weight:g}-c{self.cluster_weight:g}"
            f"-w{self.warmup}-i{self.initialization}"
        ).replace(".", "p")

    def overrides(self) -> dict[str, Any]:
        return {
            "model.loss.weighting": self.weighting,
            "model.loss.survival_weight": self.survival_weight,
            "model.loss.cluster_weight": self.cluster_weight,
            "trainer.warmup_epochs": self.warmup,
            "trainer.gmm_init_iters": self.initialization,
        }


@dataclass(frozen=True)
class Job:
    key: str
    stage: str
    kind: Literal["preproc", "screen", "selection", "evaluation"]
    values: dict[str, Any]
    recipe: Recipe | None = None

    @property
    def fits(self) -> int:
        return 27 if self.kind == "selection" else int(self.kind == "screen")

    @property
    def command_name(self) -> str:
        return {
            "preproc": "preproc",
            "screen": "run",
            "selection": "run",
            "evaluation": "evaluate",
        }[self.kind]

    @property
    def marker(self) -> str:
        return {
            "preproc": "preproc_manifest.json",
            "screen": "run_manifest.json",
            "selection": "run_manifest.json",
            "evaluation": "evaluation_summary.json",
        }[self.kind]


def overrides(values: dict[str, Any]) -> list[str]:
    return [f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in values.items()]


def resolved(name: str, values: dict[str, Any]) -> dict[str, Any]:
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
        raw = compose(config_name=f"crc_yunnan/{name}", overrides=overrides(values))
    models = {
        "preproc": CRCPreprocConfig,
        "run": TrailsApplicationConfig,
        "evaluate": CRCEvaluationConfig,
    }
    return (
        models[name]
        .model_validate(OmegaConf.to_container(raw, resolve=True))
        .model_dump(mode="json")
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.Series(value).to_json(temporary, force_ascii=False, indent=2, double_precision=15)
    temporary.replace(path)


def preproc_values(split: Path) -> dict[str, Any]:
    return {
        "outcome": "os",
        "landmark_months": 12,
        "min_observation_dates": 2,
        "features.panel": "D20",
        "preprocessing.log_transform": "auto-log1p",
        "preprocessing.scaling": "robust",
        "preprocessing.skew_threshold": 1.0,
        "inputs.split_dir": str(split),
        "inputs.train_dir": str(split / "train"),
        "inputs.test_dirs": [str(split / name) for name in ("validation", "test")],
    }


def run_values(recipe: Recipe, data: Path, *, selection: bool = False) -> dict[str, Any]:
    return (
        BASE_OVERRIDES
        | recipe.overrides()
        | {
            "split.dir": str(data),
            "n_clusters": None if selection else 3,
            "k_selection.enabled": selection,
            "k_selection.candidate_clusters": list(K_VALUES),
            "k_selection.seeds": list(SEEDS),
            "k_selection.selection_rule": "one_standard_error",
            "k_selection.require_non_empty": True,
            "k_selection.min_cluster_fraction": 0.05,
            "k_selection.compute_stability": selection,
            "k_selection.min_mean_pairwise_ari": None,
            "k_selection.result_dir": "k_selection",
            "outputs.summary": "run_manifest.json",
        }
    )


def evaluate_values(source: Path) -> dict[str, Any]:
    return {"inputs.run_dir": str(source), "survival.months": [24, 48]}


def ranking(table: pd.DataFrame) -> pd.DataFrame:
    ranked = table.copy()
    for name in ("平均AUC", "logrank强度", "C_index", "平均Brier"):
        ranked[name] = pd.to_numeric(ranked[name], errors="coerce")
    for name in ("平均AUC", "logrank强度"):
        ranked[name + "百分位"] = ranked[name].rank(pct=True).fillna(0)
    ranked["综合分"] = (ranked["平均AUC百分位"] + ranked["logrank强度百分位"]) / 2
    return ranked.sort_values(
        ["综合分", "平均AUC", "C_index", "平均Brier", "recipe_id", "seed"],
        ascending=[False, False, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


##################################################
# 二、统一进程调度；仅父进程写状态，阶段屏障后才选优。
##################################################
class ExplorationRunner:
    def __init__(self, options: Options, contract: dict[str, Any]) -> None:
        self.options, self.contract = options, contract
        self.root = options.output.resolve()
        self.state: dict[str, Any] = {}
        self.lock: TextIO | None = None
        self.started = time.monotonic()
        self.consumed = 0.0
        self.active: dict[str, tuple[subprocess.Popen[bytes], TextIO, dict[str, Any]]] = {}

    def __enter__(self) -> ExplorationRunner:
        if self.root.exists() and not self.options.resume:
            raise FileExistsError(f"拒绝覆盖既有目录：{self.root}；续跑使用--resume")
        if self.options.resume and not (self.root / "plan.json").is_file():
            raise FileNotFoundError("续跑目录缺少plan.json")
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".batch.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if self.options.resume:
            saved = OmegaConf.to_container(
                OmegaConf.load(self.root / "resolved_config.yaml"), resolve=True
            )
            if saved != self.contract:
                raise ValueError("脚本、实验配置、路径或预算变化，必须使用新目录；仅workers可调整")
            self.state = json.loads((self.root / "plan.json").read_text())
            # 子进程继承目录锁；能重新获得锁即说明旧训练均已退出。
            for attempts in self.state["tasks"].values():
                row = attempts[-1]
                if row["status"] == "running":
                    status = self.completed_status(row)
                    row["status"] = status or "interrupted"
                    output = Path(row["output"])
                    marker = output / row["marker"]
                    if status == "no_eligible_k":
                        marker = output / "k_selection/selection_summary.json"
                    finished = Path(row["log"]).stat().st_mtime
                    if status:
                        finished = max(finished, marker.stat().st_mtime)
                    row["seconds"] = max(0.0, finished - row["started_at"])
                    self.state["last_updated_at"] = max(self.state["last_updated_at"], finished)
            self.consumed = self.state["elapsed_seconds"] + max(
                0.0, self.state["last_updated_at"] - self.state["checkpoint_at"]
            )
        else:
            OmegaConf.save(self.contract, self.root / "resolved_config.yaml")
            self.state = {"stages": {}, "tasks": {}, "budget_skips": [], "elapsed_seconds": 0.0}
        self.started = time.monotonic()
        self.state["status"] = "running"
        self.persist()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, _exc: BaseException | None, _tb: object
    ) -> None:
        if exc_type is not None:
            self.state["status"] = (
                "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed"
            )
        self.persist()
        if self.lock is not None:
            self.lock.close()

    @property
    def elapsed(self) -> float:
        return self.consumed + time.monotonic() - self.started

    def persist(self) -> None:
        self.state.update(
            elapsed_seconds=self.elapsed,
            last_updated_at=time.time(),
            checkpoint_at=time.time(),
            workers=self.options.workers,
        )
        write_json(self.root / "plan.json", self.state)
        rows = [row for attempts in self.state["tasks"].values() for row in attempts]
        temporary = self.root / "progress.tmp"
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(self.root / "progress.csv")
        current = {
            "elapsed_hours": round(self.elapsed / 3600, 3),
            "workers": self.options.workers,
            "completed_fits": sum(
                row["fits"]
                for row in self.latest()
                if row["status"] in {"success", "no_eligible_k"}
            ),
            "maximum_fits": 82,
            "active": [
                {key: row[key] for key in ("task", "stage", "pid", "log")}
                for _, _, row in self.active.values()
            ],
            "status": self.state.get("status", "running"),
        }
        write_json(self.root / "current_run.txt", current)

    def latest(self) -> list[dict[str, Any]]:
        return [attempts[-1] for attempts in self.state["tasks"].values()]

    def completed_status(self, row: dict[str, Any]) -> str | None:
        output = Path(row["output"])
        if (output / row["marker"]).is_file():
            return "success"
        selection = output / "k_selection/selection_summary.json"
        if row["kind"] == "selection" and selection.is_file():
            payload = json.loads(selection.read_text())
            expected = {(seed, k) for seed, k in product(SEEDS, K_VALUES)}
            actual = {(item["seed"], item["n_clusters"]) for item in payload["run_metrics"]}
            candidates_saved = all(
                (output / f"k_selection/seed-{seed}/k{k}/config.json").is_file()
                for seed, k in expected
            )
            if payload["selected_k"] is None and actual == expected and candidates_saved:
                return "no_eligible_k"
        return None

    def done(self, job: Job) -> bool:
        attempts = self.state["tasks"].get(job.key, [])
        if not attempts or attempts[-1]["status"] not in {"success", "no_eligible_k"}:
            return False
        if self.completed_status(attempts[-1]) != attempts[-1]["status"]:
            raise FileNotFoundError(f"已完成任务的产物缺失：{job.key}")
        return True

    def output(self, key: str) -> Path:
        return Path(self.state["tasks"][key][-1]["output"])

    def affordable(self, jobs: list[Job], *, reserve_selection: bool) -> bool:
        durations = [
            row["seconds"] / row["fits"]
            for row in self.latest()
            if row["fits"] and row["status"] in {"success", "no_eligible_k"}
        ]
        estimate = max(402.0, float(np.quantile(durations[-12:], 0.75))) if durations else 402.0
        lanes = [0.0] * self.options.workers
        for job in jobs:
            if not self.done(job):
                index = int(np.argmin(lanes))
                lanes[index] += job.fits * estimate if job.fits else 300.0
        # 最终两个27次选择任务各自串行；不能误按54/3估算其墙钟时间。
        reserve = math.ceil(2 / self.options.workers) * 27 * estimate if reserve_selection else 0.0
        reserve += math.ceil(2 / self.options.workers) * 180 + 300
        return self.elapsed + max(lanes) + reserve <= self.options.budget_hours * 3600

    def launch(self, job: Job) -> None:
        attempts = self.state["tasks"].get(job.key, [])
        attempt = len(attempts) + 1
        folder = {
            "preproc": "preproc",
            "screen": "runs",
            "selection": "selection",
            "evaluation": "evaluation",
        }[job.kind]
        output = self.root / folder / job.key / f"attempt-{attempt}"
        log = self.root / "logs" / f"{job.key}-attempt-{attempt}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        values = job.values | {"paths.dir": str(output)}
        if job.kind in {"preproc", "evaluation"}:
            values["paths.run_dir"] = str(self.root / "logs" / f"{job.key}-attempt-{attempt}")
        resolved(job.command_name, values)
        argv = [
            sys.executable,
            "-u",
            "-m",
            f"scripts.crc_yunnan.{MODULES[job.command_name]}",
            *overrides(values),
        ]
        row = {
            "task": job.key,
            "stage": job.stage,
            "kind": job.kind,
            "attempt": attempt,
            "recipe": asdict(job.recipe) if job.recipe else None,
            "fits": job.fits,
            "status": "running",
            "output": str(output),
            "log": str(log),
            "marker": job.marker,
            "command": shlex.join(argv),
            "started_at": time.time(),
            "seconds": 0.0,
        }
        assert self.lock is not None
        # 每个子进程持有同一目录锁，父进程意外退出后也不能重复启动这批任务。
        with log.open("w") as stream:
            process = subprocess.Popen(
                argv, stdout=stream, stderr=subprocess.STDOUT, pass_fds=(self.lock.fileno(),)
            )
        row["pid"] = process.pid
        attempts.append(row)
        self.state["tasks"][job.key] = attempts
        self.active[job.key] = process, stream, row
        self.persist()
        print(f"开始 {job.stage} {job.key} pid={process.pid} 日志={log}", flush=True)

    def finish(self, key: str) -> dict[str, Any]:
        process, stream, row = self.active.pop(key)
        exit_code = process.wait()
        stream.close()
        status = self.completed_status(row)
        row.update(
            exit_code=exit_code,
            seconds=time.time() - row["started_at"],
            status=status
            if (exit_code == 0 or (exit_code == 1 and status == "no_eligible_k")) and status
            else "failed",
        )
        self.persist()
        print(f"结束 {row['task']} {row['status']} 用时={row['seconds']:.0f}s", flush=True)
        if row["status"] == "failed":
            print(Path(row["log"]).read_text()[-16000:], file=sys.stderr, flush=True)
        return row

    def run_jobs(self, jobs: list[Job]) -> None:
        queue = iter(job for job in jobs if not self.done(job))
        exhausted = False
        failures: list[dict[str, Any]] = []
        checkpoint = time.monotonic()
        try:
            while self.active or not exhausted:
                while not failures and not exhausted and len(self.active) < self.options.workers:
                    job = next(queue, None)
                    if job is None:
                        exhausted = True
                    else:
                        self.launch(job)
                finished = [
                    key
                    for key, (process, _, _) in self.active.items()
                    if process.poll() is not None
                ]
                for key in finished:
                    row = self.finish(key)
                    if row["status"] == "failed":
                        failures.append(row)
                if failures:
                    exhausted = True
                if self.active:
                    time.sleep(0.2)
                if time.monotonic() - checkpoint >= 5:
                    self.persist()
                    checkpoint = time.monotonic()
        finally:
            # 异常停止派发，已启动任务收尾；不丢弃其产物或原始traceback。
            for key in list(self.active):
                self.finish(key)
        if failures:
            self.state["status"] = "failed"
            self.persist()
            raise subprocess.CalledProcessError(
                failures[0]["exit_code"] or 1, failures[0]["command"]
            )


##################################################
# 三、复用CRC统计；只保存聚合评价和训练历史摘要。
##################################################
def evaluate_predictions(
    predictions: dict[str, TrailsPrediction],
    frames: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
    config: CRCEvaluationConfig,
) -> dict[str, list[dict[str, Any]]]:
    collected: dict[str, list[pd.DataFrame]] = {}
    k = metadata["k"]
    for name, frame in frames.items():
        prediction = predictions[name]
        probability = prediction.predict_proba().numpy()
        frame["pred_cluster"] = prediction.predict().numpy()
        frame["risk_score"] = prediction.risk_score(method="median_survival").numpy()
        frame["confidence"] = probability.max(axis=1)
        frame["entropy"] = -xlogy(probability, probability).sum(axis=1) / np.log(k)
    for name, frame in frames.items():
        sizes = cluster_statistics(frame, k)
        km, _, separation = cluster_survival(frame, k, config.survival.months)
        metrics, curve, calibration, overall = survival_evaluation(
            frame, frames["train"], predictions[name], config
        )
        horizons = curve.set_index("月份").loc[config.survival.months]
        auc, brier = (
            horizons["动态AUC"].to_numpy(dtype=float),
            horizons["Brier"].to_numpy(dtype=float),
        )
        p_value = separation["总体log-rank_p值"]
        row = {
            "人数": len(frame),
            "事件数": int(frame.event.sum()),
            "C_index": metrics["Harrell_C-index"],
            "IPCW_C_index": metrics["IPCW_C-index"],
            "IBS": metrics["IBS"],
            "AUC_3年": auc[0],
            "AUC_5年": auc[1],
            "平均AUC": auc.mean(),
            "Brier_3年": brier[0],
            "Brier_5年": brier[1],
            "平均Brier": brier.mean(),
            "logrank_p": p_value,
            "logrank强度": -np.log10(max(p_value, np.finfo(float).tiny))
            if p_value is not None
            else None,
            "空簇数": separation["空簇数"],
            "最小簇比例": sizes["占比（%）"].min() / 100,
            "占用标记": "合格"
            if separation["空簇数"] == 0 and sizes["占比（%）"].min() >= 5
            else "空簇或小簇",
            "分型可解释": frame.pred_cluster.nunique() >= 2,
            "不可计算原因": json.dumps(
                metrics["不可计算原因"] | {"logrank": separation["不可计算原因"]},
                ensure_ascii=False,
            ),
        }
        for label, table in {
            "指标": pd.DataFrame([row]),
            "簇概况": sizes,
            "簇生存": km,
            "月度预测": curve,
            "校准": calibration,
            "整体生存": overall,
        }.items():
            if "月份" in table:
                table = table.assign(术后月份=table["月份"] + 12)
            collected.setdefault(label, []).append(table.assign(数据集=name, **metadata))
    return {label: pd.concat(tables).to_dict("records") for label, tables in collected.items()}


def history_tables(
    history: pd.DataFrame, metadata: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    epoch = int(history.iloc[-1]["best_global_epoch"])
    saved = history.loc[history.global_epoch.eq(epoch)].iloc[0]
    summary: dict[str, Any] = {
        "保存轮次": epoch,
        "实际warmup轮数": int(history.stage.eq("warmup").sum()),
        "主训练轮数": int(history.stage.eq("vade").sum()),
    }
    for split in ("train", "val"):
        for loss in ("reconstruction_loss", "survival_loss", "vade_kl_loss"):
            for suffix in ("", "_weight"):
                key = f"{split}_{loss}{suffix}"
                saved_key = f"{'val_' if split == 'val' else ''}{loss}{suffix}"
                summary[key] = float(saved[saved_key])
    return {
        "保存模型损失": [metadata | summary],
        "逐轮损失与权重": history.assign(**metadata, 保存轮次=epoch).to_dict("records"),
    }


def screen_tables(
    runner: ExplorationRunner, config: CRCEvaluationConfig
) -> dict[str, pd.DataFrame]:
    collected: dict[str, list[pd.DataFrame]] = {}
    for row in runner.latest():
        if row["kind"] != "screen" or row["status"] != "success":
            continue
        recipe = Recipe(**row["recipe"])
        source = Path(row["output"])
        cache = runner.root / "summary/quick" / f"{row['task']}.json"
        if not cache.is_file():
            predictions = {
                name: TrailsPrediction.load(source / name / "model_prediction.pt")
                for name in DATASETS
            }
            frames = {
                name: pd.read_csv(
                    source / name / "patient_outputs.csv", dtype={"patient_id": "string"}
                )
                for name in DATASETS
            }
            metadata = asdict(recipe) | {
                "recipe_id": recipe.key,
                "seed": SEEDS[0],
                "k": 3,
                "阶段": row["stage"],
                "运行目录": str(source),
            }
            tables = evaluate_predictions(predictions, frames, metadata, config)
            tables.update(history_tables(pd.read_csv(source / "training_history.csv"), metadata))
            write_json(cache, tables)
        for label, records in json.loads(cache.read_text()).items():
            collected.setdefault(label, []).append(pd.DataFrame(records))
    return {label: pd.concat(tables, ignore_index=True) for label, tables in collected.items()}


def finalist_tables(
    runner: ExplorationRunner,
    row: dict[str, Any],
    data: Path,
    config: CRCEvaluationConfig,
) -> dict[str, pd.DataFrame]:
    recipe, source = Recipe(**row["recipe"]), Path(row["output"])
    cache = runner.root / "summary/quick" / f"final-{recipe.key}.json"
    if not cache.is_file():
        manifest = json.loads((source / "run_manifest.json").read_text())
        k = manifest["selected_k"]
        datasets = {
            name: ClinicalTimeSeriesDataset.load(data / name / "dataset.pt") for name in DATASETS
        }
        originals = {
            name: pd.read_csv(data / name / "patients.csv", dtype={"patient_id": "string"})
            for name in DATASETS
        }
        collected: dict[str, list[pd.DataFrame]] = {}
        labels: dict[tuple[str, int], np.ndarray] = {}
        for seed in SEEDS:
            model = TrailsEstimator.load(
                source / f"k_selection/seed-{seed}/k{k}/model.pt", device="cpu"
            )
            # 代表seed直接复用05预测；其余seed仅推理，不增加训练次数。
            predictions = {
                name: TrailsPrediction.load(source / name / "model_prediction.pt")
                if seed == manifest["selected_seed"]
                else model.predict(dataset)
                for name, dataset in datasets.items()
            }
            frames = {name: frame.copy() for name, frame in originals.items()}
            metadata = asdict(recipe) | {
                "recipe_id": recipe.key,
                "seed": seed,
                "k": k,
                "阶段": "D",
                "代表模型": seed == manifest["selected_seed"],
                "运行目录": str(source),
            }
            tables = evaluate_predictions(predictions, frames, metadata, config)
            tables.update(history_tables(pd.DataFrame(flatten_history(model.history)), metadata))
            for name, prediction in predictions.items():
                labels[name, seed] = prediction.predict().numpy()
            for label, records in tables.items():
                collected.setdefault(label, []).append(pd.DataFrame(records))
            del model
        tables_out = {
            label: pd.concat(tables).to_dict("records") for label, tables in collected.items()
        }
        tables_out["跨seed配对ARI"] = [
            asdict(recipe)
            | {
                "recipe_id": recipe.key,
                "k": k,
                "数据集": name,
                "seed_1": first,
                "seed_2": second,
                "ARI": adjusted_rand_score(labels[name, first], labels[name, second]),
            }
            for name in DATASETS
            for first, second in combinations(SEEDS, 2)
        ]
        write_json(cache, tables_out)
    return {
        label: pd.DataFrame(records) for label, records in json.loads(cache.read_text()).items()
    }


##################################################
# 四、单图封装：权重网格、单因素和最终重复/训练过程。
##################################################
def plot_weights(metrics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for index, mode in enumerate(MODES):
        part = metrics.loc[
            metrics.weighting.eq(mode) & metrics["阶段"].eq("A") & metrics["数据集"].eq("test")
        ]
        for column, metric in enumerate(("平均AUC", "logrank强度")):
            table = part.pivot(
                index="cluster_weight", columns="survival_weight", values=metric
            ).reindex(index=CLUSTER_WEIGHTS, columns=SURVIVAL_WEIGHTS)
            axis = axes[index, column]
            values = table.to_numpy(dtype=float)
            artist = axis.imshow(np.ma.masked_invalid(values), aspect="auto", cmap="viridis")
            for (y, x), value in np.ndenumerate(values):
                axis.text(
                    x,
                    y,
                    f"{value:.3f}" if np.isfinite(value) else "不可计算",
                    ha="center",
                    va="center",
                    color="white",
                )
            axis.set(
                xticks=range(3),
                xticklabels=SURVIVAL_WEIGHTS,
                yticks=range(3),
                yticklabels=CLUSTER_WEIGHTS,
                xlabel="生存权重",
                ylabel="聚类权重",
                title=f"{mode}：{metric}",
            )
            figure.colorbar(artist, ax=axis)
    figure.suptitle("重建权重=1；K=3单seed探索；test参与选优")
    save_figure(figure, output / "weight_ratios", 180)


def plot_factors(metrics: pd.DataFrame, references: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    tests = metrics.loc[metrics["数据集"].eq("test") & metrics["阶段"].ne("D")]
    for index, mode in enumerate(MODES):
        base = Recipe(**references[mode])
        for column, (factor, values) in enumerate(
            (("warmup", WARMUPS), ("initialization", INITIALIZATIONS))
        ):
            keys = [replace(base, **{factor: value}).key for value in values]
            part = tests.set_index("recipe_id").reindex(keys)
            axis = axes[index, column]
            axis.plot(values, part["平均AUC"].to_numpy(dtype=float), "o-", label="平均AUC")
            axis.plot(values, part["C_index"].to_numpy(dtype=float), "s--", label="C-index")
            axis.set(
                xlabel="warmup轮数" if factor == "warmup" else "初始化迭代轮数",
                title=mode,
                xticks=values,
            )
            axis.legend()
    figure.suptitle("每种加权方式固定阶段A入选比例；K=3单seed比较")
    save_figure(figure, output / "warmup_initialization", 180)


def plot_repeats(metrics: pd.DataFrame, ari: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    tests = metrics.loc[metrics["数据集"].eq("test") & metrics["阶段"].eq("D")]
    for index, (key, part) in enumerate(tests.groupby("recipe_id", sort=True)):
        label = f"{part.weighting.iloc[0]} K={part.k.iloc[0]}"
        axes[0].scatter(np.full(len(part), index), part["平均AUC"], label=label)
        axes[0].plot(index, part["平均AUC"].to_numpy(dtype=float).mean(), "k_")
        pairs = ari.loc[ari.recipe_id.eq(key) & ari["数据集"].eq("test")]
        axes[1].scatter(np.full(len(pairs), index), pairs["ARI"], label=label)
    for axis in axes:
        axis.set(xticks=[])
        axis.legend()
    axes[0].set(title="入选K的三个seed", ylabel="术后3/5年平均AUC")
    axes[1].set(title="同方案跨seed稳定性（结合簇占用解释）", ylabel="ARI")
    save_figure(figure, output / "seed_repeats", 180)


def plot_losses(history: pd.DataFrame, output: Path) -> None:
    baselines = [Recipe(mode).key for mode in MODES]
    representative = (
        history["代表模型"].fillna(False)
        if "代表模型" in history
        else pd.Series(False, index=history.index)
    )
    chosen = history.loc[
        (history.recipe_id.isin(baselines) & history["阶段"].eq("A")) | representative
    ]
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for (_key, stage, seed), group in chosen.groupby(["recipe_id", "阶段", "seed"], sort=True):
        label = f"{stage} {group.weighting.iloc[0]} K={group.k.iloc[0]} seed={seed}"
        for row, (loss, title) in enumerate(
            (("reconstruction_loss", "重建"), ("survival_loss", "生存"), ("vade_kl_loss", "聚类"))
        ):
            for column, suffix in enumerate(("", "_weight")):
                axes[row, column].plot(
                    group.global_epoch, group[f"val_{loss}{suffix}"], label=label, alpha=0.8
                )
                axes[row, column].set(
                    title=title + ("原始损失" if not suffix else "实际权重"),
                    xlabel="全局轮次",
                    yscale="symlog",
                )
                axes[row, column].yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _position: f"{value:.3g}")
                )
                axes[row, column].yaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("基准与入围代表模型；总损失下降不能代替原始损失改善")
    save_figure(figure, output / "losses_weights", 180)


##################################################
# 五、冻结分阶段计划、执行并汇总；不更改05的K选择实现。
##################################################
def main() -> int:
    parser = argparse.ArgumentParser(
        description="OS12月/D20三因素探索；test参与选优，同卡最多3并发"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/derived/crc_yunnan/splits/v2/random/seed-20260908"),
    )
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--budget-hours", type=float, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = Options.model_validate(vars(parser.parse_args(sys.argv[2:])))
    root, split = args.output.resolve(), args.split_dir.resolve()
    template_data, template_run = root / "preproc/template", root / "runs/template"
    contract = {
        "script_sha256": hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
        "budget_hours": args.budget_hours,
        "preproc": resolved(
            "preproc",
            preproc_values(split)
            | {
                "paths.dir": str(template_data),
                "paths.run_dir": str(root / "logs/preproc-template"),
            },
        ),
        "run": resolved(
            "run",
            run_values(Recipe("uncertainty"), template_data) | {"paths.dir": str(template_run)},
        ),
        "selection": resolved(
            "run",
            run_values(Recipe("uncertainty"), template_data, selection=True)
            | {"paths.dir": str(root / "selection/template")},
        ),
        "evaluate": resolved(
            "evaluate",
            evaluate_values(template_run)
            | {
                "paths.dir": str(root / "evaluation/template"),
                "paths.run_dir": str(root / "logs/evaluation-template"),
            },
        ),
    }
    stage_a = [
        Recipe(mode, survival, cluster)
        for mode, survival, cluster in product(MODES, SURVIVAL_WEIGHTS, CLUSTER_WEIGHTS)
    ]
    if args.dry_run:
        count = 0
        for mode, survival, cluster, warmup, initialization in product(
            MODES, SURVIVAL_WEIGHTS, CLUSTER_WEIGHTS, WARMUPS, INITIALIZATIONS
        ):
            recipe = Recipe(mode, survival, cluster, warmup, initialization)
            for selection in (False, True):
                resolved(
                    "run",
                    run_values(recipe, template_data, selection=selection)
                    | {"paths.dir": str(template_run)},
                )
            count += 1
        print(
            json.dumps(
                {
                    "候选模板数": count,
                    "阶段训练上限": STAGE_LIMITS,
                    "训练总上限": sum(STAGE_LIMITS.values()),
                    "阶段A唯一方案数": len({recipe.key for recipe in stage_a}),
                    "预处理次数": 1,
                    "workers": args.workers,
                    "K选择最大并发": min(2, args.workers),
                    "batch_size": contract["run"]["trainer"]["batch_size"],
                    "learning_rate": contract["run"]["trainer"]["learning_rate"],
                    "说明": (
                        "全部固定K/三seed选择模板解析通过；未读取数据、创建输出或训练；"
                        "B/C/D由完成阶段的指标决定"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    config = CRCEvaluationConfig.model_validate(contract["evaluate"])
    with ExplorationRunner(args, contract) as runner:
        # 预处理在训练前单独完成；三个worker共享同一冻结产物。
        runner.run_jobs([Job("preproc", "preproc", "preproc", preproc_values(split))])
        data = runner.output("preproc")
        tables = screen_tables(runner, config)
        for stage in ("A", "B", "C", "D"):
            stages = runner.state["stages"]
            if stage not in stages:
                recipes: list[Recipe] = []
                references: dict[str, Any] = {}
                if stage == "A":
                    recipes = stage_a
                elif tables:
                    tests = tables["指标"].loc[tables["指标"]["数据集"].eq("test")]
                    ranked = ranking(tests)
                    for mode in MODES:
                        best = ranked.loc[ranked.weighting.eq(mode)].iloc[0]
                        recipe_map = {
                            Recipe(**row["recipe"]).key: Recipe(**row["recipe"])
                            for row in runner.latest()
                            if row["kind"] == "screen"
                        }
                        base = recipe_map[str(best.recipe_id)]
                        if stage == "B":
                            references[mode] = asdict(base)
                            recipes.extend([replace(base, warmup=value) for value in (5, 30)])
                            recipes.extend(
                                [replace(base, initialization=value) for value in (20, 50)]
                            )
                        elif stage == "C" and stages.get("B", {}).get("status") == "completed":
                            reference = Recipe(**stages["B"]["references"][mode])
                            changes: dict[str, int] = {}
                            for factor, values in (
                                ("warmup", WARMUPS),
                                ("initialization", INITIALIZATIONS),
                            ):
                                keys = [
                                    replace(reference, **{factor: value}).key for value in values
                                ]
                                choice = ranked.loc[ranked.recipe_id.isin(keys)].iloc[0]
                                changes[factor] = int(choice[factor])
                            recipes.append(replace(reference, **changes))
                        elif stage == "D":
                            recipes.append(base)
                stages[stage] = {
                    "recipes": [asdict(recipe) for recipe in recipes],
                    "references": references,
                    "status": "planned",
                }
                runner.persist()
            planned = stages[stage]
            selection = stage == "D"
            jobs = [
                Job(
                    ("select-" if selection else "screen-") + recipe.key,
                    stage,
                    "selection" if selection else "screen",
                    run_values(recipe, data, selection=selection),
                    recipe,
                )
                for recipe in (Recipe(**item) for item in planned["recipes"])
            ]
            pending = [job for job in jobs if not runner.done(job)]
            if planned["status"] == "budget_skipped":
                continue
            if pending and not runner.affordable(jobs, reserve_selection=not selection):
                planned["status"] = "budget_skipped"
                runner.state["budget_skips"].append(stage)
                runner.persist()
                print(f"预算预留：跳过阶段{stage}的剩余任务", flush=True)
                continue
            runner.run_jobs(jobs)
            planned["status"] = "completed" if jobs else "not_available"
            runner.persist()
            tables = screen_tables(runner, config)

        # 入选K的三seed指标直接来自已保存模型；所有预测只在执行服务器内处理。
        evaluation_jobs: list[Job] = []
        selection_tables: dict[str, list[pd.DataFrame]] = {}
        no_k: list[str] = []
        for row in runner.latest():
            if row["kind"] != "selection" or row["status"] not in {"success", "no_eligible_k"}:
                continue
            recipe = Recipe(**row["recipe"])
            source = Path(row["output"])
            for filename, label in (
                ("run_metrics", "K选择逐次指标"),
                ("k_summary", "K选择汇总"),
                ("stability_pairs", "K选择验证ARI"),
            ):
                frame = pd.read_csv(source / "k_selection" / f"{filename}.csv").assign(
                    recipe_id=recipe.key, weighting=recipe.weighting
                )
                selection_tables.setdefault(label, []).append(frame)
            if row["status"] == "no_eligible_k":
                no_k.append(recipe.key)
                continue
            for label, frame in finalist_tables(runner, row, data, config).items():
                tables[label] = (
                    pd.concat([tables[label], frame], ignore_index=True)
                    if label in tables
                    else frame
                )
            evaluation_jobs.append(
                Job("evaluate-" + recipe.key, "E", "evaluation", evaluate_values(source), recipe)
            )
        for label, frames in selection_tables.items():
            tables[label] = pd.concat(frames, ignore_index=True)
        summary_dir = root / "summary"
        summary_dir.mkdir(exist_ok=True)
        finals = pd.DataFrame()
        if "指标" in tables:
            metrics = tables["指标"]
            tests = metrics.loc[metrics["数据集"].eq("test")]
            tables["K3筛选排行榜"] = ranking(tests.loc[tests["阶段"].ne("D")])
            final_rows: list[dict[str, Any]] = []
            for key, group in tests.loc[tests["阶段"].eq("D")].groupby("recipe_id", sort=True):
                item = {
                    field: group.iloc[0][field] for field in (*Recipe.__dataclass_fields__, "k")
                }
                item.update(
                    recipe_id=key,
                    seed=0,
                    完成seed数=len(group),
                    占用合格seed数=int(group["占用标记"].eq("合格").sum()),
                )
                for metric in ("平均AUC", "AUC_3年", "AUC_5年", "C_index", "平均Brier"):
                    values = group[metric].to_numpy(dtype=float)
                    item[metric], item[metric + "最小"], item[metric + "最大"] = (
                        values.mean(),
                        values.min(),
                        values.max(),
                    )
                item["logrank强度"] = float(np.median(group["logrank强度"].to_numpy(dtype=float)))
                final_rows.append(item)
            if final_rows:
                finals = ranking(pd.DataFrame(final_rows))
                tables["最终方案三seed汇总"] = finals
            # 差异按相同数据集和加权方式对照默认比例；D阶段可能同时改变K。
            baseline_metrics = metrics.loc[
                metrics["阶段"].eq("A")
                & metrics.recipe_id.isin([Recipe(mode).key for mode in MODES])
            ]
            columns = ["平均AUC", "C_index", "平均Brier", "最小簇比例"]
            comparison = metrics.merge(
                baseline_metrics[["weighting", "数据集", *columns]],
                on=["weighting", "数据集"],
                suffixes=("", "_参考"),
                how="left",
            )
            for metric in columns:
                comparison[metric + "差异"] = comparison[metric] - comparison[metric + "_参考"]
            comparison["参考说明"] = (
                "同加权方式，K3/seed20260909，比例1:0.2:0.1，warmup10，初始化迭代5；D可能改变K"
            )
            tables["相对基准差异"] = comparison
            available = {font.name for font in font_manager.fontManager.ttflist}
            fonts = [name for name in config.plot.font_families if name in available]
            if not fonts:
                raise ValueError("未找到配置中的中文字体")
            plt.rcParams.update(
                {"font.family": fonts, "axes.unicode_minus": False, "pdf.fonttype": 42}
            )
            figures = summary_dir / "figures"
            figures.mkdir(exist_ok=True)
            plot_weights(metrics, figures)
            if runner.state["stages"].get("B", {}).get("status") == "completed":
                plot_factors(metrics, runner.state["stages"]["B"]["references"], figures)
            if not finals.empty:
                plot_repeats(metrics, tables["跨seed配对ARI"], figures)
            plot_losses(tables["逐轮损失与权重"], figures)
        tables["运行记录"] = pd.read_csv(root / "progress.csv")
        tables["阶段计划"] = pd.DataFrame(
            [
                {"阶段": stage, "状态": plan["status"], **recipe}
                for stage, plan in runner.state["stages"].items()
                for recipe in plan["recipes"]
            ]
        )
        chinese = {
            "recipe_id": "方案",
            "weighting": "加权方式",
            "survival_weight": "生存权重",
            "cluster_weight": "聚类权重",
            "warmup": "warmup轮数",
            "initialization": "初始化迭代轮数",
            "k": "K",
            "C_index": "Harrell C-index",
            "IPCW_C_index": "IPCW C-index",
        }
        with pd.ExcelWriter(summary_dir / "exploration_tables.xlsx", engine="openpyxl") as writer:
            for label, table in tables.items():
                table.rename(columns=chinese).replace({"数据集": SPLIT_LABELS}).to_excel(
                    writer, sheet_name=label, index=False
                )
        selected = [
            row
            for row in runner.latest()
            if row["kind"] == "selection" and row["status"] in {"success", "no_eligible_k"}
        ]
        complete = len(selected) == 2 and not runner.state["budget_skips"]
        status = "completed" if complete else "partial"
        summary_payload = {
            "状态": "evaluating" if evaluation_jobs else status,
            "性质": "测试集参与选优的探索结果，不是独立测试评价",
            "固定设置": "OS/12个月/D20/auto-log1p/robust；batch128；学习率继承默认",
            "学习率": contract["run"]["trainer"]["learning_rate"],
            "时间口径": "术后36/60月，即landmark后24/48月；每月30天",
            "训练上限": 82,
            "完成训练数": sum(
                row["fits"]
                for row in runner.latest()
                if row["status"] in {"success", "no_eligible_k"}
            ),
            "workers": args.workers,
            "预算小时": args.budget_hours,
            "实际耗时小时": runner.elapsed / 3600,
            "预算跳过阶段": runner.state["budget_skips"],
            "无合格K方案": no_k,
            "最终方案": finals.to_dict("records"),
            "完整评价目录": [],
            "解释": [
                "A-C固定K3、单seed筛选；D用train/validation三seed选K",
                "综合榜使用平均AUC与-log10(logrank p)百分位等权评分，缺失百分位为0",
                "三seed汇总使用平均AUC和中位数-log10(p)，缺失值传播",
                "初始化参数表示K-means迭代轮数；uncertainty配置为初始权重",
                "空簇/小簇保留标记；单簇或退化解的高ARI不能解释为稳定分型",
                "原始损失和实际权重分别报告；默认基准差异仅为探索性比较",
                "患者表、模型、预测均留在执行服务器",
            ],
        }
        write_json(summary_dir / "exploration_summary.json", summary_payload)
        # 完整06可能遇到不可估计的调整Cox；先保存核心结果，真实错误仍向上抛出。
        try:
            runner.run_jobs(evaluation_jobs)
        finally:
            evaluation_rows = [row for row in runner.latest() if row["kind"] == "evaluation"]
            successful_evaluations = [row for row in evaluation_rows if row["status"] == "success"]
            if any(row["status"] == "failed" for row in evaluation_rows):
                status = "failed"
            elif len(successful_evaluations) != len(evaluation_jobs):
                status = "partial"
            summary_payload.update(
                {
                    "状态": status,
                    "完整评价目录": [row["output"] for row in successful_evaluations],
                    "评价任务": [
                        {"任务": row["task"], "状态": row["status"], "日志": row["log"]}
                        for row in evaluation_rows
                    ],
                }
            )
            write_json(summary_dir / "exploration_summary.json", summary_payload)
            runner.state["status"] = status
            runner.persist()
        print(f"探索{status}：{summary_dir}", flush=True)
        return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
PY
