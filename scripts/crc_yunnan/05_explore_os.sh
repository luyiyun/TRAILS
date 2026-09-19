#!/usr/bin/env bash
# OS分阶段探索：仅组合现有04/05/06命令，原始数据与患者级产物留在执行服务器。
set -euo pipefail
script_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
cd -- "$(dirname -- "$script_path")/../.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/trails-os-matplotlib}"
export HYDRA_FULL_ERROR=1
exec uv run python - "$script_path" "$@" <<'PY'
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from itertools import combinations, product
from pathlib import Path
from typing import Any

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

from trails import TrailsPrediction

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

##################################################
# 一、固定探索方案；参数通过现有Hydra入口解析。
##################################################
SEEDS = (20260909, 20260908, 20260910)
DATASETS = ("train", "validation", "test")
STAGE_LIMITS = {"A": 48, "B": 8, "C": 26, "D": 6, "E": 8}
PROFILES: dict[str, tuple[str, dict[str, Any]]] = {
    "monitor-cindex": ("monitor", {"trainer.early_stopping_monitor": "cindex"}),
    "monitor-survival": ("monitor", {"trainer.early_stopping_monitor": "survival_loss"}),
    "fixed-02": ("loss", {"model.loss.weighting": "fixed", "model.loss.survival_weight": 0.2}),
    "fixed-10": ("loss", {"model.loss.weighting": "fixed", "model.loss.survival_weight": 1.0}),
    "lr-3e-4": ("learning_rate", {"trainer.learning_rate": 3e-4}),
    "lr-1e-4": ("learning_rate", {"trainer.learning_rate": 1e-4}),
    "batch-256": ("batch", {"trainer.batch_size": 256}),
    "batch-64": ("batch", {"trainer.batch_size": 64}),
    "warmup-30": ("warmup", {"trainer.warmup_epochs": 30}),
    "init-50": ("initialization", {"trainer.gmm_init_iters": 50}),
    "latent-16": ("latent", {"model.latent_dim": 16}),
    "gru": ("architecture", {"model.encoder.mapping.kind": "gru", "model.decoder.kind": "gru"}),
    "mtan": ("architecture", {"model.encoder.input.kind": "mtan"}),
}
BASE_OVERRIDES: dict[str, Any] = {
    "model": "base",
    "trainer": "full",
    "k_selection.enabled": False,
    "split.strategy": "random",
    "split.seed": 20260908,
    "model.latent_dim": 32,
    "model.loss.weighting": "uncertainty",
    "model.loss.reconstruction_weight": 1.0,
    "model.loss.survival_weight": 0.2,
    "model.loss.cluster_weight": 0.1,
    "trainer.batch_size": 128,
    "trainer.learning_rate": 1e-3,
    "trainer.warmup_epochs": 10,
    "trainer.gmm_init_iters": 5,
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
    dry_run: bool = False
    resume: bool = False


@dataclass(frozen=True)
class Recipe:
    landmark: int
    panel: str
    log: str
    scaling: str = "robust"
    k: int = 3
    changes: tuple[tuple[str, Any], ...] = ()

    @property
    def key(self) -> str:
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]
        return f"l{self.landmark}-{self.panel}-{self.log}-{self.scaling}-k{self.k}-{digest}"

    @property
    def data_key(self) -> str:
        return f"l{self.landmark}-{self.panel}-{self.log}-{self.scaling}"

    def changed(self, values: dict[str, Any]) -> Recipe:
        return replace(self, changes=tuple(sorted((dict(self.changes) | values).items())))


def recipe_from(payload: dict[str, Any]) -> Recipe:
    return Recipe(
        landmark=payload["landmark"],
        panel=payload["panel"],
        log=payload["log"],
        scaling=payload["scaling"],
        k=payload["k"],
        changes=tuple((key, value) for key, value in payload["changes"]),
    )


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


def command(name: str, values: dict[str, Any]) -> list[str]:
    number = {"preproc": "04_preproc", "run": "05_run", "evaluate": "06_evaluate"}[name]
    return [sys.executable, "-u", "-m", f"scripts.crc_yunnan.{number}", *overrides(values)]


def preproc_values(recipe: Recipe, split_dir: Path, output: Path, logs: Path) -> dict[str, Any]:
    return {
        "outcome": "os",
        "landmark_months": recipe.landmark,
        "min_observation_dates": 2,
        "features.panel": recipe.panel,
        "preprocessing.log_transform": recipe.log,
        "preprocessing.scaling": recipe.scaling,
        "preprocessing.skew_threshold": 1.0,
        "inputs.split_dir": str(split_dir),
        "paths.dir": str(output),
        "paths.run_dir": str(logs),
    }


def run_values(recipe: Recipe, seed: int, data: Path, output: Path) -> dict[str, Any]:
    return (
        BASE_OVERRIDES
        | dict(recipe.changes)
        | {
            "n_clusters": recipe.k,
            "trainer.seed": seed,
            "split.dir": str(data),
            "paths.dir": str(output),
        }
    )


def evaluate_values(recipe: Recipe, source: Path, output: Path, logs: Path) -> dict[str, Any]:
    return {
        "inputs.run_dir": str(source),
        "survival.months": [36 - recipe.landmark, 60 - recipe.landmark],
        "paths.dir": str(output),
        "paths.run_dir": str(logs),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pd.Series(value).to_json(temporary, force_ascii=False, indent=2, double_precision=15)
    temporary.replace(path)


def ranking(table: pd.DataFrame) -> pd.DataFrame:
    ranked = table.copy()
    # 缺失指标排末位；两时点必须均可计算，不能悄悄改成单时点评分。
    for name in ("平均AUC", "logrank强度", "C_index", "平均Brier"):
        ranked[name] = pd.to_numeric(ranked[name], errors="coerce")
    for name in ("平均AUC", "logrank强度"):
        ranked[name + "百分位"] = ranked.groupby("landmark")[name].rank(pct=True).fillna(0)
    ranked["综合分"] = (ranked["平均AUC百分位"] + ranked["logrank强度百分位"]) / 2
    return ranked.sort_values(
        ["综合分", "平均AUC", "C_index", "平均Brier", "recipe_id", "seed"],
        ascending=[False, False, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def select_two(table: pd.DataFrame) -> list[str]:
    ranked = ranking(table)
    prediction = ranked.sort_values(
        ["平均AUC", "C_index", "平均Brier", "recipe_id"],
        ascending=[False, False, True, True],
        na_position="last",
    ).iloc[0]["recipe_id"]
    joint = ranked.loc[ranked.recipe_id.ne(prediction)].iloc[0]["recipe_id"]
    return [str(prediction), str(joint)]


def quick_evaluate(recipe: Recipe, source: Path, destination: Path) -> None:
    config = CRCEvaluationConfig.model_validate(
        resolved(
            "evaluate",
            evaluate_values(
                recipe, source, destination.with_suffix(""), destination.parent / "logs"
            ),
        )
    )
    frames = {
        name: pd.read_csv(source / name / "patient_outputs.csv", dtype={"patient_id": "string"})
        for name in DATASETS
    }
    original = pd.read_csv(source / "metrics.csv").set_index("split")
    collected: dict[str, list[pd.DataFrame]] = {}
    for name, frame in frames.items():
        prediction = TrailsPrediction.load(source / name / "model_prediction.pt")
        probability = prediction.predict_proba().numpy()
        frame["confidence"] = probability.max(axis=1)
        frame["entropy"] = -xlogy(probability, probability).sum(axis=1) / np.log(recipe.k)
        sizes = cluster_statistics(frame, recipe.k)
        km, _, separation = cluster_survival(frame, recipe.k, config.survival.months)
        metrics, curve, calibration, overall = survival_evaluation(
            frame, frames["train"], prediction, config
        )
        if metrics["Harrell_C-index"] is not None and not np.isclose(
            metrics["Harrell_C-index"],
            original.loc[name, "cindex"],
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("聚合C-index与05不一致")
        horizons = curve.set_index("月份").loc[config.survival.months]
        auc = horizons["动态AUC"].to_numpy(dtype=float)
        brier = horizons["Brier"].to_numpy(dtype=float)
        p_value = separation["总体log-rank_p值"]
        row = {
            "数据集": name,
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
                table = table.assign(术后月份=table["月份"] + recipe.landmark)
            collected.setdefault(label, []).append(table.assign(数据集=name))
    write_json(
        destination,
        {label: pd.concat(tables).to_dict("records") for label, tables in collected.items()},
    )


##################################################
# 二、聚合图：分阶段比较与最终重复，不导出患者级坐标。
##################################################
def plot_exploration(
    metrics: pd.DataFrame, repeats: pd.DataFrame, ari: pd.DataFrame, output: Path
) -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [
        name
        for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "PingFang SC", "Songti SC", "SimHei")
        if name in available
    ]
    if not fonts:
        raise ValueError("未找到中文字体")
    plt.rcParams.update({"font.family": fonts, "axes.unicode_minus": False, "pdf.fonttype": 42})
    selected = metrics.loc[metrics["数据集"].eq("test") & metrics.seed.eq(SEEDS[0])]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    for axis, landmark in zip(axes[0], (12, 24), strict=True):
        part = selected.loc[selected.landmark.eq(landmark)]
        for stage, group in part.groupby("阶段", sort=False):
            axis.scatter(group["平均AUC"], group["logrank强度"], label=f"阶段{stage}", alpha=0.7)
        flagged = part.loc[part["占用标记"].ne("合格")]
        axis.scatter(
            flagged["平均AUC"],
            flagged["logrank强度"],
            marker="x",
            color="black",
            label="空簇或小簇",
        )
        axis.axvline((0.572162 + 0.612353) / 2, color="gray", linestyle=":", label="原24月模型参考")
        axis.set(
            title=f"{landmark}个月landmark",
            xlabel="术后3年与5年平均AUC",
            ylabel="-log10（log-rank p）",
        )
        axis.legend(fontsize=8)
    figure.suptitle("测试集参与选优的探索结果；不同方案的合格队列可能不同")
    save_figure(figure, output / "prediction_subtyping", 180)
    if not repeats.empty:
        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        for index, (key, group) in enumerate(repeats.groupby("recipe_id", sort=False)):
            label = (
                f"{group.landmark.iloc[0]}月 {group.panel.iloc[0]} K={group.k.iloc[0]}\n"
                f"{str(key)[-10:]}"
            )
            axes[0].scatter(np.full(len(group), index), group["平均AUC"], alpha=0.8)
            axes[0].plot(index, group["平均AUC"].mean(), "k_")
            relevant = ari.loc[ari.recipe_id.eq(key) & ari["数据集"].eq("test")]
            axes[1].scatter(np.full(len(relevant), index), relevant["ARI"], alpha=0.8)
            for axis in axes:
                axis.text(
                    index,
                    -0.12,
                    label,
                    ha="center",
                    va="top",
                    transform=axis.get_xaxis_transform(),
                    fontsize=8,
                )
        for axis in axes:
            axis.set_xticks([])
        axes[0].set(title="三个seed的预测结果（横线为均值）", ylabel="术后3年与5年平均AUC")
        axes[1].set(title="同方案测试集跨seed稳定性", ylabel="ARI")
        save_figure(figure, output / "seed_repeats", 180)


def main() -> int:
    parser = argparse.ArgumentParser(description="CRC OS分阶段探索；验证集早停，测试集参与选优")
    parser.add_argument("output", type=Path, help="新的探索结果目录")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/derived/crc_yunnan/splits/v2/random/seed-20260908"),
    )
    parser.add_argument("--budget-hours", type=float, default=15)
    parser.add_argument("--dry-run", action="store_true", help="解析全部参数模板，不训练或创建输出")
    parser.add_argument("--resume", action="store_true", help="继续同一脚本及配置下的既有探索")
    args = Options.model_validate(vars(parser.parse_args(sys.argv[2:])))
    root, split = args.output.resolve(), args.split_dir.resolve()
    script_hash = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
    base = Recipe(24, "E60", "auto-log1p")
    run_config = resolved("run", run_values(base, SEEDS[0], root / "preproc", root / "runs"))
    preproc_config = resolved(
        "preproc", preproc_values(base, split, root / "preproc", root / "logs")
    )
    contract = {
        "脚本SHA256": script_hash,
        "预算小时": args.budget_hours,
        "run": run_config,
        "preproc": preproc_config,
        "evaluate": resolved(
            "evaluate", evaluate_values(base, root / "runs", root / "evaluation", root / "logs")
        ),
    }
    stage_a = [
        Recipe(landmark, panel, log, k=k)
        for panel, log, k, landmark in product(
            ("A2", "B8", "C12", "D20", "E60", "F88"),
            ("none", "auto-log1p"),
            (2, 3),
            (12, 24),
        )
    ]
    if args.dry_run:
        # 自适应阶段的赢家尚未知；验证全部取值及组合结构，不把模板当作实际入围名单。
        for landmark, panel, log, scaling in product(
            (12, 24),
            ("A2", "B8", "C12", "D20", "E60", "F88"),
            ("none", "auto-log1p"),
            ("robust", "standard", "minmax"),
        ):
            resolved(
                "preproc",
                preproc_values(
                    Recipe(landmark, panel, log, scaling), split, root / "preproc", root / "logs"
                ),
            )
        for k, seed in product((2, 3, 4, 5), SEEDS):
            for changes in ({}, *(value for _, value in PROFILES.values())):
                resolved(
                    "run",
                    run_values(
                        replace(base, k=k).changed(changes), seed, root / "preproc", root / "runs"
                    ),
                )
        combined_overrides: dict[str, Any] = {}
        for _, changes in PROFILES.values():
            combined_overrides.update(changes)
        resolved(
            "run",
            run_values(base.changed(combined_overrides), SEEDS[0], root / "preproc", root / "runs"),
        )
        for landmark in (12, 24):
            resolved(
                "evaluate",
                evaluate_values(
                    replace(base, landmark=landmark),
                    root / "runs",
                    root / "evaluation",
                    root / "logs",
                ),
            )
        print(
            json.dumps(
                {
                    "阶段训练上限": STAGE_LIMITS,
                    "训练总上限": sum(STAGE_LIMITS.values()),
                    "阶段A唯一方案数": len({recipe.key for recipe in stage_a}),
                    "阶段A预处理数": len({recipe.data_key for recipe in stage_a}),
                    "说明": "B至E根据已完成指标动态入围；全部参数模板解析通过",
                },
                ensure_ascii=False,
            )
        )
        for recipe in stage_a:
            print(
                shlex.join(
                    command(
                        "run",
                        run_values(
                            recipe,
                            SEEDS[0],
                            root / "preproc" / recipe.data_key,
                            root / "runs" / recipe.key,
                        ),
                    )
                )
            )
        return 0

    ##################################################
    # 三、锁定目录并恢复执行状态；匹配产物复用，失败尝试保留。
    ##################################################
    if root.exists() and not args.resume:
        raise FileExistsError(f"拒绝覆盖既有探索目录：{root}；续跑使用--resume")
    if args.resume and not (root / "resolved_config.yaml").is_file():
        raise FileNotFoundError("续跑目录缺少resolved_config.yaml")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".batch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        if (
            OmegaConf.to_container(OmegaConf.load(root / "resolved_config.yaml"), resolve=True)
            != contract
        ):
            raise ValueError("脚本、配置、路径或预算与既有探索不同，必须使用新目录")
    else:
        OmegaConf.save(contract, root / "resolved_config.yaml")
    summary_dir = root / "summary"
    summary_dir.mkdir(exist_ok=True)
    progress_path, plan_path = root / "progress.csv", root / "plan.json"
    progress: dict[str, dict[str, Any]] = (
        {
            row["任务"]: row
            for row in pd.read_csv(progress_path, keep_default_na=False).to_dict("records")
        }
        if progress_path.exists()
        else {}
    )
    plan: dict[str, Any] = (
        json.loads(plan_path.read_text())
        if plan_path.exists()
        else {
            "开始时间": time.time(),
            "阶段": {},
            "最终方案": [],
            "预算停止": False,
        }
    )
    invocation_start = time.time()
    consumed_before = sum(float(row["用时秒"]) for row in progress.values())
    attempted: set[str] = set()
    quick_dir = summary_dir / "quick"
    quick_dir.mkdir(exist_ok=True)

    def persist() -> None:
        write_json(plan_path, plan)
        temporary = progress_path.with_suffix(".tmp")
        pd.DataFrame(progress.values()).to_csv(temporary, index=False)
        temporary.replace(progress_path)

    def current(stage: str, recipe: Recipe, seed: int, log: Path, status: str = "running") -> None:
        completed = sum(
            row["类型"] == "train" and row["状态"] == "success" for row in progress.values()
        )
        content = {
            "status": status,
            "stage": stage,
            "recipe": recipe.key,
            "landmark": recipe.landmark,
            "panel": recipe.panel,
            "k": recipe.k,
            "seed": seed,
            "completed": completed,
            "maximum": 96,
            "elapsed_seconds": round(consumed_before + time.time() - invocation_start),
            "log": str(log),
        }
        (root / "current_run.txt").write_text(
            "\n".join(f"{key}={value}" for key, value in content.items()) + "\n"
        )

    def launch(
        task: str,
        kind: str,
        stage: str,
        recipe: Recipe,
        seed: int,
        values: dict[str, Any],
        completion: str,
    ) -> Path | None:
        previous = progress.get(task)
        if previous is not None and previous["状态"] == "success":
            output = Path(previous["输出目录"])
            if not (output / completion).is_file():
                raise FileNotFoundError(f"完成记录缺失产物：{output / completion}")
            return output
        if task in attempted:
            return None
        attempted.add(task)
        if previous is not None and previous["状态"] == "running" and previous.get("PID"):
            running = subprocess.run(
                ["ps", "-p", str(int(previous["PID"])), "-o", "args="],
                capture_output=True,
                text=True,
                check=False,
            )
            if previous["输出目录"] in running.stdout:
                raise RuntimeError(f"上次子进程仍在运行，不能重复提交：{task}")
        # 中断后若入口的最后完成文件已落盘，仅恢复记录，不重复训练。
        if (
            previous is not None
            and previous["状态"] == "running"
            and (Path(previous["输出目录"]) / completion).is_file()
        ):
            previous["状态"] = "success"
            persist()
            return Path(previous["输出目录"])
        attempt = int(previous["尝试"]) + 1 if previous else 1
        output = Path(values["paths.dir"])
        if output.exists():
            output = output.with_name(output.name + f"-attempt{attempt}")
            values = values | {"paths.dir": str(output)}
        name = {"train": "run", "preproc": "preproc", "evaluate": "evaluate"}[kind]
        values = values | (
            {"paths.run_dir": str(root / "logs" / f"{task}-attempt{attempt}-hydra")}
            if kind != "train"
            else {}
        )
        argv = command(name, values)
        log = root / "logs" / f"{task}-attempt{attempt}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        old_seconds = float(previous["用时秒"]) if previous else 0.0
        progress[task] = {
            "任务": task,
            "类型": kind,
            "阶段": stage,
            "recipe_id": recipe.key,
            "seed": seed,
            "配方": json.dumps(asdict(recipe), ensure_ascii=False),
            "状态": "running",
            "尝试": attempt,
            "退出码": "",
            "用时秒": old_seconds,
            "输出目录": str(output),
            "日志": str(log),
            "命令": shlex.join(argv),
        }
        persist()
        current(stage, recipe, seed, log)
        print(f"开始 {stage} {kind} {recipe.key} seed={seed} attempt={attempt}", flush=True)
        started = time.time()
        with log.open("w") as stream:
            with subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT) as process:
                progress[task]["PID"] = process.pid
                persist()
                exit_code = process.wait()
        elapsed = time.time() - started
        progress[task].update(
            {
                "状态": "success" if exit_code == 0 else "failed",
                "退出码": exit_code,
                "用时秒": old_seconds + elapsed,
            }
        )
        persist()
        print(f"结束 {stage} {kind} exit={exit_code} 用时={elapsed:.0f}s", flush=True)
        return output if exit_code == 0 else None

    def fit(recipe: Recipe, seed: int, stage: str) -> None:
        data_task = "preproc-" + recipe.data_key
        data = launch(
            data_task,
            "preproc",
            stage,
            recipe,
            0,
            preproc_values(
                recipe,
                split,
                root / "preproc" / recipe.data_key,
                root / "logs" / data_task,
            ),
            "preproc_manifest.json",
        )
        if data is None:
            return
        task = recipe.key + f"-s{seed}"
        source = launch(
            task,
            "train",
            stage,
            recipe,
            seed,
            run_values(recipe, seed, data, root / "runs" / task),
            "run_manifest.json",
        )
        if source is None or (quick_dir / f"{task}.json").is_file():
            return
        log = root / "logs" / f"{task}-metrics.log"
        current(stage + "-metrics", recipe, seed, log)
        # 指标失败后其余组合仍可探索；保留traceback，续跑时只重算指标。
        try:
            quick_evaluate(recipe, source, quick_dir / f"{task}.json")
        except Exception:
            log.write_text(traceback.format_exc())
            progress[task]["指标状态"] = "failed"
            print(f"指标失败 {task}，见 {log}", flush=True)
        else:
            progress[task]["指标状态"] = "success"
        persist()

    def all_tables() -> dict[str, pd.DataFrame]:
        collected: dict[str, list[pd.DataFrame]] = {}
        for task, step in progress.items():
            path = quick_dir / f"{task}.json"
            if step["类型"] != "train" or step["状态"] != "success" or not path.is_file():
                continue
            recipe = recipe_from(json.loads(step["配方"]))
            for label, rows in json.loads(path.read_text()).items():
                metadata = {
                    "recipe_id": recipe.key,
                    "landmark": recipe.landmark,
                    "panel": recipe.panel,
                    "log": recipe.log,
                    "scaling": recipe.scaling,
                    "k": recipe.k,
                    "seed": int(step["seed"]),
                    "阶段": step["阶段"],
                    "运行目录": step["输出目录"],
                    "训练参数覆盖": json.dumps(dict(recipe.changes), ensure_ascii=False),
                }
                collected.setdefault(label, []).append(pd.DataFrame(rows).assign(**metadata))
        return {label: pd.concat(tables, ignore_index=True) for label, tables in collected.items()}

    def exploration_scores() -> pd.DataFrame:
        tables = all_tables()
        if not tables:
            return pd.DataFrame()
        metrics = tables["指标"]
        return ranking(metrics.loc[metrics["数据集"].eq("test") & metrics.seed.eq(SEEDS[0])])

    def budget_available() -> bool:
        if plan["预算停止"]:
            return False
        durations = [
            float(row["用时秒"]) / int(row["尝试"])
            for row in progress.values()
            if row["类型"] == "train" and row["状态"] == "success"
        ]
        estimate = (
            max(6.7 * 60, float(np.quantile(durations[-12:], 0.75))) if durations else 6.7 * 60
        )
        # 为8次重复及4次完整评价预留时间，主训练不因时间预算而缩水。
        needed = 9 * estimate + 4 * 180 + 300
        elapsed = consumed_before + time.time() - invocation_start
        if elapsed + needed > args.budget_hours * 3600:
            plan["预算停止"] = True
            plan["预算说明"] = {"已用秒": elapsed, "单次估计秒": estimate, "预留秒": needed}
            persist()
            return False
        return True

    ##################################################
    # 四、按已完成的测试聚合指标逐阶段入围；计划一旦形成即冻结。
    ##################################################
    for stage in ("A", "B", "C", "D"):
        if stage not in plan["阶段"]:
            items: list[dict[str, Any]] = []
            if stage == "A":
                items = [{"recipe": asdict(recipe), "seed": SEEDS[0]} for recipe in stage_a]
            elif not plan["预算停止"]:
                scores = exploration_scores()
                if scores.empty:
                    plan["停止原因"] = "没有成功的候选指标，无法继续自适应探索"
                    persist()
                    break
                recipes = {
                    recipe_from(json.loads(row["配方"])).key: recipe_from(json.loads(row["配方"]))
                    for row in progress.values()
                    if row["类型"] == "train"
                }
                for landmark in (12, 24):
                    candidates = scores.loc[scores.landmark.eq(landmark)]
                    if len(candidates) < 2:
                        plan.setdefault("不足说明", []).append(
                            f"{stage}：{landmark}个月不足两个成功方案"
                        )
                        continue
                    if stage == "B":
                        for key in select_two(candidates):
                            for scaling in ("standard", "minmax"):
                                items.append(
                                    {
                                        "recipe": asdict(replace(recipes[key], scaling=scaling)),
                                        "seed": SEEDS[0],
                                        "参考": key,
                                    }
                                )
                    elif stage == "C":
                        reference = recipes[str(ranking(candidates).iloc[0].recipe_id)]
                        for profile, (factor, changes) in PROFILES.items():
                            items.append(
                                {
                                    "recipe": asdict(reference.changed(changes)),
                                    "seed": SEEDS[0],
                                    "参考": reference.key,
                                    "参数块": factor,
                                    "变化": profile,
                                }
                            )
                    else:
                        profiles = [
                            item
                            for item in plan["阶段"]["C"]
                            if item["recipe"]["landmark"] == landmark
                        ]
                        if not profiles:
                            continue
                        reference = recipes[profiles[0]["参考"]]
                        keys = [
                            reference.key,
                            *(recipe_from(item["recipe"]).key for item in profiles),
                        ]
                        ranked = ranking(candidates.loc[candidates.recipe_id.isin(keys)]).set_index(
                            "recipe_id"
                        )
                        changes: dict[str, Any] = {}
                        for factor in dict.fromkeys(item["参数块"] for item in profiles):
                            choices = [
                                item
                                for item in profiles
                                if item["参数块"] == factor
                                and recipe_from(item["recipe"]).key in ranked.index
                            ]
                            choices.sort(
                                key=lambda item: (
                                    -float(ranked.loc[recipe_from(item["recipe"]).key, "综合分"])
                                )
                            )
                            if (
                                choices
                                and ranked.loc[recipe_from(choices[0]["recipe"]).key, "综合分"]
                                > ranked.loc[reference.key, "综合分"]
                            ):
                                changes.update(dict(recipe_from(choices[0]["recipe"]).changes))
                        combined = reference.changed(changes)
                        items.extend(
                            {
                                "recipe": asdict(replace(combined, k=k)),
                                "seed": SEEDS[0],
                                "参考": reference.key,
                            }
                            for k in dict.fromkeys((reference.k, 4, 5))
                        )
            if stage != "A" and items:
                first = [item for item in items if item["recipe"]["landmark"] == 12]
                second = [item for item in items if item["recipe"]["landmark"] == 24]
                if len(first) == len(second):
                    items = [item for pair in zip(first, second, strict=True) for item in pair]
            plan["阶段"][stage] = items
            persist()
        for item in plan["阶段"][stage]:
            recipe, seed = recipe_from(item["recipe"]), item["seed"]
            task = recipe.key + f"-s{seed}"
            if (
                (quick_dir / f"{task}.json").is_file()
                or (task in progress and progress[task]["状态"] == "success")
                or budget_available()
            ):
                fit(recipe, seed, stage)
            else:
                item["跳过原因"] = "预算预留：进入最终重复阶段"
        persist()

    if "E" not in plan["阶段"]:
        scores = exploration_scores()
        final_keys: list[str] = []
        if not scores.empty:
            for landmark in (12, 24):
                candidates = scores.loc[scores.landmark.eq(landmark)]
                if len(candidates) >= 2:
                    final_keys.extend(select_two(candidates))
        plan["最终方案"] = final_keys
        recipes = {
            recipe_from(json.loads(row["配方"])).key: recipe_from(json.loads(row["配方"]))
            for row in progress.values()
            if row["类型"] == "train"
        }
        plan["阶段"]["E"] = [
            {"recipe": asdict(recipes[key]), "seed": seed}
            for key in final_keys
            for seed in SEEDS[1:]
        ]
        persist()
    for item in plan["阶段"]["E"]:
        fit(recipe_from(item["recipe"]), item["seed"], "E")

    ##################################################
    # 五、三个seed统计、ARI和最终完整评价；既有06产物保持原样。
    ##################################################
    tables = all_tables()
    if not tables:
        write_json(
            summary_dir / "exploration_summary.json",
            {"状态": "失败", "原因": "没有可汇总的候选结果"},
        )
        (root / "current_run.txt").write_text("status=failed\nreason=no_successful_metrics\n")
        return 1
    metrics = tables["指标"]
    tests = metrics.loc[metrics["数据集"].eq("test")]
    ranked = ranking(tests)
    repeats = ranked.loc[ranked.recipe_id.isin(plan["最终方案"])].copy()
    # 每个方案只在自身固定患者名单内对齐，不跨landmark或面板比较ARI。
    ari_rows: list[dict[str, Any]] = []
    for key, group in repeats.groupby("recipe_id", sort=False):
        for name in DATASETS:
            assignments: dict[int, pd.Series] = {}
            for row in group.to_dict("records"):
                frame = pd.read_csv(
                    Path(row["运行目录"]) / name / "patient_outputs.csv",
                    dtype={"patient_id": "string"},
                ).set_index("patient_id")
                assignments[row["seed"]] = frame["pred_cluster"]  # type: ignore[reportArgumentType]
            for first, second in combinations(SEEDS, 2):
                if first not in assignments or second not in assignments:
                    continue
                left, right = assignments[first], assignments[second]
                aligned = right.reindex(left.index)
                if len(left) != len(right) or aligned.isna().any():
                    raise ValueError(f"同方案seed患者不一致：{key}/{name}")
                occupancy = metrics.loc[
                    metrics.recipe_id.eq(key) & metrics["数据集"].eq(name)
                ].set_index("seed")
                ari_rows.append(
                    {
                        "recipe_id": key,
                        "数据集": name,
                        "seed_1": first,
                        "seed_2": second,
                        "人数": len(left),
                        "ARI": adjusted_rand_score(left, aligned),
                        "seed_1占用": occupancy.loc[first, "占用标记"],
                        "seed_2占用": occupancy.loc[second, "占用标记"],
                    }
                )
    ari = pd.DataFrame(
        ari_rows,
        columns=[
            "recipe_id",
            "数据集",
            "seed_1",
            "seed_2",
            "人数",
            "ARI",
            "seed_1占用",
            "seed_2占用",
        ],
    )
    final_rows: list[dict[str, Any]] = []
    for key, group in repeats.groupby("recipe_id", sort=False):
        row = {
            name: group.iloc[0][name]
            for name in ("recipe_id", "landmark", "panel", "log", "scaling", "k")
        }
        complete = len(group) == 3
        row.update({"seed": 0, "完成seed数": len(group), "重复完整": complete})
        for name in ("平均AUC", "AUC_3年", "AUC_5年", "C_index", "平均Brier"):
            values = group[name].to_numpy(dtype=float)
            row[name] = float(values.mean()) if complete else None
            row[name + "最小"] = float(values.min())
            row[name + "最大"] = float(values.max())
        row["logrank强度"] = (
            float(np.median(group["logrank强度"].to_numpy(dtype=float))) if complete else None
        )
        row["空簇seed数"] = int(group["空簇数"].gt(0).sum())
        row["占用合格seed数"] = int(group["占用标记"].eq("合格").sum())
        row["测试ARI均值"] = ari.loc[ari.recipe_id.eq(key) & ari["数据集"].eq("test"), "ARI"].mean()
        final_rows.append(row)
        best = ranking(group).iloc[0]
        step = progress[key + f"-s{int(best.seed)}"]
        recipe = recipe_from(json.loads(step["配方"]))
        launch(
            "evaluate-" + key + f"-s{int(best.seed)}",
            "evaluate",
            "F",
            recipe,
            int(best.seed),
            evaluate_values(
                recipe, Path(best["运行目录"]), root / "evaluation" / key, root / "logs" / key
            ),
            "evaluation_summary.json",
        )
    finals = ranking(pd.DataFrame(final_rows)) if final_rows else pd.DataFrame()
    if not finals.empty:
        tables["最终方案重复汇总"] = finals
        tables["最终方案各seed"] = repeats
    tables["预测排行榜"] = ranked.sort_values(
        ["平均AUC", "C_index", "平均Brier", "recipe_id", "seed"],
        ascending=[False, False, True, True, True],
        na_position="last",
    )
    tables["综合排行榜"] = ranked
    tables["跨seed配对ARI"] = ari
    tables["运行记录"] = pd.DataFrame(progress.values())
    tables["阶段计划"] = pd.DataFrame(
        [
            {
                "阶段": stage,
                "recipe_id": recipe_from(item["recipe"]).key,
                "seed": item["seed"],
                "跳过原因": item.get("跳过原因", ""),
                "参考": item.get("参考", ""),
                "参数变化": item.get("变化", ""),
            }
            for stage, items in plan["阶段"].items()
            for item in items
        ]
    )
    if plan["预算停止"]:
        tables["未展开阶段"] = pd.DataFrame(
            [
                {
                    "阶段": stage,
                    "训练上限": STAGE_LIMITS[stage],
                    "原因": "达到预算预留边界，未生成自适应组合",
                }
                for stage in ("B", "C", "D")
                if not plan["阶段"].get(stage)
            ]
        )
    chinese = {
        "recipe_id": "方案",
        "landmark": "landmark月",
        "panel": "面板",
        "log": "log变换",
        "scaling": "缩放",
        "k": "K",
        "seed": "seed",
        "C_index": "Harrell C-index",
        "IPCW_C_index": "IPCW C-index",
    }
    with pd.ExcelWriter(summary_dir / "exploration_tables.xlsx", engine="openpyxl") as writer:
        for name, table in tables.items():
            translated = table.rename(columns=chinese).replace({"数据集": SPLIT_LABELS})
            translated.to_excel(writer, sheet_name=name, index=False)
    plot_exploration(metrics, repeats, ari, summary_dir / "figures")
    failures = [
        row["任务"]
        for row in progress.values()
        if row["状态"] != "success" or row.get("指标状态") == "failed"
    ]
    complete = len(finals) == 4 and bool(finals["重复完整"].all()) and not failures
    best_average = (
        finals.loc[finals["重复完整"]].head(1).to_dict("records") if not finals.empty else []
    )
    write_json(
        summary_dir / "exploration_summary.json",
        {
            "状态": "完成" if complete else "部分完成",
            "性质": "测试集参与选优的探索结果，不是独立测试评价",
            "结局": "OS",
            "时间口径": "每月30天；AUC目标为术后36/60个月，模型预测时间扣除landmark",
            "训练上限": 96,
            "成功训练数": sum(
                row["类型"] == "train" and row["状态"] == "success" for row in progress.values()
            ),
            "成功预处理数": sum(
                row["类型"] == "preproc" and row["状态"] == "success" for row in progress.values()
            ),
            "预算小时": args.budget_hours,
            "预算停止": plan["预算停止"],
            "失败任务": failures,
            "未完整重复方案": finals.loc[~finals["重复完整"], "recipe_id"].tolist()
            if not finals.empty
            else [],
            "最佳平均综合方案": best_average,
            "最佳单次综合结果": ranked.head(1).to_dict("records"),
            "最佳单次预测结果": tables["预测排行榜"].head(1).to_dict("records"),
            "排序口径": (
                "预测榜优先两时点平均AUC；综合榜按landmark分组，"
                "以AUC和-log10(logrank p)百分位等权评分；缺失排末位"
            ),
            "重复口径": "仅三次齐全时报告平均值，NaN传播；综合榜使用跨seed平均AUC和中位数-log10(p)",
            "解释": [
                "空簇和小簇只标记；单簇不能解释为有效分型",
                "各landmark及面板的合格队列可能不同",
                "06中的独立评价集仅指数据来源，本轮test实际参与选优",
                "全部患者级文件、模型和预测留在远端",
            ],
            "06评价目录": [
                row["输出目录"]
                for row in progress.values()
                if row["类型"] == "evaluate" and row["状态"] == "success"
            ],
            "耗时小时": (consumed_before + time.time() - invocation_start) / 3600,
        },
    )
    persist()
    (root / "current_run.txt").write_text(
        f"status={'completed' if complete else 'partial'}\nsummary={summary_dir}\n"
    )
    print(f"探索{'完成' if complete else '部分完成'}：{summary_dir}", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
PY
