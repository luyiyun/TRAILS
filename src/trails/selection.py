"""候选簇数训练、描述性评价及可迁移的结果重载。"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score

from .artifacts import save_history_csv, save_json
from .config import ClusterNumberEvaluatorConfig, TrailsConfig, resolve_batch_size
from .data import ClinicalTimeSeriesDataset
from .estimator import TrailsEstimator
from .metrics import cluster_assignment_diagnostics, concordance_index, gaussian_log_prob
from .prediction import TrailsPrediction
from .survival import survival_risk

if TYPE_CHECKING:
    from matplotlib.figure import Figure

LOGGER = logging.getLogger(__name__)
_PAIR_COLUMNS = ["n_clusters", "seed_a", "seed_b", "ari"]
_PLOT_LABELS = {
    "cindex": "C-index",
    "latent_mixture_bic": "Latent-mixture BIC",
    "latent_mixture_mean_nll": "Latent-mixture mean NLL",
    "selection_score": "Descriptive composite score",
    "cluster_min_fraction": "Minimum cluster fraction",
    "cluster_entropy": "Normalized cluster entropy",
    "mean_pairwise_ari": "Validation pairwise ARI",
}


@dataclass(frozen=True)
class ClusterNumberEvaluationResult:
    """候选记录与聚合表；不持有模型、患者预测或原始数据。

    candidates中的路径相对于result_dir。run_metrics每个seed/K/dataset一行，
    k_summary按K/dataset汇总；stability_pairs仅描述独立validation上的跨seed ARI。
    """

    config: ClusterNumberEvaluatorConfig
    candidates: list[dict[str, Any]]
    run_metrics: pd.DataFrame
    stability_pairs: pd.DataFrame
    k_summary: pd.DataFrame
    stability_reason: str = ""
    result_dir: Path | None = None

    def to_payload(self) -> dict[str, Any]:
        """返回版本化的轻量元信息，指标本体保存在CSV中。"""
        return {
            "format_version": 1,
            "config": self.config.model_dump(mode="json"),
            "candidates": self.candidates,
            "datasets": self.run_metrics["dataset"].drop_duplicates().tolist(),
            "tables": {
                "run_metrics": "run_metrics.csv",
                "k_summary": "k_summary.csv",
                "stability_pairs": "stability_pairs.csv",
            },
            "stability_reason": self.stability_reason,
        }

    def save(self, result_dir: str | Path | None = None) -> None:
        """保存聚合表和重载元信息；不复制模型或患者级文件。"""
        destination = Path(result_dir) if result_dir is not None else self.result_dir
        if destination is None:
            raise ValueError("保存结果需要result_dir")
        destination.mkdir(parents=True, exist_ok=True)
        self.run_metrics.to_csv(destination / "run_metrics.csv", index=False)
        self.k_summary.to_csv(destination / "k_summary.csv", index=False)
        self.stability_pairs.to_csv(destination / "stability_pairs.csv", index=False)
        save_json(destination / "evaluator.json", self.to_payload())

    def plot_metrics(
        self,
        metrics: Sequence[str] | None = None,
        *,
        dataset: str = "validation",
        path: str | Path | None = None,
    ) -> Figure:
        """绘制指定数据集的候选指标；多seed显示均值及SE，不标记最佳K。"""
        from matplotlib import pyplot as plt

        rows = self.run_metrics.loc[self.run_metrics["dataset"].eq(dataset)]
        if rows.empty:
            raise ValueError(f"结果不包含数据集：{dataset}")
        names = (
            list(metrics)
            if metrics is not None
            else [
                "cindex",
                "latent_mixture_bic",
                "selection_score",
                "cluster_min_fraction",
                "cluster_entropy",
            ]
        )
        if "mean_pairwise_ari" in names and dataset != "validation":
            raise ValueError("跨seed ARI仅描述独立validation，不属于其他数据集")
        if not names:
            raise ValueError("At least one metric is required for plotting.")
        if missing := set(names) - (set(rows.columns) | {"mean_pairwise_ari"}):
            raise ValueError(f"评价结果缺少绘图指标：{sorted(missing)}")
        fig, axes = plt.subplots(
            math.ceil(len(names) / 2),
            min(2, len(names)),
            squeeze=False,
            figsize=(11, 3.6 * math.ceil(len(names) / 2)),
            layout="constrained",
        )
        for axis, metric in zip(axes.flat, names, strict=False):
            if metric == "mean_pairwise_ari":
                values = self.stability_pairs.groupby("n_clusters")["ari"]
            else:
                if not pd.api.types.is_numeric_dtype(rows[metric]):
                    raise ValueError(f"绘图指标必须为数值：{metric}")
                values = rows.groupby("n_clusters")[metric]
            means = pd.Series(values.mean())
            errors = pd.Series(values.sem()).fillna(0.0)
            axis.errorbar(means.index, means, yerr=errors, marker="o", color="#0072B2")
            axis.set(title=_PLOT_LABELS.get(metric, metric), xlabel="Number of clusters (K)")
            axis.set_xticks(sorted(rows["n_clusters"].unique()))
            axis.spines[["top", "right"]].set_visible(False)
        for axis in list(axes.flat)[len(names) :]:
            axis.remove()
        fig.suptitle(f"{dataset}: candidate metrics (mean ± SE)")
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=220)
        return fig


class ClusterNumberEvaluator:
    """逐个训练全部seed/K并报告指标；支持不依赖GPU或源数据的结果浏览。"""

    def __init__(
        self,
        candidates: Sequence[int],
        *,
        seeds: int | Sequence[int] = 2026,
        require_non_empty: bool = True,
        min_cluster_fraction: float | None = None,
        compute_stability: bool = True,
        estimator_config: TrailsConfig | Mapping[str, object] | None = None,
    ) -> None:
        self.config = ClusterNumberEvaluatorConfig(
            candidates=tuple(candidates),
            seeds=(seeds,) if isinstance(seeds, int) else tuple(seeds),
            require_non_empty=require_non_empty,
            min_cluster_fraction=min_cluster_fraction,
            compute_stability=compute_stability,
            estimator=(
                estimator_config
                if isinstance(estimator_config, TrailsConfig)
                else TrailsConfig.model_validate(estimator_config or {})
            ),
        )
        self._result: ClusterNumberEvaluationResult | None = None

    @classmethod
    def from_config(cls, config: ClusterNumberEvaluatorConfig) -> ClusterNumberEvaluator:
        """由Pydantic配置构造尚未拟合的评价器。"""
        evaluator = cls.__new__(cls)
        evaluator.config, evaluator._result = config, None
        return evaluator

    @property
    def result(self) -> ClusterNumberEvaluationResult:
        """访问最近的拟合、汇总或重载结果。"""
        if self._result is None:
            raise RuntimeError("请先fit、summarize_completed或load")
        return self._result

    @property
    def candidates(self) -> list[dict[str, Any]]:
        """候选模型及相对产物路径。"""
        return self.result.candidates

    @property
    def run_metrics(self) -> pd.DataFrame:
        """逐seed/K/dataset指标。"""
        return self.result.run_metrics

    @property
    def k_summary(self) -> pd.DataFrame:
        """按K/dataset汇总的指标。"""
        return self.result.k_summary

    @property
    def stability_pairs(self) -> pd.DataFrame:
        """独立验证集上的跨seed ARI。"""
        return self.result.stability_pairs

    @classmethod
    def load(cls, result_dir: str | Path) -> ClusterNumberEvaluator:
        """只读取轻量元信息和汇总表；允许模型/预测文件未下载。"""
        root = Path(result_dir).resolve()
        payload = json.loads((root / "evaluator.json").read_text(encoding="utf-8"))
        if payload["format_version"] != 1:
            raise ValueError("不支持的evaluator格式；旧选K结果须使用原代码")
        evaluator = cls.from_config(ClusterNumberEvaluatorConfig.model_validate(payload["config"]))
        # relative路径保证搬迁后仍可定位；CSV空理由恢复为空字符串，指标缺失保留NaN。
        tables = {
            name: pd.read_csv(root / relative, float_precision="round_trip")
            for name, relative in payload["tables"].items()
        }
        for column in ("metric_reason", "occupancy_reason"):
            if column in tables["run_metrics"]:
                tables["run_metrics"][column] = tables["run_metrics"][column].fillna("")
        evaluator._result = ClusterNumberEvaluationResult(
            config=evaluator.config,
            candidates=payload["candidates"],
            run_metrics=tables["run_metrics"],
            k_summary=tables["k_summary"],
            stability_pairs=tables["stability_pairs"],
            stability_reason=payload["stability_reason"],
            result_dir=root,
        )
        return evaluator

    def _candidate_path(self, seed: int, n_clusters: int, field: str) -> Path:
        root = self.result.result_dir
        if root is None:
            raise ValueError("此汇总对象没有模型产物目录")
        for row in self.candidates:
            if row["seed"] == seed and row["n_clusters"] == n_clusters:
                return root / row[field]
        raise KeyError(f"没有候选seed={seed}, K={n_clusters}")

    def load_history(self, *, seed: int, n_clusters: int) -> pd.DataFrame:
        """按需读取一个候选的训练历史。"""
        return pd.read_csv(self._candidate_path(seed, n_clusters, "training_history"))

    def load_estimator(self, *, seed: int, n_clusters: int, device: str = "cpu") -> TrailsEstimator:
        """按需恢复一个候选模型；默认CPU，不触发训练。"""
        return TrailsEstimator.load(self._candidate_path(seed, n_clusters, "model"), device=device)

    def load_prediction(self, *, seed: int, n_clusters: int, dataset: str) -> TrailsPrediction:
        """按需读取某个候选在指定数据集上的冻结预测。"""
        if dataset not in set(self.run_metrics["dataset"]):
            raise KeyError(f"没有数据集：{dataset}")
        root = self._candidate_path(seed, n_clusters, "prediction_dir")
        return TrailsPrediction.load(root / dataset / "model_prediction.pt")

    def plot_metrics(
        self,
        metrics: Sequence[str] | None = None,
        *,
        dataset: str = "validation",
        path: str | Path | None = None,
    ) -> Figure:
        """直接绘制当前结果，重载后无需重新计算指标。"""
        return self.result.plot_metrics(metrics, dataset=dataset, path=path)

    def fit(
        self,
        data: ClinicalTimeSeriesDataset,
        *,
        validation_data: ClinicalTimeSeriesDataset | None = None,
        evaluation_data: Mapping[str, ClinicalTimeSeriesDataset | str | Path] | None = None,
        result_dir: str | Path,
    ) -> ClusterNumberEvaluationResult:
        """先保存全部候选，再加载额外评价集并生成预测与汇总。

        不隐式划分train；validation_data仅用于既定早停。额外评价集名称不可占用
        train/validation或候选文件名。异常直接抛出，已保存模型不覆盖、不自动续训。
        """
        root = Path(result_dir).resolve()
        extra = dict(evaluation_data or {})
        reserved = {
            "train",
            "validation",
            "model.pt",
            "training_history.csv",
            "metrics.csv",
            "resolved_config.yaml",
            "config.json",
        }
        if any(
            not name or Path(name).name != name or name in reserved or name in {".", ".."}
            for name in extra
        ):
            raise ValueError("评价集名称必须为不冲突的单级目录名")
        targets = [
            root / name
            for name in (
                "models",
                "evaluator.json",
                "run_metrics.csv",
                "k_summary.csv",
                "stability_pairs.csv",
            )
        ]
        if existing := [str(path) for path in targets if path.exists()]:
            raise FileExistsError(f"拒绝覆盖既有候选产物：{existing}")
        records: list[dict[str, Any]] = []
        base = self.config.estimator
        for seed in self.config.seeds:
            for k in self.config.candidates:
                LOGGER.info(
                    "候选 %d/%d：seed=%d，K=%d",
                    len(records) + 1,
                    len(self.config.seeds) * len(self.config.candidates),
                    seed,
                    k,
                )
                config = base.model_copy(
                    update={
                        "seed": seed,
                        "model": base.model.model_copy(update={"n_clusters": k}),
                        "trainer": base.trainer.model_copy(
                            update={
                                "seed": seed,
                                "valid_size": 0.0,
                                "batch_size": resolve_batch_size(
                                    len(data), base.trainer.batch_size
                                ),
                            }
                        ),
                    }
                )
                estimator = TrailsEstimator(config).fit(data, validation_data=validation_data)
                directory = Path("models") / f"seed-{seed}" / f"k{k}"
                estimator.save(root / directory / "model.pt")
                save_history_csv(root / directory / "training_history.csv", estimator.history)
                # 模型配置使用JSON；命令层另外保存原工作流要求的YAML。
                save_json(
                    root / directory / "config.json", estimator.config.model_dump(mode="json")
                )
                records.append(
                    {
                        "seed": seed,
                        "n_clusters": k,
                        "model": str(directory / "model.pt"),
                        "prediction_dir": str(directory),
                        "training_history": str(directory / "training_history.csv"),
                        "metrics": str(directory / "metrics.csv"),
                        "config": str(directory / "config.json"),
                    }
                )
                del estimator

        datasets = {"train": data}
        if validation_data is not None:
            datasets["validation"] = validation_data
        for name, source in extra.items():
            datasets[name] = (
                source
                if isinstance(source, ClinicalTimeSeriesDataset)
                else ClinicalTimeSeriesDataset.load(source)
            )
        rows: list[dict[str, Any]] = []
        assignments: dict[tuple[int, int], np.ndarray] = {}
        for record in records:
            estimator = TrailsEstimator.load(root / record["model"], device=base.trainer.device)
            for name, dataset in datasets.items():
                prediction = estimator.predict(dataset)
                prediction.save(root / record["prediction_dir"] / name / "model_prediction.pt")
                labels = prediction.predict()
                events = torch.stack([sample.event for sample in dataset.samples])
                metrics = self.evaluate_fitted(estimator, dataset)
                fractions = torch.bincount(labels, minlength=record["n_clusters"]) / len(dataset)
                rows.append(
                    {
                        "seed": record["seed"],
                        "n_clusters": record["n_clusters"],
                        "dataset": name,
                        "n_patients": len(dataset),
                        "n_events": int(events.sum()),
                        "metric_reason": "无事件，C-index不可计算"
                        if not bool(events.any())
                        else "",
                        **metrics,
                        **{f"cluster_{k}_fraction": float(v) for k, v in enumerate(fractions)},
                    }
                )
                if name == "validation" and self.config.compute_stability:
                    assignments[record["seed"], record["n_clusters"]] = labels.numpy()
            del estimator
        pairs = [
            {
                "n_clusters": k,
                "seed_a": a,
                "seed_b": b,
                "ari": float(adjusted_rand_score(assignments[a, k], assignments[b, k])),
            }
            for k in self.config.candidates
            for a, b in combinations(self.config.seeds, 2)
            if (a, k) in assignments and (b, k) in assignments
        ]
        reason = (
            "已关闭稳定性计算"
            if not self.config.compute_stability
            else "没有独立验证集"
            if validation_data is None
            else "不足两个候选seed"
            if len(self.config.seeds) < 2
            else ""
        )
        summary = self.summarize_completed(
            pd.DataFrame(rows), pd.DataFrame(pairs, columns=_PAIR_COLUMNS)
        )
        self._result = ClusterNumberEvaluationResult(
            config=self.config,
            candidates=records,
            run_metrics=summary.run_metrics,
            stability_pairs=summary.stability_pairs,
            k_summary=summary.k_summary,
            stability_reason=reason,
            result_dir=root,
        )
        for record in records:
            selected = self.run_metrics.loc[
                self.run_metrics["seed"].eq(record["seed"])
                & self.run_metrics["n_clusters"].eq(record["n_clusters"])
            ]
            selected.to_csv(root / record["metrics"], index=False)
        self.result.save()
        return self.result

    @staticmethod
    def evaluate_fitted(
        estimator: TrailsEstimator, data: ClinicalTimeSeriesDataset
    ) -> dict[str, float]:
        """沿用潜在混合BIC和风险定义，评价冻结模型；无事件集保留缺失C-index。"""
        outputs, batch = estimator.trainer._collect_outputs(data)
        latent = outputs.latent_mean.detach()
        n_samples = int(latent.shape[0])
        if n_samples == 0:
            raise ValueError("Evaluation metrics require at least one sample.")
        log_prior = torch.log_softmax(estimator.model.mixture_logits.detach(), dim=-1).unsqueeze(0)
        component = gaussian_log_prob(
            latent.unsqueeze(1),
            estimator.model.mixture_means.detach().unsqueeze(0),
            estimator.model.mixture_log_variances.detach().unsqueeze(0),
        ).sum(dim=-1)
        likelihood = torch.logsumexp(log_prior + component, dim=-1)
        k, latent_dim = estimator.config.model.n_clusters, estimator.config.model.latent_dim
        # 保持既有BIC定义，固定混合权重时也保留历史K-1项。
        n_parameters = k * (2 * latent_dim) + k - 1
        labels = outputs.cluster_probabilities.detach().cpu().argmax(dim=-1).long()
        cindex = float("nan")
        if bool(batch["event"].any()):
            cindex = float(
                concordance_index(
                    survival_risk(
                        outputs.survival_log_params,
                        outputs.cluster_probabilities,
                        estimator.config.trainer.risk_horizon,
                        method=estimator.config.trainer.cindex_risk_score,
                    ).detach(),
                    batch["survival_time"],
                    batch["event"],
                )
            )
        return {
            "cindex": cindex,
            "latent_mixture_bic": -2 * float(likelihood.sum().item())
            + math.log(n_samples) * n_parameters,
            "latent_mixture_mean_nll": float((-likelihood).mean().item()),
            "latent_mixture_n_parameters": float(n_parameters),
            **cluster_assignment_diagnostics(labels, n_clusters=k),
        }

    def summarize_completed(
        self,
        run_metrics: pd.DataFrame,
        stability_pairs: pd.DataFrame,
    ) -> ClusterNumberEvaluationResult:
        """汇总既有指标，不训练、不排序或选择最佳K；允许候选缺失。"""
        metrics = run_metrics.copy()
        if "dataset" not in metrics:
            metrics["dataset"] = "validation"
        if metrics.duplicated(["seed", "n_clusters", "dataset"]).any():
            raise ValueError("每个seed/K/dataset只能提供一个候选")
        if set(metrics["seed"]) - set(self.config.seeds) or set(metrics["n_clusters"]) - set(
            self.config.candidates
        ):
            raise ValueError("汇总指标含有未配置的seed或K")
        metrics["latent_mixture_bic_normalized"] = np.nan
        metrics["selection_score"] = np.nan
        for _, rows in metrics.groupby(["dataset", "seed"], sort=False):
            finite = np.isfinite(rows[["cindex", "latent_mixture_bic"]].to_numpy(dtype=float)).all(
                axis=1
            )
            if "status" in rows:
                finite &= rows["status"].eq("completed").to_numpy()
            valid = rows.loc[finite]
            if valid.empty:
                continue
            bics = valid["latent_mixture_bic"]
            width = float(bics.max() - bics.min())
            normalized = (bics - bics.min()) / width if width else bics * 0.0
            metrics.loc[valid.index, "latent_mixture_bic_normalized"] = normalized
            metrics.loc[valid.index, "selection_score"] = np.hypot(valid["cindex"], 1 - normalized)
        empty = (
            metrics["cluster_empty_count"].gt(0)
            if self.config.require_non_empty
            else pd.Series(False, index=metrics.index)
        )
        small = (
            metrics["cluster_min_fraction"].lt(self.config.min_cluster_fraction)
            if self.config.min_cluster_fraction is not None
            else pd.Series(False, index=metrics.index)
        )
        finite_occupancy = np.isfinite(
            metrics[["cluster_min_fraction", "cluster_empty_count"]].to_numpy(dtype=float)
        ).all(axis=1)
        metrics["passes_occupancy"] = ~empty & ~small & finite_occupancy
        metrics["occupancy_reason"] = [
            "; ".join(
                reason
                for reason, applies in (
                    ("存在空簇", e),
                    (f"最小簇占比低于{self.config.min_cluster_fraction or 0:.0%}", s),
                    ("占簇指标缺失", not finite),
                )
                if applies
            )
            for e, s, finite in zip(empty, small, finite_occupancy, strict=True)
        ]
        numeric = [
            col
            for col in metrics.select_dtypes(include="number")
            if col not in {"seed", "n_clusters"}
        ]
        grouped = metrics.groupby(["n_clusters", "dataset"], sort=True)
        summary = grouped[numeric].agg(["mean", "std"])
        summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
        summary = summary.join(
            grouped.agg(
                n_seeds=("seed", "nunique"),
                all_seeds_pass_occupancy=("passes_occupancy", "all"),
                minimum_cluster_fraction=("cluster_min_fraction", "min"),
                maximum_empty_clusters=("cluster_empty_count", "max"),
            )
        ).reset_index()
        summary["complete"] = summary["n_seeds"].eq(len(self.config.seeds))
        summary["mean_pairwise_ari"] = np.nan
        if not stability_pairs.empty:
            means = stability_pairs.groupby("n_clusters")["ari"].mean()
            valid = summary["dataset"].eq("validation")
            summary.loc[valid, "mean_pairwise_ari"] = summary.loc[valid, "n_clusters"].map(means)
        self._result = ClusterNumberEvaluationResult(
            config=self.config,
            candidates=[],
            run_metrics=metrics,
            stability_pairs=stability_pairs.copy(),
            k_summary=summary,
            stability_reason=(
                "已关闭稳定性计算"
                if not self.config.compute_stability
                else "没有独立验证集"
                if "validation" not in set(metrics["dataset"])
                else "不足两个候选seed"
                if len(self.config.seeds) < 2
                else "未提供验证集跨seed配对结果"
                if stability_pairs.empty
                else ""
            ),
        )
        return self.result
