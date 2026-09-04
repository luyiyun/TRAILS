"""TRAILS 候选簇数选择器及其结构化结果对象。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score

from .artifacts import save_json
from .config import ClusterNumberSelectorConfig, TrailsConfig
from .data import ClinicalTimeSeriesDataset
from .estimator import TrailsEstimator
from .metrics import (
    cluster_assignment_diagnostics,
    concordance_index,
    gaussian_log_prob,
    weibull_event_probability,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


_PLOT_METRIC_LABELS = {
    "cindex": "Validation C-index",
    "latent_mixture_bic": "Latent-mixture BIC",
    "latent_mixture_mean_nll": "Latent-mixture mean NLL",
    "selection_score": "Selection score",
    "cluster_min_fraction": "Minimum cluster fraction",
    "cluster_max_fraction": "Maximum cluster fraction",
    "cluster_entropy": "Normalized cluster entropy",
    "mean_pairwise_ari": "Mean pairwise ARI",
}
_DEFAULT_PLOT_METRICS = (
    "cindex",
    "latent_mixture_bic",
    "selection_score",
    "cluster_min_fraction",
    "cluster_entropy",
)


@dataclass(frozen=True)
class ClusterNumberSelectionResult:
    """保存候选 K 在一个或多个随机种子下的完整选择结果。

    属性：
        config: 本次选择使用的已校验配置。
        selected_k: 根据配置规则选出的 K；没有候选通过门槛时为 ``None``。
        run_metrics: 每个 ``seed × K`` 一行的评价和选择指标，包括 C-index、
            潜在混合分布 BIC、簇占用、归一化 BIC、复合分数和 seed 内排名。
        stability_pairs: 同一 K 下不同 seed 预测标签的两两稳定性指标；单 seed
            运行时为空表。
        k_summary: 按 K 聚合的复合分数、标准误、C-index、簇占用、稳定性和
            门槛结果。
        seed_winners: 每个 seed 独立排名第一的 K。
        estimators: 以 ``(seed, K)`` 为键保存的已拟合估计器。

    结果对象不替多 seed 运行决定最终代表模型；论文或应用流程可从
    :attr:`selected_estimators` 获取入选 K 的所有估计器并使用自己的锁模规则。
    """

    config: ClusterNumberSelectorConfig
    selected_k: int | None
    run_metrics: pd.DataFrame
    stability_pairs: pd.DataFrame
    k_summary: pd.DataFrame
    seed_winners: dict[int, int]
    estimators: dict[tuple[int, int], TrailsEstimator]

    @property
    def selected_estimators(self) -> dict[int, TrailsEstimator]:
        """返回入选 K 按 seed 组织的估计器；未选出 K 时返回空字典。"""
        if self.selected_k is None:
            return {}
        return {
            seed: estimator
            for (seed, n_clusters), estimator in self.estimators.items()
            if n_clusters == self.selected_k
        }

    def to_payload(self) -> dict[str, object]:
        """将配置、选择结论和各层指标转换为 JSON 兼容载荷。

        返回：
            包含配置、入选 K、各 seed 最优 K、逐运行指标、稳定性和 K 汇总的
            字典；已拟合估计器不包含在载荷中。
        """

        def records(frame: pd.DataFrame) -> object:
            serialized = frame.to_json(orient="records")
            assert serialized is not None
            return json.loads(serialized)

        return {
            "config": self.config.model_dump(mode="json"),
            "selected_k": self.selected_k,
            "seed_winners": self.seed_winners,
            "run_metrics": records(self.run_metrics),
            "stability_pairs": records(self.stability_pairs),
            "k_summary": records(self.k_summary),
        }

    def save(self, result_dir: str | Path) -> None:
        """保存选择表、JSON 摘要及每个 ``seed × K`` 候选估计器。

        参数：
            result_dir: 结果根目录。总体表保存在根目录，每个候选的模型、历史、
                指标和配置保存在 ``seed-<seed>/k<K>/`` 下。
        """
        root = Path(result_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.run_metrics.to_csv(root / "run_metrics.csv", index=False)
        self.stability_pairs.to_csv(root / "stability_pairs.csv", index=False)
        self.k_summary.to_csv(root / "k_summary.csv", index=False)
        save_json(root / "selection_summary.json", self.to_payload())
        for (seed, n_clusters), estimator in self.estimators.items():
            candidate_dir = root / f"seed-{seed}" / f"k{n_clusters}"
            estimator.save(candidate_dir / "model.pt")
            save_json(candidate_dir / "history.json", estimator.history)
            candidate_metrics = self.run_metrics.loc[
                (self.run_metrics["seed"] == seed) & (self.run_metrics["n_clusters"] == n_clusters)
            ].iloc[0]
            serialized = candidate_metrics.to_json()
            assert serialized is not None
            save_json(candidate_dir / "metrics.json", json.loads(serialized))
            save_json(candidate_dir / "config.json", estimator.config.model_dump(mode="json"))

    def plot_metrics(
        self,
        metrics: Sequence[str] | None = None,
        *,
        path: str | Path | None = None,
    ) -> Figure:
        """绘制候选 K 变化时的选择指标，并可保存为静态图片。

        每个指标独占一个面板。多 seed 结果显示跨 seed 均值及标准误阴影；
        单 seed 只显示折线。若已选出 K，则在所有面板中用竖直虚线标记。

        参数：
            metrics: ``run_metrics`` 或 ``k_summary`` 中需要绘制的数值列。
                默认绘制 C-index、latent-mixture BIC、综合选择分、最小簇比例
                和簇熵；多 seed 有稳定性结果时还会绘制平均成对 ARI。
            path: 可选输出路径，图片格式由扩展名决定。

        返回：
            未关闭的 Matplotlib Figure，调用方可继续调整或自行关闭。
        """
        from matplotlib import pyplot as plt

        selected_metrics: list[str]
        if metrics is None:
            selected_metrics = list(_DEFAULT_PLOT_METRICS)
            if "mean_pairwise_ari" in self.k_summary and bool(
                np.asarray(self.k_summary["mean_pairwise_ari"].notna()).any()
            ):
                selected_metrics.append("mean_pairwise_ari")
        else:
            selected_metrics = list(metrics)
        if not selected_metrics:
            raise ValueError("At least one metric is required for plotting.")
        missing = [
            metric
            for metric in selected_metrics
            if metric not in self.run_metrics and metric not in self.k_summary
        ]
        if missing:
            raise ValueError(f"选择结果缺少绘图指标：{missing}")
        non_numeric = [
            metric
            for metric in selected_metrics
            if not pd.api.types.is_numeric_dtype(
                self.run_metrics[metric] if metric in self.run_metrics else self.k_summary[metric]
            )
        ]
        if non_numeric:
            raise ValueError(f"绘图指标必须是数值列：{non_numeric}")

        grouped = self.run_metrics.groupby("n_clusters", sort=True)
        ncols = min(2, len(selected_metrics))
        nrows = math.ceil(len(selected_metrics) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6.0 * ncols, 3.8 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        plot_axes = list(axes.flat)
        n_seeds = self.run_metrics["seed"].nunique()
        color = "#2F6B9A"

        for ax, metric in zip(plot_axes, selected_metrics, strict=False):
            has_uncertainty = metric in self.run_metrics and n_seeds > 1
            if metric in self.run_metrics:
                mean = cast(pd.Series, grouped[metric].mean())
                standard_error = cast(pd.Series, grouped[metric].sem()).fillna(0.0)
            else:
                summary = self.k_summary.sort_values("n_clusters").set_index("n_clusters")
                mean = summary[metric]
                standard_error = pd.Series(0.0, index=mean.index)
            x_values = np.asarray(mean.index.tolist(), dtype=np.int64)
            y_values = mean.to_numpy().astype(np.float64, copy=False)
            ax.plot(x_values, y_values, color=color, marker="o", linewidth=2.0)
            if has_uncertainty:
                error = standard_error.to_numpy().astype(np.float64, copy=False)
                ax.fill_between(
                    x_values, y_values - error, y_values + error, color=color, alpha=0.18
                )
            if self.selected_k is not None and self.selected_k in x_values:
                ax.axvline(self.selected_k, color="#333333", linestyle="--", linewidth=1.2)
            ax.set_title(_PLOT_METRIC_LABELS.get(metric, metric), loc="left", fontsize=11)
            ax.set_xticks(x_values)
            ax.grid(axis="y", color="#D9DEE3", linewidth=0.8, alpha=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for ax in plot_axes[len(selected_metrics) :]:
            ax.remove()
        fig.supxlabel("Number of clusters (K)")
        uncertainty = "mean ± SE" if n_seeds > 1 else "single seed"
        fig.suptitle(f"Cluster-number selection metrics ({uncertainty})", fontsize=13)
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, dpi=180)
        return fig


class ClusterNumberSelector:
    """通过一个或多个随机种子比较候选 K，并返回结构化选择结果。

    直接参数生成 Pydantic 配置，也可用 :meth:`from_config` 恢复；所有候选
    共用同一数据划分，先在 seed 内评分，再跨 seed 汇总。
    """

    def __init__(
        self,
        candidates: Sequence[int],
        *,
        seeds: int | Sequence[int] = 2026,
        split_seed: int = 2026,
        valid_fraction: float = 0.2,
        selection_rule: Literal["best_mean", "one_standard_error"] = "best_mean",
        require_non_empty: bool = False,
        min_cluster_fraction: float | None = None,
        min_mean_pairwise_ari: float | None = None,
        estimator_config: TrailsConfig | Mapping[str, object] | None = None,
    ) -> None:
        """校验直接参数并建立 K 选择配置，尚不训练模型。

        参数：
            candidates: 要比较的候选簇数。
            seeds: 单个模型随机种子或重复运行使用的种子序列。
            split_seed: 内部训练/验证划分使用的固定随机种子。
            valid_fraction: 未提供显式验证集时的内部验证比例。
            selection_rule: 选择最高平均分，或应用 one-standard-error 规则。
            require_non_empty: 是否排除任何运行产生空簇的候选 K。
            min_cluster_fraction: 可选的最小簇占比门槛。
            min_mean_pairwise_ari: 可选的多 seed 平均成对 ARI 门槛。
            estimator_config: 基础 TRAILS 配置或可由 Pydantic 解析的映射。
        """
        estimator = (
            estimator_config
            if isinstance(estimator_config, TrailsConfig)
            else TrailsConfig.model_validate(estimator_config or {})
        )
        self.config = ClusterNumberSelectorConfig(
            candidates=tuple(candidates),
            seeds=(seeds,) if isinstance(seeds, int) else tuple(seeds),
            split_seed=split_seed,
            valid_fraction=valid_fraction,
            selection_rule=selection_rule,
            require_non_empty=require_non_empty,
            min_cluster_fraction=min_cluster_fraction,
            min_mean_pairwise_ari=min_mean_pairwise_ari,
            estimator=estimator,
        )

    @classmethod
    def from_config(cls, config: ClusterNumberSelectorConfig) -> ClusterNumberSelector:
        """从已校验配置恢复选择器，适合读取保存的 JSON/YAML 配置。

        参数：
            config: 已通过 Pydantic 校验的选择器配置。

        返回：
            使用原配置且尚未训练候选模型的选择器。
        """
        selector = cls.__new__(cls)
        selector.config = config
        return selector

    def select(
        self,
        data: ClinicalTimeSeriesDataset,
        *,
        validation_data: ClinicalTimeSeriesDataset | None = None,
    ) -> ClusterNumberSelectionResult:
        """训练全部 ``seed × K`` 候选并按配置规则选择 K。

        显式验证集会用于全部运行；否则只按 ``split_seed`` 划分一次。每个 seed
        内先将潜空间混合 BIC 做 min-max 归一化，再按
        ``sqrt(C-index² + (1 - BIC_norm)²)`` 计算复合分数；多 seed 时进一步
        计算验证集簇标签的两两 ARI。没有候选通过配置门槛时不会静默回退。

        参数：
            data: 候选模型的训练数据，或内部划分前的完整训练数据。
            validation_data: 可选的独立验证数据集。

        返回：
            包含入选 K、逐运行指标、稳定性、K 汇总和候选估计器的结果对象。
        """
        if validation_data is None:
            train_data, valid_data = data.split(
                [1.0 - self.config.valid_fraction, self.config.valid_fraction],
                seed=self.config.split_seed,
            )
        else:
            train_data, valid_data = data, validation_data

        run_scores: list[pd.DataFrame] = []
        estimators: dict[tuple[int, int], TrailsEstimator] = {}
        cluster_assignments: dict[tuple[int, int], np.ndarray] = {}
        seed_winners: dict[int, int] = {}
        base_config = self.config.estimator
        for seed in self.config.seeds:
            candidate_metrics: list[dict[str, float | int]] = []
            for n_clusters in self.config.candidates:
                config = base_config.model_copy(
                    update={
                        "seed": seed,
                        "model": base_config.model.model_copy(update={"n_clusters": n_clusters}),
                        "trainer": base_config.trainer.model_copy(
                            update={"seed": seed, "valid_size": 0.0}
                        ),
                    }
                )
                estimator = TrailsEstimator(config).fit(train_data, validation_data=valid_data)
                candidate_metrics.append(
                    {
                        "n_clusters": n_clusters,
                        **self._calculate_candidate_metrics(estimator, valid_data),
                    }
                )
                estimators[(seed, n_clusters)] = estimator
                cluster_assignments[(seed, n_clusters)] = (
                    estimator.predict(valid_data).predict().numpy()
                )
            candidate_scores = self._score_and_rank_candidates(pd.DataFrame(candidate_metrics))
            candidate_scores.insert(0, "seed", seed)
            run_scores.append(candidate_scores)
            seed_winners[seed] = int(candidate_scores.iloc[0]["n_clusters"])

        run_metrics = pd.concat(run_scores, ignore_index=True).sort_values(["n_clusters", "seed"])
        stability_pairs = self._calculate_stability(cluster_assignments)
        k_summary = self._summarize_candidates(run_metrics, stability_pairs)
        selected_k = self._select_k(k_summary)

        return ClusterNumberSelectionResult(
            config=self.config,
            selected_k=selected_k,
            run_metrics=run_metrics,
            stability_pairs=stability_pairs,
            k_summary=k_summary,
            seed_winners=seed_winners,
            estimators=estimators,
        )

    @staticmethod
    def _calculate_candidate_metrics(
        estimator: TrailsEstimator,
        data: ClinicalTimeSeriesDataset,
    ) -> dict[str, float]:
        """计算一个已拟合候选模型的生存、潜空间混合分布和簇占用指标。"""
        outputs, batch = estimator.trainer._collect_outputs(data)
        latent = outputs.latent_mean.detach()
        n_samples = int(latent.shape[0])
        if n_samples == 0:
            raise ValueError("Selection metrics require at least one sample.")

        log_prior = torch.log_softmax(estimator.model.mixture_logits.detach(), dim=-1).unsqueeze(0)
        component_log_prob = gaussian_log_prob(
            latent.unsqueeze(1),
            estimator.model.mixture_means.detach().unsqueeze(0),
            estimator.model.mixture_log_variances.detach().unsqueeze(0),
        ).sum(dim=-1)
        log_likelihood = torch.logsumexp(log_prior + component_log_prob, dim=-1)
        n_clusters = estimator.config.model.n_clusters
        latent_dim = estimator.config.model.latent_dim
        n_parameters = n_clusters * (2 * latent_dim) + (n_clusters - 1)
        latent_mixture_bic = (
            -2.0 * float(log_likelihood.sum().item()) + math.log(float(n_samples)) * n_parameters
        )
        pred_cluster = torch.argmax(outputs.cluster_probabilities.detach().cpu(), dim=-1).long()
        return {
            "cindex": float(
                concordance_index(
                    weibull_event_probability(
                        outputs.weibull_shape,
                        outputs.weibull_scale,
                        estimator.config.trainer.risk_horizon,
                    )
                    .detach()
                    .cpu()
                    .float(),
                    batch["survival_time"].detach().cpu().float(),
                    batch["event"].detach().cpu().float(),
                )
            ),
            "latent_mixture_bic": float(latent_mixture_bic),
            "latent_mixture_mean_nll": float((-log_likelihood).mean().item()),
            "latent_mixture_n_parameters": float(n_parameters),
            **cluster_assignment_diagnostics(pred_cluster, n_clusters=n_clusters),
        }

    @staticmethod
    def _score_and_rank_candidates(candidate_metrics: pd.DataFrame) -> pd.DataFrame:
        """在一个 seed 内归一化潜在混合 BIC，并按复合分数确定候选 K 排名。"""
        if candidate_metrics.empty:
            raise ValueError("K selection requires at least one candidate.")
        candidate_scores = candidate_metrics.copy()
        bics = candidate_scores["latent_mixture_bic"].astype(float)
        bic_range = float(bics.max() - bics.min())
        candidate_scores["latent_mixture_bic_normalized"] = (
            0.0 if bic_range == 0.0 else (bics - bics.min()) / bic_range
        )
        candidate_scores["selection_score"] = np.hypot(
            candidate_scores["cindex"].astype(float),
            1.0 - candidate_scores["latent_mixture_bic_normalized"],
        )
        candidate_scores.sort_values(
            ["selection_score", "cindex", "latent_mixture_bic", "n_clusters"],
            ascending=[False, False, True, True],
            inplace=True,
        )
        candidate_scores["rank"] = np.arange(1, len(candidate_scores) + 1)
        return candidate_scores

    def _calculate_stability(
        self, cluster_assignments: Mapping[tuple[int, int], np.ndarray]
    ) -> pd.DataFrame:
        """计算每个候选 K 在不同 seed 预测标签之间的成对 ARI。"""
        stability_records: list[dict[str, float | int]] = []
        for n_clusters in self.config.candidates:
            for seed_a, seed_b in combinations(self.config.seeds, 2):
                ari = adjusted_rand_score(
                    cluster_assignments[(seed_a, n_clusters)],
                    cluster_assignments[(seed_b, n_clusters)],
                )
                stability_records.append(
                    {"n_clusters": n_clusters, "seed_a": seed_a, "seed_b": seed_b, "ari": ari}
                )
        return pd.DataFrame(stability_records, columns=["n_clusters", "seed_a", "seed_b", "ari"])

    def _summarize_candidates(
        self, run_metrics: pd.DataFrame, stability_pairs: pd.DataFrame
    ) -> pd.DataFrame:
        """按 K 汇总跨 seed 得分、簇占用和稳定性门槛。"""
        summary_records: list[dict[str, float | int | bool]] = []
        for n_clusters in self.config.candidates:
            candidate_runs = run_metrics.loc[run_metrics["n_clusters"] == n_clusters]
            scores = candidate_runs["selection_score"].astype(float)
            aris = stability_pairs.loc[stability_pairs["n_clusters"] == n_clusters, "ari"]
            mean_ari = float(aris.mean()) if not aris.empty else float("nan")
            min_fraction = float(candidate_runs["cluster_min_fraction"].min())
            max_empty = int(candidate_runs["cluster_empty_count"].max())
            passes_gate = not self.config.require_non_empty or max_empty == 0
            if self.config.min_cluster_fraction is not None:
                passes_gate &= min_fraction >= self.config.min_cluster_fraction
            if self.config.min_mean_pairwise_ari is not None:
                passes_gate &= mean_ari >= self.config.min_mean_pairwise_ari
            summary_records.append(
                {
                    "n_clusters": n_clusters,
                    "mean_selection_score": float(scores.mean()),
                    "se_selection_score": 0.0 if len(scores) == 1 else float(scores.sem()),
                    "mean_cindex": float(candidate_runs["cindex"].mean()),
                    "min_cluster_fraction": min_fraction,
                    "max_empty_clusters": max_empty,
                    "mean_pairwise_ari": mean_ari,
                    "passes_gate": passes_gate,
                }
            )
        return pd.DataFrame(summary_records).sort_values("n_clusters")

    def _select_k(self, k_summary: pd.DataFrame) -> int | None:
        """根据门槛和配置规则从 K 汇总表中选择最终簇数。"""
        eligible = k_summary.loc[k_summary["passes_gate"]]
        if eligible.empty:
            return None
        best = eligible.sort_values(
            ["mean_selection_score", "n_clusters"], ascending=[False, True]
        ).iloc[0]
        if self.config.selection_rule == "one_standard_error":
            cutoff = float(best["mean_selection_score"] - best["se_selection_score"])
            return int(eligible.loc[eligible["mean_selection_score"] >= cutoff, "n_clusters"].min())
        return int(best["n_clusters"])
