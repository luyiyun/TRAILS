"""候选诊断、训练历史和仅在train拟合的UMAP补充图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from trails import ClusterNumberEvaluator, TrailsPrediction
from trails.artifacts import save_json

from .config import MimicSupplementaryConfig
from .figure_plots import COLORS
from .reporting import EvaluationReport


def plot_history(history: pd.DataFrame, path: Path) -> None:
    """只消费已保存的逐轮指标，不读取或重建模型。"""
    groups = [
        [name for name in history if "loss" in name],
        [name for name in history if "cindex" in name or "cluster" in name],
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for axis, columns in zip(axes, groups, strict=True):
        for column in columns:
            axis.plot(history.index + 1, history[column], label=column)
        axis.set_xlabel("Epoch")
        if columns:
            axis.legend(fontsize=5.5)
    figure.savefig(path)
    plt.close(figure)


def candidate_diagnostics(
    config: MimicSupplementaryConfig, output: Path
) -> dict[str, pd.DataFrame]:
    """保留所有seed/K聚合结果，历史缺失不阻止仅聚合产物浏览。"""
    sources = list(
        dict.fromkeys([config.inputs.run_dir, *(item.run_dir for item in config.comparison_runs)])
    )
    metrics, summaries, pairs, histories = [], [], [], []
    unavailable = []
    for index, source in enumerate(sources):
        evaluator = ClusterNumberEvaluator.load(source)
        directory = output / "candidates" / f"run-{index}"
        directory.mkdir(parents=True, exist_ok=True)
        metrics.append(evaluator.run_metrics.assign(source_run=str(source)))
        summaries.append(evaluator.k_summary.assign(source_run=str(source)))
        pairs.append(evaluator.stability_pairs.assign(source_run=str(source)))
        for dataset in evaluator.run_metrics["dataset"].unique():
            figure = evaluator.plot_metrics(
                dataset=dataset, path=directory / f"k_metrics_{dataset}.pdf"
            )
            plt.close(figure)
        if not evaluator.stability_pairs.empty:
            figure = evaluator.plot_metrics(
                ["mean_pairwise_ari"],
                dataset="validation",
                path=directory / "cross_seed_stability.pdf",
            )
            plt.close(figure)
        else:
            unavailable.append(
                {"source_run": str(source), "stability_reason": evaluator.result.stability_reason}
            )
        for record in evaluator.candidates:
            if not (source / record["training_history"]).is_file():
                unavailable.append(
                    {
                        "source_run": str(source),
                        "seed": record["seed"],
                        "n_clusters": record["n_clusters"],
                        "reason": "History file not downloaded",
                    }
                )
                continue
            history = evaluator.load_history(seed=record["seed"], n_clusters=record["n_clusters"])
            histories.append(
                history.assign(
                    source_run=str(source), seed=record["seed"], n_clusters=record["n_clusters"]
                )
            )
            plot_history(
                history, directory / f"history_seed-{record['seed']}_k{record['n_clusters']}.pdf"
            )
    tables = {
        "candidate_metrics": pd.concat(metrics, ignore_index=True),
        "candidate_k_summary": pd.concat(summaries, ignore_index=True),
        "candidate_stability": pd.concat(pairs, ignore_index=True),
        "candidate_history": pd.concat(histories, ignore_index=True)
        if histories
        else pd.DataFrame(),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    save_json(output / "candidate_unavailable.json", {"items": unavailable})
    return tables


def plot_umap(report: EvaluationReport, config: MimicSupplementaryConfig) -> pd.DataFrame:
    """训练集拟合UMAP，其余集仅transform；输出图及二维聚合箱，不导出患者坐标。"""
    directory = Path(report.methods[0]["directory"])
    predictions = {
        name: TrailsPrediction.load(directory / name / "model_prediction.pt")
        for name in report.datasets
    }
    latent = {
        name: prediction.latent_representation.numpy() for name, prediction in predictions.items()
    }
    if len(latent["train"]) < 4:
        save_json(
            report.output / "umap_summary.json",
            {"unavailable_reason": "At least four training patients required"},
        )
        return pd.DataFrame()
    reducer = umap.UMAP(
        n_neighbors=min(15, len(latent["train"]) - 1),
        random_state=config.umap_seed,
        transform_seed=config.umap_seed,
        n_jobs=1,
    )
    coordinates = {"train": np.asarray(reducer.fit_transform(latent["train"]))}
    coordinates.update(
        {
            name: np.asarray(reducer.transform(values))
            for name, values in latent.items()
            if name != "train"
        }
    )
    figure, axes = plt.subplots(1, len(coordinates), figsize=(12, 3.7), layout="constrained")
    rows = []
    for axis, (name, xy) in zip(np.atleast_1d(axes), coordinates.items(), strict=True):
        labels = predictions[name].predict().numpy()
        for cluster in range(config.n_clusters):
            values = xy[labels == cluster]
            axis.scatter(
                values[:, 0],
                values[:, 1],
                s=5,
                alpha=0.5,
                color=COLORS[cluster % len(COLORS)],
                label=f"C{cluster}",
                rasterized=True,
            )
            if len(values):
                counts, xedges, yedges = np.histogram2d(values[:, 0], values[:, 1], bins=25)
                xx, yy = np.meshgrid(
                    (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2, indexing="ij"
                )
                rows.append(
                    pd.DataFrame(
                        {
                            "dataset": name,
                            "cluster": cluster,
                            "umap1_bin": xx.ravel(),
                            "umap2_bin": yy.ravel(),
                            "n_patients": counts.ravel().astype(int),
                        }
                    )
                )
        axis.set(title=f"{name}: train-fitted UMAP", xlabel="UMAP 1", ylabel="UMAP 2")
    axes[0].legend(frameon=False)
    for suffix in ("pdf", "png"):
        figure.savefig(report.output / f"umap.{suffix}", dpi=300)
    plt.close(figure)
    save_json(
        report.output / "umap_summary.json",
        {"fit_dataset": "train", "seed": config.umap_seed, "patient_coordinates_exported": False},
    )
    return pd.concat(rows, ignore_index=True)


def plot_agreement(report: EvaluationReport) -> None:
    """全部重复的ARI/NMI仅解释为标签一致性。"""
    for dataset in report.datasets:
        pairs = report.agreement.loc[report.agreement["split"].eq(dataset)]
        keys = [item["key"] for item in report.methods if "cluster" in item["capabilities"]]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for axis, metric in zip(axes, ("ari", "nmi"), strict=True):
            matrix = pd.DataFrame(np.eye(len(keys)), index=keys, columns=keys)
            for _, row in pairs.iterrows():
                matrix.loc[row["left"], row["right"]] = matrix.loc[row["right"], row["left"]] = row[
                    metric
                ]
            image = axis.imshow(matrix, vmin=-1 if metric == "ari" else 0, vmax=1, cmap="cividis")
            axis.set_xticks(range(len(keys)), keys, rotation=60, ha="right")
            axis.set_yticks(range(len(keys)), keys)
            axis.set_title(f"{dataset}: {metric.upper()} label agreement")
            figure.colorbar(image, ax=axis, shrink=0.6)
        figure.savefig(report.output / f"label_agreement_{dataset}.pdf")
        plt.close(figure)
