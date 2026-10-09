from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import hydra
import matplotlib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from omegaconf import DictConfig, OmegaConf
from scripts.crc_yunnan.config import CRCSupplementaryConfig
from scripts.crc_yunnan.evaluation import (
    SPLIT_LABELS,
    EvaluationModel,
    adjusted_cox,
    available_models,
    clinical_characteristics,
    cluster_statistics,
    cluster_survival,
    evaluation_requests,
    evaluation_summary,
    latent_projection,
    load_evaluation_data,
    survival_evaluation,
    trajectory_statistics,
    write_summary,
)
from scripts.crc_yunnan.evaluation_plots import (
    configure_plotting,
    english_label,
    plot_calibration,
    plot_categorical_baseline,
    plot_cluster_sizes,
    plot_confidence,
    plot_cox,
    plot_k_metrics,
    plot_km,
    plot_numeric_baseline,
    plot_overall_survival,
    plot_stability,
    plot_survival_metrics,
    plot_training,
    plot_trajectories,
    plot_trajectory_coverage,
    plot_umap,
)
from sklearn.metrics import adjusted_rand_score

from trails import ClinicalTimeSeriesDataset, TrailsEstimator, TrailsPrediction

matplotlib.use("Agg")

LOGGER = logging.getLogger(__name__)


def run(config: CRCSupplementaryConfig) -> None:
    """对明确指定的K分别输出补充评价，不依赖06产物。"""
    manifest, models = evaluation_requests(config)
    if (config.paths.dir / "k_comparison.csv").exists():
        raise FileExistsError("拒绝覆盖既有k_comparison.csv，请使用新输出目录")
    comparisons = []
    for model in models:
        summary = evaluate_model(config, manifest, model)
        for name, values in summary["数据集"].items():
            metrics = values["生存预测"]
            comparisons.append(
                {
                    "K": model.n_clusters,
                    "seed": model.seed,
                    "数据集": name,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if not isinstance(value, dict | list)
                    },
                    "最小簇占比": values["最小簇占比"],
                }
            )
    pd.DataFrame(comparisons).to_csv(config.paths.dir / "k_comparison.csv", index=False)


def evaluate_model(
    config: CRCSupplementaryConfig, manifest: dict[str, Any], model: EvaluationModel
) -> dict[str, Any]:
    ##################################################
    # 一、共用模型追溯、预测和参考簇，设置独立补充输出。
    ##################################################
    source = config.inputs.run_dir.resolve()
    n_clusters, output = model.n_clusters, model.output
    data = load_evaluation_data(config, manifest, model)
    data_root, metadata, parameters = data.data_root, data.metadata, data.parameters
    predictions, frames, history, reference = (
        data.predictions,
        data.frames,
        data.history,
        data.reference,
    )
    colors = configure_plotting(config, n_clusters)
    figures, dpi, landmark = output / "figures", config.plot.dpi, float(manifest["landmark_days"])
    output.mkdir(parents=True, mode=0o700)
    selected_frames = {name: frames[name] for name in config.datasets}
    train = frames["train"]
    tables: dict[str, pd.DataFrame] = {}
    summary = evaluation_summary(config, manifest, model, data)

    ##################################################
    # 二、按评价集复用统计，保存簇、临床、轨迹和生存图表。
    ##################################################
    collected: dict[str, list[pd.DataFrame]] = {}
    km_models: dict[str, dict[int, KaplanMeierFitter]] = {}
    km_summaries: dict[str, dict[str, Any]] = {}
    cox_summaries: dict[str, dict[str, Any]] = {}
    for name, frame in selected_frames.items():
        LOGGER.info("评价数据集 %s：%d人", name, len(frame))
        label = SPLIT_LABELS.get(name, name)
        clusters = cluster_statistics(frame, n_clusters)
        survival, km_models[name], km_summaries[name] = cluster_survival(
            frame, n_clusters, config.survival.months
        )
        cox, cox_summaries[name] = adjusted_cox(frame, reference)
        baseline, tests = clinical_characteristics(frame, n_clusters, config)
        observations = pd.read_csv(
            data_root / name / "observations.csv", dtype={"patient_id": "string"}
        )
        trajectories = trajectory_statistics(
            observations,
            frame,
            parameters,
            metadata,
            n_clusters,
            landmark,
            config.trajectory_bin_days,
        )
        metrics, time_metrics, calibration, overall = survival_evaluation(
            frame, train, predictions[name], config
        )
        summary["数据集"][label] = {
            "生存预测": metrics,
            "簇生存": km_summaries[name],
            "校正Cox": cox_summaries[name],
            "最小簇占比": float(clusters["占比（%）"].min() / 100),
            "空簇数": int(clusters["人数"].eq(0).to_numpy().sum()),
            "通过占簇诊断": bool(
                clusters["人数"].gt(0).all()
                and clusters["占比（%）"].min() / 100 >= manifest.get("min_cluster_fraction", 0.05)
            ),
        }
        metric_row = {
            key: value for key, value in metrics.items() if not isinstance(value, dict | list)
        }
        for sheet, table in {
            "簇概况": clusters,
            "簇生存": survival,
            "校正Cox": cox,
            "临床特征": baseline,
            "临床总体检验": tests,
            "纵向轨迹": trajectories,
            "生存预测": pd.DataFrame([metric_row]),
            "时间预测指标": time_metrics,
            "生存校准": calibration,
            "整体生存曲线": overall,
        }.items():
            table.insert(0, "数据集", label)
            collected.setdefault(sheet, []).append(table)
        plot_trajectory_coverage(
            trajectories,
            parameters["feature"].tolist(),
            colors,
            label,
            figures / f"trajectory_coverage_{name}",
            dpi,
        )
        for group_index, (group, group_table) in enumerate(
            trajectories.groupby("特征组", sort=False), start=1
        ):
            features = group_table["特征"].drop_duplicates().tolist()
            width = config.trajectory_panels_per_page
            for start in range(0, len(features), width):
                plot_trajectories(
                    group_table,
                    features[start : start + width],
                    f"{english_label(label)}: {english_label(str(group))}",
                    colors,
                    figures
                    / "trajectories"
                    / name
                    / f"group-{group_index:02d}-page-{start // width + 1}",
                    dpi,
                )
    tables.update({name: pd.concat(parts, ignore_index=True) for name, parts in collected.items()})
    plot_cluster_sizes(tables["簇概况"], colors, figures / "cluster_sizes", dpi)
    plot_confidence(selected_frames, colors, figures / "cluster_confidence", dpi)
    plot_km(km_models, km_summaries, colors, manifest["outcome"], figures / "cluster_survival", dpi)
    plot_cox(tables["校正Cox"], cox_summaries, colors, figures / "adjusted_cox", dpi)
    plot_numeric_baseline(
        selected_frames, config.columns.numeric_baseline, colors, figures / "clinical_numeric", dpi
    )
    categorical = config.columns.categorical_baseline
    for start in range(0, len(categorical), 4):
        plot_categorical_baseline(
            selected_frames,
            categorical[start : start + 4],
            n_clusters,
            figures / f"clinical_categorical_{start // 4 + 1}",
            dpi,
        )
    plot_survival_metrics(
        tables["时间预测指标"],
        tables["生存预测"],
        config.survival.months,
        figures / "survival_metrics",
        dpi,
    )
    plot_calibration(tables["生存校准"], figures / "survival_calibration", dpi)
    plot_overall_survival(tables["整体生存曲线"], figures / "overall_survival", dpi)

    ##################################################
    # 三、训练集拟合UMAP，各评价集共享投影；不导出患者坐标。
    ##################################################
    projections, summary["UMAP"] = latent_projection(data, config)
    if projections:
        plot_umap(projections, selected_frames, colors, figures / "latent_umap", dpi)

    ##################################################
    # 四、既有训练历史和K选择；固定验证集上的全部候选ARI。
    ##################################################
    plot_training(history, figures / "training_history", dpi)
    if manifest["schema_version"] == 3:
        run_metrics = pd.read_csv(source / manifest["metrics"])
        k_summary = pd.read_csv(source / manifest["k_summary"])
        minimum = float(manifest["min_cluster_fraction"])
    elif manifest["k_selection_dir"] is not None:
        selection_dir = Path(manifest["k_selection_dir"])
        original = json.loads((selection_dir / "selection_summary.json").read_text())
        run_metrics = pd.read_csv(selection_dir / "run_metrics.csv").assign(dataset="validation")
        k_summary = pd.read_csv(selection_dir / "k_summary.csv")
        minimum = float(original["config"]["min_cluster_fraction"] or 0)
    else:
        run_metrics = pd.read_csv(source / manifest["metrics"]).rename(columns={"split": "dataset"})
        run_metrics["n_clusters"], run_metrics["seed"] = (
            manifest["selected_k"],
            manifest["selected_seed"],
        )
        k_summary = run_metrics.copy()
        minimum = 0.05
    tables["K候选指标"], tables["K指标汇总"] = run_metrics, k_summary
    plot_k_metrics(run_metrics, minimum, figures / "k_metrics", dpi)
    candidates = available_models(source, manifest)
    if "validation" not in manifest["datasets"] or len({row["seed"] for row in candidates}) < 2:
        summary["跨seedARI"] = {"不可计算原因": "没有独立验证集或不足两个候选seed"}
    else:
        labels: dict[tuple[int, int], np.ndarray] = {}
        occupancy: dict[tuple[int, int], tuple[int, float]] = {}
        validation = None
        for row in candidates:
            seed, k = row["seed"], row["n_clusters"]
            if row["predictions"] is not None:
                assignment = (
                    TrailsPrediction.load(row["predictions"] / "validation/model_prediction.pt")
                    .predict()
                    .numpy()
                )
            else:
                if validation is None:
                    validation = ClinicalTimeSeriesDataset.load(data_root / "validation/dataset.pt")
                estimator = TrailsEstimator.load(row["checkpoint"], device=config.device)
                assignment = estimator.predict(validation).predict().numpy()
                del estimator
            labels[seed, k] = assignment
            counts = np.bincount(assignment, minlength=k)
            occupancy[seed, k] = int((counts == 0).sum()), float(counts.min() / len(assignment))
        rows = []
        for k in sorted({key[1] for key in labels}):
            seeds = sorted(seed for seed, cluster in labels if cluster == k)
            for first, second in combinations(seeds, 2):
                empty1, fraction1 = occupancy[first, k]
                empty2, fraction2 = occupancy[second, k]
                rows.append(
                    {
                        "K": k,
                        "seed_1": first,
                        "seed_2": second,
                        "ARI": float(adjusted_rand_score(labels[first, k], labels[second, k])),
                        "候选1空簇数": empty1,
                        "候选2空簇数": empty2,
                        "候选1最小簇比例": fraction1,
                        "候选2最小簇比例": fraction2,
                        "两候选均通过占用门槛": empty1 == empty2 == 0
                        and min(fraction1, fraction2) >= minimum,
                    }
                )
        pairs = pd.DataFrame(rows)
        tables["跨seed配对ARI"] = pairs
        tables["跨seedARI汇总"] = (
            pairs.groupby("K")["ARI"]
            .agg(配对数="size", 均值="mean", 最小值="min", 最大值="max")
            .reset_index()
        )
        plot_stability(pairs, n_clusters, figures / "cross_seed_ari", dpi)
        summary["跨seedARI"] = {
            "评价集": "验证集",
            "配对数": len(pairs),
            "用途": "事后稳定性描述，不重新选模型",
            "不可计算原因": "",
        }

    ##################################################
    # 五、统一中文工作簿、摘要和解析配置；只保存聚合产物。
    ##################################################
    with pd.ExcelWriter(output / "evaluation_tables.xlsx", engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
    OmegaConf.save(config.model_dump(mode="json"), output / "resolved_config.yaml")
    summary["产物"] = {
        "表格数": len(tables),
        "图数": len(list(figures.rglob("*.png"))),
        "图片格式": ["PNG", "PDF", "SVG"],
        "工作簿": "evaluation_tables.xlsx",
    }
    write_summary(output / "evaluation_summary.json", summary)
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    LOGGER.info("CRC补充评价完成：%s，%s", output, summary["产物"])
    return summary


@hydra.main(config_path="../../configs", config_name="crc_yunnan/supplementary", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    config = CRCSupplementaryConfig.model_validate(OmegaConf.to_container(raw_config, resolve=True))
    run(config)


if __name__ == "__main__":
    main()
