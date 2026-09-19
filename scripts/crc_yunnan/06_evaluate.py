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
import umap
from lifelines import KaplanMeierFitter
from omegaconf import DictConfig, OmegaConf
from scipy.special import xlogy
from scripts.crc_yunnan.config import CRCEvaluationConfig
from scripts.crc_yunnan.evaluation import (
    SPLIT_LABELS,
    adjusted_cox,
    clinical_characteristics,
    cluster_statistics,
    cluster_survival,
    survival_evaluation,
    trajectory_statistics,
)
from scripts.crc_yunnan.evaluation_plots import (
    plot_calibration,
    plot_categorical_baseline,
    plot_cluster_sizes,
    plot_confidence,
    plot_cox,
    plot_k_selection,
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
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)


def run(config: CRCEvaluationConfig) -> None:
    ##################################################
    # 一、沿用05的数据和模型约定，设置独立输出及中文绘图。
    ##################################################
    source, output = config.inputs.run_dir.resolve(), config.paths.dir.resolve()
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    data_root = Path(manifest["split_dir"])
    preproc = json.loads((data_root / "preproc_manifest.json").read_text(encoding="utf-8"))
    split_source = Path(preproc["source_datasets"]["train"])
    split_manifest = json.loads((split_source / "dataset_manifest.json").read_text())
    cohort_root = Path(split_manifest["cohort_root"])
    if missing := set(config.datasets) - set(manifest["datasets"]):
        raise ValueError(f"05不存在请求的数据集：{sorted(missing)}")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖既有评价目录：{output}")
    if output == data_root or data_root in output.parents:
        raise ValueError("评价输出不得写入04预处理目录")
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_fonts = [name for name in config.plot.font_families if name in fonts]
    if not selected_fonts:
        raise ValueError("未找到配置中的中文字体")
    plt.rcParams.update(
        {
            "font.family": selected_fonts,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "font.size": 10,
        }
    )
    output.mkdir(parents=True, mode=0o700)
    figures = output / "figures"
    dpi, n_clusters = config.plot.dpi, int(manifest["selected_k"])
    landmark = float(manifest["landmark_days"])
    colors = dict(enumerate(plt.get_cmap("tab10").resampled(n_clusters)(np.arange(n_clusters))))
    metadata = pd.read_csv(cohort_root / "feature_metadata.csv")
    parameters = pd.read_csv(data_root / "preprocessing_parameters.csv")
    fitted_model = TrailsEstimator.load(source / manifest["model"], device="cpu")
    horizon = fitted_model.config.trainer.risk_horizon
    del fitted_model
    predictions: dict[str, TrailsPrediction] = {}
    frames: dict[str, pd.DataFrame] = {}
    for name in dict.fromkeys(["train", *config.datasets]):
        frame = pd.read_csv(data_root / name / "patients.csv", dtype={"patient_id": "string"})
        prediction = TrailsPrediction.load(source / name / "model_prediction.pt")
        probabilities = prediction.predict_proba().numpy()
        frame["pred_cluster"] = prediction.predict().numpy()
        frame["risk_score"] = prediction.risk_score(
            horizon, method=manifest["cindex_risk_score"]
        ).numpy()
        frame["confidence"] = probabilities.max(axis=1)
        frame["entropy"] = (
            -xlogy(probabilities, probabilities).sum(axis=1) / np.log(n_clusters)
            if n_clusters > 1
            else 0.0
        )
        predictions[name], frames[name] = prediction, frame
    selected_frames = {name: frames[name] for name in config.datasets}
    train = frames["train"]
    mean_risk: pd.DataFrame = train.groupby("pred_cluster").agg(risk=("risk_score", "mean"))  # type: ignore[reportAssignmentType]
    reference = int(mean_risk.index.to_numpy()[np.argmin(mean_risk["risk"].to_numpy())])
    tables: dict[str, pd.DataFrame] = {}
    summary: dict[str, Any] = {
        "来源": {"建模目录": str(source), "预处理目录": str(data_root)},
        "结局": manifest["outcome"].upper(),
        "landmark天数": landmark,
        "面板": manifest["panel"],
        "最终K": n_clusters,
        "最终seed": manifest["selected_seed"],
        "统计口径": {
            "生存时间": "从landmark起算，每月30天",
            "轨迹时间": "从手术起算",
            "Cox参考簇": reference,
            "参考规则": "训练集平均预测风险最低的簇",
            "C-index风险定义": manifest["cindex_risk_score"],
            "解释边界": [
                "UMAP仅描述潜空间；ARI结合占用解释",
                "调整Cox为描述性关联；训练和验证集参与过模型选择",
            ],
            "训练范围": manifest["training_scope"],
            "独立评价集": [name for name in config.datasets if name not in {"train", "validation"}],
        },
        "数据集": {},
    }
    LOGGER.info(
        "读取05完成：K=%d，seed=%s，评价集=%s",
        n_clusters,
        manifest["selected_seed"],
        config.datasets,
    )

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
                    f"{label}：{group}",
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
    if len(train) <= config.umap.n_neighbors:
        summary["UMAP"] = {"不可计算原因": "训练人数须大于配置的邻居数，不自动改变UMAP参数"}
    else:
        LOGGER.info("UMAP：仅拟合训练集，随后投影配置的评价集")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=config.umap.n_neighbors,
            min_dist=config.umap.min_dist,
            metric=config.umap.metric,
            random_state=config.umap.seed,
            transform_seed=config.umap.seed,
            n_jobs=1,
        )
        reducer.fit(predictions["train"].latent_representation.numpy())
        projections = {
            name: np.asarray(
                reducer.embedding_
                if name == "train"
                else reducer.transform(predictions[name].latent_representation.numpy())
            )
            for name in config.datasets
        }
        plot_umap(projections, selected_frames, colors, figures / "latent_umap", dpi)
        summary["UMAP"] = {
            "拟合数据集": "训练集",
            "参数": config.umap.model_dump(),
            "不可计算原因": "",
        }

    ##################################################
    # 四、既有训练历史和K选择；固定验证集上的全部候选ARI。
    ##################################################
    history = pd.read_csv(source / "training_history.csv")
    plot_training(history, figures / "training_history", dpi)
    selection_path = manifest["k_selection_dir"]
    if selection_path is None:
        summary["K选择与ARI"] = {"不可计算原因": "固定K运行没有候选结果"}
    else:
        selection_dir = Path(selection_path)
        selection_summary = json.loads((selection_dir / "selection_summary.json").read_text())
        run_metrics = pd.read_csv(selection_dir / "run_metrics.csv")
        k_summary = pd.read_csv(selection_dir / "k_summary.csv")
        minimum = float(selection_summary["config"]["min_cluster_fraction"] or 0)
        plot_k_selection(run_metrics, k_summary, n_clusters, minimum, figures / "k_selection", dpi)
        column_labels = {
            "seed": "seed",
            "n_clusters": "K",
            "cindex": "验证C-index",
            "latent_mixture_bic": "潜在混合BIC",
            "latent_mixture_mean_nll": "潜在混合平均NLL",
            "latent_mixture_n_parameters": "潜在混合参数数",
            "cluster_empty_count": "空簇数",
            "cluster_min_fraction": "最小簇比例",
            "cluster_max_fraction": "最大簇比例",
            "cluster_entropy": "簇分配熵",
            "latent_mixture_bic_normalized": "归一化BIC",
            "selection_score": "选择分数",
            "rank": "seed内排名",
            "mean_selection_score": "平均选择分数",
            "se_selection_score": "选择分数标准误",
            "mean_cindex": "平均验证C-index",
            "min_cluster_fraction": "跨seed最小簇比例",
            "max_empty_clusters": "跨seed最大空簇数",
            "mean_pairwise_ari": "平均两两ARI",
            "passes_gate": "通过占用门槛",
        }
        tables["K候选指标"] = run_metrics.rename(columns=column_labels)
        tables["K选择汇总"] = k_summary.rename(columns=column_labels)
        if "validation" not in manifest["datasets"] or run_metrics["seed"].nunique() < 2:
            summary["K选择与ARI"] = {"不可计算原因": "没有独立验证集或不足两个候选seed"}
        else:
            validation = ClinicalTimeSeriesDataset.load(data_root / "validation/dataset.pt")
            labels: dict[tuple[int, int], np.ndarray] = {}
            occupancy: dict[tuple[int, int], tuple[int, float]] = {}
            for index, (seed, k) in enumerate(
                run_metrics[["seed", "n_clusters"]].itertuples(index=False, name=None), start=1
            ):
                seed, k = int(seed), int(k)
                LOGGER.info("稳定性候选 %d/%d：seed=%d，K=%d", index, len(run_metrics), seed, k)
                estimator = TrailsEstimator.load(
                    selection_dir / f"seed-{seed}/k{k}/model.pt", device=config.device
                )
                assignment = estimator.predict(validation).predict().numpy()
                counts = np.bincount(assignment, minlength=k)
                labels[seed, k] = assignment
                occupancy[seed, k] = (
                    int((counts == 0).sum()),
                    float(counts.min() / len(assignment)),
                )
                del estimator
            rows: list[dict[str, Any]] = []
            for _, candidates in run_metrics.groupby("n_clusters", sort=True):
                k = int(candidates["n_clusters"].iloc[0])
                for first, second in combinations(sorted(candidates["seed"].astype(int)), 2):
                    empty1, fraction1 = occupancy[first, int(k)]
                    empty2, fraction2 = occupancy[second, int(k)]
                    rows.append(
                        {
                            "K": int(k),
                            "seed_1": first,
                            "seed_2": second,
                            "ARI": float(
                                adjusted_rand_score(labels[first, int(k)], labels[second, int(k)])
                            ),
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
            summary["K选择与ARI"] = {
                "评价集": "验证集",
                "候选数": len(run_metrics),
                "配对数": len(pairs),
                "选中K平均ARI": float(pairs.loc[pairs["K"].eq(n_clusters), "ARI"].mean()),
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
        "图片格式": ["PNG", "PDF"],
        "工作簿": "evaluation_tables.xlsx",
    }
    pd.Series(summary).to_json(output / "evaluation_summary.json", force_ascii=False, indent=2)
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    LOGGER.info("CRC评价完成：%s，%s", output, summary["产物"])


@hydra.main(config_path="../../configs", config_name="crc_yunnan/evaluate", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    config = CRCEvaluationConfig.model_validate(OmegaConf.to_container(raw_config, resolve=True))
    run(config)


if __name__ == "__main__":
    main()
