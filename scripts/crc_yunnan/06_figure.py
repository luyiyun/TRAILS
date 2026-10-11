"""从明确指定的05候选模型制作正文组图和无标识Source Data。"""

from __future__ import annotations

import logging
from typing import Any

import hydra
import pandas as pd
from lifelines import KaplanMeierFitter
from omegaconf import DictConfig, OmegaConf
from scripts.crc_yunnan.config import CRCFigureConfig
from scripts.crc_yunnan.evaluation import (
    adjusted_cox,
    cluster_statistics,
    cluster_survival,
    evaluation_requests,
    evaluation_summary,
    km_source_data,
    latent_projection,
    load_evaluation_data,
    observation_summaries,
    trajectory_selection,
    trajectory_statistics,
    write_summary,
)
from scripts.crc_yunnan.evaluation_plots import configure_plotting
from scripts.crc_yunnan.figure_plots import plot_main_figure

LOGGER = logging.getLogger(__name__)


def run(config: CRCFigureConfig) -> None:
    ##################################################
    # 一、预检全部K及正文配置，沿用05数据和训练尺度。
    ##################################################
    manifest, models = evaluation_requests(config)
    landmark = float(manifest["landmark_days"])
    parameters = pd.read_csv(f"{manifest['split_dir']}/preprocessing_parameters.csv")
    requested = {spec.feature for spec in [*config.figure.atlas, *config.figure.trajectories]}
    if missing := requested - set(parameters["feature"]):
        raise ValueError(f"正文配置包含模型面板不存在的指标，请明确调整配置：{sorted(missing)}")
    if config.figure.phase_days[-1] != landmark:
        raise ValueError(f"phase_days须完整覆盖05的0–{landmark:g}天窗口，请调整配置")
    for model in models:
        data = load_evaluation_data(config, manifest, model)
        colors = configure_plotting(config, model.n_clusters)
        summary = evaluation_summary(config, manifest, model, data)
        frames = {name: data.frames[name] for name in config.datasets}
        collected: dict[str, list[pd.DataFrame]] = {}
        km_models: dict[str, dict[int, KaplanMeierFitter]] = {}
        km_summaries: dict[str, dict[str, Any]] = {}

        ##################################################
        # 二、患者等权阶段画像、原始轨迹、密度及完整随访KM。
        ##################################################
        for name, frame in frames.items():
            LOGGER.info(
                "正文汇总：K=%d，seed=%d，%s=%d人", model.n_clusters, model.seed, name, len(frame)
            )
            observations = pd.read_csv(
                data.data_root / name / "observations.csv", dtype={"patient_id": "string"}
            )
            tables = observation_summaries(
                observations,
                frame,
                data.parameters,
                data.metadata,
                model.n_clusters,
                config.figure.phase_days,
                config.trajectory_bin_days,
            )
            tables["Trajectories"] = trajectory_statistics(
                observations,
                frame,
                data.parameters,
                data.metadata,
                model.n_clusters,
                landmark,
                config.trajectory_bin_days,
            )
            tables["Composition"] = cluster_statistics(frame, model.n_clusters)
            _, km_models[name], km_summaries[name] = cluster_survival(
                frame, model.n_clusters, config.survival.months
            )
            tables["Survival"], tables["At_risk"] = km_source_data(
                frame, km_models[name], model.n_clusters, config.figure.risk_months
            )
            sizes = tables["Composition"]
            minimum = float(sizes["占比（%）"].min() / 100)
            empty = int(sizes["人数"].eq(0).to_numpy().sum())
            summary["数据集"][name] = {
                "人数": len(frame),
                "事件数": int(frame["event"].to_numpy().sum()),
                "最小簇占比": minimum,
                "空簇数": empty,
                "通过占簇诊断": empty == 0
                and minimum >= manifest.get("min_cluster_fraction", 0.05),
                "簇生存": km_summaries[name],
            }
            for key, table in tables.items():
                table.insert(0, "dataset", name)
                collected.setdefault(key, []).append(table)
        tables = {key: pd.concat(parts, ignore_index=True) for key, parts in collected.items()}
        if config.trajectory_selection is not None:
            selection, pairwise, chosen = trajectory_selection(
                tables["Trajectories"], data.parameters, model.n_clusters, config
            )
            if chosen != [spec.feature for spec in config.figure.trajectories]:
                raise ValueError(f"固定轨迹名单与训练集选择不同：{chosen}；请显式复核配置")
            tables["Trajectory_selection"] = selection
            tables["Trajectory_pairs"] = pairwise
        cox, cox_summary = adjusted_cox(frames[config.figure.cox_dataset], data.reference)
        cox.insert(0, "dataset", config.figure.cox_dataset)
        tables["Cox"] = cox
        tables["Cox_details"] = pd.DataFrame(
            [
                {
                    "dataset": config.figure.cox_dataset,
                    **cox_summary,
                    "调整变量": ", ".join(cox_summary["调整变量"]),
                }
            ]
        )
        summary["校正Cox"] = cox_summary
        summary["总体log-rank"] = km_summaries
        tables["Logrank"] = pd.DataFrame(
            [{"dataset": name, **values} for name, values in km_summaries.items()]
        )
        tables["Atlas"]["low_support"] = tables["Atlas"]["n"] < config.figure.min_observed
        tables["Atlas"]["shown_in_main_atlas"] = tables["Atlas"]["feature"].isin(
            [spec.feature for spec in config.figure.atlas]
        )
        tables["Trajectories"]["low_support"] = (
            tables["Trajectories"]["观测人数"] < config.figure.min_observed
        )
        tables["Trajectories"]["shown_in_main_profiles"] = tables["Trajectories"]["特征"].isin(
            [spec.feature for spec in config.figure.trajectories]
        )
        units = {spec.feature: spec.unit for spec in config.figure.trajectories}
        tables["Trajectories"]["recorded_unit"] = tables["Trajectories"]["特征"].map(units.get)
        tables["Atlas"]["recorded_unit"] = tables["Atlas"]["feature"].map(units.get)

        ##################################################
        # 三、训练拟合UMAP并直接绘组图，坐标只在当前进程中使用。
        ##################################################
        projections, summary["UMAP"] = latent_projection(data, config)
        tables["UMAP_details"] = pd.DataFrame(
            [
                {
                    "fit_dataset": "train",
                    "n_train": len(data.frames["train"]),
                    **config.umap.model_dump(),
                    "available": bool(projections),
                    "reason": summary["UMAP"]["不可计算原因"],
                }
            ]
        )
        tables["UMAP_counts"] = pd.DataFrame(
            [
                {
                    "dataset": name,
                    "cluster": cluster,
                    "n": int(frames[name]["pred_cluster"].eq(cluster).to_numpy().sum()),
                }
                for name in config.figure.display_datasets
                for cluster in range(model.n_clusters)
            ]
        )
        definitions = {
            "Model": f"K={model.n_clusters}; seed={model.seed}; manual K request; no retraining",
            "Input": (
                f"{manifest['panel']}; {manifest['outcome'].upper()}; landmark day {landmark:g}"
            ),
            "Atlas": (
                "Patient medians within each phase, then cluster medians and IQR; "
                "unchanged training-fitted transformed scale; color range -2 to 2"
            ),
            "Phase boundaries (days)": str(config.figure.phase_days)
            + "; left closed, right open; final endpoint included",
            "Trajectories": (
                f"Original units; patient medians within {config.trajectory_bin_days}-day bins, "
                "then cluster medians and IQR; final endpoint included"
            ),
            "Trajectory selection": (
                "Fixed configured features; optional train-only pair coverage of level and "
                "baseline-subtracted change; d/(1+d), coverage weighting, exhaustive combinations. "
                "See Trajectory_selection / Trajectory_pairs when enabled. "
                "Descriptive, not model importance or within-patient change."
            ),
            "Density": (
                "Observed dates and nonmissing feature values per patient; "
                "patient median intervals, then population quartiles"
            ),
            "Coverage": (
                "Observed patients / eligible population; "
                "mean equally over model features in each domain"
            ),
            "Low support": (
                f"n < {config.figure.min_observed}: grey atlas cells and disconnected "
                "trajectory points; full values retained"
            ),
            "UMAP": (
                "Fit training latent means only; transform display datasets; "
                "common limits and equal aspect; no coordinate export"
            ),
            "KM": (
                "95% CI; ticks at actual censoring times; "
                f"display 0–{config.figure.display_months:g} months after landmark; "
                "full available follow-up used for log-rank and Cox"
            ),
            "At risk": "survival_time >= month * 30, before removals at exact time",
            "Cox": (
                f"Age per 10 years, sex, AJCC stage and primary site; reference C{data.reference} "
                "determined from training mean predicted risk; complete cases, no penalty"
            ),
            "Interpretation": (
                "Manual K choice may use reported test metrics; exploratory evaluation; "
                "previously inspected test is not new independent validation; "
                "single seed does not establish stability"
            ),
        }
        tables["Definitions"] = pd.DataFrame(
            [{"item": key, "definition": value} for key, value in definitions.items()]
        )
        model.output.mkdir(parents=True, mode=0o700)
        dimensions = plot_main_figure(
            tables,
            frames,
            projections,
            km_models,
            km_summaries,
            cox_summary,
            manifest["outcome"],
            landmark,
            colors,
            config,
            model.output / f"Figure_K{model.n_clusters}",
        )

        ##################################################
        # 四、只输出群体表、配置及摘要，不改写现有文章文字。
        ##################################################
        with pd.ExcelWriter(model.output / "SourceData.xlsx", engine="openpyxl") as writer:
            for name, table in tables.items():
                table.to_excel(writer, sheet_name=name, index=False)
                writer.sheets[name].freeze_panes = "A2"
        OmegaConf.save(config.model_dump(mode="json"), model.output / "resolved_config.yaml")
        summary["产物"] = {
            "表格数": len(tables),
            "图片格式": ["PDF", "SVG", "PNG"],
            "尺寸mm": dimensions,
            "PNG_dpi": config.plot.dpi,
            "工作簿": "SourceData.xlsx",
        }
        write_summary(model.output / "figure_summary.json", summary)
        for path in model.output.rglob("*"):
            if path.is_file():
                path.chmod(0o600)
        LOGGER.info("正文组图完成：%s", model.output)


@hydra.main(config_path="../../configs", config_name="crc_yunnan/figure", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    config = CRCFigureConfig.model_validate(OmegaConf.to_container(raw_config, resolve=True))
    run(config)


if __name__ == "__main__":
    main()
