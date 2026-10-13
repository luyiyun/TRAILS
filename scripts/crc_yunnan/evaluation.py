"""CRC各评价集复用的统计；患者级中间结果仅留在进程内。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import umap
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from omegaconf import OmegaConf
from scipy import stats
from scipy.special import xlogy

from trails import (
    ClinicalTimeSeriesDataset,
    ClusterNumberEvaluator,
    TrailsEstimator,
    TrailsPrediction,
)
from trails.artifacts import flatten_history
from trails.metrics import SurvivalMetrics, concordance_index

from .config import BASELINE_LABELS, CRCFigureConfig, CRCPostprocessConfig, CRCSupplementaryConfig

SPLIT_LABELS = {"train": "训练集", "validation": "验证集", "test": "测试集"}


def k_scan_metrics(config: CRCSupplementaryConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """合并同配置的05聚合指标，复用公开评价器计算描述性综合分。"""
    roots = config.k_scan.run_dirs or [config.inputs.run_dir]
    signature = None
    parts = []
    minimum = 0.05
    for root in roots:
        manifest = json.loads((root / "run_manifest.json").read_text())
        if manifest["schema_version"] != 3:
            raise ValueError("跨目录K曲线要求schema_version=3的05产物")
        raw = OmegaConf.to_container(OmegaConf.load(root / "resolved_config.yaml"), resolve=True)
        assert isinstance(raw, dict)
        model, trainer = dict(raw["model"]), dict(raw["trainer"])
        model.pop("n_clusters", None)
        trainer.pop("device", None)
        current = {
            "model": model,
            "trainer": trainer,
            **{
                key: manifest[key]
                for key in [
                    "split_dir",
                    "outcome",
                    "landmark_days",
                    "panel",
                    "training_scope",
                    "min_cluster_fraction",
                ]
            },
        }
        if signature is not None and current != signature:
            raise ValueError(f"K曲线输入的超参数或冻结队列不一致：{root}")
        signature = current
        minimum = float(manifest["min_cluster_fraction"])
        parts.append(pd.read_csv(root / manifest["metrics"]).assign(source_run=str(root)))
    metrics = pd.concat(parts, ignore_index=True)
    if config.model_seed is not None:
        metrics = metrics.loc[metrics["seed"].eq(config.model_seed)].copy()
    if metrics.duplicated(["seed", "n_clusters", "dataset"]).any():
        raise ValueError("同一seed/K/dataset重复；请明确只保留一个05来源")
    expected = config.k_scan.expected_ks or sorted(metrics["n_clusters"].unique().tolist())
    if set(metrics["n_clusters"]) - set(expected):
        raise ValueError("K曲线输入含有expected_ks以外的K")
    scored = []
    for name, rows in metrics.groupby("dataset", sort=False):
        evaluator = ClusterNumberEvaluator(
            candidates=expected,
            seeds=sorted(rows["seed"].unique().tolist()),
            require_non_empty=True,
            min_cluster_fraction=minimum,
            compute_stability=False,
        )
        result = evaluator.summarize_completed(rows, pd.DataFrame(columns=["n_clusters", "ari"]))
        scored.append(result.run_metrics.assign(dataset=name))
    return pd.concat(scored, ignore_index=True), {
        "expected_ks": expected,
        "available_ks": sorted(metrics["n_clusters"].unique().tolist()),
        "missing_ks": sorted(set(expected) - set(metrics["n_clusters"])),
        "minimum_cluster_fraction": minimum,
        "score_definition": (
            "sqrt(C-index^2 + (1 - minmax(BIC))^2); "
            "each dataset/seed separately; all finite available K"
        ),
        "interpretation": ("Descriptive score only; K and seed are explicitly chosen by the user."),
        "bic_definition": (
            "Existing latent-mixture BIC, not full joint-model BIC; includes "
            "the historical K-1 mixture term even with fixed logits."
        ),
        "sources": [str(root) for root in roots],
    }


def trajectory_selection(
    trajectories: pd.DataFrame, parameters: pd.DataFrame, k: int, config: CRCFigureConfig
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """复现G14训练集的水平/趋势覆盖选择，不使用结局或验证/测试集。"""
    options = config.trajectory_selection
    assert options is not None
    features = parameters["feature"].tolist()
    table = trajectories.loc[trajectories["dataset"].eq("train")]
    bins = sorted(table["时间箱"].unique())
    if len(bins) < options.minimum_shared_bins:
        raise ValueError("时间箱不足，不能完成配置的轨迹覆盖选择")
    index = pd.MultiIndex.from_product([features, range(k), bins], names=["特征", "簇", "时间箱"])
    ordered = table.set_index(["特征", "簇", "时间箱"]).reindex(index)
    shape = (len(features), k, len(bins))
    z = ordered["中位数"].to_numpy(dtype=float, copy=True).reshape(shape)
    counts = ordered["观测人数"].to_numpy(dtype=float).reshape(shape)
    coverage = ordered["覆盖率（%）"].to_numpy(dtype=float).reshape(shape) / 100
    pars = parameters.set_index("feature").loc[features]
    logged = pars["transform"].eq("log1p").to_numpy()
    z[logged] = np.log1p(z[logged])
    z = (z - pars["center"].to_numpy()[:, None, None]) / pars["scale"].to_numpy()[:, None, None]
    pairs = list(combinations(range(k), 2))
    first, second = np.array(pairs).T
    contrast = z[:, first] - z[:, second]
    valid = (
        (counts[:, first] >= config.figure.min_observed)
        & (counts[:, second] >= config.figure.min_observed)
        & np.isfinite(contrast)
    )
    enough = valid.sum(axis=2) >= options.minimum_shared_bins
    weight = np.where(
        valid & enough[:, :, None], np.sqrt(coverage[:, first] * coverage[:, second]), 0
    )
    # 固定箱数为分母；缺失与低覆盖不会因分母缩小而提高描述分。
    distance = np.nan_to_num(np.abs(contrast))
    level = (weight * distance / (1 + distance)).sum(axis=2) / len(bins)
    distance = np.nan_to_num(np.abs(contrast - contrast[:, :, :1]))
    shape_score = (
        (weight * valid[:, :, :1])[:, :, 1:] * distance[:, :, 1:] / (1 + distance[:, :, 1:])
    ).sum(axis=2) / (len(bins) - 1)
    alpha = options.level_weight
    count = len(config.figure.trajectories)
    if count > len(features):
        raise ValueError("请求轨迹数量大于模型指标数")

    def objective(chosen: tuple[int, ...]) -> float:
        return float(
            (
                alpha * level[list(chosen)].max(axis=0)
                + (1 - alpha) * shape_score[list(chosen)].max(axis=0)
            ).mean()
        )

    best, score, secondary = (), -1.0, -1.0
    for choice in combinations(range(len(features)), count):
        value = objective(choice)
        individual = float(
            (alpha * level[list(choice)] + (1 - alpha) * shape_score[list(choice)]).mean()
        )
        if value > score + 1e-12 or (abs(value - score) <= 1e-12 and individual > secondary):
            best, score, secondary = choice, value, individual
    selected = [features[i] for i in best]
    rows = [
        {
            "feature": feature,
            "level": level[i].mean(),
            "shape": shape_score[i].mean(),
            "individual_score": (alpha * level[i] + (1 - alpha) * shape_score[i]).mean(),
            "selected": i in best,
            "coverage_objective": score,
            "leave_one_out_gain": score - objective(tuple(j for j in best if j != i))
            if i in best and len(best) > 1
            else 0.0,
        }
        for i, feature in enumerate(features)
    ]
    pairs_table = pd.DataFrame(
        [
            {
                "feature": feature,
                "cluster_a": a,
                "cluster_b": b,
                "level": level[i, p],
                "shape": shape_score[i, p],
            }
            for i, feature in enumerate(features)
            for p, (a, b) in enumerate(pairs)
        ]
    )
    return pd.DataFrame(rows), pairs_table, selected


@dataclass(frozen=True)
class EvaluationModel:
    """已明确请求的单个模型；路径来自05清单，不另存患者中间产物。"""

    n_clusters: int
    seed: int
    checkpoint: Path
    predictions: Path | None
    output: Path


def available_models(source: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """统一新版全部候选和历史入选/候选模型的路径。"""
    if manifest["schema_version"] == 3:
        return [
            {
                **row,
                "checkpoint": source / row["model"],
                "predictions": source / row["prediction_dir"],
            }
            for row in manifest["models"]
        ]
    if manifest["schema_version"] != 2:
        raise ValueError(f"不支持05清单版本：{manifest['schema_version']}")
    models = [
        {
            "n_clusters": manifest["selected_k"],
            "seed": manifest["selected_seed"],
            "checkpoint": source / manifest["model"],
            "predictions": source,
        }
    ]
    if manifest["k_selection_dir"] is not None:
        directory = Path(manifest["k_selection_dir"])
        if not directory.is_absolute():
            directory = directory.resolve()
        table = pd.read_csv(directory / "run_metrics.csv")
        for seed, k in table[["seed", "n_clusters"]].itertuples(index=False, name=None):
            if (seed, k) != (manifest["selected_seed"], manifest["selected_k"]):
                models.append(
                    {
                        "n_clusters": int(k),
                        "seed": int(seed),
                        "checkpoint": directory / f"seed-{seed}/k{k}/model.pt",
                        "predictions": None,
                    }
                )
    return models


def evaluation_requests(
    config: CRCPostprocessConfig,
) -> tuple[dict[str, Any], list[EvaluationModel]]:
    """全部请求预检通过后才能创建输出，绝不自动选择K或多seed代表模型。"""
    source = config.inputs.run_dir.resolve()
    manifest = json.loads((source / "run_manifest.json").read_text())
    if missing := set(config.datasets) - set(manifest["datasets"]):
        raise ValueError(f"05不存在请求的数据集：{sorted(missing)}")
    models = available_models(source, manifest)
    requested = config.n_clusters if isinstance(config.n_clusters, list) else [config.n_clusters]
    seeds = {row["seed"] for row in models}
    seed = config.model_seed
    if seed is None:
        if len(seeds) != 1:
            raise ValueError(f"存在多个模型seed，请显式指定model_seed：{sorted(seeds)}")
        seed = seeds.pop()
    requests = []
    for k in requested:
        matched = [row for row in models if row["n_clusters"] == k and row["seed"] == seed]
        if not matched:
            raise ValueError(f"05没有保存请求的K={k}、seed={seed}")
        row = matched[0]
        if not row["checkpoint"].is_file():
            raise FileNotFoundError(row["checkpoint"])
        output = config.paths.dir.resolve() / f"k{k}"
        if output.exists():
            raise FileExistsError(f"拒绝覆盖既有结果：{output}")
        data_root = Path(manifest["split_dir"])
        if output == data_root or data_root in output.parents or output in data_root.parents:
            raise ValueError("评价输出须独立于04预处理目录")
        requests.append(EvaluationModel(k, seed, row["checkpoint"], row["predictions"], output))
    return manifest, requests


@dataclass
class EvaluationData:
    data_root: Path
    metadata: pd.DataFrame
    parameters: pd.DataFrame
    predictions: dict[str, TrailsPrediction]
    frames: dict[str, pd.DataFrame]
    history: pd.DataFrame
    reference: int
    risk_method: str


def load_evaluation_data(
    config: CRCPostprocessConfig, manifest: dict[str, Any], model: EvaluationModel
) -> EvaluationData:
    """共用冻结数据追溯、预测读取和训练风险参考簇，不重新拟合模型。"""
    root = Path(manifest["split_dir"])
    preproc = json.loads((root / "preproc_manifest.json").read_text())
    split = json.loads(
        (Path(preproc["source_datasets"]["train"]) / "dataset_manifest.json").read_text()
    )
    metadata = pd.read_csv(Path(split["cohort_root"]) / "feature_metadata.csv")
    parameters = pd.read_csv(root / "preprocessing_parameters.csv")
    estimator, predictions = load_evaluation_predictions(
        model.checkpoint,
        root,
        config.datasets,
        model.n_clusters,
        model.seed,
        config.device,
        saved_predictions=model.predictions,
    )
    trainer = estimator.config.trainer
    frames = {}
    for name, prediction in predictions.items():
        frame = pd.read_csv(root / name / "patients.csv", dtype={"patient_id": "string"})
        probabilities = prediction.predict_proba().numpy()
        frame["pred_cluster"] = prediction.predict().numpy()
        frame["risk_score"] = prediction.risk_score(
            trainer.risk_horizon, method=trainer.cindex_risk_score
        ).numpy()
        frame["confidence"] = probabilities.max(axis=1)
        frame["entropy"] = -xlogy(probabilities, probabilities).sum(axis=1) / np.log(
            model.n_clusters
        )
        frames[name] = frame
    risks = frames["train"].groupby("pred_cluster")["risk_score"].mean()
    reference = int(risks.idxmin())
    return EvaluationData(
        root,
        metadata,
        parameters,
        predictions,
        frames,
        pd.DataFrame(flatten_history(estimator.history)),
        reference,
        trainer.cindex_risk_score,
    )


def latent_projection(
    data: EvaluationData, config: CRCPostprocessConfig
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """仅拟合训练latent mean；患者坐标只保留在当前进程。"""
    if len(data.frames["train"]) <= config.umap.n_neighbors:
        return {}, {"不可计算原因": "训练人数须大于配置的邻居数，不自动改变UMAP参数"}
    reducer = umap.UMAP(
        n_components=2,
        **config.umap.model_dump(exclude={"seed"}),
        random_state=config.umap.seed,
        transform_seed=config.umap.seed,
        n_jobs=1,
    )
    reducer.fit(data.predictions["train"].latent_representation.numpy())
    projections = {
        name: np.asarray(
            reducer.embedding_
            if name == "train"
            else reducer.transform(data.predictions[name].latent_representation.numpy())
        )
        for name in config.datasets
    }
    return projections, {
        "拟合数据集": "训练集",
        "参数": config.umap.model_dump(),
        "不可计算原因": "",
    }


def evaluation_summary(
    config: CRCPostprocessConfig,
    manifest: dict[str, Any],
    model: EvaluationModel,
    data: EvaluationData,
) -> dict[str, Any]:
    return {
        "来源": {
            "建模目录": str(config.inputs.run_dir.resolve()),
            "预处理目录": str(data.data_root),
            "模型": str(model.checkpoint),
            "评价目录": str(model.output),
        },
        "结局": manifest["outcome"].upper(),
        "landmark天数": manifest["landmark_days"],
        "面板": manifest["panel"],
        "评价K": model.n_clusters,
        "模型seed": model.seed,
        "统计口径": {
            "生存时间": "从landmark起算，每月30天",
            "轨迹时间": "从手术起算",
            "Cox参考簇": data.reference,
            "参考规则": "训练集平均预测风险最低的簇",
            "C-index风险定义": data.risk_method,
            "训练范围": manifest["training_scope"],
            "解释边界": [
                "人工指定K，不重新训练或自动选择模型",
                "候选可报告test指标；人工参考这些指标选择K属于探索评价",
                "历史已查看的test不构成新的独立验证",
                "UMAP分离不证明有效性，单seed不证明稳定性",
            ],
        },
        "数据集": {},
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    """保留小P值精度，并将不可计算数值写成标准JSON null。"""

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.write_text(
        json.dumps(clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def observation_summaries(
    observations: pd.DataFrame,
    patients: pd.DataFrame,
    parameters: pd.DataFrame,
    metadata: pd.DataFrame,
    n_clusters: int,
    phase_days: list[float],
    bin_days: int,
) -> dict[str, pd.DataFrame]:
    """阶段中位数先按患者计算；覆盖率及采样间隔保留明确分母。"""
    features = parameters["feature"].tolist()
    params = parameters.set_index("feature")
    groups = metadata.set_index("feature")["feature_group"].to_dict()
    counts = patients["pred_cluster"].value_counts().reindex(range(n_clusters), fill_value=0)
    visits = observations.groupby("patient_id")["time_days"].nunique()
    measurements = (
        observations[features].notna().sum(axis=1).groupby(observations["patient_id"]).sum()
    )
    gaps = (
        observations.sort_values(["patient_id", "time_days"])
        .groupby("patient_id")["time_days"]
        .diff()
    )
    patient_density = pd.DataFrame(
        {
            "visits": visits,
            "measurements": measurements,
            "median_gap_days": gaps.groupby(observations["patient_id"]).median(),
        }
    ).join(patients.set_index("patient_id")["pred_cluster"])
    density = []
    for cluster in [-1, *range(n_clusters)]:
        subset = (
            patient_density
            if cluster == -1
            else patient_density.loc[patient_density["pred_cluster"].eq(cluster)]
        )
        record: dict[str, Any] = {"cluster": cluster, "n": len(subset)}
        for column in ["visits", "measurements", "median_gap_days"]:
            for suffix, value in zip(
                ["q25", "median", "q75"],
                subset[column].quantile([0.25, 0.5, 0.75]).to_numpy(),
                strict=True,
            ):
                record[f"{column}_{suffix}"] = value
        density.append(record)
    long = observations.melt(
        id_vars=["patient_id", "time_days"],
        value_vars=features,
        var_name="feature",
        value_name="scaled_value",
    ).dropna(subset=["scaled_value"])
    long = long.merge(patients[["patient_id", "pred_cluster"]], on="patient_id")
    n_phases = len(phase_days) - 1
    long["phase"] = np.minimum(
        np.searchsorted(phase_days, long["time_days"], side="right") - 1, n_phases - 1
    )
    n_bins = int(np.ceil(phase_days[-1] / bin_days))
    long["bin"] = np.minimum((long["time_days"] // bin_days).astype(int), n_bins - 1)
    original = long["scaled_value"] * long["feature"].map(params["scale"].to_dict().get) + long[
        "feature"
    ].map(params["center"].to_dict().get)
    logged = long["feature"].map(params["transform"].to_dict().get).eq("log1p")
    original.loc[logged] = np.expm1(original.loc[logged])
    long["original_value"] = original
    per_patient = long.groupby(["patient_id", "pred_cluster", "phase", "feature"], observed=True)[
        ["scaled_value", "original_value"]
    ].median()
    grouped = per_patient.groupby(["pred_cluster", "phase", "feature"], observed=True)
    profiles = grouped.agg(
        n=("scaled_value", "size"),
        scaled_median=("scaled_value", "median"),
        original_median=("original_value", "median"),
    )
    for value, prefix in [("scaled_value", "scaled"), ("original_value", "original")]:
        quartiles = grouped[value].quantile(np.array([0.25, 0.75])).unstack()
        quartiles.columns = [f"{prefix}_q25", f"{prefix}_q75"]
        profiles = profiles.join(quartiles)
    index = pd.MultiIndex.from_product(
        [range(n_clusters), range(n_phases), features], names=["pred_cluster", "phase", "feature"]
    )
    profiles = profiles.reindex(index).reset_index().rename(columns={"pred_cluster": "cluster"})
    profiles["n"] = profiles["n"].fillna(0).astype(int)
    profiles["cluster_n"] = profiles["cluster"].map(counts.to_dict().get)
    profiles["coverage_fraction"] = profiles["n"] / profiles["cluster_n"].replace(0, np.nan)
    profiles["feature_group"] = profiles["feature"].map(groups.get)
    observed_counts: pd.Series = long.groupby(["bin", "feature"])["patient_id"].nunique()  # type: ignore[reportAssignmentType]
    coverage = (
        observed_counts.reindex(
            pd.MultiIndex.from_product([range(n_bins), features], names=["bin", "feature"]),
            fill_value=0,
        )
        .rename("n_observed")
        .reset_index()
    )
    coverage["n_population"] = len(patients)
    coverage["coverage_fraction"] = coverage["n_observed"] / len(patients)
    coverage["feature_group"] = coverage["feature"].map(groups.get)
    domains: pd.DataFrame = coverage.groupby(
        ["bin", "feature_group"], as_index=False, observed=True
    ).agg(  # type: ignore[reportAssignmentType]
        mean_feature_coverage=("coverage_fraction", "mean"), n_features=("feature", "size")
    )
    return {
        "Atlas": profiles,
        "Density": pd.DataFrame(density),
        "Feature_coverage": coverage,
        "Domain_coverage": domains,
    }


def km_source_data(
    patients: pd.DataFrame,
    models: dict[int, KaplanMeierFitter],
    n_clusters: int,
    months: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """完整KM及其95%CI，不因绘图显示上限而截断统计随访。"""
    curves = []
    for cluster, km in models.items():
        table = km.survival_function_.join(km.confidence_interval_)
        table.columns = ["survival", "ci_lower", "ci_upper"]
        table["censored"] = km.event_table["censored"].reindex(table.index, fill_value=0)
        table = table.rename_axis("month").reset_index()
        table.insert(0, "cluster", cluster)
        curves.append(table)
    risk = [
        {
            "cluster": cluster,
            "month": month,
            "at_risk": int(
                (
                    patients.loc[patients["pred_cluster"].eq(cluster), "survival_time"]
                    >= month * 30
                ).sum()
            ),
        }
        for cluster in range(n_clusters)
        for month in months
    ]
    return pd.concat(curves, ignore_index=True), pd.DataFrame(risk)


def load_evaluation_predictions(
    checkpoint: Path,
    data_root: Path,
    datasets: list[str],
    n_clusters: int,
    seed: int,
    device: str,
    saved_predictions: Path | None = None,
) -> tuple[TrailsEstimator, dict[str, TrailsPrediction]]:
    """按指定K加载同一seed的模型；非入选K在冻结数据上推理，不训练或保存患者产物。"""
    estimator = TrailsEstimator.load(
        checkpoint, device="cpu" if saved_predictions is not None else device
    )
    if estimator.config.model.n_clusters != n_clusters or estimator.config.trainer.seed != seed:
        raise ValueError(f"模型不匹配请求的K={n_clusters}、seed={seed}：{checkpoint}")
    predictions = {
        name: (
            TrailsPrediction.load(saved_predictions / name / "model_prediction.pt")
            if saved_predictions is not None
            else estimator.predict(ClinicalTimeSeriesDataset.load(data_root / name / "dataset.pt"))
        )
        for name in dict.fromkeys(["train", *datasets])
    }
    return estimator, predictions


def cluster_statistics(frame: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """空簇保留为零人数，分配不确定性按最终模型后验计算。"""
    grouped = frame.groupby("pred_cluster", observed=True)
    table: pd.DataFrame = grouped.agg(  # type: ignore[reportAssignmentType]
        人数=("event", "size"),
        事件数=("event", "sum"),
        平均最大后验概率=("confidence", "mean"),
        平均归一化熵=("entropy", "mean"),
        平均预测风险=("risk_score", "mean"),
    ).reindex(range(n_clusters))
    table[["人数", "事件数"]] = table[["人数", "事件数"]].fillna(0).astype(int)
    table["占比（%）"] = table["人数"] / len(frame) * 100
    table["事件比例（%）"] = table["事件数"] / table["人数"].replace(0, np.nan) * 100
    return table.rename_axis("簇").reset_index()


def cluster_survival(
    frame: pd.DataFrame, n_clusters: int, months: list[int]
) -> tuple[pd.DataFrame, dict[int, KaplanMeierFitter], dict[str, Any]]:
    """KM对象供风险表绘图复用，时间点统计与曲线采用同一拟合。"""
    models: dict[int, KaplanMeierFitter] = {}
    records: list[dict[str, Any]] = []
    for _, group in frame.groupby("pred_cluster", observed=True):
        label = int(group["pred_cluster"].iloc[0])
        km = KaplanMeierFitter(label=f"Cluster {label}").fit(
            group["survival_time"] / 30, group["event"]
        )
        models[label] = km
        for month in months:
            supported = month <= group["survival_time"].max() / 30
            bounds = km.confidence_interval_.reindex([float(month)], method="ffill").iloc[0]
            records.append(
                {
                    "簇": label,
                    "月份": month,
                    "人数": len(group),
                    "风险人数": int((group["survival_time"].to_numpy() >= month * 30).sum()),
                    "生存率": float(km.predict(month)) if supported else np.nan,
                    "95%CI下限": float(bounds.iloc[0]) if supported else np.nan,
                    "95%CI上限": float(bounds.iloc[1]) if supported else np.nan,
                    "不可计算原因": "" if supported else "超过该簇观察随访范围",
                }
            )
    reason = (
        "" if len(models) >= 2 and frame["event"].to_numpy().any() else "不足两个有患者的簇或无事件"
    )
    p_value = (
        float(
            multivariate_logrank_test(
                frame["survival_time"], frame["pred_cluster"], frame["event"]
            ).p_value
        )
        if not reason
        else None
    )
    return (
        pd.DataFrame(records),
        models,
        {
            "总体log-rank_p值": p_value,
            "不可计算原因": reason,
            "空簇数": n_clusters - len(models),
        },
    )


def adjusted_cox(frame: pd.DataFrame, reference: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """固定四项调整变量；收敛失败不改模型、不自动添加惩罚。"""
    columns = [
        "pred_cluster",
        "survival_time",
        "event",
        "Age",
        "Sex",
        "AJCC_8th_ed_Stage",
        "Primary_site",
    ]
    analysis: pd.DataFrame = frame.loc[:, columns].dropna().copy()
    summary: dict[str, Any] = {
        "参考簇": reference,
        "输入人数": len(frame),
        "完整病例人数": len(analysis),
        "缺失排除人数": len(frame) - len(analysis),
        "调整变量": ["年龄（每10岁）", "性别", "AJCC第8版分期", "原发部位"],
    }
    reason = ""
    if analysis.empty or not analysis["event"].to_numpy().any():
        reason = "完整病例无事件"
    elif analysis["pred_cluster"].nunique() < 2:
        reason = "完整病例不足两个有患者的簇"
    elif reference not in analysis["pred_cluster"].unique():
        reason = "当前数据集没有训练集确定的参考簇"
    result_columns = ["簇", "HR", "95%CI下限", "95%CI上限", "p值", "人数", "参考簇"]
    if reason:
        summary["不可计算原因"] = reason
        return pd.DataFrame(columns=result_columns), summary
    analysis["age_per_10"] = analysis["Age"] / 10
    formula = (
        f"C(pred_cluster, Treatment(reference={reference})) + age_per_10"
        " + C(Sex) + C(AJCC_8th_ed_Stage) + C(Primary_site)"
    )
    fitter = CoxPHFitter().fit(analysis, "survival_time", "event", formula=formula)
    coefficients = fitter.summary
    rows: list[dict[str, Any]] = []
    counts = analysis["pred_cluster"].value_counts().sort_index()
    for cluster, count in zip(counts.index.to_numpy(), counts.to_numpy(), strict=True):
        if int(cluster) == reference:
            hr, lower, upper, p_value = 1.0, 1.0, 1.0, np.nan
        else:
            row = coefficients.loc[
                f"C(pred_cluster, Treatment(reference={reference}))[T.{int(cluster)}]"
            ]
            hr, lower, upper, p_value = row[
                ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
            ].to_numpy(dtype=float)
        rows.append(
            {
                "簇": int(cluster),
                "HR": hr,
                "95%CI下限": lower,
                "95%CI上限": upper,
                "p值": p_value,
                "人数": int(count),
                "参考簇": int(cluster) == reference,
            }
        )
    summary["不可计算原因"] = ""
    return pd.DataFrame(rows, columns=result_columns), summary


def clinical_characteristics(
    frame: pd.DataFrame, n_clusters: int, config: CRCSupplementaryConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """基线描述和总体检验；分类比例的分母包括缺失患者。"""
    numeric, categorical = config.columns.numeric_baseline, config.columns.categorical_baseline
    groups = [
        ("总体", frame),
        *[(f"簇{k}", frame.loc[frame["pred_cluster"].eq(k)]) for k in range(n_clusters)],
    ]
    rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed)
    for variable in [*numeric, *categorical]:
        label = BASELINE_LABELS.get(variable, variable)
        p_value, method, reason = np.nan, "", ""
        observed = frame.dropna(subset=[variable])
        if variable in numeric:
            samples = [
                group[variable].to_numpy(dtype=float)
                for _, group in observed.groupby("pred_cluster", observed=True)
            ]
            if len(samples) < 2:
                reason = "不足两个非空组"
            elif observed[variable].nunique() <= 1:
                reason = "所有有效值相同"
            else:
                p_value, method = float(stats.kruskal(*samples).pvalue), "Kruskal–Wallis"
            for group_name, group in groups:
                values = group[variable].dropna()
                rows.append(
                    {
                        "变量": label,
                        "类型": "连续",
                        "类别": "",
                        "组别": group_name,
                        "人数": len(group),
                        "有效人数": len(values),
                        "缺失人数": int(group[variable].isna().sum()),
                        "中位数": values.median(),
                        "下四分位数": values.quantile(0.25),
                        "上四分位数": values.quantile(0.75),
                        "类别人数": np.nan,
                        "比例（%）": np.nan,
                    }
                )
        else:
            contingency = pd.crosstab(observed["pred_cluster"], observed[variable]).to_numpy()
            if min(contingency.shape, default=0) < 2:
                reason = "有效簇或类别不足两个"
            else:
                chi: Any = stats.chi2_contingency(contingency, correction=False)
                if (chi.expected_freq < 5).any():
                    fisher: Any = stats.fisher_exact(
                        contingency,
                        method=stats.MonteCarloMethod(n_resamples=config.fisher_resamples, rng=rng),
                    )
                    p_value = float(fisher.pvalue)
                    method = "Monte Carlo Fisher"
                else:
                    p_value, method = float(chi.pvalue), "χ²"
            levels = sorted(observed[variable].astype(str).unique()) + ["缺失"]
            for group_name, group in groups:
                counts = group[variable].fillna("缺失").astype(str).value_counts()
                for level in levels:
                    count = int(counts.get(level, 0))
                    rows.append(
                        {
                            "变量": label,
                            "类型": "分类",
                            "类别": level,
                            "组别": group_name,
                            "人数": len(group),
                            "有效人数": int(group[variable].notna().sum()),
                            "缺失人数": int(group[variable].isna().sum()),
                            "中位数": np.nan,
                            "下四分位数": np.nan,
                            "上四分位数": np.nan,
                            "类别人数": count,
                            "比例（%）": count / len(group) * 100 if len(group) else np.nan,
                        }
                    )
        tests.append({"变量": label, "检验": method, "p值": p_value, "不可计算原因": reason})
    test_table = pd.DataFrame(tests)
    valid = test_table["p值"].notna()
    test_table["BH_q值"] = np.nan
    if valid.to_numpy().any():
        test_table.loc[valid, "BH_q值"] = stats.false_discovery_control(
            test_table.loc[valid, "p值"].to_numpy(dtype=float)
        )
    return pd.DataFrame(rows), test_table


def trajectory_statistics(
    observations: pd.DataFrame,
    patients: pd.DataFrame,
    parameters: pd.DataFrame,
    metadata: pd.DataFrame,
    n_clusters: int,
    landmark: float,
    bin_days: int,
) -> pd.DataFrame:
    """先还原原始尺度，再按患者分箱汇总；患者等权且不插值。"""
    features = parameters["feature"].tolist()
    params = parameters.set_index("feature").loc[features]
    values = observations[features].mul(params["scale"]).add(params["center"])
    logged = params.index[params["transform"].eq("log1p")].tolist()
    values[logged] = np.expm1(values[logged])
    restored = pd.concat([observations[["patient_id", "time_days"]], values], axis=1)
    restored = restored.merge(patients[["patient_id", "pred_cluster"]], on="patient_id")
    n_bins = int(np.ceil(landmark / bin_days))
    restored["时间箱"] = np.minimum((restored["time_days"] // bin_days).astype(int), n_bins - 1)
    long = restored.melt(
        id_vars=["patient_id", "pred_cluster", "时间箱"],
        value_vars=features,
        var_name="特征",
        value_name="数值",
    ).dropna(subset=["数值"])
    per_patient = (
        long.groupby(["patient_id", "pred_cluster", "时间箱", "特征"], observed=True)["数值"]
        .agg(患者中位数="median", 观测数="size")
        .reset_index()
    )
    grouped = per_patient.groupby(["pred_cluster", "时间箱", "特征"], observed=True)
    table: pd.DataFrame = grouped.agg(  # type: ignore[reportAssignmentType]
        观测人数=("patient_id", "size"),
        有效观测数=("观测数", "sum"),
        中位数=("患者中位数", "median"),
    )
    quantiles = grouped["患者中位数"].quantile(np.array([0.25, 0.75])).unstack()
    quantiles.columns = ["下四分位数", "上四分位数"]
    full = pd.MultiIndex.from_product(
        [range(n_clusters), range(n_bins), features], names=["pred_cluster", "时间箱", "特征"]
    )
    table = table.join(quantiles).reindex(full).reset_index()  # type: ignore[reportAssignmentType]
    table = table.rename(columns={"pred_cluster": "簇"})
    table[["观测人数", "有效观测数"]] = table[["观测人数", "有效观测数"]].fillna(0).astype(int)
    denominators = patients["pred_cluster"].value_counts().reindex(range(n_clusters), fill_value=0)
    table["簇总人数"] = table["簇"].map(denominators.to_dict())  # type: ignore[reportArgumentType]
    table["覆盖率（%）"] = table["观测人数"] / table["簇总人数"].replace(0, np.nan) * 100
    table["起始天"] = table["时间箱"] * bin_days
    table["结束天"] = np.minimum(table["起始天"] + bin_days, landmark)
    table["中点月份"] = (table["起始天"] + table["结束天"]) / 60
    feature_groups = dict(zip(metadata["feature"], metadata["feature_group"], strict=True))
    table["特征组"] = table["特征"].map(feature_groups)  # type: ignore[reportArgumentType]
    return table


def survival_evaluation(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    prediction: TrailsPrediction,
    config: CRCSupplementaryConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """预测指标只使用训练删失分布；校准和KM均按当前评价集观察结局计算。"""
    event = frame["event"].to_numpy(dtype=bool)
    # 05从float32张量计算指标，沿用同一精度以保留相同的并列时间定义。
    time = frame["survival_time"].to_numpy(dtype=np.float32).astype(float)
    risk = frame["risk_score"].to_numpy(dtype=float)
    event_tensor, time_tensor, risk_tensor = (
        torch.tensor(event),
        torch.tensor(time),
        torch.tensor(risk),
    )
    metrics = SurvivalMetrics(
        torch.tensor(train["event"].to_numpy(dtype=bool)),
        torch.tensor(train["survival_time"].to_numpy(dtype=np.float32).astype(float)),
    )
    months = np.arange(
        config.survival.curve_step_months,
        config.survival.months[-1] + 1,
        config.survival.curve_step_months,
    )
    times = months.astype(float) * 30
    probabilities = prediction.survival(times.tolist()).numpy().astype(float)
    summary: dict[str, Any] = {
        "人数": len(frame),
        "事件数": int(event.sum()),
        "Harrell_C-index": None,
        "IPCW_C-index": None,
        "IPCW截断月份": config.survival.months[-1],
        "IBS": None,
        "IBS起止月份": [int(months[0]), int(months[-1])],
        "删失分布来源": "训练集",
        "不可计算原因": {},
    }
    reasons: dict[str, str] = summary["不可计算原因"]
    comparable = event.any() and (
        time[event].min() < time.max()
        or ((~event).any() and time[event].min() <= time[~event].max())
    )
    if comparable:
        summary["Harrell_C-index"] = float(
            concordance_index(risk_tensor, time_tensor, event_tensor)
        )
    else:
        reasons["Harrell_C-index"] = "无有效生存比较对"
    train_has_events = bool(train["event"].to_numpy().any())
    censoring = metrics if train_has_events else None
    train_max = float(metrics.max_time)
    support = (times >= time.min()) & (times < time.max()) & (times < train_max)
    censor_positive = np.zeros(len(times), dtype=bool)
    if censoring is not None and support.any():
        censor_positive[support] = censoring.supported(torch.tensor(times[support])).numpy()
    support &= censor_positive
    # AUC的库实现会为全部事件计算IPCW；不隐式截断不受支持的事件。
    event_support = censoring is not None and event.any() and time[event].max() <= train_max
    if event_support and censoring is not None:
        event_support = bool(censoring.supported(time_tensor[event_tensor]).all())
    in_tau = event & (time < times[-1])
    tau_comparable = in_tau.any() and (
        time[in_tau].min() < time.max()
        or ((~event).any() and time[in_tau].min() <= time[~event].max())
    )
    if tau_comparable and support[-1] and censoring is not None:
        summary["IPCW_C-index"] = float(
            metrics.cindex(risk_tensor, event_tensor, time_tensor, tau=float(times[-1]))
        )
    else:
        reasons["IPCW_C-index"] = "截断内无可比较事件或训练删失分布不支持"
    curve = pd.DataFrame(
        {
            "月份": months,
            "风险人数": [int((time >= t).sum()) for t in times],
            "累积事件数": [int((event & (time <= t)).sum()) for t in times],
            "动态AUC": np.nan,
            "Brier": np.nan,
            "AUC不可计算原因": "",
            "Brier不可计算原因": "",
        }
    )
    auc_supported = support & event_support & (curve["累积事件数"].to_numpy() > 0)
    if auc_supported.any():
        auc, _ = metrics.auc(
            torch.tensor(1 - probabilities[:, auc_supported]),
            event_tensor,
            time_tensor,
            torch.tensor(times[auc_supported]),
        )
        curve.loc[auc_supported, "动态AUC"] = auc.numpy()
    curve.loc[~auc_supported, "AUC不可计算原因"] = "缺少病例/对照或训练事件及随访删失支持不足"
    brier_supported = support & (time.max() <= train_max) & event.any()
    integrated_brier = None
    if brier_supported.any():
        scores, integrated_brier = metrics.brier(
            torch.tensor(probabilities[:, brier_supported]),
            event_tensor,
            time_tensor,
            torch.tensor(times[brier_supported]),
        )
        curve.loc[brier_supported, "Brier"] = scores.numpy()
    curve.loc[~brier_supported, "Brier不可计算原因"] = "当前集无事件或训练事件及随访删失支持不足"
    for metric in ("AUC", "Brier"):
        unavailable = curve.loc[curve[metric + "不可计算原因"].ne("")]
        if not unavailable.empty:
            reasons[metric] = "；".join(unavailable[metric + "不可计算原因"].unique())
            summary[metric + "不可计算月份"] = unavailable["月份"].tolist()
    if brier_supported.all():
        summary["IBS"] = float(integrated_brier) if integrated_brier is not None else None
    else:
        reasons["IBS"] = "预设积分网格未全部得到支持，不缩短积分区间"
    calibration: list[dict[str, Any]] = []
    for month in config.survival.months:
        column = int(np.flatnonzero(months == month)[0])
        predicted = probabilities[:, column]
        bins = np.asarray(
            pd.qcut(
                predicted,
                q=min(config.survival.calibration_bins, len(frame)),
                labels=False,
                duplicates="drop",
            ),
            dtype=float,
        )
        if np.isnan(bins).all():
            bins = np.zeros(len(frame))
        for group in np.unique(bins):
            selected = bins == group
            km = KaplanMeierFitter().fit(time[selected] / 30, event[selected])
            supported = month <= time[selected].max() / 30
            bounds = km.confidence_interval_.reindex([float(month)], method="ffill").iloc[0]
            calibration.append(
                {
                    "月份": month,
                    "分位组": int(group) + 1,
                    "人数": int(selected.sum()),
                    "实际分组数": len(np.unique(bins)),
                    "平均预测生存率": float(predicted[selected].mean()),
                    "KM生存率": float(km.predict(month)) if supported else np.nan,
                    "95%CI下限": float(bounds.iloc[0]) if supported else np.nan,
                    "95%CI上限": float(bounds.iloc[1]) if supported else np.nan,
                    "不可计算原因": "" if supported else "超过该组观察随访范围",
                }
            )
    km = KaplanMeierFitter().fit(time / 30, event)
    bounds = km.confidence_interval_.reindex(months.astype(float), method="ffill")
    overall = pd.DataFrame(
        {
            "月份": months,
            "平均预测生存率": probabilities.mean(axis=0),
            "KM生存率": km.predict(months).to_numpy(),
            "95%CI下限": bounds.iloc[:, 0].to_numpy(),
            "95%CI上限": bounds.iloc[:, 1].to_numpy(),
        }
    )
    beyond = months > time.max() / 30
    overall.loc[beyond, ["KM生存率", "95%CI下限", "95%CI上限"]] = np.nan
    overall["不可计算原因"] = np.where(beyond, "超过观察随访范围", "")
    return summary, curve, pd.DataFrame(calibration), overall
