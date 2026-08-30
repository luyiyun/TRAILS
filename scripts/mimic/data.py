from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from trails.data import ClinicalTimeSeriesDataset

BASELINE_COVARIATE_COLUMNS = ("age", "gender", "race", "sofa_score")


class LongitudinalFeatureTransformer:
    """按变量截尾并标准化异步纵向观测长表。

    ``fit`` 仅应接收训练集，并按 ``feature`` 保存分位点、截尾后均值与标准差；
    ``transform`` 将同一参数应用到任意同结构数据，保留标识、时间和变量列，仅替换
    ``value``。输入不会被原位修改，拟合参数保存在 ``parameters_`` 中供审计和还原。
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("截尾分位点必须满足 0 <= lower < upper <= 1")
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.parameters_: pd.DataFrame | None = None

    def fit(self, observations: pd.DataFrame) -> LongitudinalFeatureTransformer:
        limits = (
            observations.groupby("feature")["value"]
            .quantile(np.array([self.lower_quantile, self.upper_quantile]))
            .unstack()
        )
        limits.columns = ["lower", "upper"]
        clipped = observations.merge(limits, on="feature", validate="many_to_one")
        clipped["value"] = clipped["value"].clip(lower=clipped["lower"], upper=clipped["upper"])
        normalization = clipped.groupby("feature")["value"].agg(
            center="mean", scale=lambda values: values.std(ddof=0)
        )
        if (normalization["scale"] <= 0).any():
            raise ValueError("至少一个变量在训练集截尾后没有变异")
        self.parameters_ = limits.join(normalization).reset_index()
        return self

    def transform(self, observations: pd.DataFrame) -> pd.DataFrame:
        if self.parameters_ is None:
            raise RuntimeError("必须先使用训练集拟合预处理参数")
        transformed = observations.merge(self.parameters_, on="feature", validate="many_to_one")
        transformed["value"] = transformed["value"].clip(
            lower=transformed["lower"], upper=transformed["upper"]
        )
        transformed["value"] = (transformed["value"] - transformed["center"]) / transformed["scale"]
        return transformed.drop(columns=["lower", "upper", "center", "scale"])

    def fit_transform(self, observations: pd.DataFrame) -> pd.DataFrame:
        return self.fit(observations).transform(observations)


def prepare_mimic_datasets(
    patients_csv: Path,
    observations_csv: Path,
    split_dir: Path,
    split_seed: int,
    feature_order: tuple[str, ...],
    description: str,
) -> tuple[dict[str, ClinicalTimeSeriesDataset], LongitudinalFeatureTransformer]:
    patients = pd.read_csv(patients_csv, dtype={"patient_id": str})
    observations = pd.read_csv(observations_csv, dtype={"patient_id": str})
    resolved_feature_order = feature_order or tuple(
        str(feature) for feature in observations["feature"].drop_duplicates()
    )
    required = {
        "patient_id",
        "survival_time",
        "event",
        "left_icu_before_48h",
        *BASELINE_COVARIATE_COLUMNS,
    }
    if missing := sorted(required - set(patients.columns)):
        raise ValueError(f"patients.csv 缺少正式划分字段：{missing}")
    if bool(patients["patient_id"].duplicated().to_numpy().any()):
        raise ValueError("patients.csv 的 patient_id 必须唯一")

    id_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        ids_path = split_dir / f"{split_name}_ids.csv"
        if not ids_path.is_file():
            raise FileNotFoundError(f"缺少正式划分文件：{ids_path}")
        ids = pd.read_csv(ids_path, dtype={"patient_id": str})
        if list(ids.columns) != ["patient_id"]:
            raise ValueError(f"{ids_path} 必须只包含 patient_id 列")
        id_frames[split_name] = ids

    combined_ids = pd.concat(id_frames.values(), ignore_index=True)
    if bool(combined_ids["patient_id"].duplicated().to_numpy().any()):
        raise ValueError("train、validation 和 test patient_id 存在重复")
    if set(combined_ids["patient_id"]) != set(patients["patient_id"]):
        raise ValueError("正式划分 patient_id 并集与 patients.csv 不一致")
    patient_splits = {
        name: ids.merge(patients, on="patient_id", how="left", validate="one_to_one", sort=False)
        for name, ids in id_frames.items()
    }

    split_observations = {
        name: observations.loc[observations["patient_id"].isin(frame["patient_id"])]
        for name, frame in patient_splits.items()
    }
    transformer = LongitudinalFeatureTransformer()
    split_observations["train"] = transformer.fit_transform(split_observations["train"])
    split_observations["validation"] = transformer.transform(split_observations["validation"])
    split_observations["test"] = transformer.transform(split_observations["test"])

    datasets: dict[str, ClinicalTimeSeriesDataset] = {}
    with TemporaryDirectory(prefix="trails-mimic-splits-") as temporary:
        temporary_root = Path(temporary)
        for split_name, split_patients in patient_splits.items():
            patients_path = temporary_root / f"{split_name}_patients.csv"
            observations_path = temporary_root / f"{split_name}_observations.csv"
            split_patients.to_csv(patients_path, index=False)
            split_observations[split_name].to_csv(observations_path, index=False)
            datasets[split_name] = ClinicalTimeSeriesDataset.load_from_csv(
                patients_csv=patients_path,
                observations_csv=observations_path,
                use_features=resolved_feature_order,
                description=f"{description} ({split_name})",
                metadata={
                    "split_name": split_name,
                    "split_seed": split_seed,
                    "split_ids_csv": str(split_dir / f"{split_name}_ids.csv"),
                    "baseline_covariates": split_patients.loc[
                        :, ["patient_id", *BASELINE_COVARIATE_COLUMNS]
                    ].to_dict(orient="records"),
                },
            )
    return datasets, transformer
