from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from trails.data import ClinicalTimeSeriesDataset

SPLIT_SEED = 20260517


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
    feature_order: tuple[str, ...],
    description: str,
) -> tuple[dict[str, ClinicalTimeSeriesDataset], LongitudinalFeatureTransformer]:
    patients = pd.read_csv(patients_csv, dtype={"patient_id": str})
    observations = pd.read_csv(observations_csv, dtype={"patient_id": str})
    required = {"patient_id", "survival_time", "event", "left_icu_before_48h"}
    if missing := sorted(required - set(patients.columns)):
        raise ValueError(f"patients.csv 缺少正式划分字段：{missing}")

    # 固定划分只依赖结局与48小时前转出状态，模型 seed 的变化不会改变样本组成。
    strata = patients["event"].astype(str) + ":" + patients["left_icu_before_48h"].astype(str)
    first_split = train_test_split(
        patients, test_size=0.20, random_state=SPLIT_SEED, stratify=strata
    )
    train_valid = cast(pd.DataFrame, first_split[0])
    test = cast(pd.DataFrame, first_split[1])
    train_valid_strata = (
        train_valid["event"].astype(str) + ":" + train_valid["left_icu_before_48h"].astype(str)
    )
    second_split = train_test_split(
        train_valid, test_size=0.20, random_state=SPLIT_SEED, stratify=train_valid_strata
    )
    patient_splits = {
        "train": cast(pd.DataFrame, second_split[0]),
        "validation": cast(pd.DataFrame, second_split[1]),
        "test": test,
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
                use_features=feature_order,
                description=f"{description} ({split_name})",
                metadata={"split_name": split_name, "split_seed": SPLIT_SEED},
            )
    return datasets, transformer
