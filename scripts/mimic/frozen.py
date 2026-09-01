"""MIMIC冻结划分的共同读取与来源校验。"""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path

from trails import ClinicalTimeSeriesDataset

from ..utils.baseline_features import dataset_patient_ids


def sha256_file(path: Path) -> str:
    """流式计算大模型或数据文件的指纹，不展开患者内容。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_frozen_datasets(source: Path) -> dict[str, ClinicalTimeSeriesDataset]:
    """加载三划分并拒绝患者交叉或变量顺序不一致。"""
    datasets = {
        split: ClinicalTimeSeriesDataset.load(source / split / "dataset.pt")
        for split in ("train", "validation", "test")
    }
    for split, dataset in datasets.items():
        ids = dataset_patient_ids(dataset)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"{split}患者ID为空或重复")
        if dataset.feature_names != datasets["train"].feature_names:
            raise ValueError("三划分变量顺序不一致")
    for left, right in combinations(datasets.values(), 2):
        if set(dataset_patient_ids(left)) & set(dataset_patient_ids(right)):
            raise ValueError("三划分患者ID存在交叉")
    return datasets
