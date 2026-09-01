"""双09入口共享的冻结预测读取，区分TRAILS与其他基线产物。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trails import ClinicalTimeSeriesDataset, TrailsPrediction

from ..utils.baseline_features import dataset_patient_ids, dataset_survival_arrays
from ..utils.baselines import BaselinePrediction
from .config import MimicEvaluationConfig
from .frozen import sha256_file
from .paths import resolve_input_path


def evaluation_methods(config: MimicEvaluationConfig) -> list[dict[str, Any]]:
    """读取完成manifest，并验证08确实使用了当前07的冻结输入。"""
    source = resolve_input_path(config.input_dir)
    original = json.loads((source / "run_manifest.json").read_text())
    methods: list[dict[str, Any]] = [
        {
            "name": "trails",
            "seed": original["training"]["seed"],
            "directory": source,
            "key": "trails",
            "capabilities": ["cluster", "survival"],
            "prediction_format": "trails",
        }
    ]
    for root in config.baseline_dirs:
        root = resolve_input_path(root)
        manifest = json.loads((root / "baselines_manifest.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError(f"拒绝把未完整成功的08运行纳入评价：{root}")
        if manifest["risk_horizon"] != config.tau:
            raise ValueError("08和09风险时间窗不一致")
        for relative, expected in manifest["source_sha256"].items():
            if sha256_file(source / relative) != expected:
                raise ValueError(f"08来源与当前07不一致：{relative}")
        for record in manifest["methods"]:
            method = dict(record)
            if method["status"] != "completed":
                raise ValueError("08方法尚未完成")
            method["directory"] = root / method["directory"]
            for relative, expected in method["artifacts"].items():
                if sha256_file(method["directory"] / relative) != expected:
                    raise ValueError(f"基线冻结产物指纹不符：{method['name']}/{relative}")
            method["key"] = f"{method['name']}/seed-{method['seed']}"
            methods.append(method)
    keys = [method["key"] for method in methods]
    if len(set(keys)) != len(keys):
        raise ValueError("评价方法×seed重复")
    return methods


def load_prediction(
    method: dict[str, Any],
    split: str,
    dataset: ClinicalTimeSeriesDataset,
    config: MimicEvaluationConfig,
) -> BaselinePrediction:
    """按产物类型分支读取，再提取评价真正需要的标签和曲线。"""
    directory = Path(method["directory"]) / split
    if method["prediction_format"] == "trails":
        saved = TrailsPrediction.load(directory / "model_prediction.pt")
        survival = "survival" in method["capabilities"]
        prediction = BaselinePrediction(
            method_name=method["name"],
            patient_ids=dataset_patient_ids(dataset),
            cluster_labels=saved.predict().numpy().astype(np.int64),
            n_clusters=saved.predict_proba().shape[1],
            risk_score=saved.risk_score(config.tau).numpy().astype(np.float64)
            if survival
            else None,
            risk_horizon=config.tau if survival else None,
            survival_times=np.asarray(config.probability_times) if survival else None,
            survival_probabilities=saved.survival(config.probability_times)
            .numpy()
            .astype(np.float64)
            if survival
            else None,
        )
    else:
        prediction = BaselinePrediction.load(directory / "baseline_prediction.npz")
    if prediction.patient_ids != dataset_patient_ids(dataset):
        raise ValueError("预测患者顺序与冻结dataset不一致")
    if prediction.capabilities != frozenset(method["capabilities"]):
        raise ValueError("预测能力与manifest不一致")
    return prediction


def prediction_frame(
    dataset: ClinicalTimeSeriesDataset, prediction: BaselinePrediction
) -> pd.DataFrame:
    """按原始患者顺序构造评价表；患者行只保留在内存中。"""
    event, time = dataset_survival_arrays(dataset)
    frame = pd.DataFrame(
        {"patient_id": prediction.patient_ids, "event": event, "survival_time": time}
    )
    if prediction.cluster_labels is not None:
        frame["pred_cluster"] = prediction.cluster_labels
        covariates = pd.DataFrame(dataset.metadata["baseline_covariates"])
        covariates["patient_id"] = covariates["patient_id"].astype(str)
        if set(covariates["patient_id"]) != set(prediction.patient_ids):
            raise ValueError("协变量患者集合与预测不一致")
        frame = frame.merge(covariates, on="patient_id", validate="one_to_one", sort=False)
    if prediction.risk_score is not None:
        frame["risk_score"] = prediction.risk_score
    return frame
