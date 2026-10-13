"""候选评价器产物的共享命令层导出；不负责训练或选择模型。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from trails import ClinicalTimeSeriesDataset, ClusterNumberEvaluator


def save_candidate_outputs(
    evaluator: ClusterNumberEvaluator,
    datasets: Mapping[str, ClinicalTimeSeriesDataset | Path],
    *,
    baseline_columns: Sequence[str] = (),
) -> None:
    """沿用CRC患者输出字段和候选YAML，MIMIC可附加已有基线协变量。"""
    root = evaluator.result.result_dir
    if root is None:
        raise ValueError("候选结果缺少输出目录")
    for record in evaluator.candidates:
        config = json.loads((root / record["config"]).read_text())
        OmegaConf.save(config, root / record["prediction_dir"] / "resolved_config.yaml")
    for name, source in datasets.items():
        dataset = (
            source
            if isinstance(source, ClinicalTimeSeriesDataset)
            else ClinicalTimeSeriesDataset.load(source)
        )
        times = torch.stack([sample.survival_time for sample in dataset.samples]).numpy()
        events = torch.stack([sample.event for sample in dataset.samples]).numpy()
        for record in evaluator.candidates:
            prediction = evaluator.load_prediction(
                seed=record["seed"], n_clusters=record["n_clusters"], dataset=name
            )
            trainer = evaluator.config.estimator.trainer
            frame = pd.DataFrame(
                {
                    "patient_id": dataset.metadata["patient_ids"],
                    "pred_cluster": prediction.predict().numpy(),
                    "predicted_median_time_days": prediction.median_survival_time().numpy(),
                    "risk_score": prediction.risk_score(
                        trainer.risk_horizon, method=trainer.cindex_risk_score
                    ).numpy(),
                    "survival_time": times,
                    "event": events,
                }
            )
            if baseline_columns:
                covariates = pd.DataFrame(dataset.metadata["baseline_covariates"])
                frame = frame.merge(
                    covariates.loc[:, ["patient_id", *baseline_columns]],
                    on="patient_id",
                    how="left",
                    sort=False,
                    validate="one_to_one",
                )
            probabilities = pd.DataFrame(
                prediction.predict_proba().numpy(),
                columns=[f"cluster_probability_{k}" for k in range(record["n_clusters"])],
            )
            pd.concat([frame, probabilities], axis=1).to_csv(
                root / record["prediction_dir"] / name / "patient_outputs.csv", index=False
            )
