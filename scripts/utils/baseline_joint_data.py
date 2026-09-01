"""Joint-model基线共享的landmark长表导出契约。"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trails import ClinicalTimeSeriesDataset

from .baseline_features import dataset_patient_ids, dataset_survival_arrays


@dataclass(frozen=True)
class JointModelInputPaths:
    """一个冻结划分导出后的joint-model输入路径。"""

    patients_csv: Path
    observations_csv: Path
    features_csv: Path
    landmark_entry_time: float


def export_joint_model_input(
    data: ClinicalTimeSeriesDataset,
    output_dir: str | Path,
    split_name: str,
    *,
    landmark_time: float,
    observation_time_factor: float = 1.0,
    survival_time_factor: float = 1.0,
    patient_batch_size: int = 128,
) -> JointModelInputPaths:
    """将冻结dataset流式导出为joint model共用的患者表和观测长表。

    纵向时间从原研究时间原点开始，并乘以``observation_time_factor``；
    ``landmark_time``转换后作为所有患者的左截断时间。dataset中的生存时间
    视为landmark后的随访时间，乘以``survival_time_factor``后加到左截断时间。
    仅导出mask标记的真实观测，不进行插值或缺失填补。

    三个时间参数只是用于统一纵向与生存时间轴的数据契约，不是模型超参数。
    MIMIC冻结数据直接使用``landmark_time=48``、
    ``observation_time_factor=1/24``和``survival_time_factor=1``：0--48小时
    纵向历史转换为0--2天，患者在第2天进入风险集，绝对终点时间等于2天加
    landmark后的随访天数。
    """
    if re.fullmatch(r"[A-Za-z0-9_-]+", split_name) is None:
        raise ValueError("split_name只能包含字母、数字、下划线和连字符")
    if not np.isfinite(landmark_time) or landmark_time < 0.0:
        raise ValueError("landmark_time必须是非负有限数")
    if not np.isfinite(observation_time_factor) or observation_time_factor <= 0.0:
        raise ValueError("observation_time_factor必须是正有限数")
    if not np.isfinite(survival_time_factor) or survival_time_factor <= 0.0:
        raise ValueError("survival_time_factor必须是正有限数")
    if patient_batch_size <= 0:
        raise ValueError("patient_batch_size必须为正整数")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    patients_path = destination / f"{split_name}.patients.csv"
    observations_path = destination / f"{split_name}.observations.csv.gz"
    features_path = destination / f"{split_name}.features.csv"

    patient_ids = dataset_patient_ids(data)
    events, followup_times = dataset_survival_arrays(data)
    subject_ids = np.arange(1, len(data) + 1, dtype=np.int64)
    entry_time = float(landmark_time * observation_time_factor)
    event_times = entry_time + followup_times * survival_time_factor
    pd.DataFrame(
        {
            "subject_id": subject_ids,
            "patient_id": patient_ids,
            "entry_time": np.full(len(data), entry_time, dtype=np.float64),
            "event_time": event_times,
            "event": events.astype(np.int8),
        }
    ).to_csv(patients_path, index=False)
    pd.DataFrame(
        {
            "feature_id": np.arange(1, data.n_features + 1, dtype=np.int64),
            "feature_name": data.feature_names,
        }
    ).to_csv(features_path, index=False)

    wrote_header = False
    observation_count = 0
    with gzip.open(observations_path, "wt", encoding="utf-8", newline="") as stream:
        for batch_start in range(0, len(data), patient_batch_size):
            subject_blocks: list[np.ndarray] = []
            time_blocks: list[np.ndarray] = []
            feature_blocks: list[np.ndarray] = []
            value_blocks: list[np.ndarray] = []
            batch_stop = min(batch_start + patient_batch_size, len(data))

            for patient_index in range(batch_start, batch_stop):
                sample = data.samples[patient_index].to_aligned()
                positions = sample.mask.nonzero(as_tuple=False)
                if positions.shape[0] == 0:
                    raise ValueError(f"subject_id={patient_index + 1}没有真实纵向观测")
                visit_indices = positions[:, 0]
                feature_indices = positions[:, 1]
                times = (
                    sample.times[visit_indices].detach().cpu().numpy().astype(np.float64)
                    * observation_time_factor
                )
                values = (
                    sample.x[visit_indices, feature_indices]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                if not np.isfinite(times).all() or not np.isfinite(values).all():
                    raise ValueError(f"subject_id={patient_index + 1}包含非有限纵向观测")
                if np.any(times < 0.0) or np.any(times > entry_time + 1e-8):
                    raise ValueError(f"subject_id={patient_index + 1}的观测时间超出[0, landmark]")

                row_count = positions.shape[0]
                subject_blocks.append(np.full(row_count, patient_index + 1, dtype=np.int64))
                time_blocks.append(times)
                feature_blocks.append(feature_indices.detach().cpu().numpy().astype(np.int64) + 1)
                value_blocks.append(values)

            frame = pd.DataFrame(
                {
                    "subject_id": np.concatenate(subject_blocks),
                    "time": np.concatenate(time_blocks),
                    "feature_id": np.concatenate(feature_blocks),
                    "value": np.concatenate(value_blocks),
                }
            )
            frame.to_csv(stream, index=False, header=not wrote_header)
            wrote_header = True
            observation_count += len(frame)

    if observation_count == 0:
        raise ValueError("dataset没有可导出的真实纵向观测")
    return JointModelInputPaths(
        patients_csv=patients_path,
        observations_csv=observations_path,
        features_csv=features_path,
        landmark_entry_time=entry_time,
    )
