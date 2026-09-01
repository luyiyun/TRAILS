"""将R端mpjlcmm实现适配到共享基线预测契约。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from trails import ClinicalTimeSeriesDataset

from .baseline_features import dataset_patient_ids
from .baseline_joint_data import export_joint_model_input
from .baseline_r import RScriptBackend
from .baselines import BaselineCapability, BaselinePrediction


class MPJLCMMBaseline:
    """用train拟合mpjlcmm，并对外部纵向历史产生聚类和生存预测。"""

    capabilities: frozenset[BaselineCapability] = frozenset({"cluster", "survival"})

    def __init__(
        self,
        name: str,
        n_clusters: int,
        seed: int,
        work_dir: Path,
        *,
        landmark_time: float,
        observation_time_factor: float,
        survival_time_factor: float,
        max_iterations: int,
        grid_repetitions: int,
        grid_iterations: int,
        n_processes: int,
        rscript_executable: str = "Rscript",
        timeout_seconds: float | None = None,
        entrypoint: Path | None = None,
    ) -> None:
        iteration_options = (max_iterations, grid_repetitions, grid_iterations, n_processes)
        if not name or n_clusters < 2 or min(iteration_options) <= 0:
            raise ValueError("mpjlcmm名称、簇数、迭代次数和进程数配置无效")
        self.name = name
        self.n_clusters = n_clusters
        self.seed = seed
        self.landmark_time = landmark_time
        self.observation_time_factor = observation_time_factor
        self.survival_time_factor = survival_time_factor
        self.max_iterations = max_iterations
        self.grid_repetitions = grid_repetitions
        self.grid_iterations = grid_iterations
        self.n_processes = n_processes
        r_entrypoint = entrypoint or Path(__file__).with_name("r") / "mpjlcmm.R"
        self.backend = RScriptBackend(
            r_entrypoint,
            work_dir,
            executable=rscript_executable,
            timeout_seconds=timeout_seconds,
        )
        self.model_path = self.backend.work_dir / "mpjlcmm_model.rds"
        self._fitted = False
        self._prediction_index = 0

    def fit(
        self,
        train: ClinicalTimeSeriesDataset,
        validation: ClinicalTimeSeriesDataset,
    ) -> Self:
        """只用train的纵向观测和结局拟合joint latent class model。"""
        del validation
        self.backend.preflight()
        paths = export_joint_model_input(
            train,
            self.backend.work_dir / "inputs",
            "train",
            landmark_time=self.landmark_time,
            observation_time_factor=self.observation_time_factor,
            survival_time_factor=self.survival_time_factor,
        )
        result = self.backend.run(
            "fit",
            "fit",
            {
                "patients_csv": str(paths.patients_csv.resolve()),
                "observations_csv": str(paths.observations_csv.resolve()),
                "features_csv": str(paths.features_csv.resolve()),
                "model_path": str(self.model_path),
                "n_clusters": self.n_clusters,
                "seed": self.seed,
                "max_iterations": self.max_iterations,
                "grid_repetitions": self.grid_repetitions,
                "grid_iterations": self.grid_iterations,
                "n_processes": self.n_processes,
            },
        )
        self._validate_common_result(result, len(train))
        if int(result.get("convergence", -1)) != 1:
            raise RuntimeError(f"mpjlcmm未收敛：convergence={result.get('convergence')}")
        if not self.model_path.is_file():
            raise FileNotFoundError("mpjlcmm未生成模型文件")
        self._fitted = True
        return self

    def predict(
        self,
        data: ClinicalTimeSeriesDataset,
        *,
        prediction_times: NDArray[np.float64],
        risk_horizon: float,
    ) -> BaselinePrediction:
        """仅用目标划分的纵向历史计算类别后验和条件生存曲线。"""
        if not self._fitted or not self.model_path.is_file():
            raise RuntimeError("MPJLCMMBaseline必须先拟合")
        times = np.asarray(prediction_times, dtype=np.float64)
        if times.ndim != 1 or len(times) == 0 or times[-1] >= risk_horizon:
            raise ValueError("预测时间必须正且递增，并严格小于有限risk_horizon")
        self._prediction_index += 1
        run_name = f"predict-{self._prediction_index}"
        paths = export_joint_model_input(
            data,
            self.backend.work_dir / "inputs",
            run_name,
            landmark_time=self.landmark_time,
            observation_time_factor=self.observation_time_factor,
            survival_time_factor=self.survival_time_factor,
        )
        # 预测阶段不允许R进程读取目标结局；患者表只是共享导出器的临时副产物。
        paths.patients_csv.unlink()
        predictions_path = self.backend.work_dir / f"{run_name}.predictions.csv"
        result = self.backend.run(
            run_name,
            "predict",
            {
                "observations_csv": str(paths.observations_csv.resolve()),
                "features_csv": str(paths.features_csv.resolve()),
                "model_path": str(self.model_path),
                "predictions_csv": str(predictions_path),
                "entry_time": paths.landmark_entry_time,
                "prediction_times": times.tolist(),
                "risk_horizon": risk_horizon,
            },
        )
        self._validate_prediction_result(result, len(data))
        frame = pd.read_csv(predictions_path)
        probability_columns = [f"prob_{index}" for index in range(1, self.n_clusters + 1)]
        survival_columns = [str(value) for value in result["survival_columns"]]
        expected = ["subject_id", "cluster", "risk_score", *probability_columns, *survival_columns]
        if frame.columns.tolist() != expected or len(frame) != len(data):
            raise ValueError("mpjlcmm预测表的列或患者数不符合契约")
        subject_ids = frame["subject_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(subject_ids, np.arange(1, len(data) + 1)):
            raise ValueError("mpjlcmm预测表的subject_id顺序无效")
        return BaselinePrediction(
            method_name=self.name,
            patient_ids=dataset_patient_ids(data),
            cluster_labels=frame["cluster"].to_numpy(dtype=np.int64) - 1,
            n_clusters=self.n_clusters,
            risk_score=frame["risk_score"].to_numpy(dtype=np.float64),
            risk_horizon=float(risk_horizon),
            survival_times=times.copy(),
            survival_probabilities=frame[survival_columns].to_numpy(dtype=np.float64),
        )

    def save_model(self, path: Path) -> None:
        """复制已拟合的RDS模型到编排器指定位置。"""
        if not self._fitted or not self.model_path.is_file():
            raise RuntimeError("MPJLCMMBaseline必须先拟合")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != self.model_path:
            shutil.copy2(self.model_path, path)

    def _validate_common_result(self, result: dict[str, Any], n_patients: int) -> None:
        if (
            result.get("format_version") != 1
            or result.get("method") != "lcmm::mpjlcmm"
            or result.get("n_patients") != n_patients
            or result.get("n_clusters") != self.n_clusters
        ):
            raise ValueError("mpjlcmm结果元数据不符合契约")

    def _validate_prediction_result(
        self,
        result: dict[str, Any],
        n_patients: int,
    ) -> None:
        self._validate_common_result(result, n_patients)
        if (
            result.get("outcome_columns_consumed") != []
            or result.get("class_assignment_inputs") != "longitudinal_only"
            or result.get("survival_conditioning") != "alive_at_entry"
        ):
            raise ValueError("mpjlcmm外部预测元数据不符合无结局泄漏契约")
