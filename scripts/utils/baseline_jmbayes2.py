"""将R端JMbayes2动态joint model适配到共享生存预测契约。"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
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


class JMbayes2Baseline:
    """用train拟合多变量joint model，并从外部纵向历史动态预测生存。"""

    capabilities: frozenset[BaselineCapability] = frozenset({"survival"})

    def __init__(
        self,
        name: str,
        seed: int,
        work_dir: Path,
        *,
        landmark_time: float,
        observation_time_factor: float,
        survival_time_factor: float,
        fit_options: Mapping[str, int | str],
        prediction_options: Mapping[str, int | str],
        rscript_executable: str = "Rscript",
        timeout_seconds: float | None = None,
        entrypoint: Path | None = None,
    ) -> None:
        self.name = name
        self.seed = seed
        self.landmark_time = landmark_time
        self.observation_time_factor = observation_time_factor
        self.survival_time_factor = survival_time_factor
        self.fit_options = dict(fit_options)
        self.prediction_options = dict(prediction_options)
        r_entrypoint = entrypoint or Path(__file__).with_name("r") / "jmbayes2.R"
        self.backend = RScriptBackend(
            r_entrypoint,
            work_dir,
            executable=rscript_executable,
            timeout_seconds=timeout_seconds,
        )
        self.model_path = self.backend.work_dir / "jmbayes2_model.rds"
        self.fit_metadata: dict[str, Any] | None = None
        self._prediction_index = 0

    def fit(
        self,
        train: ClinicalTimeSeriesDataset,
        validation: ClinicalTimeSeriesDataset,
    ) -> Self:
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
                "seed": self.seed,
                **self.fit_options,
            },
        )
        self._validate_common_result(result, len(train), train.n_features)
        if (
            not isinstance(result.get("mcmc"), dict)
            or not isinstance(result.get("rhat"), dict)
            or not isinstance(result["rhat"].get("available"), bool)
        ):
            raise ValueError("JMbayes2训练结果的MCMC诊断不符合契约")
        if not self.model_path.is_file():
            raise FileNotFoundError("JMbayes2未生成模型文件")
        self.fit_metadata = result
        return self

    def predict(
        self,
        data: ClinicalTimeSeriesDataset,
        *,
        prediction_times: NDArray[np.float64],
        risk_horizon: float,
    ) -> BaselinePrediction:
        if self.fit_metadata is None or not self.model_path.is_file():
            raise RuntimeError("JMbayes2Baseline必须先拟合")
        times = np.asarray(prediction_times, dtype=np.float64)
        if (
            times.ndim != 1
            or len(times) == 0
            or not np.isfinite(times).all()
            or np.any(times <= 0.0)
            or np.any(np.diff(times) <= 0.0)
            or not np.isfinite(risk_horizon)
            or times[-1] >= risk_horizon
        ):
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
        # R端不能读取目标结局；共享导出器产生的患者表在启动预测前删除。
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
                "seed": self.seed,
                **self.prediction_options,
            },
        )
        self._validate_prediction_result(result, len(data), data.n_features, times, risk_horizon)
        survival_columns = [str(value) for value in result["survival_columns"]]
        frame = pd.read_csv(predictions_path)
        expected = ["subject_id", "risk_score", *survival_columns]
        if frame.columns.tolist() != expected or len(frame) != len(data):
            raise ValueError("JMbayes2预测表的列或患者数不符合契约")
        if not np.array_equal(
            frame["subject_id"].to_numpy(dtype=np.int64),
            np.arange(1, len(data) + 1),
        ):
            raise ValueError("JMbayes2预测表的subject_id顺序无效")
        return BaselinePrediction(
            method_name=self.name,
            patient_ids=dataset_patient_ids(data),
            risk_score=frame["risk_score"].to_numpy(dtype=np.float64),
            risk_horizon=float(risk_horizon),
            survival_times=times.copy(),
            survival_probabilities=frame[survival_columns].to_numpy(dtype=np.float64),
        )

    def save_model(self, path: Path) -> None:
        if self.fit_metadata is None or not self.model_path.is_file():
            raise RuntimeError("JMbayes2Baseline必须先拟合")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != self.model_path:
            shutil.copy2(self.model_path, path)

    @staticmethod
    def _validate_common_result(result: dict[str, Any], n_patients: int, n_features: int) -> None:
        if (
            result.get("format_version") != 1
            or result.get("method") != "JMbayes2::jm"
            or result.get("n_patients") != n_patients
            or result.get("n_features") != n_features
        ):
            raise ValueError("JMbayes2结果元数据不符合契约")

    def _validate_prediction_result(
        self,
        result: dict[str, Any],
        n_patients: int,
        n_features: int,
        times: NDArray[np.float64],
        risk_horizon: float,
    ) -> None:
        self._validate_common_result(result, n_patients, n_features)
        if (
            result.get("outcome_columns_consumed") != []
            or result.get("prediction_inputs") != "longitudinal_only"
            or result.get("survival_conditioning") != "alive_at_entry"
            or result.get("prediction_times") != times.tolist()
            or result.get("risk_horizon") != risk_horizon
        ):
            raise ValueError("JMbayes2动态预测元数据不符合无结局泄漏契约")
