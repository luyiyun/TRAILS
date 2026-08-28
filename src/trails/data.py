"""TRAILS 纵向临床样本、数据集、批处理和数据加载工具。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, TrainerConfig, resolve_batch_size

type SampleKind = Literal["aligned", "compact"]
Batch = dict[str, Tensor]
type AlignedBatch = dict[str, Tensor]
type CompactBatch = dict[str, Tensor]
PATIENT_LEVEL_METADATA_KEYS = frozenset({"latent_z", "sequence_lengths"})


logger = getLogger(__name__)


@dataclass(frozen=True)
class AlignedClinicalSample:
    """表示共享访视时间轴上的单个患者纵向样本。

    所有特征共享 ``times`` 中的访视时点；某个特征在某次访视是否实际观测由
    ``mask`` 决定，未观测位置的 ``x`` 不参与模型目标。

    属性：
        times: 形状为 ``(n_visits,)`` 的访视时间。
        x: 形状为 ``(n_visits, n_features)`` 的纵向特征值。
        mask: 与 ``x`` 同形状的观测指示矩阵，取值范围为 ``[0, 1]``。
        delta_time: 与 ``x`` 同形状，记录各特征距上一次观测经过的时间。
        survival_time: 正的标量随访或事件时间。
        event: 标量事件指示，取值范围为 ``[0, 1]``。
        cluster_label: 可选的标量参考簇标签，仅用于已知真值的评价。
    """

    times: Tensor
    x: Tensor
    mask: Tensor
    delta_time: Tensor
    survival_time: Tensor
    event: Tensor
    cluster_label: Tensor | None

    def __post_init__(self) -> None:
        """在冻结样本创建后校验张量形状和值域。"""
        validate_aligned_clinical_sample(self)

    def to_aligned(self) -> AlignedClinicalSample:
        """返回当前 aligned 视图，不复制底层张量。"""
        return self

    def to_compact(self) -> CompactClinicalSample:
        """转换为每个特征拥有独立时间轴的 compact 视图。

        转换仅保留 ``mask`` 标记的真实观测，并按特征分别左对齐；生存结局和
        可选参考簇标签保持不变。

        返回：
            与当前样本等价的 :class:`CompactClinicalSample`。
        """
        observed = self.mask > 0
        feature_lengths = observed.sum(dim=0).long()
        max_length = max(1, int(feature_lengths.max().item()))
        n_features = int(self.x.shape[-1])
        compact_times = self.x.new_zeros(max_length, n_features)
        compact_x = self.x.new_zeros(max_length, n_features)
        compact_mask = self.x.new_zeros(max_length, n_features)

        # compact 视图按变量独立左对齐，只保留真实观测点。
        for feature_index in range(n_features):
            feature_observed = observed[:, feature_index]
            length = int(feature_lengths[feature_index].item())
            if length == 0:
                continue
            compact_times[:length, feature_index] = self.times[feature_observed]
            compact_x[:length, feature_index] = self.x[feature_observed, feature_index]
            compact_mask[:length, feature_index] = 1.0

        return CompactClinicalSample(
            times=compact_times,
            x=compact_x,
            mask=compact_mask,
            feature_lengths=feature_lengths,
            survival_time=self.survival_time,
            event=self.event,
            cluster_label=self.cluster_label,
        )


@dataclass(frozen=True)
class CompactClinicalSample:
    """表示按特征独立左对齐的单个患者纵向样本。

    每列对应一个特征自己的观测流，列内前 ``feature_lengths`` 个位置有效，
    因而不同特征不必共享访视时点。

    属性：
        times: 形状为 ``(max_observations, n_features)`` 的特征级观测时间。
        x: 与 ``times`` 同形状的纵向特征值。
        mask: 与 ``x`` 同形状的左对齐观测指示矩阵。
        feature_lengths: 形状为 ``(n_features,)`` 的各特征有效观测数。
        survival_time: 正的标量随访或事件时间。
        event: 标量事件指示，取值范围为 ``[0, 1]``。
        cluster_label: 可选的标量参考簇标签，仅用于已知真值的评价。
    """

    times: Tensor
    x: Tensor
    mask: Tensor
    feature_lengths: Tensor
    survival_time: Tensor
    event: Tensor
    cluster_label: Tensor | None

    def __post_init__(self) -> None:
        """在冻结样本创建后校验张量形状、值域和左对齐约束。"""
        validate_compact_clinical_sample(self)

    def to_compact(self) -> CompactClinicalSample:
        """返回当前 compact 视图，不复制底层张量。"""
        return self

    def to_aligned(self) -> AlignedClinicalSample:
        """将各特征的独立时间轴合并为共享的 aligned 视图。

        合并后的时间轴是所有真实观测时间的有序并集，未在某时点观测的特征由
        ``mask`` 标记为缺失，并根据新时间轴重新计算 ``delta_time``。

        返回：
            与当前样本等价的 :class:`AlignedClinicalSample`。

        异常：
            ValueError: 当样本没有真实观测，或同一特征包含重复时间时抛出。
        """
        observed_times = self.times[self.mask > 0].float()
        if observed_times.numel() == 0:
            raise ValueError("compact samples must contain at least one observed value.")
        aligned_times = torch.unique(observed_times, sorted=True)
        n_visits = int(aligned_times.shape[0])
        n_features = int(self.x.shape[-1])
        aligned_x = self.x.new_zeros(n_visits, n_features)
        aligned_mask = self.x.new_zeros(n_visits, n_features)

        # 将每个变量自己的 compact 时间轴放回统一 aligned 时间轴。
        for feature_index in range(n_features):
            length = int(self.feature_lengths[feature_index].item())
            if length == 0:
                continue
            feature_times = self.times[:length, feature_index].float().contiguous()
            if int(torch.unique(feature_times).shape[0]) != length:
                raise ValueError("compact samples cannot contain duplicate feature-time pairs.")
            target_indices = torch.searchsorted(aligned_times, feature_times)
            aligned_x[target_indices, feature_index] = self.x[:length, feature_index]
            aligned_mask[target_indices, feature_index] = 1.0

        return AlignedClinicalSample(
            times=aligned_times,
            x=aligned_x,
            mask=aligned_mask,
            delta_time=compute_delta_time(aligned_times, aligned_mask),
            survival_time=self.survival_time,
            event=self.event,
            cluster_label=self.cluster_label,
        )


type DatasetSample = AlignedClinicalSample | CompactClinicalSample


class ClinicalTimeSeriesDataset(Dataset[DatasetSample]):
    """管理具有统一特征定义的变长临床时间序列样本集合。

    数据集在初始化时将所有样本转换为指定的 ``return_kind``，而不是在
    ``__getitem__`` 中按次转换。所有样本必须具有相同特征维度，并且必须全部
    包含参考簇标签或全部不包含标签。

    属性：
        samples: 已转换为当前返回视图的患者样本列表。
        feature_names: 按特征维度顺序排列的变量名称。
        description: 数据集的人类可读说明。
        metadata: 模拟机制、患者标识或其他调用方提供的附加信息。
        feature_means: 根据真实观测位置计算的各特征均值。
        return_kind: 样本采用 ``"aligned"`` 还是 ``"compact"`` 视图。
        has_cluster_labels: 是否所有样本都带有参考簇标签。
    """

    def __init__(
        self,
        samples: Sequence[DatasetSample],
        *,
        feature_names: list[str],
        description: str = "",
        metadata: dict[str, Any] | None = None,
        return_kind: SampleKind = "aligned",
    ) -> None:
        """构造并校验临床时间序列数据集。

        参数：
            samples: 至少包含一个 aligned 或 compact 患者样本的序列。
            feature_names: 与样本特征维度一一对应的变量名称。
            description: 可选的数据集说明。
            metadata: 可选的附加元数据字典。
            return_kind: 数据集内部存储和返回的样本视图。

        异常：
            ValueError: 当视图类型无效、样本为空、特征维度不一致，或混合使用
                有标签和无标签样本时抛出。
        """
        if return_kind not in {"aligned", "compact"}:
            raise ValueError("return_kind must be 'aligned' or 'compact'.")
        input_samples = list(samples)
        if not input_samples:
            raise ValueError("ClinicalTimeSeriesDataset requires at least one sample.")
        if len(feature_names) != int(input_samples[0].x.shape[-1]):
            raise ValueError("feature_names length must match sample feature dimension.")
        for sample in input_samples:
            if int(sample.x.shape[-1]) != len(feature_names):
                raise ValueError("All samples must share the same feature dimension.")
        converted_samples: list[DatasetSample] = [
            sample.to_aligned() if return_kind == "aligned" else sample.to_compact()
            for sample in input_samples
        ]
        self.has_cluster_labels: bool = converted_samples[0].cluster_label is not None
        for sample in converted_samples:
            if (sample.cluster_label is not None) != self.has_cluster_labels:
                raise ValueError(
                    "ClinicalTimeSeriesDataset cannot mix labeled and unlabeled samples."
                )
        self.samples = converted_samples
        self.feature_names = feature_names
        self.description = description
        self.metadata = metadata or {}
        self.feature_means = compute_feature_means(converted_samples)
        self.return_kind: SampleKind = return_kind

    def __len__(self) -> int:
        """返回数据集中的患者样本数。"""
        return len(self.samples)

    def __getitem__(self, index: int) -> DatasetSample:
        """按位置返回当前视图中的患者样本。"""
        return self.samples[index]

    @property
    def n_features(self) -> int:
        """返回每个患者样本包含的纵向特征数。"""
        return len(self.feature_names)

    def with_return_kind(self, return_kind: SampleKind) -> ClinicalTimeSeriesDataset:
        """创建采用指定样本视图的新数据集。

        每个患者样本会通过自身的视图转换方法转换；特征名称、说明和元数据继续
        传入新数据集，并重新计算特征均值。当前数据集不会被修改。

        参数：
            return_kind: 目标视图，可选 ``"aligned"`` 或 ``"compact"``。

        返回：
            采用目标视图的新 :class:`ClinicalTimeSeriesDataset`。

        异常：
            ValueError: 当目标视图无效或样本无法完成转换时抛出。
        """
        return ClinicalTimeSeriesDataset(
            self.samples,
            feature_names=self.feature_names,
            description=self.description,
            metadata=self.metadata,
            return_kind=return_kind,
        )

    def save(self, path: str | Path) -> None:
        """使用 PyTorch 序列化格式保存数据集。

        无论当前返回视图为何，磁盘载荷始终保存 aligned 样本及其
        ``delta_time``，同时保留特征名称、说明和元数据。目标父目录不存在时
        会自动创建。

        参数：
            path: 目标 ``.pt`` 文件路径。
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        aligned_samples = [sample.to_aligned() for sample in self.samples]
        payload = {
            "description": self.description,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
            "samples": [
                {
                    "times": sample.times,
                    "x": sample.x,
                    "mask": sample.mask,
                    "delta_time": sample.delta_time,
                    "survival_time": sample.survival_time,
                    "event": sample.event,
                    "cluster_label": sample.cluster_label,
                }
                for sample in aligned_samples
            ],
        }
        torch.save(payload, destination)

    def save_to_csv(
        self,
        *,
        patients_csv: str | Path,
        observations_csv: str | Path,
        patient_ids: Sequence[str] | None = None,
        patient_id_col: str = "patient_id",
        survival_time_col: str = "survival_time",
        event_col: str = "event",
        cluster_label_col: str = "cluster_label",
        observation_patient_id_col: str = "patient_id",
        time_col: str = "time",
        feature_col: str = "feature",
        value_col: str = "value",
    ) -> None:
        """将数据集导出为患者表和长格式纵向观测表。

        患者表每位患者一行，包含生存结局和可选参考簇；观测表仅写出
        ``mask > 0`` 的真实观测。患者 ID 优先使用显式输入，其次使用元数据中
        的 ``patient_ids``，否则生成 ``sample_<index>``。

        参数：
            patients_csv: 患者级 CSV 输出路径。
            observations_csv: 长格式观测 CSV 输出路径。
            patient_ids: 可选的唯一非空患者 ID 序列。
            patient_id_col: 患者表 ID 列名。
            survival_time_col: 患者表生存时间列名。
            event_col: 患者表事件指示列名。
            cluster_label_col: 患者表参考簇列名。
            observation_patient_id_col: 观测表患者 ID 列名。
            time_col: 观测时间列名。
            feature_col: 特征名称列名。
            value_col: 观测值列名。

        异常：
            ValueError: 当患者 ID 数量或唯一性无效，或任一患者没有真实观测时
                抛出。
        """
        resolved_patient_ids = _resolve_csv_patient_ids(
            self,
            explicit_patient_ids=patient_ids,
        )
        aligned_samples = [sample.to_aligned() for sample in self.samples]

        patient_rows: list[dict[str, Any]] = []
        observation_rows: list[dict[str, Any]] = []
        for patient_id, sample in zip(resolved_patient_ids, aligned_samples, strict=True):
            patient_row: dict[str, Any] = {
                patient_id_col: patient_id,
                survival_time_col: float(sample.survival_time),
                event_col: float(sample.event),
            }
            if sample.cluster_label is not None:
                patient_row[cluster_label_col] = int(sample.cluster_label)
            patient_rows.append(patient_row)

            observed_positions = torch.nonzero(sample.mask > 0, as_tuple=False)
            if int(observed_positions.shape[0]) == 0:
                raise ValueError("save_to_csv requires every patient to have observed values.")
            for visit_index, feature_index in observed_positions.tolist():
                observation_rows.append(
                    {
                        observation_patient_id_col: patient_id,
                        time_col: float(sample.times[int(visit_index)]),
                        feature_col: self.feature_names[int(feature_index)],
                        value_col: float(sample.x[int(visit_index), int(feature_index)]),
                    }
                )

        patient_columns = [patient_id_col, survival_time_col, event_col]
        if self.has_cluster_labels:
            patient_columns.append(cluster_label_col)
        observation_columns = [observation_patient_id_col, time_col, feature_col, value_col]

        patients_path = Path(patients_csv)
        observations_path = Path(observations_csv)
        patients_path.parent.mkdir(parents=True, exist_ok=True)
        observations_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(patient_rows, columns=patient_columns).to_csv(patients_path, index=False)
        pd.DataFrame(observation_rows, columns=observation_columns).to_csv(
            observations_path,
            index=False,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        return_kind: SampleKind = "aligned",
    ) -> ClinicalTimeSeriesDataset:
        """从 PyTorch 序列化文件加载临床时间序列数据集。

        保存的张量先映射到 CPU，并重建为经过校验的 aligned 样本，再根据
        ``return_kind`` 转换为调用方需要的视图。

        参数：
            path: 由 :meth:`save` 写出的数据集文件路径。
            return_kind: 加载后采用的样本视图。

        返回：
            在 CPU 上重建并采用指定视图的 :class:`ClinicalTimeSeriesDataset`。

        异常：
            ValueError: 当载荷中的样本不满足数据契约或视图类型无效时抛出。
        """
        payload: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=False)
        samples = [
            make_clinical_sample(
                times=sample["times"],
                x=sample["x"],
                mask=sample["mask"],
                delta_time=sample["delta_time"],
                survival_time=sample["survival_time"],
                event=sample["event"],
                cluster_label=sample.get("cluster_label"),
            )
            for sample in payload["samples"]
        ]
        return ClinicalTimeSeriesDataset(
            samples,
            feature_names=list(payload["feature_names"]),
            description=str(payload.get("description", "")),
            metadata=dict(payload.get("metadata", {})),
            return_kind=return_kind,
        )

    @classmethod
    def load_from_csv(
        cls,
        *,
        patients_csv: str | Path,
        observations_csv: str | Path,
        patient_id_col: str = "patient_id",
        survival_time_col: str = "survival_time",
        event_col: str = "event",
        cluster_label_col: str = "cluster_label",
        observation_id_col: str = "patient_id",
        time_col: str = "time",
        feature_col: str = "feature",
        value_col: str = "value",
        use_features: Sequence[str] = (),
        description: str = "",
        metadata: dict[str, Any] | None = None,
        return_kind: SampleKind = "aligned",
    ) -> ClinicalTimeSeriesDataset:
        """从患者表和长格式纵向观测表构建临床时间序列数据集。

        观测按患者和时间透视为 aligned 样本，缺失位置填零并由 ``mask`` 标记；
        ``delta_time`` 根据排序后的时间轴计算。若患者表含参考簇列，其原始类别
        会编码为连续整数，并在元数据中保存类别编码。生成的元数据还记录列映射、
        特征顺序、患者 ID 和患者级观测摘要。

        参数：
            patients_csv: 含患者 ID、生存时间、事件和可选参考簇的 CSV。
            observations_csv: 含患者 ID、时间、特征和值的长格式 CSV。
            patient_id_col: 患者表 ID 列名。
            survival_time_col: 患者表生存时间列名。
            event_col: 患者表事件指示列名。
            cluster_label_col: 可选参考簇列名；列不存在时按无标签数据处理。
            observation_id_col: 观测表患者 ID 列名。
            time_col: 观测时间列名。
            feature_col: 特征名称列名。
            value_col: 观测值列名。
            use_features: 可选的特征筛选及排序序列；为空时使用文件出现顺序。
            description: 数据集说明。
            metadata: 与自动生成元数据合并的调用方元数据，同名键由调用方覆盖。
            return_kind: 返回 ``"aligned"`` 或 ``"compact"`` 样本视图。

        返回：
            从两个 CSV 文件构建并校验的 :class:`ClinicalTimeSeriesDataset`。

        异常：
            AssertionError: 当必需列、患者 ID 或观测表基本约束不满足时抛出。
            ValueError: 当患者没有观测、样本值不满足契约或视图类型无效时抛出。
        """
        patients_path = Path(patients_csv)
        observations_path = Path(observations_csv)
        metadata = dict(metadata or {})

        # -----------------------------------------
        # 1. 读取患者信息
        # -----------------------------------------
        patient_frame = pd.read_csv(
            patients_path,
            na_values=[
                "nan",
                "+nan",
                "-nan",
                "inf",
                "+inf",
                "-inf",
                "infinity",
                "+infinity",
                "-infinity",
                " ",
            ],
        )
        for col in [patient_id_col, survival_time_col, event_col]:
            assert col in patient_frame.columns, f"{patients_path} must contain column {col!r}."

        patient_frame = patient_frame.astype(
            {patient_id_col: str, survival_time_col: float, event_col: float}
        )
        assert not patient_frame[patient_id_col].duplicated().any(), (
            f"{patients_csv} has duplicated ids."
        )
        assert (patient_frame[survival_time_col].to_numpy() >= 0.0).any(), (
            f"{patients_csv} must contain positive survival_time values."
        )
        assert bool((patient_frame[event_col].isin([0.0, 1.0])).any()), (
            f"{patients_csv} must contain 0 or 1 event values."
        )

        has_cluster_labels = cluster_label_col is not None and cluster_label_col in patient_frame
        if has_cluster_labels:
            le = LabelEncoder()
            patient_frame[cluster_label_col] = le.fit_transform(
                patient_frame[cluster_label_col].to_numpy()
            )
            metadata["cluster_label_codes"] = le.classes_

        assert patient_frame.shape[0] > 0, f"{patients_csv} must contain at least one patient row."

        patient_frame.set_index(patient_id_col, inplace=True)

        # -----------------------------------------
        # 2. 读取纵向观测数值
        # -----------------------------------------
        observation_frame = pd.read_csv(
            observations_path,
            na_values=[
                "nan",
                "+nan",
                "-nan",
                "inf",
                "+inf",
                "-inf",
                "infinity",
                "+infinity",
                "-infinity",
                " ",
            ],
        )
        for col in [observation_id_col, time_col, feature_col, value_col]:
            assert col in observation_frame.columns, (
                f"{observations_csv} must contain column {col!r}."
            )
        # configured_features = list(feature_order)
        # _validate_unique_names(configured_features, label="feature_order")
        # known_patients = {str(record["patient_id"]) for record in patient_records}

        observation_frame = observation_frame.astype(
            {
                observation_id_col: str,
                feature_col: str,
                value_col: float,
                time_col: float,
            }
        )

        unknown_patient = ~(
            observation_frame[observation_id_col].isin(patient_frame.index.tolist())
        )
        assert not bool(unknown_patient.any()), f"{observations_path} contains unknown patient_ids."

        if use_features:
            mask = observation_frame[feature_col].isin(use_features)
            observation_frame: pd.DataFrame = observation_frame.loc[mask, :]

        canonical_observations: pd.DataFrame = observation_frame.loc[
            :, [observation_id_col, time_col, feature_col, value_col]
        ].rename(
            columns={
                observation_id_col: "patient_id",
                time_col: "time",
                feature_col: "feature",
                value_col: "value",
            }
        )

        assert not canonical_observations.duplicated(["patient_id", "time", "feature"]).any(), (
            f"{observations_path} contains duplicate observations."
        )
        assert bool(canonical_observations["value"].notna().all()), (
            f"{observations_path} contains invalid numeric values."
        )

        # -----------------------------------------
        # 3. 转换格式，形成dataset
        # -----------------------------------------
        feature_names = (
            list(dict.fromkeys(use_features))
            if use_features
            else canonical_observations["feature"].unique().tolist()
        )
        grouped_observations = {
            str(patient_id): patient_observations
            for patient_id, patient_observations in canonical_observations.groupby(
                canonical_observations["patient_id"],
                sort=False,
            )
        }
        missing_patient_ids = [
            patient_id
            for patient_id in patient_frame.index.tolist()
            if patient_id not in grouped_observations
        ]
        if missing_patient_ids:
            preview = ", ".join(str(patient_id) for patient_id in missing_patient_ids[:5])
            suffix = (
                "" if len(missing_patient_ids) <= 5 else f", ... ({len(missing_patient_ids)} total)"
            )
            raise ValueError(
                f"Every patient must have at least one observation; missing: {preview}{suffix}"
            )

        samples = []
        patient_summaries = []
        for i, (pid, dfi) in enumerate(grouped_observations.items()):
            dfi_wide = dfi.pivot(index="time", columns="feature", values="value")
            dfi_wide.sort_index(inplace=True)
            dfi_wide = dfi_wide.reindex(columns=feature_names)
            mask = torch.as_tensor(dfi_wide.notna().to_numpy(copy=True), dtype=torch.float32)
            times = torch.as_tensor(dfi_wide.index.to_numpy(copy=True), dtype=torch.float32)
            x = torch.as_tensor(dfi_wide.fillna(0.0).to_numpy(copy=True), dtype=torch.float32)

            samples.append(
                AlignedClinicalSample(
                    times=times,
                    x=x,
                    mask=mask,
                    delta_time=compute_delta_time(times, mask),
                    survival_time=torch.as_tensor(
                        patient_frame.loc[pid, survival_time_col], dtype=torch.float32
                    ),
                    event=torch.as_tensor(patient_frame.loc[pid, event_col], dtype=torch.float32),
                    cluster_label=(
                        None
                        if not has_cluster_labels
                        else torch.as_tensor(
                            patient_frame.loc[pid, cluster_label_col], dtype=torch.long
                        )
                    ),
                )
            )

            n_observations = int(mask.sum().item())
            total_slots = x.shape[0] * x.shape[1]
            patient_summaries.append(
                {
                    "patient_id": pid,
                    "sample_index": i,
                    "n_observations": n_observations,
                    "n_visits": x.shape[0],
                    "first_time": times[0].item(),
                    "last_time": times[-1].item(),
                    "missing_fraction": 1.0 - (n_observations / float(total_slots)),
                }
            )

        csv_columns = {
            "patients": {
                "patient_id": patient_id_col,
                "survival_time": survival_time_col,
                "event": event_col,
                "cluster_label": cluster_label_col,
            },
            "observations": {
                "patient_id": observation_id_col,
                "time": time_col,
                "feature": feature_col,
                "value": value_col,
            },
        }
        generated_metadata = {
            "csv_columns": csv_columns,
            "feature_names": feature_names,
            "has_cluster_labels": has_cluster_labels,
            "n_features": len(feature_names),
            "n_observations": int(canonical_observations.shape[0]),
            "n_patients": len(samples),
            "observations_csv": str(observations_path),
            "patient_ids": [str(record["patient_id"]) for record in patient_summaries],
            "patient_summaries": patient_summaries,
            "patients_csv": str(patients_path),
            "source": "csv",
        }
        generated_metadata.update(metadata)
        metadata = generated_metadata

        return cls(
            samples,
            feature_names=feature_names,
            description=description,
            metadata=metadata,
            return_kind=return_kind,
        )

    def split(self, fraction: list[float], seed: int = 0) -> list[ClinicalTimeSeriesDataset]:
        """按给定比例随机拆分患者，并保持样本视图和患者级元数据对齐。

        参数：
            fraction: 总和为 ``1`` 的拆分比例；每个比例必须产生正的样本数。
            seed: 患者索引洗牌使用的随机种子。

        返回：
            按输入比例顺序排列的独立数据集列表；单个比例时返回当前数据集。

        异常：
            ValueError: 当比例和不为 ``1`` 或产生无效拆分计数时抛出。
        """
        if not np.isclose(float(sum(fraction)), 1.0):
            raise ValueError("Split fractions must sum to 1.")

        if len(fraction) == 1:
            return [self]

        n_samples = len(self.samples)
        split_indices = np.cumsum(np.array(fraction) * n_samples).astype(int).tolist()
        counts = [
            end - start for start, end in zip([0] + split_indices[:-1], split_indices, strict=True)
        ]
        return self.split_counts(counts, seed=seed, split_fractions=fraction)

    def split_counts(
        self,
        counts: list[int],
        seed: int = 0,
        *,
        split_fractions: list[float] | None = None,
    ) -> list[ClinicalTimeSeriesDataset]:
        """按精确患者数随机拆分数据集。

        拆分会使用 ``seed`` 确定性地打乱索引，并同步切分已知的患者级元数据。
        ``split_fractions`` 仅用于把原始比例记录到各子集元数据。

        参数：
            counts: 每个子集的正患者数，总和必须等于数据集长度。
            seed: 患者索引洗牌使用的随机种子。
            split_fractions: 可选的原始拆分比例，仅用于元数据。

        返回：
            按 ``counts`` 顺序排列的独立数据集列表。

        异常：
            ValueError: 当计数为空、非正或总和与数据集长度不符时抛出。
        """
        if not counts:
            raise ValueError("Split counts must contain at least one split.")
        if any(count <= 0 for count in counts):
            raise ValueError("Split counts must be positive.")
        if sum(counts) != len(self.samples):
            raise ValueError("Split counts must sum to dataset length.")
        if len(counts) == 1:
            return [self]

        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.samples))
        rng.shuffle(indices)
        split_indices = np.cumsum(np.array(counts)).astype(int).tolist()
        res = []
        for split_index, (start, end) in enumerate(
            zip([0] + split_indices[:-1], split_indices, strict=True)
        ):
            split_sample_indices = indices[start:end]
            samples_i = [self.samples[int(i)] for i in split_sample_indices]
            res.append(
                ClinicalTimeSeriesDataset(
                    samples_i,
                    feature_names=self.feature_names,
                    description=f"{self.description} (split {split_index + 1}/{len(counts)})",
                    metadata=self._split_metadata(
                        split_sample_indices,
                        split_index=split_index,
                        split_count=len(counts),
                        split_fraction=(
                            None if split_fractions is None else split_fractions[split_index]
                        ),
                    ),
                    return_kind=self.return_kind,
                )
            )

        return res

    def _split_metadata(
        self,
        indices: np.ndarray,
        *,
        split_index: int,
        split_count: int,
        split_fraction: float | None,
    ) -> dict[str, Any]:
        """复制公共元数据并按患者索引切分已知患者级字段。"""
        metadata = dict(self.metadata)
        for key in PATIENT_LEVEL_METADATA_KEYS:
            if key in metadata:
                metadata[key] = _slice_patient_metadata(
                    metadata[key],
                    indices,
                    source_count=len(self.samples),
                )
        metadata.update(
            {
                "source_patient_count": len(self.samples),
                "split_index": split_index,
                "split_count": split_count,
                "split_patient_count": int(indices.shape[0]),
            }
        )
        if split_fraction is not None:
            metadata["split_fraction"] = split_fraction
        return metadata


def _slice_patient_metadata(value: Any, indices: np.ndarray, *, source_count: int) -> Any:
    """按索引切分首维等于原患者数的常见容器。"""
    if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == source_count:
        tensor_indices = torch.as_tensor(indices, dtype=torch.long)
        return value[tensor_indices]
    if isinstance(value, np.ndarray) and value.ndim > 0 and int(value.shape[0]) == source_count:
        return value[indices]
    if isinstance(value, list) and len(value) == source_count:
        return [value[int(index)] for index in indices]
    if isinstance(value, tuple) and len(value) == source_count:
        return tuple(value[int(index)] for index in indices)
    return value


def _resolve_csv_patient_ids(
    dataset: ClinicalTimeSeriesDataset,
    *,
    explicit_patient_ids: Sequence[str] | None,
) -> list[str]:
    """按显式输入、元数据和自动生成的优先级解析导出患者 ID。"""
    if explicit_patient_ids is not None:
        return _validate_csv_patient_ids(
            explicit_patient_ids,
            expected_count=len(dataset),
            label="patient_ids",
        )

    metadata_patient_ids = dataset.metadata.get("patient_ids")
    if isinstance(metadata_patient_ids, Sequence) and not isinstance(
        metadata_patient_ids,
        str | bytes,
    ):
        values = [str(value).strip() for value in metadata_patient_ids]
        if _is_valid_csv_patient_ids(values, expected_count=len(dataset)):
            return values

    return [f"sample_{index}" for index in range(len(dataset))]


def _validate_csv_patient_ids(
    patient_ids: Sequence[str],
    *,
    expected_count: int,
    label: str,
) -> list[str]:
    """规范化并校验患者 ID 的数量、非空性和唯一性。"""
    values = [str(value).strip() for value in patient_ids]
    if len(values) != expected_count:
        raise ValueError(f"{label} length must match dataset length.")
    empty = [index for index, value in enumerate(values) if value == ""]
    if empty:
        raise ValueError(f"{label} cannot contain empty values.")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"{label} cannot contain duplicates: {', '.join(duplicates)}.")
    return values


def _is_valid_csv_patient_ids(values: Sequence[str], *, expected_count: int) -> bool:
    """判断患者 ID 序列是否具有预期数量且非空唯一。"""
    return (
        len(values) == expected_count
        and all(value != "" for value in values)
        and len(set(values)) == len(values)
    )


def make_clinical_sample(
    *,
    times: Tensor,
    x: Tensor,
    mask: Tensor,
    delta_time: Tensor,
    survival_time: float | Tensor,
    event: float | Tensor,
    cluster_label: int | Tensor | None = None,
) -> AlignedClinicalSample:
    """构造经过 dtype 规范化和完整校验的 aligned 临床样本。

    特征、掩码、时间和结局统一转换为 ``float32``，参考簇标签转换为
    ``long``。本函数不会推导 ``delta_time``，调用方应传入与 ``x`` 同形状的
    已计算张量。返回对象初始化时会自动执行形状和值域校验。

    参数：
        times: 形状为 ``(n_visits,)`` 的访视时间。
        x: 形状为 ``(n_visits, n_features)`` 的特征值。
        mask: 与 ``x`` 同形状的观测指示矩阵。
        delta_time: 与 ``x`` 同形状的距上次观测时间。
        survival_time: 正的随访或事件时间。
        event: 取值范围为 ``[0, 1]`` 的事件指示。
        cluster_label: 可选的整数参考簇标签。

    返回：
        规范化后的 :class:`AlignedClinicalSample`。

    异常：
        ValueError: 当张量形状、值域或标量约束不满足样本契约时抛出。
    """
    return AlignedClinicalSample(
        times=times.float(),
        x=x.float(),
        mask=mask.float(),
        delta_time=delta_time.float(),
        survival_time=torch.as_tensor(survival_time, dtype=torch.float32),
        event=torch.as_tensor(event, dtype=torch.float32),
        cluster_label=(
            None if cluster_label is None else torch.as_tensor(cluster_label, dtype=torch.long)
        ),
    )


def compute_delta_time(times: Tensor, mask: Tensor) -> Tensor:
    """计算每个访视时点上各特征距其上一次观测经过的时间。

    首次访视的间隔固定为零；后续时点若该特征在前一次访视中被观测，则从本次
    时间差重新累计，否则继续累加缺失期间经过的时间。

    参数：
        times: 形状为 ``(n_visits,)`` 的共享访视时间。
        mask: 形状为 ``(n_visits, n_features)`` 的观测指示矩阵。

    返回：
        与 ``mask`` 同形状、继承其 dtype 和设备的时间间隔张量。
    """
    delta_time = torch.zeros_like(mask)
    for step in range(1, int(times.shape[0])):
        gap = times[step] - times[step - 1]
        delta_time[step] = torch.where(mask[step - 1] > 0, gap, delta_time[step - 1] + gap)
    return delta_time


def validate_aligned_clinical_sample(sample: AlignedClinicalSample) -> None:
    """校验 aligned 样本的张量形状、值域和标量约束。"""
    if sample.times.ndim != 1:
        raise ValueError("times must have shape (n_visits,).")
    if sample.x.ndim != 2:
        raise ValueError("x must have shape (n_visits, n_features).")
    if sample.mask.shape != sample.x.shape:
        raise ValueError("mask must have the same shape as x.")
    if sample.delta_time.shape != sample.x.shape:
        raise ValueError("delta_time must have the same shape as x.")
    if sample.times.shape[0] != sample.x.shape[0]:
        raise ValueError("times length must match x visit dimension.")
    if torch.any(sample.delta_time < 0):
        raise ValueError("delta_time values must be non-negative.")
    if torch.any((sample.mask < 0) | (sample.mask > 1)):
        raise ValueError("mask values must be in [0, 1].")
    if float(sample.survival_time) <= 0:
        raise ValueError("survival_time must be positive.")
    if float(sample.event) < 0 or float(sample.event) > 1:
        raise ValueError("event must be in [0, 1].")
    if sample.cluster_label is not None and sample.cluster_label.ndim > 0:
        raise ValueError("cluster_label must be a scalar tensor when provided.")


def validate_compact_clinical_sample(sample: CompactClinicalSample) -> None:
    """校验 compact 样本的张量形状、值域和左对齐约束。"""
    if sample.times.ndim != 2:
        raise ValueError("compact times must have shape (max_observations, n_features).")
    if sample.x.shape != sample.times.shape:
        raise ValueError("compact x must have the same shape as compact times.")
    if sample.mask.shape != sample.x.shape:
        raise ValueError("compact mask must have the same shape as compact x.")
    if sample.feature_lengths.shape != (sample.x.shape[-1],):
        raise ValueError("feature_lengths must have shape (n_features,).")
    if torch.any((sample.mask < 0) | (sample.mask > 1)):
        raise ValueError("compact mask values must be in [0, 1].")
    if torch.any(sample.feature_lengths < 0):
        raise ValueError("feature_lengths values must be non-negative.")
    if torch.any(sample.feature_lengths > sample.x.shape[0]):
        raise ValueError("feature_lengths cannot exceed compact sequence length.")
    positions = torch.arange(int(sample.x.shape[0]), device=sample.mask.device).unsqueeze(1)
    expected_mask = positions < sample.feature_lengths.to(sample.mask.device).unsqueeze(0)
    if not torch.equal(sample.mask > 0, expected_mask):
        raise ValueError("compact mask must contain left-aligned observations only.")
    if float(sample.survival_time) <= 0:
        raise ValueError("survival_time must be positive.")
    if float(sample.event) < 0 or float(sample.event) > 1:
        raise ValueError("event must be in [0, 1].")
    if sample.cluster_label is not None and sample.cluster_label.ndim > 0:
        raise ValueError("cluster_label must be a scalar tensor when provided.")


def compute_feature_means(samples: Sequence[DatasetSample]) -> Tensor:
    """根据样本掩码计算各特征观测值均值，无观测特征返回零。"""
    n_features = int(samples[0].x.shape[-1])
    numerator = torch.zeros(n_features, dtype=torch.float32)
    denominator = torch.zeros(n_features, dtype=torch.float32)
    for sample in samples:
        numerator += (sample.x * sample.mask).sum(dim=0).float()
        denominator += sample.mask.sum(dim=0).float()
    return numerator / denominator.clamp_min(1.0)


def clinical_collate_fn(samples: list[DatasetSample]) -> Batch:
    """将同一视图的变长患者样本整理为补零批次。

    根据首个样本选择 aligned 或 compact 批处理路径，并拒绝混合视图或混合
    标签状态。输出包含模型输入、生存结局和可选参考簇标签。

    参数：
        samples: 非空且视图类型一致的患者样本列表。

    返回：
        aligned 或 compact 张量批次字典。

    异常：
        ValueError: 当输入为空、混合样本视图或混合标签状态时抛出。
    """
    if not samples:
        raise ValueError("clinical_collate_fn requires at least one sample.")
    if isinstance(samples[0], CompactClinicalSample):
        if not all(isinstance(sample, CompactClinicalSample) for sample in samples):
            raise ValueError("clinical_collate_fn cannot mix aligned and compact samples.")
        return collate_compact_samples(cast(list[CompactClinicalSample], samples))
    if not all(isinstance(sample, AlignedClinicalSample) for sample in samples):
        raise ValueError("clinical_collate_fn cannot mix aligned and compact samples.")
    return collate_aligned_samples(cast(list[AlignedClinicalSample], samples))


def collate_aligned_samples(samples: list[AlignedClinicalSample]) -> AlignedBatch:
    """将 aligned 样本补齐到批次最大访视数并堆叠结局。

    返回批次包含 ``times``、``x``、``mask``、``delta_time``、
    ``sequence_lengths``、``survival_time``、``event`` 和可选
    ``cluster_label``。
    """
    batch_size = len(samples)
    max_length = max(int(sample.times.shape[0]) for sample in samples)
    n_features = int(samples[0].x.shape[-1])

    times = torch.zeros(batch_size, max_length, dtype=torch.float32)
    x = torch.zeros(batch_size, max_length, n_features, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_length, n_features, dtype=torch.float32)
    delta_time = torch.zeros(batch_size, max_length, n_features, dtype=torch.float32)
    sequence_lengths = torch.zeros(batch_size, dtype=torch.long)

    for row, sample in enumerate(samples):
        length = int(sample.times.shape[0])
        times[row, :length] = sample.times
        x[row, :length] = sample.x
        mask[row, :length] = sample.mask
        delta_time[row, :length] = sample.delta_time
        sequence_lengths[row] = length

    has_cluster_labels = samples[0].cluster_label is not None
    for sample in samples:
        if (sample.cluster_label is not None) != has_cluster_labels:
            raise ValueError("clinical_collate_fn cannot mix labeled and unlabeled samples.")

    batch = {
        "times": times,
        "x": x,
        "mask": mask,
        "delta_time": delta_time,
        "sequence_lengths": sequence_lengths,
        "survival_time": torch.stack([sample.survival_time for sample in samples]).float(),
        "event": torch.stack([sample.event for sample in samples]).float(),
    }
    if has_cluster_labels:
        cluster_labels = [
            sample.cluster_label for sample in samples if sample.cluster_label is not None
        ]
        batch["cluster_label"] = torch.stack(cluster_labels).long()
    return batch


def collate_compact_samples(samples: list[CompactClinicalSample]) -> CompactBatch:
    """将 compact 样本补齐到批次最大特征流长度并堆叠结局。

    返回批次包含 ``times``、``x``、``mask``、``feature_lengths``、
    ``survival_time``、``event`` 和可选 ``cluster_label``。
    """
    batch_size = len(samples)
    max_length = max(int(sample.x.shape[0]) for sample in samples)
    n_features = int(samples[0].x.shape[-1])

    times = torch.zeros(batch_size, max_length, n_features, dtype=torch.float32)
    x = torch.zeros(batch_size, max_length, n_features, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_length, n_features, dtype=torch.float32)
    feature_lengths = torch.zeros(batch_size, n_features, dtype=torch.long)

    for row, sample in enumerate(samples):
        length = int(sample.x.shape[0])
        times[row, :length] = sample.times
        x[row, :length] = sample.x
        mask[row, :length] = sample.mask
        feature_lengths[row] = sample.feature_lengths

    has_cluster_labels = samples[0].cluster_label is not None
    for sample in samples:
        if (sample.cluster_label is not None) != has_cluster_labels:
            raise ValueError("clinical_collate_fn cannot mix labeled and unlabeled samples.")

    batch = {
        "times": times,
        "x": x,
        "mask": mask,
        "feature_lengths": feature_lengths,
        "survival_time": torch.stack([sample.survival_time for sample in samples]).float(),
        "event": torch.stack([sample.event for sample in samples]).float(),
    }
    if has_cluster_labels:
        cluster_labels = [
            sample.cluster_label for sample in samples if sample.cluster_label is not None
        ]
        batch["cluster_label"] = torch.stack(cluster_labels).long()
    return batch


def infer_data_config(dataset: ClinicalTimeSeriesDataset) -> DataConfig:
    """根据数据集特征数构造模型数据配置。"""
    return DataConfig(n_features=dataset.n_features)


def make_data_loader(
    data: ClinicalTimeSeriesDataset,
    trainer_config: TrainerConfig,
    *,
    shuffle: bool,
) -> DataLoader[Batch]:
    """使用训练配置中的批大小规则创建临床数据加载器。

    参数：
        data: 要迭代的临床时间序列数据集。
        trainer_config: 提供显式或自动批大小的训练配置。
        shuffle: 是否在每轮迭代前打乱患者顺序。

    返回：
        使用 :func:`clinical_collate_fn` 的 PyTorch 数据加载器。
    """
    return cast(
        DataLoader[Batch],
        DataLoader(
            data,
            batch_size=resolve_batch_size(len(data), trainer_config.batch_size),
            shuffle=shuffle,
            collate_fn=clinical_collate_fn,
        ),
    )
