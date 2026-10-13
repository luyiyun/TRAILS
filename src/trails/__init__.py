"""用于配置、加载和拟合 TRAILS 模型的公共 API。

大多数应用只需使用本模块重新导出的对象。更底层的模型、训练器、指标和产物
工具仍可从各自的 :mod:`trails` 子模块中导入。
"""

from .config import (
    ClusterNumberEvaluatorConfig,
    DataConfig,
    DecoderConfig,
    EncoderConfig,
    EncoderInputConfig,
    EncoderMappingConfig,
    ModelConfig,
    TrailsConfig,
    TrainerConfig,
    resolve_batch_size,
)
from .data import (
    AlignedClinicalSample,
    ClinicalTimeSeriesDataset,
    CompactClinicalSample,
    DatasetSample,
    clinical_collate_fn,
)
from .estimator import TrailsEstimator
from .prediction import TrailsPrediction
from .selection import ClusterNumberEvaluationResult, ClusterNumberEvaluator

__all__ = [
    "AlignedClinicalSample",
    "CompactClinicalSample",
    "ClinicalTimeSeriesDataset",
    "ClusterNumberEvaluationResult",
    "ClusterNumberEvaluator",
    "ClusterNumberEvaluatorConfig",
    "DataConfig",
    "DecoderConfig",
    "DatasetSample",
    "EncoderConfig",
    "EncoderInputConfig",
    "EncoderMappingConfig",
    "ModelConfig",
    "TrailsConfig",
    "TrainerConfig",
    "TrailsEstimator",
    "TrailsPrediction",
    "clinical_collate_fn",
    "resolve_batch_size",
]
