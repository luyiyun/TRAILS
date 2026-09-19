"""CRC Yunnan命令配置；默认值由Hydra YAML提供。"""

import math
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

BASELINE_LABELS = {
    "Preoperative_CEA": "术前CEA",
    "Preoperative_CA242": "术前CA242",
    "Age": "年龄",
    "Sex": "性别",
    "Primary_site": "原发部位",
    "Surgical_approach": "手术方式",
    "Tumor_differentiation": "肿瘤分化",
    "AJCC_8th_ed_Stage": "AJCC第8版分期",
    "Lymph_node_yield": "淋巴结检出数",
    "Mucinous_colloid_type": "黏液胶样类型",
    "Lymphovascular_invasion": "淋巴血管侵犯",
    "Perineural_invasion": "神经侵犯",
    "Adjuvant_chemotherapy": "辅助化疗",
}


class CohortPathsConfig(BaseModel):
    """将Hydra解析后的路径统一转换为相对启动目录的绝对路径。"""

    model_config = ConfigDict(extra="forbid")

    raw_root: Path
    patients_csv: Path
    observations_csv: Path
    derived_dir: Path
    run_dir: Path

    @field_validator("*")
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        return value.resolve()


CohortColumnName = Annotated[str, Field(min_length=1, pattern=r"\S")]


class CohortColumnsConfig(BaseModel):
    """源表列名统一为非空字符串，基线列名单不允许重复。"""

    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    export_index: CohortColumnName
    patient_id: CohortColumnName
    surgery_date: CohortColumnName
    last_followup_date: CohortColumnName
    survival_time: CohortColumnName
    death: CohortColumnName
    recurrence_free_time: CohortColumnName
    recurrence: CohortColumnName
    baseline: list[CohortColumnName]
    feature: CohortColumnName
    feature_group: CohortColumnName
    measurement_date: CohortColumnName
    value: CohortColumnName

    @field_validator("baseline")
    @classmethod
    def validate_baseline(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("columns.baseline不能包含重复列名")
        return value


class CohortProcessingConfig(BaseModel):
    """基础清洗选项及数值约束，在读取数据前完成校验。"""

    model_config = ConfigDict(extra="forbid")

    drop_export_index: bool
    research_id_seed: int = Field(ge=0)
    apply_date_corrections: bool
    same_day_aggregation: Literal["median", "mean"]
    exclude_invalid_surgery_date: bool
    exclude_negative_outcome_time: bool
    exclude_followup_before_surgery: bool
    max_outcome_days: FiniteFloat | None = Field(gt=0)
    outcome_time_unit_days: int = Field(gt=0)
    refuse_overwrite: bool


class CohortDateCorrectionsConfig(BaseModel):
    """已核实的日期修正映射；保留字符串供源数据直接替换。"""

    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    patients: dict[str, str]
    observations: dict[str, str]


class CRCCohortConfig(BaseModel):
    """01基础队列命令的完整配置，YAML继续提供所有默认值。"""

    model_config = ConfigDict(extra="forbid")

    dataset: Literal["crc_yunnan"]
    version: str
    paths: CohortPathsConfig
    columns: CohortColumnsConfig
    processing: CohortProcessingConfig
    date_corrections: CohortDateCorrectionsConfig


class EDAPathsConfig(BaseModel):
    """基础队列输入及EDA输出路径，统一相对启动目录解析。"""

    model_config = ConfigDict(extra="forbid")

    cohort_root: Path
    patients_csv: Path
    observations_csv: Path
    feature_metadata_csv: Path
    output_dir: Path
    run_dir: Path

    @field_validator("*")
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        return value.resolve()


class EDAColumnsConfig(BaseModel):
    """连续与分类基线变量分别配置，列名非空且不重复。"""

    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    numeric_baseline: list[Annotated[str, Field(min_length=1, pattern=r"\S")]]
    categorical_baseline: list[Annotated[str, Field(min_length=1, pattern=r"\S")]]

    @model_validator(mode="after")
    def validate_columns(self) -> Self:
        names = [*self.numeric_baseline, *self.categorical_baseline]
        if len(set(names)) != len(names):
            raise ValueError("基线列名不能重复，也不能同时配置为连续和分类变量")
        return self


class EDAQualityConfig(BaseModel):
    """EDA产物的覆盖策略。"""

    model_config = ConfigDict(extra="forbid")

    refuse_overwrite: bool


class EDATrajectoryConfig(BaseModel):
    """正整数时间窗口；landmark按升序解析且不允许重复。"""

    model_config = ConfigDict(extra="forbid")

    landmark_days: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)
    time_bin_days: int = Field(gt=0)
    annual_segment_days: int = Field(gt=0)

    @field_validator("landmark_days")
    @classmethod
    def normalize_landmarks(cls, values: list[int]) -> list[int]:
        if len(set(values)) != len(values):
            raise ValueError("trajectory.landmark_days不能重复")
        return sorted(values)


class EDAPlotConfig(BaseModel):
    """正整数绘图分辨率及按优先级排列的候选字体。"""

    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    dpi: int = Field(gt=0)
    font_families: list[Annotated[str, Field(min_length=1, pattern=r"\S")]] = Field(min_length=1)


class CRCEDAConfig(BaseModel):
    """02描述性分析配置，在读取数据前统一解析和验证。"""

    model_config = ConfigDict(extra="forbid")

    dataset: Literal["crc_yunnan"]
    version: str
    outcome: Literal["dfs", "os"]
    paths: EDAPathsConfig
    columns: EDAColumnsConfig
    quality: EDAQualityConfig
    trajectory: EDATrajectoryConfig
    plot: EDAPlotConfig


class SplitPathsConfig(BaseModel):
    """源表、共享主划分与冻结产物路径。"""

    model_config = ConfigDict(extra="forbid")

    cohort_root: Path
    patients_csv: Path
    observations_csv: Path
    assignment_root: Path
    dir: Path
    run_dir: Path


class RandomSplitConfig(BaseModel):
    """随机划分的三套比例。"""

    model_config = ConfigDict(extra="forbid")

    train_fraction: float = Field(gt=0, lt=1)
    validation_fraction: float = Field(gt=0, lt=1)
    test_fraction: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        """三套比例共同覆盖完整基础队列。"""
        if not math.isclose(self.train_fraction + self.validation_fraction + self.test_fraction, 1):
            raise ValueError("random三套比例之和须为1")
        return self


class TemporalSplitConfig(BaseModel):
    """按手术年份划分test，再从较早队列随机划分validation。"""

    model_config = ConfigDict(extra="forbid")

    test_start_year: int
    validation_fraction: float = Field(gt=0, lt=1)


class PatientSplitConfig(BaseModel):
    """本次需要生成的划分策略和种子。"""

    model_config = ConfigDict(extra="forbid")

    strategies: list[Literal["random", "temporal"]]
    seeds: list[Annotated[int, Field(ge=0, lt=2**32)]] = Field(min_length=1)
    random: RandomSplitConfig
    temporal: TemporalSplitConfig

    @model_validator(mode="after")
    def validate_unique_requests(self) -> Self:
        """重复请求会指向同一个冻结目录，应在执行前拒绝。"""
        if len(set(self.strategies)) != len(self.strategies):
            raise ValueError("split.strategies不能重复")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("split.seeds不能重复")
        return self


class FeaturePanelsConfig(BaseModel):
    """显式特征面板，不自动删减或扩充所选名单。"""

    model_config = ConfigDict(extra="forbid")

    panel: str
    panels: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_selected_panel(self) -> Self:
        """只要求实际使用的面板存在、非空且不含重复特征。"""
        if self.panel not in self.panels:
            raise ValueError("features.panel须为已配置的面板名称")
        features = self.panels[self.panel]
        if (
            not features
            or len(set(features)) != len(features)
            or any(not name.strip() for name in features)
        ):
            raise ValueError("面板的特征名单须非空且不重复")
        if {"patient_id", "time_days"}.intersection(features):
            raise ValueError("特征面板不能包含patient_id或time_days观测键")
        return self


class PreprocessingConfig(BaseModel):
    """训练集拟合的log及缩放步骤。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    log_transform: Literal["none", "auto-log1p"]
    skew_threshold: FiniteFloat
    scaling: Literal["none", "robust", "standard", "minmax"]


class CRCSplitConfig(BaseModel):
    """Hydra解析后的CRC划分配置，在读取数据前完成校验。"""

    model_config = ConfigDict(extra="forbid")

    dataset: Literal["crc_yunnan"]
    version: str
    split_version: str
    paths: SplitPathsConfig
    split: PatientSplitConfig
    save_full_dataset: bool


class PreprocInputsConfig(BaseModel):
    """训练目录负责拟合，其他目录仅应用训练参数。"""

    model_config = ConfigDict(extra="forbid")

    split_dir: Path | None = None
    train_dir: Path
    test_dirs: list[Path] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_directories(self) -> Self:
        paths = [self.train_dir.resolve(), *(path.resolve() for path in self.test_dirs)]
        names = ["train", *(path.name for path in self.test_dirs)]
        if len(set(paths)) != len(paths) or len(set(names)) != len(names):
            raise ValueError("输入目录或输出数据集名称不能重复，train为保留名称")
        if any(name in {"", ".", ".."} for name in names):
            raise ValueError("输入目录须有明确的数据集名称")
        if set(names).intersection(
            {"preprocessing_parameters.csv", "resolved_config.yaml", "preproc_manifest.json"}
        ):
            raise ValueError("数据集名称不能与预处理产物文件名冲突")
        return self


class PreprocPathsConfig(BaseModel):
    """变换产物及Hydra日志目录。"""

    model_config = ConfigDict(extra="forbid")

    dir: Path
    run_dir: Path


class CRCPreprocConfig(BaseModel):
    """分析窗口、固定面板和训练集拟合的预处理配置。"""

    model_config = ConfigDict(extra="forbid")

    dataset: Literal["crc_yunnan"]
    version: str
    outcome: Literal["dfs", "os"]
    landmark_months: FiniteFloat = Field(gt=0)
    min_observation_dates: int = Field(ge=2)
    inputs: PreprocInputsConfig
    preprocessing: PreprocessingConfig
    paths: PreprocPathsConfig
    features: FeaturePanelsConfig


class EvaluationInputsConfig(BaseModel):
    """05建模目录；沿清单追溯04和01的产物。"""

    model_config = ConfigDict(extra="forbid")
    run_dir: Path


class EvaluationUMAPConfig(BaseModel):
    """仅在训练潜表示上拟合的二维UMAP。"""

    model_config = ConfigDict(extra="forbid")
    n_neighbors: int = Field(ge=2)
    min_dist: FiniteFloat = Field(ge=0, le=1)
    metric: Literal["euclidean"]
    seed: int = Field(ge=0, lt=2**32)


class EvaluationSurvivalConfig(BaseModel):
    """所有时间均从landmark起算，每月30天。"""

    model_config = ConfigDict(extra="forbid")
    months: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)
    curve_step_months: int = Field(gt=0)
    calibration_bins: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        if self.months != sorted(set(self.months)):
            raise ValueError("survival.months须严格递增且不重复")
        if any(month % self.curve_step_months for month in self.months):
            raise ValueError("评价月份须落在曲线网格上")
        if self.months[-1] // self.curve_step_months < 2:
            raise ValueError("IBS至少需要两个曲线时间点")
        return self


class CRCEvaluationConfig(BaseModel):
    """06的统计和绘图选项；结局、landmark及面板沿用05。"""

    model_config = ConfigDict(extra="forbid")
    inputs: EvaluationInputsConfig
    datasets: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    columns: EDAColumnsConfig
    umap: EvaluationUMAPConfig
    survival: EvaluationSurvivalConfig
    trajectory_bin_days: int = Field(gt=0)
    trajectory_panels_per_page: int = Field(ge=1, le=24)
    fisher_resamples: int = Field(ge=100)
    seed: int = Field(ge=0, lt=2**32)
    device: str = Field(min_length=1)
    plot: EDAPlotConfig
    paths: PreprocPathsConfig

    @model_validator(mode="after")
    def validate_requests(self) -> Self:
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("datasets不能重复")
        output, logs = self.paths.dir.resolve(), self.paths.run_dir.resolve()
        source = self.inputs.run_dir.resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("评价输出须独立于05建模目录")
        if logs == output or output in logs.parents:
            raise ValueError("Hydra日志目录须独立于评价输出目录")
        return self
