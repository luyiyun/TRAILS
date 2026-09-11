from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from trails import ClinicalTimeSeriesDataset

LOGGER = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖manifest：{path}")
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def _check_ids(splits: dict[str, pd.DataFrame], expected: set[str]) -> None:
    combined = cast(
        pd.DataFrame,
        pd.concat([frame[["patient_id"]] for frame in splits.values()], ignore_index=True),
    )
    if any(frame.empty for frame in splits.values()):
        raise ValueError("train、validation、test均须非空")
    if (
        combined["patient_id"].isna().to_numpy().any()
        or combined["patient_id"].duplicated().to_numpy().any()
    ):
        raise ValueError("三套划分的患者ID必须完整且互斥")
    if set(combined["patient_id"]) != expected:
        raise ValueError("三套划分的患者ID并集与目标队列不一致")


def _feature_parameters(observations: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    grouped = observations.groupby("feature", observed=True)["value"]
    parameters = cast(
        pd.DataFrame,
        grouped.agg(
            n_observations="size", center="median", standard_deviation=lambda s: s.std(ddof=0)
        ).reindex(feature_order),
    )
    quartiles = grouped.quantile(np.array([0.25, 0.75])).unstack().reindex(feature_order)
    parameters["iqr"] = quartiles[0.75] - quartiles[0.25]
    parameters["scale"] = parameters["iqr"].where(
        parameters["iqr"] > 0, parameters["standard_deviation"]
    )
    parameters["scale_method"] = np.where(parameters["iqr"] > 0, "iqr", "standard_deviation")
    parameters["n_observations"] = parameters["n_observations"].fillna(0).astype(int)
    parameters["drop_reason"] = np.select(
        [parameters["n_observations"] == 0, parameters["standard_deviation"] == 0],
        ["no_train_observations", "constant_in_train"],
        default="",
    )
    retained = parameters.loc[parameters["drop_reason"] == ""]
    if not np.isfinite(retained[["center", "scale"]].to_numpy()).all():
        raise ValueError("训练集特征处理参数出现非有限数值")
    return parameters.rename_axis("feature").reset_index()


@hydra.main(config_path="../../configs", config_name="crc_yunnan/split", version_base="1.3")
def main(config: DictConfig) -> None:
    # ==================================================================================
    # 一、解析配置、检查全部目标目录，并读取预处理后的患者与纵向观测。
    # ==================================================================================
    outcome = str(config.outcome)
    landmark_days = float(config.landmark_months) * 30.0
    minimum_dates = int(config.cohort.min_observation_dates)
    maximum_days = (
        None if config.cohort.max_outcome_days is None else float(config.cohort.max_outcome_days)
    )
    exclude_reversed_dates = bool(config.cohort.exclude_followup_before_surgery)
    panel = str(config.features.panel)
    strategies = [str(value) for value in config.split.strategies]
    seeds = [int(value) for value in config.split.seeds]
    fractions = {name: float(config.split.random[f"{name}_fraction"]) for name in SPLIT_NAMES}
    temporal_year = int(config.split.temporal.test_start_year)
    temporal_validation = float(config.split.temporal.validation_fraction)
    if outcome not in {"dfs", "os"} or not math.isfinite(landmark_days) or landmark_days <= 0:
        raise ValueError("outcome须为dfs/os，landmark_months须为有限正数")
    if minimum_dates < 2 or minimum_dates != config.cohort.min_observation_dates:
        raise ValueError("cohort.min_observation_dates须为至少2的整数")
    if maximum_days is not None and (not math.isfinite(maximum_days) or maximum_days <= 0):
        raise ValueError("cohort.max_outcome_days须为有限正数或null")
    if not strategies or len(set(strategies)) != len(strategies):
        raise ValueError("split.strategies须非空且不重复")
    if set(strategies) - {"random", "temporal"}:
        raise ValueError("split.strategies仅支持random和temporal")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("split.seeds须非空且不重复")
    if any(
        seed != raw or not 0 <= seed < 2**32
        for seed, raw in zip(seeds, config.split.seeds, strict=True)
    ):
        raise ValueError("seed须为0至2**32-1的整数")
    if any(not 0 < value < 1 for value in fractions.values()) or not math.isclose(
        sum(fractions.values()), 1.0
    ):
        raise ValueError("random三套比例须在0与1之间且和为1")
    if not 0 < temporal_validation < 1 or temporal_year != config.split.temporal.test_start_year:
        raise ValueError("temporal验证比例须在0与1之间，测试起始年份须为整数")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", panel) or panel not in config.features.panels:
        raise ValueError("features.panel须为已配置的面板名称")
    feature_order = [str(value) for value in config.features.panels[panel]]
    if not feature_order or len(set(feature_order)) != len(feature_order):
        raise ValueError("面板的特征名单须非空且不重复")

    patient_path = Path(str(config.paths.patients_csv)).resolve()
    observation_path = Path(str(config.paths.observations_csv)).resolve()
    assignment_root = Path(str(config.paths.assignment_root)).resolve()
    output_root = Path(str(config.paths.dir)).resolve()
    aggregate_root = Path(str(config.paths.output_dir)).resolve()
    strategy_labels = {
        strategy: strategy if strategy == "random" else f"temporal-{temporal_year}"
        for strategy in strategies
    }
    bundle_paths = {
        (strategy, seed): output_root / strategy_labels[strategy] / f"seed-{seed}"
        for strategy in strategies
        for seed in seeds
    }
    targets = [*bundle_paths.values(), aggregate_root]
    if len(set(targets)) != len(targets) or any(
        left in right.parents for left in targets for right in targets if left != right
    ):
        raise ValueError("冻结包和聚合目录不得重复或相互嵌套")
    if existing := [str(path) for path in targets if path.exists()]:
        raise FileExistsError(f"拒绝覆盖既有冻结产物：{existing}")
    if missing := [str(path) for path in (patient_path, observation_path) if not path.is_file()]:
        raise FileNotFoundError(f"缺少CRC云南预处理输入：{missing}")

    patients = pd.read_csv(
        patient_path,
        dtype={
            "patient_id": "string",
            "dfs_time_days": "float64",
            "dfs_event": "float64",
            "os_time_days": "float64",
            "os_event": "float64",
        },
    )
    observations = pd.read_csv(
        observation_path,
        dtype={
            "patient_id": "string",
            "feature": "string",
            "feature_group": "string",
            "time_days": "float64",
            "value": "float64",
        },
    )

    # ==================================================================================
    # 二、校验数据契约，按已确认规则准备与窗口、面板、结局选择无关的基础队列。
    # ==================================================================================
    required_patients = {
        "patient_id",
        "surgery_date",
        "last_followup_date",
        "surgery_year",
        "dfs_time_days",
        "dfs_event",
        "os_time_days",
        "os_event",
        "recurrence",
    }
    required_observations = {"patient_id", "feature", "feature_group", "time_days", "value"}
    if required_patients - set(patients.columns) or required_observations - set(
        observations.columns
    ):
        raise ValueError("预处理输入缺少必需字段")
    if patients.empty or observations.empty:
        raise ValueError("预处理输入不能为空")
    if (
        patients[list(required_patients)].isna().to_numpy().any()
        or observations.isna().to_numpy().any()
    ):
        raise ValueError("患者关键字段或纵向观测字段存在缺失")
    for frame, columns in (
        (patients, ["patient_id"]),
        (observations, ["patient_id", "feature", "feature_group"]),
    ):
        if any(
            cast(pd.Series, frame[name]).str.strip().eq("").to_numpy().any() for name in columns
        ):
            raise ValueError("患者ID或特征标识存在空字符串")
    if (
        patients["patient_id"].duplicated().to_numpy().any()
        or observations.duplicated(["patient_id", "time_days", "feature"]).to_numpy().any()
    ):
        raise ValueError("患者ID或患者-时间-特征复合键重复")
    if not observations["patient_id"].isin(patients["patient_id"]).to_numpy().all():
        raise ValueError("纵向表包含无法关联患者表的ID")
    if missing_features := sorted(set(feature_order) - set(observations["feature"])):
        raise ValueError(f"面板特征未出现在源表：{missing_features}")
    numeric_patient_columns = [
        "dfs_time_days",
        "os_time_days",
        "dfs_event",
        "os_event",
        "recurrence",
        "surgery_year",
    ]
    if not np.isfinite(patients[numeric_patient_columns].to_numpy(dtype=float)).all():
        raise ValueError("患者关键数值包含非有限值")
    if not np.isfinite(observations[["time_days", "value"]].to_numpy()).all():
        raise ValueError("纵向时间或数值包含非有限值")
    if not patients[["dfs_event", "os_event", "recurrence"]].isin([0, 1]).to_numpy().all():
        raise ValueError("事件编码须为0或1")
    if (patients[["dfs_time_days", "os_time_days"]] < 0).any().any():
        raise ValueError("结局时间不能为负")
    if (patients["dfs_time_days"] > patients["os_time_days"]).any() or (
        patients["dfs_event"] != (patients["recurrence"].eq(1) | patients["os_event"].eq(1))
    ).any():
        raise ValueError("DFS与OS的时间或事件关系不一致")
    for name in ("surgery_date", "last_followup_date"):
        patients[name] = pd.to_datetime(patients[name], errors="raise")
        if patients[name].isna().to_numpy().any():
            raise ValueError("关键日期存在缺失")
    if (patients["surgery_year"] != patients["surgery_date"].dt.year).any():
        raise ValueError("surgery_year与手术日期不一致")
    patients["calendar_followup_days"] = (
        patients["last_followup_date"] - patients["surgery_date"]
    ).dt.days
    patients["os_calendar_difference_days"] = (
        patients["os_time_days"] - patients["calendar_followup_days"]
    )
    reversed_dates = patients["calendar_followup_days"] < 0
    excessive_times = cast(
        pd.Series,
        (
            patients[["dfs_time_days", "os_time_days"]].gt(maximum_days).any(axis=1)
            if maximum_days is not None
            else pd.Series(False, index=patients.index)
        ),
    )
    excluded = (reversed_dates & exclude_reversed_dates) | excessive_times
    base = patients.loc[~excluded].sort_values("patient_id").copy()
    if base.empty:
        raise ValueError("基础纳排后没有患者")
    source_sha256 = {
        "patients.csv": _sha256(patient_path),
        "observations.csv": _sha256(observation_path),
    }
    quality_summary = {
        "source_patients": len(patients),
        "source_observations": len(observations),
        "followup_before_surgery": int(reversed_dates.sum()),
        "above_configured_maximum": int(excessive_times.sum()),
        "excluded_patients": int(excluded.sum()),
        "base_patients": len(base),
        "os_calendar_absolute_difference_gt_60_days": int(
            patients["os_calendar_difference_days"].abs().gt(60).to_numpy().sum()
        ),
    }
    LOGGER.info("基础队列：%s", quality_summary)

    # ==================================================================================
    # 三、冻结患者主划分；后续修改面板、landmark或结局仍复用相同的患者归属。
    # ==================================================================================
    master_splits: dict[tuple[str, int], dict[str, pd.DataFrame]] = {}
    master_manifests: dict[tuple[str, int], Path] = {}
    pending_masters: list[tuple[Path, dict[str, pd.DataFrame], dict[str, Any]]] = []
    for strategy in strategies:
        for seed in seeds:
            master_dir = assignment_root / strategy_labels[strategy] / f"seed-{seed}"
            if any(
                master_dir == path or master_dir in path.parents or path in master_dir.parents
                for path in targets
            ):
                raise ValueError("主划分目录与派生输出目录必须独立")
            manifest_path = master_dir / "assignment_manifest.json"
            contract = {
                "schema_version": 1,
                "dataset": "crc_yunnan",
                "source_sha256": source_sha256,
                "cohort": {
                    "exclude_followup_before_surgery": exclude_reversed_dates,
                    "max_outcome_days": maximum_days,
                },
                "strategy": strategy,
                "seed": seed,
                "stratification": ["dfs_event", "os_event"],
                "random": fractions if strategy == "random" else None,
                "temporal": {
                    "test_start_year": temporal_year,
                    "validation_fraction": temporal_validation,
                }
                if strategy == "temporal"
                else None,
            }
            if master_dir.exists():
                if not manifest_path.is_file():
                    raise ValueError(f"主划分目录不完整，请使用新版本：{master_dir}")
                saved = json.loads(manifest_path.read_text(encoding="utf-8"))
                if saved["contract"] != contract:
                    raise ValueError(f"主划分源数据或规则不匹配，请使用新版本：{master_dir}")
                splits = {}
                for name in SPLIT_NAMES:
                    path = master_dir / f"{name}_ids.csv"
                    if _sha256(path) != saved["artifact_sha256"][path.name]:
                        raise ValueError(f"主划分文件哈希不匹配：{path}")
                    ids = pd.read_csv(path, dtype={"patient_id": "string"})
                    if ids.columns.tolist() != ["patient_id"]:
                        raise ValueError("主划分ID文件只能包含patient_id列")
                    splits[name] = ids.merge(
                        base, on="patient_id", how="inner", validate="one_to_one"
                    )
                    if len(splits[name]) != len(ids):
                        raise ValueError("主划分含基础队列外的患者")
            else:
                if strategy == "random":
                    first_split = train_test_split(
                        base,
                        test_size=fractions["test"],
                        random_state=seed,
                        stratify=base["dfs_event"].astype(str) + ":" + base["os_event"].astype(str),
                    )
                    development = cast(pd.DataFrame, first_split[0])
                    test = cast(pd.DataFrame, first_split[1])
                    validation_fraction = fractions["validation"] / (
                        fractions["train"] + fractions["validation"]
                    )
                else:
                    development = base.loc[base["surgery_year"] < temporal_year]
                    test = base.loc[base["surgery_year"] >= temporal_year]
                    validation_fraction = temporal_validation
                second_split = train_test_split(
                    development,
                    test_size=validation_fraction,
                    random_state=seed,
                    stratify=development["dfs_event"].astype(str)
                    + ":"
                    + development["os_event"].astype(str),
                )
                splits = {
                    "train": cast(pd.DataFrame, second_split[0]),
                    "validation": cast(pd.DataFrame, second_split[1]),
                    "test": test,
                }
                pending_masters.append((master_dir, splits, contract))
            _check_ids(splits, set(base["patient_id"]))
            if strategy == "temporal" and set(splits["test"]["patient_id"]) != set(
                base.loc[base["surgery_year"] >= temporal_year, "patient_id"]
            ):
                raise ValueError("temporal测试ID不符合手术年份切点")
            master_splits[strategy, seed] = splits
            master_manifests[strategy, seed] = manifest_path

    # 全部配置、目标和既有主划分检查完成后，才创建新目录。
    for master_dir, splits, contract in pending_masters:
        master_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        for name, frame in splits.items():
            path = master_dir / f"{name}_ids.csv"
            frame.loc[:, ["patient_id"]].sort_values("patient_id").to_csv(path, index=False)
            path.chmod(0o600)
        _write_json(
            master_dir / "assignment_manifest.json",
            {
                "contract": contract,
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "split_counts": {name: len(frame) for name, frame in splits.items()},
                "artifact_sha256": {
                    f"{name}_ids.csv": _sha256(master_dir / f"{name}_ids.csv")
                    for name in SPLIT_NAMES
                },
            },
        )
    aggregate_root.mkdir(parents=True, exist_ok=False)
    split_rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []
    feature_rows: list[pd.DataFrame] = []

    # ==================================================================================
    # 四、按所选结局、landmark和面板构造分析队列，保持主划分归属。
    # ==================================================================================
    landmark_patients = base.loc[base[f"{outcome}_time_days"] > landmark_days].copy()
    landmark_patients["survival_time"] = landmark_patients[f"{outcome}_time_days"] - landmark_days
    landmark_patients["event"] = landmark_patients[f"{outcome}_event"]
    window_observations = observations.loc[
        observations["time_days"].between(0, landmark_days)
        & observations["feature"].isin(feature_order)
        & observations["patient_id"].isin(landmark_patients["patient_id"])
    ].copy()
    date_counts = window_observations.groupby("patient_id")["time_days"].nunique()
    initial_cohort = landmark_patients.loc[
        landmark_patients["patient_id"].isin(date_counts.index[date_counts >= minimum_dates])
    ].copy()
    if initial_cohort.empty:
        raise ValueError("landmark与面板观测日期筛选后没有患者")

    for (strategy, seed), master in master_splits.items():
        assignments = pd.concat(
            [frame[["patient_id"]].assign(split=name) for name, frame in master.items()],
            ignore_index=True,
        )
        cohort = initial_cohort.merge(assignments, on="patient_id", validate="one_to_one")
        active_features = feature_order.copy()
        dropped: list[dict[str, str]] = []

        # ==============================================================================
        # 五、只用train识别退化特征并拟合尺度，validation/test不参与参数决定。
        # ==============================================================================
        # 特征剔除可能使部分患者不足两个日期；单调收缩到稳定状态，避免最终train再次退化。
        while True:
            train_ids = cohort.loc[cohort["split"] == "train", "patient_id"]
            if train_ids.empty:
                raise ValueError("观测筛选后train为空")
            train_observations = window_observations.loc[
                window_observations["patient_id"].isin(train_ids)
                & window_observations["feature"].isin(active_features)
            ]
            parameters = _feature_parameters(train_observations, active_features)
            dropped_now = parameters.loc[
                parameters["drop_reason"] != "", ["feature", "drop_reason"]
            ]
            dropped.extend(dropped_now.to_dict(orient="records"))
            active_features = parameters.loc[parameters["drop_reason"] == "", "feature"].tolist()
            if not active_features:
                raise ValueError("所选面板在train中没有可用的非恒定特征")
            counts = (
                window_observations.loc[window_observations["feature"].isin(active_features)]
                .groupby("patient_id")["time_days"]
                .nunique()
            )
            filtered = cohort.loc[
                cohort["patient_id"].isin(counts.index[counts >= minimum_dates])
            ].copy()
            stable = dropped_now.empty and len(filtered) == len(cohort)
            cohort = filtered
            if stable:
                break
        splits = {
            name: cohort.loc[cohort["split"] == name].sort_values("patient_id").copy()
            for name in SPLIT_NAMES
        }
        _check_ids(splits, set(cohort["patient_id"]))
        parameters = parameters.drop(columns="drop_reason")
        bundle = bundle_paths[strategy, seed]
        bundle.mkdir(parents=True, exist_ok=False, mode=0o700)
        parameter_path = bundle / "preprocessing_parameters.csv"
        parameters.to_csv(parameter_path, index=False)
        OmegaConf.save(config, bundle / "resolved_config.yaml", resolve=True)
        manifest_splits: dict[str, dict[str, int]] = {}
        for name, frame in splits.items():
            flow_rows.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "split": name,
                    "base_patients": len(master[name]),
                    "landmark_patients": int(
                        master[name][f"{outcome}_time_days"].gt(landmark_days).to_numpy().sum()
                    ),
                    "initial_observation_patients": int(
                        assignments.loc[assignments["split"] == name, "patient_id"]
                        .isin(initial_cohort["patient_id"])
                        .sum()
                    ),
                    "final_patients": len(frame),
                }
            )

            # ==========================================================================
            # 六、应用已冻结尺度并调用公开API构造张量；评价信息与纵向输入分开保存。
            # ==========================================================================
            split_dir = bundle / name
            split_dir.mkdir(mode=0o700)
            frame[["patient_id"]].to_csv(bundle / f"{name}_ids.csv", index=False)
            patient_output = split_dir / "patients.csv"
            frame.drop(columns="split").to_csv(patient_output, index=False)
            panel_observations = window_observations.loc[
                window_observations["patient_id"].isin(frame["patient_id"])
            ]
            selected_observations = panel_observations.loc[
                panel_observations["feature"].isin(active_features)
            ].copy()
            coverage = (
                panel_observations.groupby("feature", observed=True)
                .agg(n_patients=("patient_id", "nunique"), n_observations=("value", "size"))
                .reindex(feature_order)
                .fillna(0)
                .astype(int)
                .rename_axis("feature")
                .reset_index()
            )
            coverage["patient_fraction"] = coverage["n_patients"] / len(frame)
            coverage["retained"] = coverage["feature"].isin(active_features)
            coverage["drop_reason"] = (
                coverage["feature"]
                .map({row["feature"]: row["drop_reason"] for row in dropped})
                .fillna("")
            )
            feature_rows.append(coverage.assign(strategy=strategy, seed=seed, split=name))
            transformed = selected_observations.merge(
                parameters[["feature", "center", "scale"]], on="feature", validate="many_to_one"
            )
            transformed["value"] = (transformed["value"] - transformed["center"]) / transformed[
                "scale"
            ]
            if not np.isfinite(transformed["value"].to_numpy(dtype=np.float32)).all():
                raise ValueError("标准化数值无法表示为有限float32张量")
            baseline_columns = [
                name
                for name in patients.columns
                if name not in required_patients
                and name not in {"calendar_followup_days", "os_calendar_difference_days"}
            ]
            baseline = frame[["patient_id", *baseline_columns]].astype(object)
            baseline = baseline.where(pd.notna(baseline), None)
            with TemporaryDirectory(prefix="tensor-input-", dir=bundle) as temporary:
                observation_output = Path(temporary) / "observations.csv"
                transformed[["patient_id", "time_days", "feature", "value"]].to_csv(
                    observation_output, index=False
                )
                dataset = ClinicalTimeSeriesDataset.load_from_csv(
                    patients_csv=patient_output,
                    observations_csv=observation_output,
                    time_col="time_days",
                    use_features=active_features,
                    description=(
                        f"CRC Yunnan {outcome.upper()} landmark {landmark_days:g} days ({name})"
                    ),
                    metadata={
                        "source": "crc_yunnan_frozen_split",
                        "split_name": name,
                        "split_seed": seed,
                        "split_strategy": strategy,
                        "outcome": outcome,
                        "landmark_days": landmark_days,
                        "observation_time_unit": "days_since_surgery",
                        "survival_time_unit": "days_since_landmark",
                        "observations_csv": str(observation_path),
                        "preprocessing_parameters_csv": str(parameter_path),
                        "split_manifest": str(bundle / "split_manifest.json"),
                        "baseline_covariates": baseline.to_dict(orient="records"),
                    },
                )
            if dataset.metadata["patient_ids"] != frame["patient_id"].tolist():
                raise RuntimeError("张量患者顺序与ID及评价表不一致")
            dataset.save(split_dir / "dataset.pt")
            manifest_splits[name] = {"patients": len(frame), "events": int(frame["event"].sum())}
            split_rows.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "split": name,
                    "outcome": outcome,
                    "landmark_days": landmark_days,
                    "panel": panel,
                    "n_features": len(active_features),
                    "n_patients": len(frame),
                    "n_events": int(frame["event"].sum()),
                    "realized_fraction": len(frame) / len(cohort),
                    "remaining_time_median_days": float(frame["survival_time"].median()),
                    "surgery_year_min": int(frame["surgery_year"].min()),
                    "surgery_year_max": int(frame["surgery_year"].max()),
                }
            )

        # ==============================================================================
        # 七、直接构造manifest并最后原子写入；聚合目录不保存任何患者级数据。
        # ==============================================================================
        artifacts = sorted(path for path in bundle.rglob("*") if path.is_file())
        for path in artifacts:
            path.chmod(0o600)
        manifest = {
            "schema_version": 1,
            "dataset": "crc_yunnan",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "input_paths": {
                "patients_csv": str(patient_path),
                "observations_csv": str(observation_path),
            },
            "source_sha256": source_sha256,
            "assignment_manifest": str(master_manifests[strategy, seed]),
            "assignment_manifest_sha256": _sha256(master_manifests[strategy, seed]),
            "strategy": strategy,
            "seed": seed,
            "outcome": outcome,
            "landmark_months": float(config.landmark_months),
            "landmark_days": landmark_days,
            "observation_interval": "0 <= time_days <= landmark_days",
            "eligibility_rule": f"{outcome}_time_days > landmark_days",
            "survival_time_rule": f"{outcome}_time_days - landmark_days",
            "cohort": OmegaConf.to_container(config.cohort, resolve=True),
            "panel": panel,
            "requested_features": feature_order,
            "feature_order": active_features,
            "dropped_features": dropped,
            "split_counts": manifest_splits,
            "preprocessing": {
                "fit_split": "train",
                "center": "median",
                "scale": "IQR with population-SD fallback",
                "clipping": False,
                "log_transform": False,
                "missing_values": "zero with mask",
            },
            "quality_summary": quality_summary,
            "checks": {
                "ids_disjoint_complete": True,
                "tensor_patient_order_matches": True,
                "master_assignment_preserved": True,
                "train_only_preprocessing": True,
            },
            "artifact_sha256": {str(path.relative_to(bundle)): _sha256(path) for path in artifacts},
        }
        _write_json(bundle / "split_manifest.json", manifest)
        _write_json(
            aggregate_root / f"{strategy_labels[strategy]}-seed-{seed}-manifest.json", manifest
        )
        LOGGER.info(
            "已冻结 %s seed=%s：%s，特征数=%s",
            strategy,
            seed,
            manifest_splits,
            len(active_features),
        )

    pd.DataFrame(flow_rows).to_csv(aggregate_root / "cohort_flow.csv", index=False)
    pd.DataFrame(split_rows).to_csv(aggregate_root / "split_summary.csv", index=False)
    pd.concat(feature_rows, ignore_index=True).to_csv(
        aggregate_root / "feature_summary.csv", index=False
    )
    _write_json(
        aggregate_root / "aggregate_audit.json",
        {
            "schema_version": 1,
            "quality_summary": quality_summary,
            "n_bundles": len(bundle_paths),
            "artifact_sha256": {
                path.name: _sha256(path)
                for path in sorted(aggregate_root.iterdir())
                if path.is_file()
            },
            "patient_level_outputs_retrievable": False,
        },
    )
    LOGGER.info("全部冻结完成，聚合结果：%s", aggregate_root)


if __name__ == "__main__":
    main()
