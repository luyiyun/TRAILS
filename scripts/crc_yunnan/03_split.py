from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scripts.crc_yunnan.config import CRCSplitConfig
from scripts.crc_yunnan.data import read_tables, write_json
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "validation", "test")


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


def _master_split(
    config: CRCSplitConfig,
    base: pd.DataFrame,
    bundle_paths: dict[tuple[str, int], Path],
) -> dict[tuple[str, int], dict[str, pd.DataFrame]]:
    """创建或复用患者ID，核对划分规则和基础队列，不审计文件哈希。"""
    assignment_root = Path(str(config.paths.assignment_root)).resolve()
    master_splits: dict[tuple[str, int], dict[str, pd.DataFrame]] = {}
    for (strategy, seed), bundle in bundle_paths.items():
        master_dir = assignment_root / bundle.parent.name / bundle.name
        manifest_path = master_dir / "assignment_manifest.json"
        # 只比较影响主划分的规则；旧manifest的哈希等额外字段不参与复用判断。
        rules = {
            "strategy": strategy,
            "seed": seed,
            "stratification": ["dfs_event", "os_event"],
            "random": {
                name: float(getattr(config.split.random, f"{name}_fraction"))
                for name in SPLIT_NAMES
            }
            if strategy == "random"
            else None,
            "temporal": config.split.temporal.model_dump(mode="json")
            if strategy == "temporal"
            else None,
        }
        reused = master_dir.exists()
        if reused:
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))["contract"]
            if any(saved[key] != value for key, value in rules.items()):
                raise ValueError(f"主划分规则不匹配，请使用新版本：{master_dir}")
            id_frames = {
                name: pd.read_csv(master_dir / f"{name}_ids.csv", dtype={"patient_id": "string"})
                for name in SPLIT_NAMES
            }
            _check_ids(id_frames, set(base["patient_id"]))
            splits = {
                name: ids[["patient_id"]].merge(base, on="patient_id", validate="one_to_one")
                for name, ids in id_frames.items()
            }
        else:
            if strategy == "random":
                development, test = train_test_split(
                    base,
                    test_size=float(config.split.random.test_fraction),
                    random_state=seed,
                    stratify=base["dfs_event"].astype(str) + ":" + base["os_event"].astype(str),  # type: ignore
                )
                development, test = cast(pd.DataFrame, development), cast(pd.DataFrame, test)
                validation_fraction = float(config.split.random.validation_fraction) / (
                    float(config.split.random.train_fraction)
                    + float(config.split.random.validation_fraction)
                )
            else:
                cutoff = int(config.split.temporal.test_start_year)
                development, test = (
                    base.loc[base["surgery_year"] < cutoff],
                    base.loc[base["surgery_year"] >= cutoff],
                )
                validation_fraction = float(config.split.temporal.validation_fraction)
            train, validation = train_test_split(
                development,
                test_size=validation_fraction,
                random_state=seed,
                stratify=development["dfs_event"].astype(str)  # type: ignore
                + ":"
                + development["os_event"].astype(str),
            )
            splits = {
                "train": cast(pd.DataFrame, train),
                "validation": cast(pd.DataFrame, validation),
                "test": test,
            }
            _check_ids(splits, set(base["patient_id"]))
        if strategy == "temporal" and set(splits["test"]["patient_id"]) != set(
            base.loc[
                base["surgery_year"] >= int(config.split.temporal.test_start_year), "patient_id"
            ]
        ):
            raise ValueError("temporal测试ID不符合手术年份切点")
        if not reused:
            master_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
            for name, frame in splits.items():
                path = master_dir / f"{name}_ids.csv"
                frame[["patient_id"]].sort_values("patient_id").to_csv(path, index=False)
                path.chmod(0o600)
            write_json(manifest_path, {"contract": rules})
        master_splits[strategy, seed] = splits
    return master_splits


def _save_split(
    directory: Path,
    name: str,
    patients: pd.DataFrame,
    observations: pd.DataFrame,
    cohort_root: Path,
    assignment_manifest: Path | None,
) -> None:
    """保存一套原始尺度中格式数据，full与普通划分使用相同接口。"""
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    patients = patients.sort_values("patient_id")
    patients.to_csv(directory / "patients.csv", index=False)
    patients[["patient_id"]].to_csv(directory / "ids.csv", index=False)
    selected = observations.loc[observations["patient_id"].isin(patients["patient_id"])]
    selected.to_csv(directory / "observations.csv", index=False)
    write_json(
        directory / "dataset_manifest.json",
        {
            "schema_version": 2,
            "format": "middle",
            "split_name": name,
            "cohort_root": str(cohort_root.resolve()),
            "assignment_manifest": str(assignment_manifest) if assignment_manifest else None,
            "n_patients": len(patients),
            "n_timepoints": len(selected),
        },
    )
    for path in directory.iterdir():
        path.chmod(0o600)


def run(config: CRCSplitConfig) -> None:
    # ===========================================================================
    # 一、读取01基础队列；本阶段不再做患者纳排、窗口或特征筛选。
    # ===========================================================================
    output_root = config.paths.dir.resolve()
    bundle_paths = {
        (strategy, seed): output_root
        / (
            "random"
            if strategy == "random"
            else f"temporal-{config.split.temporal.test_start_year}"
        )
        / f"seed-{seed}"
        for strategy in config.split.strategies
        for seed in config.split.seeds
    }
    targets = list(bundle_paths.values())
    if config.save_full_dataset:
        targets.append(output_root / "full")
    if existing := [str(path) for path in targets if path.exists()]:
        raise FileExistsError(f"拒绝覆盖既有划分：{existing}")
    patients, observations = read_tables(config.paths.patients_csv, config.paths.observations_csv)
    # 01允许按配置保留缺失手术日期，仅时间划分需要完整年份。
    if "temporal" in config.split.strategies and patients["surgery_year"].isna().to_numpy().any():
        raise ValueError("时间划分需要完整的手术年份")

    # ===========================================================================
    # 二、创建或复用基础队列主划分；不同面板和预处理共用患者归属。
    # ===========================================================================
    masters = _master_split(config, patients, bundle_paths)
    for (strategy, seed), splits in masters.items():
        bundle = bundle_paths[strategy, seed]
        assignment = (
            config.paths.assignment_root.resolve()
            / bundle.parent.name
            / bundle.name
            / "assignment_manifest.json"
        )
        bundle.mkdir(parents=True, exist_ok=False, mode=0o700)
        for name, frame in splits.items():
            _save_split(
                bundle / name, name, frame, observations, config.paths.cohort_root, assignment
            )
        write_json(
            bundle / "split_manifest.json",
            {
                "schema_version": 2,
                "dataset": config.dataset,
                "format": "middle",
                "strategy": strategy,
                "seed": seed,
                "assignment_manifest": str(assignment),
                "split_counts": {name: len(frame) for name, frame in splits.items()},
            },
        )
        OmegaConf.save(config.model_dump(mode="json"), bundle / "resolved_config.yaml")
        (bundle / "resolved_config.yaml").chmod(0o600)
        LOGGER.info("已划分 %s seed=%s：%s", strategy, seed, {k: len(v) for k, v in splits.items()})

    # ===========================================================================
    # 三、按需保存一份完整基础队列，不按策略或种子重复。
    # ===========================================================================
    if config.save_full_dataset:
        _save_split(
            output_root / "full", "full", patients, observations, config.paths.cohort_root, None
        )
        OmegaConf.save(
            config.model_dump(mode="json"),
            output_root / "full/resolved_config.yaml",
        )
        (output_root / "full/resolved_config.yaml").chmod(0o600)
    LOGGER.info("划分完成：%s", output_root)


@hydra.main(config_path="../../configs", config_name="crc_yunnan/split", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    config = CRCSplitConfig.model_validate(OmegaConf.to_container(raw_config, resolve=True))
    run(config)


if __name__ == "__main__":
    main()
