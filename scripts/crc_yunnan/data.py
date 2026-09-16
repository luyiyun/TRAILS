"""CRC上游中格式产物的读取和私有JSON写出。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_tables(patients_csv: Path, observations_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """恢复CSV中的ID与数值类型，沿用上游保证的结构、缺失位置和行顺序。"""
    patients = pd.read_csv(patients_csv, dtype={"patient_id": "string"})
    observations = pd.read_csv(observations_csv, dtype={"patient_id": "string"})
    observations = observations.astype(
        dict.fromkeys(observations.columns.drop("patient_id"), "float64")
    )
    return patients, observations


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """拒绝覆盖或非标准JSON数值，并限制患者产物的读取权限。"""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o600)
