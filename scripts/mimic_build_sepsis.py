from __future__ import annotations

import re
import subprocess
from pathlib import Path

import duckdb

DATA_ROOT = Path("data/real/mimic-iv-3.1")
MIMIC_CODE_ROOT = Path(".remote-run/vendor/mimic-code")
DATABASE = DATA_ROOT / "derived" / "mimiciv.duckdb"
MIMIC_CODE_COMMIT = "d20b49a71ebb8cafc6febb0821432778592192d5"
TARGET_CONCEPT = "sepsis/sepsis3.sql"


class MimicSepsisBuilder:
    """Import raw MIMIC-IV CSVs and build official concepts through Sepsis-3."""

    def __init__(self) -> None:
        self.data_root = DATA_ROOT.resolve()
        self.mimic_code_root = MIMIC_CODE_ROOT.resolve()
        self.database = DATABASE.resolve()
        self.mimic_iv_root = self.mimic_code_root / "mimic-iv"
        self.create_sql = self.mimic_iv_root / "buildmimic" / "postgres" / "create.sql"
        self.concepts_root = self.mimic_iv_root / "concepts_duckdb"

    def build(self) -> None:
        self._validate_inputs()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(self.database))
        connection.execute(
            "CREATE TABLE IF NOT EXISTS trails_mimic_build_progress "
            "(step VARCHAR PRIMARY KEY, completed_at TIMESTAMPTZ DEFAULT now())"
        )

        if not self._is_done(connection, "schema"):
            schema_sql = self._patch_schema(self.create_sql.read_text(encoding="utf-8"))
            self._run_step(connection, "schema", schema_sql)

        tables = {
            f"{schema}.{table}"
            for schema, table in connection.execute(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_schema IN ('mimiciv_hosp', 'mimiciv_icu')"
            ).fetchall()
        }
        for csv_path in self._csv_paths():
            table = f"mimiciv_{csv_path.parent.name}.{csv_path.name.split('.', maxsplit=1)[0]}"
            if table not in tables:
                print(f"{table}: schema 中不存在，跳过")
                continue
            escaped_path = str(csv_path).replace("'", "''")
            self._run_step(
                connection,
                f"load:{table}",
                f"COPY {table} FROM '{escaped_path}' (HEADER, DELIM ',', QUOTE '\"', ESCAPE '\"')",
            )

        for relative_path in self._concept_paths():
            sql = (self.concepts_root / relative_path).read_text(encoding="utf-8")
            self._run_step(connection, f"concept:{relative_path}", sql)
        connection.close()
        print(f"Sepsis-3 concepts 已构建至 {self.database}")

    def _validate_inputs(self) -> None:
        for directory in (self.data_root / "hosp", self.data_root / "icu"):
            if not directory.is_dir():
                raise FileNotFoundError(f"缺少 MIMIC-IV 目录：{directory}")
        for path in (self.create_sql, self.concepts_root / "duckdb.sql"):
            if not path.is_file():
                raise FileNotFoundError(f"缺少 mimic-code 文件：{path}")
        commit = subprocess.run(
            ["git", "-C", str(self.mimic_code_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if commit != MIMIC_CODE_COMMIT:
            raise ValueError(f"mimic-code 提交不匹配：期望 {MIMIC_CODE_COMMIT}，实际 {commit}")

    def _run_step(self, connection: duckdb.DuckDBPyConnection, step: str, sql: str) -> None:
        if self._is_done(connection, step):
            print(f"{step}: 已完成，跳过")
            return
        print(f"{step}: 开始")
        connection.execute("BEGIN")
        connection.execute(sql)
        connection.execute("INSERT INTO trails_mimic_build_progress(step) VALUES (?)", [step])
        connection.execute("COMMIT")
        print(f"{step}: 完成")

    @staticmethod
    def _is_done(connection: duckdb.DuckDBPyConnection, step: str) -> bool:
        row = connection.execute(
            "SELECT EXISTS(SELECT 1 FROM trails_mimic_build_progress WHERE step = ?)", [step]
        ).fetchone()
        assert row is not None
        return bool(row[0])

    @staticmethod
    def _patch_schema(sql: str) -> str:
        sql = re.sub(r"TIMESTAMP\([0-9]+\)", "TIMESTAMP", sql)
        sql = re.sub(r"spec_type_desc(.+)NOT NULL", r"spec_type_desc\1", sql)
        return re.sub(r"drug +(VARCHAR.+)NOT NULL", r"drug \1", sql)

    def _csv_paths(self) -> list[Path]:
        return sorted(
            path
            for module in ("hosp", "icu")
            for path in (self.data_root / module).iterdir()
            if path.name.endswith((".csv", ".csv.gz"))
        )

    def _concept_paths(self) -> list[str]:
        paths: list[str] = []
        for line in (self.concepts_root / "duckdb.sql").read_text(encoding="utf-8").splitlines():
            if not line.strip().startswith(".read "):
                continue
            relative_path = line.strip()[6:]
            paths.append(relative_path)
            if relative_path == TARGET_CONCEPT:
                return paths
        raise ValueError(f"官方 concept 清单中未找到 {TARGET_CONCEPT}")


def main() -> None:
    MimicSepsisBuilder().build()


if __name__ == "__main__":
    main()
