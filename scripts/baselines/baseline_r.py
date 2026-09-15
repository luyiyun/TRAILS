"""通过隔离的Rscript进程运行共享基线后端。"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self


class RScriptBackend:
    """以JSON文件交换配置和结果，并持久化R进程日志。"""

    def __init__(
        self,
        entrypoint: Path,
        work_dir: Path,
        *,
        executable: str = "Rscript",
        environment: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        if timeout_seconds is not None and timeout_seconds <= 0.0:
            raise ValueError("Rscript超时时间必须为正数")
        self.entrypoint = entrypoint.resolve()
        self.work_dir = work_dir.resolve()
        self.executable = executable
        self.environment = dict(environment or {})
        self.timeout_seconds = timeout_seconds

    def preflight(self) -> Self:
        """在读取患者数据前确认R运行时和方法入口均可用。"""
        if not self.entrypoint.is_file():
            raise FileNotFoundError(f"R基线入口不存在：{self.entrypoint}")
        if shutil.which(self.executable) is None:
            raise RuntimeError(f"找不到R运行时：{self.executable}")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return self

    def run(
        self,
        run_name: str,
        action: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """运行一次R动作；患者内容只进入输入文件而不进入命令行。"""
        self.preflight()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name) is None:
            raise ValueError("R运行名称只能包含字母、数字、下划线和连字符")
        if not action:
            raise ValueError("R运行action不能为空")

        config_path = self.work_dir / f"{run_name}.config.json"
        result_path = self.work_dir / f"{run_name}.result.json"
        stdout_path = self.work_dir / f"{run_name}.stdout.log"
        stderr_path = self.work_dir / f"{run_name}.stderr.log"
        config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

        # 直接写文件，长程MCMC运行中即可读取有界进度，超时也保留诊断。
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            completed = subprocess.run(
                [self.executable, str(self.entrypoint), action, str(config_path), str(result_path)],
                cwd=self.work_dir,
                env=os.environ | self.environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"R基线运行失败（exit={completed.returncode}）；日志：{stdout_path}，{stderr_path}"
            )
        if not result_path.is_file():
            raise RuntimeError(f"R基线未生成结果JSON：{result_path}")

        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError(f"R基线结果必须是JSON对象：{result_path}")
        return result
