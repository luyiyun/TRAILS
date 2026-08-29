"""MIMIC 工作流的输入与输出路径解析。"""

from pathlib import Path


def resolve_input_path(path: Path, base_dir: Path | None = None) -> Path:
    """将相对输入路径解释为命令启动目录下的路径。"""
    return path if path.is_absolute() else (base_dir or Path.cwd()) / path


def resolve_output_path(path: Path, run_dir: Path) -> Path:
    """将相对输出路径解释为本次运行目录下的路径。"""
    return path if path.is_absolute() else run_dir / path
