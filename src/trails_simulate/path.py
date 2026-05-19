from dataclasses import dataclass
from pathlib import Path

from .config import ApplicationConfig


@dataclass(frozen=True)
class TrainPaths:
    data: Path
    test_data: Path | None
    train_root: Path
    save: Path | None


def optional_split_path(
    explicit_path: Path | None,
    data_root_path: Path | None,
    name: str,
    project_root: Path,
) -> Path | None:
    if explicit_path is not None:
        return resolve_path(explicit_path, project_root)
    if data_root_path is None:
        return None
    candidate = data_root_path / f"{name}.pt"
    return candidate if candidate.exists() else None


def data_root(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> Path:
    if config.paths.data_root is not None:
        return resolve_path(config.paths.data_root, project_root)
    return hydra_run_dir / "data"


def single_dataset_path(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> Path:
    if config.paths.data is not None:
        return resolve_path(config.paths.data, project_root)
    return data_root(config, hydra_run_dir, project_root) / "dataset.pt"


def train_root(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> Path:
    if config.paths.train_root is not None:
        return resolve_path(config.paths.train_root, project_root)
    return hydra_run_dir / "train"


def repeat_checkpoint_path(
    config: ApplicationConfig,
    repeat_dir: Path,
    project_root: Path,
    index: int,
) -> Path | None:
    if config.artifacts.save is None:
        return None
    configured = resolve_path(config.artifacts.save, project_root)
    if config.experiment.repeats == 1:
        return configured
    suffix = configured.suffix
    stem = configured.stem if suffix else configured.name
    return repeat_dir / "train" / f"{stem}-r{index:03d}{suffix}"


def resolve_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    return project_root / path


def train_paths_from_config(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> TrainPaths:
    if config.paths.data is None and config.paths.data_root is None:
        raise ValueError("command=train requires paths.data=... or paths.data_root=...")

    data_root_path = (
        None
        if config.paths.data_root is None
        else resolve_path(config.paths.data_root, project_root)
    )
    if config.paths.data is not None:
        data_path = resolve_path(config.paths.data, project_root)
    else:
        if data_root_path is None:
            raise ValueError("paths.data_root is required when paths.data is not set.")
        data_path = data_root_path / "train.pt"

    test_data = optional_split_path(config.paths.test_data, data_root_path, "test", project_root)
    train_root_path = train_root(config, hydra_run_dir, project_root)
    save = None
    if config.artifacts.save is not None:
        save = resolve_path(config.artifacts.save, project_root)
    return TrainPaths(
        data=data_path,
        test_data=test_data,
        train_root=train_root_path,
        save=save,
    )
