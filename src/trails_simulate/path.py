from dataclasses import dataclass
from pathlib import Path

from .config import ApplicationConfig


@dataclass(frozen=True)
class TrainPaths:
    data: Path
    test_data: Path | None
    train_root: Path
    save: Path | None


@dataclass(frozen=True)
class DatasetRunPaths:
    run_id: str
    data_root: Path
    train_data: Path
    test_data: Path


def data_root(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> Path:
    if config.paths.data_root is not None:
        return resolve_path(config.paths.data_root, project_root)
    if config.command == "simulate":
        return hydra_run_dir
    return hydra_run_dir / "data"


def checkpoint_path_for_run(
    config: ApplicationConfig,
    *,
    hydra_run_dir: Path,
    project_root: Path,
    run_id: str,
    n_runs: int,
) -> Path | None:
    if config.training.artifacts.save is None:
        return None
    configured = resolve_path(config.training.artifacts.save, project_root)
    if n_runs == 1:
        return configured
    suffix = configured.suffix
    stem = configured.stem if suffix else configured.name
    return hydra_run_dir / run_id / f"{stem}{suffix}"


def resolve_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    return project_root / path


def discover_dataset_runs(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> list[DatasetRunPaths]:
    if config.paths.data is not None:
        if config.paths.test_data is None:
            raise ValueError("Split commands require paths.test_data=... when paths.data is set.")
        data_path = resolve_path(config.paths.data, project_root)
        test_path = resolve_path(config.paths.test_data, project_root)
        return [
            DatasetRunPaths(
                run_id="0",
                data_root=data_path.parent,
                train_data=data_path,
                test_data=test_path,
            )
        ]

    root = (
        None if config.paths.data_root is None else data_root(config, hydra_run_dir, project_root)
    )
    if root is None:
        raise ValueError(
            f"command={config.command} requires paths.data_root=... or explicit "
            "paths.data/test_data."
        )

    single_train = root / "train.pt"
    single_test = root / "test.pt"
    if single_train.exists() and single_test.exists():
        return [
            DatasetRunPaths(
                run_id="0",
                data_root=root,
                train_data=single_train,
                test_data=single_test,
            )
        ]

    runs = discover_recursive_dataset_runs(root)
    if not runs:
        raise ValueError(
            "Could not find train/test split data. Expected train.pt and test.pt under "
            f"{root} or any nested subdirectory."
        )
    return runs


def discover_recursive_dataset_runs(root: Path) -> list[DatasetRunPaths]:
    if not root.exists():
        return []

    run_roots = sorted(
        {
            train_path.parent
            for train_path in root.rglob("train.pt")
            if (train_path.parent / "test.pt").exists()
        },
        key=lambda path: path.relative_to(root).as_posix(),
    )
    runs: list[DatasetRunPaths] = []
    for run_root in run_roots:
        relative = run_root.relative_to(root).as_posix()
        runs.append(
            DatasetRunPaths(
                run_id="0" if relative == "." else relative,
                data_root=run_root,
                train_data=run_root / "train.pt",
                test_data=run_root / "test.pt",
            )
        )
    return runs
