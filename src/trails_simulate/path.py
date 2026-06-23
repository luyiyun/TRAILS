from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import ArtifactsConfig, PathsConfig


class DatasetDiscoveryConfig(Protocol):
    paths: PathsConfig


class CheckpointConfig(Protocol):
    paths: PathsConfig
    artifacts: ArtifactsConfig


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


def data_root(config: DatasetDiscoveryConfig) -> Path:
    return resolve_input_path(config.paths.data_root)


def checkpoint_path_for_run(
    config: CheckpointConfig,
    *,
    run_id: str,
    n_runs: int,
) -> Path | None:
    if config.artifacts.save is None:
        return None
    configured = resolve_output_path(config.artifacts.save, config.paths.dir)
    if n_runs == 1:
        return configured
    suffix = configured.suffix
    stem = configured.stem if suffix else configured.name
    return config.paths.dir / run_id / f"{stem}{suffix}"


def resolve_input_path(path: Path, base_dir: Path | None = None) -> Path:
    if path.is_absolute():
        return path
    return (base_dir or Path.cwd()) / path


def resolve_output_path(path: Path, run_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return run_dir / path


def discover_dataset_runs(config: DatasetDiscoveryConfig) -> list[DatasetRunPaths]:
    if config.paths.explicit_split.enabled:
        data_path = resolve_input_path(config.paths.explicit_split.train_data)
        test_path = resolve_input_path(config.paths.explicit_split.test_data)
        return [
            DatasetRunPaths(
                run_id="0",
                data_root=data_path.parent,
                train_data=data_path,
                test_data=test_path,
            )
        ]

    root = data_root(config)

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
