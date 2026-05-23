import json
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
    if config.command == "simulate":
        return hydra_run_dir
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
    if config.training.artifacts.save is None:
        return None
    configured = resolve_path(config.training.artifacts.save, project_root)
    if config.simulation.repeats == 1:
        return configured
    suffix = configured.suffix
    stem = configured.stem if suffix else configured.name
    return repeat_dir / "train" / f"{stem}-r{index:03d}{suffix}"


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
    if config.training.artifacts.save is not None:
        save = resolve_path(config.training.artifacts.save, project_root)
    return TrainPaths(
        data=data_path,
        test_data=test_data,
        train_root=train_root_path,
        save=save,
    )


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
        latest_simulation_data_root(config, project_root)
        if config.paths.data_root is None and config.command in {"train", "baseline"}
        else data_root(config, hydra_run_dir, project_root)
    )
    numeric_runs = discover_numbered_dataset_runs(root)
    if numeric_runs:
        return numeric_runs

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

    repeat_roots = sorted(path for path in root.glob("repeat_*") if path.is_dir())
    runs = [
        DatasetRunPaths(
            run_id=repeat_root.name,
            data_root=repeat_root,
            train_data=repeat_root / "train.pt",
            test_data=repeat_root / "test.pt",
        )
        for repeat_root in repeat_roots
        if (repeat_root / "train.pt").exists() and (repeat_root / "test.pt").exists()
    ]
    if not runs:
        raise ValueError(
            "Could not find train/test split data. Expected train.pt and test.pt under "
            f"{root}, numbered run directories, or repeat_* subdirectories."
        )
    return runs


def discover_numbered_dataset_runs(root: Path) -> list[DatasetRunPaths]:
    if not root.exists():
        return []

    def numeric_key(path: Path) -> int:
        return int(path.name)

    run_roots = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=numeric_key,
    )
    return [
        DatasetRunPaths(
            run_id=run_root.name,
            data_root=run_root,
            train_data=run_root / "train.pt",
            test_data=run_root / "test.pt",
        )
        for run_root in run_roots
        if (run_root / "train.pt").exists() and (run_root / "test.pt").exists()
    ]


def latest_simulation_data_root(config: ApplicationConfig, project_root: Path) -> Path:
    outputs_root = project_root / "outputs"
    candidates = list(
        (outputs_root / config.simulation.name).glob("simulate-*/simulation_summary.json")
    )
    if not candidates:
        candidates = list(outputs_root.glob("*/simulate-*/simulation_summary.json"))
    if not candidates:
        candidates = list(
            (outputs_root / config.simulation.name).glob("*/data/simulation_summary.json")
        )
    if not candidates:
        candidates = list(outputs_root.glob("*/*/data/simulation_summary.json"))
    if not candidates:
        raise ValueError(
            "command=train requires paths.data=... or paths.data_root=... when no previous "
            "simulation output can be found under outputs/*/simulate-*."
        )
    summary_path = max(candidates, key=lambda path: path.stat().st_mtime)
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return summary_path.parent
    data_root_value = payload.get("data_root")
    if isinstance(data_root_value, str) and data_root_value:
        return resolve_path(Path(data_root_value), project_root)
    return summary_path.parent
