import csv
import json
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]


def compose_payload(*overrides: str) -> dict[str, Any]:
    config_dir = str((ROOT / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(overrides))
    payload = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def write_synthetic_split(root: Path, *, n_clusters: int, seed: int) -> None:
    from trails_simulate.generators import (
        ClinicalTimeSeriesDatasetGenerator,
        ClinicalTimeSeriesDatasetGeneratorConfig,
    )

    source = ClinicalTimeSeriesDatasetGenerator(
        ClinicalTimeSeriesDatasetGeneratorConfig(
            n_clusters=n_clusters,
            min_visits=3,
            max_visits=4,
            hidden_size=12,
            latent_dim=4,
            attention_layers=1,
        ),
        mechanism_seed=seed,
    ).simulate(n_patients=max(12, n_clusters + 8), seed=seed)
    train_data, test_data = source.split_counts([8, 4], seed=seed)
    train_data.save(root / "train.pt")
    test_data.save(root / "test.pt")


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({name for row in rows for name in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_command_enum_rejects_removed_paper_grid() -> None:
    from trails_simulate.config import ApplicationConfig

    payload = compose_payload("command=paper_grid")

    with pytest.raises(ValidationError, match="Input should be"):
        ApplicationConfig.model_validate(payload)


def test_summary_command_config_validates_required_roots(tmp_path: Path) -> None:
    from trails_simulate.config import ApplicationConfig

    payload = compose_payload("command=summary")

    with pytest.raises(ValidationError, match="at least one"):
        ApplicationConfig.model_validate(payload)

    roots_payload = compose_payload("command=summary")
    roots_payload["summary"]["train_roots"] = [str(tmp_path / "train")]
    roots_payload["summary"]["baseline_roots"] = [str(tmp_path / "baseline")]
    roots_config = ApplicationConfig.model_validate(roots_payload)

    assert roots_config.command == "summary"
    assert roots_config.summary.train_roots == (tmp_path / "train",)
    assert roots_config.summary.baseline_roots == (tmp_path / "baseline",)
    assert roots_config.summary.metrics == ("acc", "ari", "nmi", "cindex")

    plural_payload = compose_payload("command=summary")
    plural_payload["summary"]["train_roots"] = [
        str(tmp_path / "train-base"),
        str(tmp_path / "train-mtan"),
    ]
    plural_payload["summary"]["train_labels"] = ["base", "mtan"]
    plural_config = ApplicationConfig.model_validate(plural_payload)

    assert plural_config.summary.train_roots == (
        tmp_path / "train-base",
        tmp_path / "train-mtan",
    )
    assert plural_config.summary.train_labels == ("base", "mtan")

    bad_label_payload = compose_payload("command=summary")
    bad_label_payload["summary"]["baseline_roots"] = [
        str(tmp_path / "baseline-a"),
        str(tmp_path / "baseline-b"),
    ]
    bad_label_payload["summary"]["baseline_labels"] = ["only-one"]
    with pytest.raises(ValidationError, match="baseline_labels"):
        ApplicationConfig.model_validate(bad_label_payload)


def test_simulation_configs_validate() -> None:
    from trails_simulate.config import ApplicationConfig

    expected_features = {
        "quick": 10,
        "base": 10,
        "imbalance": 8,
        "censored": 14,
        "high_dimension": 24,
    }
    for simulation, n_features in expected_features.items():
        app_config = ApplicationConfig.model_validate(compose_payload(f"simulation={simulation}"))

        assert app_config.command == "simulate"
        assert app_config.simulation.name == simulation
        assert len(app_config.simulation.train_size) == len(app_config.simulation.test_size)
        assert all(value > 0 for value in app_config.simulation.train_size)
        assert all(value > 0 for value in app_config.simulation.test_size)
        assert all(k > 1 for k in app_config.simulation.generator.n_clusters_tuple_)
        assert len(app_config.simulation.generator.feature_names) == n_features
        assert (
            "patients"
            not in cast(dict[str, Any], compose_payload(f"simulation={simulation}")["simulation"])[
                "generator"
            ]
        )

    base_config = ApplicationConfig.model_validate(compose_payload("simulation=base"))
    assert base_config.simulation.train_size == (500, 1000, 2000, 3000, 5000)
    assert base_config.simulation.test_size == (300, 300, 300, 300, 300)
    assert base_config.simulation.generator.n_clusters_tuple_ == (2, 3, 4, 5)
    assert base_config.simulation.repeats == 5


def test_baseline_config_includes_fpca_and_rejects_duplicate_methods() -> None:
    from trails_simulate.config import ApplicationConfig

    app_config = ApplicationConfig.model_validate(compose_payload())
    assert app_config.baseline.methods == (
        "summary_kmeans",
        "risk_stratified_kmeans",
        "fpca_kmeans",
    )

    payload = compose_payload("baseline.methods=[summary_kmeans,summary_kmeans]")
    with pytest.raises(ValidationError, match="duplicates"):
        ApplicationConfig.model_validate(payload)


def test_run_config_defaults_infer_prefix_and_keep_paths_explicit() -> None:
    from trails_simulate.config import ApplicationConfig

    simulate_config = ApplicationConfig.model_validate(compose_payload("simulation=base"))
    train_config = ApplicationConfig.model_validate(
        compose_payload("command=train", "paths.data_root=data/simulated/base")
    )

    assert simulate_config.run.output_root == Path("outputs")
    assert simulate_config.run.prefix == "base"
    assert simulate_config.run.name.startswith("base-")
    assert simulate_config.paths.data_root == Path("data/simulated")
    assert not simulate_config.paths.explicit_split.enabled
    assert simulate_config.paths.explicit_split.train_data == Path("data/simulated/train.pt")
    assert train_config.run.prefix == "base"


def test_training_configs_validate() -> None:
    from trails_simulate.config import ApplicationConfig

    expected_input = {
        "small": "grud",
        "base": "grud",
        "large": "grud",
        "mtan": "mtan",
    }
    for training, input_kind in expected_input.items():
        app_config = ApplicationConfig.model_validate(compose_payload(f"training={training}"))
        assert app_config.training.model.encoder.input.kind == input_kind
        assert app_config.training.model.n_clusters > 1
        assert app_config.training.trainer.batch_size is None
        assert app_config.training.trainer.valid_size == 0.2
        assert app_config.training.parallel.workers == 1
        assert app_config.training.parallel.devices == ()
        assert app_config.training.parallel.torch_threads is None

    explicit_config = ApplicationConfig.model_validate(
        compose_payload(
            "training=base",
            "training.trainer.batch_size=64",
            "training.parallel.workers=4",
            "training.parallel.devices=[cuda:0,cuda:1]",
            "training.parallel.torch_threads=2",
        )
    )
    assert explicit_config.training.trainer.batch_size == 64
    assert explicit_config.training.parallel.workers == 4
    assert explicit_config.training.parallel.devices == ("cuda:0", "cuda:1")
    assert explicit_config.training.parallel.torch_threads == 2

    invalid_payload = compose_payload("training=base")
    invalid_payload["training"]["parallel"]["workers"] = 0
    with pytest.raises(ValidationError, match="greater than 0"):
        ApplicationConfig.model_validate(invalid_payload)


def test_simulation_list_overrides_are_validated() -> None:
    from trails_simulate.config import ApplicationConfig

    app_config = ApplicationConfig.model_validate(
        compose_payload(
            "simulation=quick",
            "simulation.train_size=[10]",
            "simulation.test_size=[5]",
            "simulation.generator.n_clusters=[2]",
        )
    )

    assert app_config.simulation.train_size == (10,)
    assert app_config.simulation.test_size == (5,)
    assert app_config.simulation.generator.n_clusters_tuple_ == (2,)


def test_simulation_rejects_mismatched_train_test_lists() -> None:
    from trails_simulate.config import ApplicationConfig

    payload = compose_payload("simulation=quick")
    simulation = dict(cast(dict[str, Any], payload["simulation"]))
    simulation["train_size"] = [10, 20]
    simulation["test_size"] = [5]
    payload["simulation"] = simulation

    with pytest.raises(ValidationError, match="equal length"):
        ApplicationConfig.model_validate(payload)


def test_simulation_rejects_sample_size_not_larger_than_k() -> None:
    from trails_simulate.config import ApplicationConfig

    payload = compose_payload(
        "simulation=quick",
        "simulation.train_size=[2]",
        "simulation.test_size=[1]",
        "simulation.generator.n_clusters=[3]",
    )

    with pytest.raises(ValidationError, match="greater than every requested K"):
        ApplicationConfig.model_validate(payload)


def test_simulate_command_writes_grid_manifest_and_splits(tmp_path: Path) -> None:
    from trails.data import ClinicalTimeSeriesDataset
    from trails_simulate import workflow
    from trails_simulate.config import ApplicationConfig

    data_root = tmp_path / "data"
    app_config = ApplicationConfig.model_validate(
        compose_payload(
            "simulation=quick",
            "simulation.train_size=[6,8]",
            "simulation.test_size=[2,4]",
            "simulation.generator.n_clusters=[2,3]",
            "simulation.repeats=2",
            f"paths.data_root={data_root}",
        )
    )

    result = workflow.run_simulate_command(app_config, tmp_path / "run", ROOT)
    scenario_root = data_root / "quick"
    train_path = scenario_root / "train_6_test_2" / "k2" / "0" / "train.pt"
    test_path = scenario_root / "train_6_test_2" / "k2" / "0" / "test.pt"
    manifest_path = scenario_root / "simulation_manifest.csv"
    hydra_manifest_path = tmp_path / "run" / "simulation_manifest.csv"
    hydra_summary_path = tmp_path / "run" / "simulation_summary.json"

    assert result["command"] == "simulate"
    assert result["data_root"] == str(scenario_root)
    assert len(result["runs"]) == 8
    assert train_path.exists()
    assert test_path.exists()
    assert manifest_path.exists()
    assert hydra_manifest_path.exists()
    assert hydra_summary_path.exists()
    assert result["outputs"]["manifest"] == str(hydra_manifest_path)
    assert result["outputs"]["data_manifest"] == str(manifest_path)

    train_data = ClinicalTimeSeriesDataset.load(train_path)
    test_data = ClinicalTimeSeriesDataset.load(test_path)
    assert len(train_data) == 6
    assert len(test_data) == 2
    assert train_data.metadata["generation_params"]["n_patients"] == 8
    assert train_data.metadata["generation_params"]["n_clusters"] == 2
    assert torch.allclose(train_data.metadata["cluster_means"], test_data.metadata["cluster_means"])

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert rows[0]["run_id"] == "train_6_test_2/k2/0"
    assert rows[0]["train_size"] == "6"
    assert rows[0]["test_size"] == "2"
    assert rows[0]["n_clusters"] == "2"


def test_recursive_dataset_discovery_uses_mirrored_run_ids(tmp_path: Path) -> None:
    from trails_simulate.config import ApplicationConfig
    from trails_simulate.path import discover_dataset_runs

    data_root = tmp_path / "data"
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k2" / "0", n_clusters=2, seed=43)
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k3" / "0", n_clusters=3, seed=44)
    app_config = ApplicationConfig.model_validate(
        compose_payload("command=train", f"paths.data_root={data_root}")
    )

    runs = discover_dataset_runs(app_config, tmp_path / "run", ROOT)

    assert [run.run_id for run in runs] == [
        "base/train_6_test_2/k2/0",
        "base/train_6_test_2/k3/0",
    ]


def test_train_command_runs_all_discovered_splits_and_uses_dataset_k(
    monkeypatch,
    caplog,
    tmp_path: Path,
) -> None:
    from trails.data import ClinicalTimeSeriesDataset
    from trails_simulate import workflow
    from trails_simulate.config import ApplicationConfig
    from trails_simulate.evaluation import prediction_payload_from_dataset
    from trails_simulate.training import TrainResult

    data_root = tmp_path / "data"
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k2" / "0", n_clusters=2, seed=43)
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k3" / "0", n_clusters=3, seed=44)
    seen_clusters: list[int] = []

    app_config = ApplicationConfig.model_validate(
        compose_payload(
            "command=train",
            "training=small",
            f"paths.data_root={data_root}",
            "training.artifacts.names=[none]",
        )
    )

    def fake_fit_training_run(*args: Any, **kwargs: Any) -> TrainResult:
        run_config = args[0]
        train_paths = kwargs["train_paths"]
        seen_clusters.append(run_config.training.model.n_clusters)
        loaded_test = ClinicalTimeSeriesDataset.load(train_paths.test_data)
        prediction = prediction_payload_from_dataset(
            loaded_test,
            pred_cluster=torch.zeros(len(loaded_test), dtype=torch.long),
            risk_score=torch.arange(len(loaded_test), dtype=torch.float32),
        )
        return TrainResult(
            history=[],
            metrics={"ari": 0.2, "cindex": 1.0, "cluster_empty_count": 0.0},
            prediction=prediction,
            run_dir=None,
        )

    monkeypatch.setattr(workflow, "fit_training_run", fake_fit_training_run)
    caplog.set_level(logging.INFO)

    result = workflow.run_train_command(app_config, tmp_path / "run", ROOT)

    assert [run["run_id"] for run in result["runs"]] == [
        "base/train_6_test_2/k2/0",
        "base/train_6_test_2/k3/0",
    ]
    assert seen_clusters == [2, 3]
    assert "Completed train run: base/train_6_test_2/k2/0" in caplog.text
    assert "cindex=1" in caplog.text
    assert "ari=0.2" in caplog.text
    assert (tmp_path / "run" / "base" / "train_6_test_2" / "k2" / "0" / "trails.pt").exists()
    assert (tmp_path / "run" / "train_summary.json").exists()
    assert (tmp_path / "run" / "train_metrics.csv").exists()


def test_train_command_parallel_branch_sorts_results_and_rotates_devices(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from concurrent.futures import Future

    from trails.data import ClinicalTimeSeriesDataset
    from trails_simulate import workflow
    from trails_simulate.config import ApplicationConfig
    from trails_simulate.evaluation import prediction_payload_from_dataset
    from trails_simulate.training import TrainResult

    data_root = tmp_path / "data"
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k2" / "0", n_clusters=2, seed=43)
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k3" / "0", n_clusters=3, seed=44)
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k4" / "0", n_clusters=4, seed=45)
    seen_devices: list[str] = []
    submitted_slots: list[int] = []
    seen_max_workers: list[int] = []

    app_config = ApplicationConfig.model_validate(
        compose_payload(
            "command=train",
            "training=small",
            f"paths.data_root={data_root}",
            "training.artifacts.names=[none]",
            "training.parallel.workers=2",
            "training.parallel.devices=[cuda:0,cuda:1]",
        )
    )

    def fake_fit_training_run(*args: Any, **kwargs: Any) -> TrainResult:
        run_config = args[0]
        train_paths = kwargs["train_paths"]
        seen_devices.append(run_config.training.trainer.device)
        loaded_test = ClinicalTimeSeriesDataset.load(train_paths.test_data)
        prediction = prediction_payload_from_dataset(
            loaded_test,
            pred_cluster=torch.zeros(len(loaded_test), dtype=torch.long),
            risk_score=torch.arange(len(loaded_test), dtype=torch.float32),
        )
        return TrainResult(
            history=[],
            metrics={"ari": 0.2, "cindex": 1.0, "cluster_empty_count": 0.0},
            prediction=prediction,
            run_dir=None,
        )

    class FakeExecutor:
        def __init__(self, max_workers: int, **_kwargs: Any) -> None:
            seen_max_workers.append(max_workers)

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def submit(self, fn: Any, job: Any) -> Future[Any]:
            submitted_slots.append(job.worker_slot)
            future: Future[Any] = Future()
            try:
                future.set_result(fn(job))
            except Exception as error:
                future.set_exception(error)
            return future

    monkeypatch.setattr(workflow, "fit_training_run", fake_fit_training_run)
    monkeypatch.setattr(workflow, "ProcessPoolExecutor", FakeExecutor)

    result = workflow.run_train_command(app_config, tmp_path / "run", ROOT)

    assert seen_max_workers == [2]
    assert submitted_slots[:2] == [0, 1]
    assert set(seen_devices[:2]) == {"cuda:0", "cuda:1"}
    assert [run["run_id"] for run in result["runs"]] == [
        "base/train_6_test_2/k2/0",
        "base/train_6_test_2/k3/0",
        "base/train_6_test_2/k4/0",
    ]
    assert (tmp_path / "run" / "train_summary.json").exists()
    assert (tmp_path / "run" / "train_metrics.csv").exists()


def test_parallel_device_helpers_keep_same_device_without_device_list() -> None:
    from trails_simulate.config import ApplicationConfig
    from trails_simulate.workflow import config_with_training_device, device_for_worker_slot

    same_device_config = ApplicationConfig.model_validate(
        compose_payload("training=small", "training.trainer.device=cuda:0")
    )
    assert device_for_worker_slot(same_device_config, 0) is None
    assert same_device_config.training.trainer.device == "cuda:0"

    rotated_config = ApplicationConfig.model_validate(
        compose_payload("training=small", "training.parallel.devices=[cuda:0,cuda:1]")
    )
    assert device_for_worker_slot(rotated_config, 0) == "cuda:0"
    assert device_for_worker_slot(rotated_config, 1) == "cuda:1"
    assert device_for_worker_slot(rotated_config, 2) == "cuda:0"
    assert config_with_training_device(rotated_config, "cuda:1").training.trainer.device == "cuda:1"


def test_train_progress_time_formatting() -> None:
    from trails_simulate.workflow import (
        estimate_remaining_seconds,
        format_completed_train_run,
        format_duration,
        format_start_train_run,
    )

    assert format_duration(None) == "estimating"
    assert format_duration(12.34) == "12.3s"
    assert format_duration(65.0) == "1m05s"
    assert estimate_remaining_seconds([10.0, 20.0], remaining_runs=2) == pytest.approx(30.0)

    start_message = format_start_train_run(
        index=1,
        total=3,
        run_id="base/train_500_test_300/k3/0",
        elapsed_seconds=65.0,
        remaining_seconds=130.0,
    )
    assert "Training run 2/3" in start_message
    assert "elapsed=1m05s" in start_message
    assert "remaining=2m10s" in start_message

    completed_message = format_completed_train_run(
        run_id="base/train_500_test_300/k3/0",
        n_clusters=3,
        seed=7,
        prediction_path=Path("trails.pt"),
        metrics={"cindex": 0.9, "ari": 0.5},
        run_duration_seconds=10.0,
        elapsed_seconds=75.0,
        remaining_seconds=20.0,
    )
    assert "duration=10.0s" in completed_message
    assert "elapsed=1m15s" in completed_message
    assert "remaining=20.0s" in completed_message
    assert "cindex=0.9" in completed_message


def test_fit_training_run_resolves_auto_batch_size_and_records_effective_value(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from trails_simulate import training
    from trails_simulate.config import ApplicationConfig
    from trails_simulate.path import TrainPaths

    split_root = tmp_path / "data"
    write_synthetic_split(split_root, n_clusters=2, seed=47)
    seen_batch_sizes: list[int | None] = []

    class FakeEstimator:
        def __init__(self, config: Any) -> None:
            self.config = config
            self.history: list[Any] = []
            seen_batch_sizes.append(config.trainer.batch_size)

        def fit(self, *_args: Any, **_kwargs: Any) -> Any:
            return self

        def predict(self, data: Any) -> torch.Tensor:
            return torch.zeros(len(data), dtype=torch.long)

        def predict_risk(self, data: Any) -> torch.Tensor:
            return torch.arange(len(data), dtype=torch.float32)

        def predict_proba(self, data: Any) -> torch.Tensor:
            return torch.ones(len(data), self.config.model.n_clusters, dtype=torch.float32)

    monkeypatch.setattr(training, "TrailsEstimator", FakeEstimator)
    app_config = ApplicationConfig.model_validate(
        compose_payload(
            "command=train",
            "training=small",
            "training.artifacts.names=[config]",
        )
    )

    training.fit_training_run(
        app_config,
        train_paths=TrainPaths(
            data=split_root / "train.pt",
            test_data=split_root / "test.pt",
            train_root=tmp_path / "run",
            save=None,
        ),
        seed=61,
        swanlab_repeat_label=None,
    )
    config_payload = json.loads((tmp_path / "run" / "config.json").read_text(encoding="utf-8"))

    assert seen_batch_sizes == [8]
    assert config_payload["config"]["trainer"]["batch_size"] == 8
    assert config_payload["train_args"]["batch_size"] == 8


def test_resolve_batch_size_uses_logging(caplog) -> None:
    from trails.config import resolve_batch_size

    caplog.set_level(logging.INFO)

    assert resolve_batch_size(8, None) == 8
    assert "Resolving batch size to 8" in caplog.text


def test_progress_context_assigns_tqdm_positions(monkeypatch) -> None:
    from trails import progress

    calls: list[dict[str, Any]] = []

    class FakeTqdm:
        def __init__(self, iterable: Any, **kwargs: Any) -> None:
            self.iterable = iterable
            calls.append(kwargs)

        def __iter__(self) -> Any:
            return iter(self.iterable)

    monkeypatch.setattr(progress, "tqdm", FakeTqdm)

    with progress.progress_context(
        outer_position=3,
        inner_position=4,
        leave=False,
        description_prefix="run-a",
    ):
        list(progress.progress_bar(range(1), desc="Epoch", level="outer"))
        list(progress.progress_bar(range(1), desc="Train", level="inner"))

    assert calls[0]["position"] == 3
    assert calls[0]["leave"] is False
    assert calls[0]["desc"] == "run-a Epoch"
    assert calls[1]["position"] == 4
    assert calls[1]["leave"] is False
    assert calls[1]["desc"] == "run-a Train"


def test_progress_logging_suppresses_nested_tensor_warning() -> None:
    from trails.progress import configure_tqdm_logging

    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    try:
        with warnings.catch_warnings(record=True) as caught:
            configure_tqdm_logging()
            warnings.warn(
                "The PyTorch API of nested tensors is in prototype stage and will change in the "
                "near future.",
                UserWarning,
                stacklevel=1,
            )
    finally:
        root_logger.handlers[:] = original_handlers
        root_logger.setLevel(original_level)
        logging.captureWarnings(False)

    assert caught == []


def test_baseline_command_infers_k_per_split(tmp_path: Path) -> None:
    from trails_simulate import workflow
    from trails_simulate.config import ApplicationConfig

    data_root = tmp_path / "data"
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k2" / "0", n_clusters=2, seed=43)
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k3" / "0", n_clusters=3, seed=44)

    app_config = ApplicationConfig.model_validate(
        compose_payload("command=baseline", f"paths.data_root={data_root}")
    )

    result = workflow.run_baseline_command(app_config, tmp_path / "run", ROOT)

    assert result["command"] == "baseline"
    assert [run["n_clusters"] for run in result["runs"]] == [2, 3]
    assert (
        tmp_path / "run" / "base" / "train_6_test_2" / "k2" / "0" / "summary_kmeans.pt"
    ).exists()
    assert (tmp_path / "run" / "baseline_summary.json").exists()
    assert (tmp_path / "run" / "baseline_metrics.csv").exists()


def test_optim_trial_config_updates_training_namespace() -> None:
    from trails_simulate.config import ApplicationConfig
    from trails_simulate.optim import optim_trial_config

    class FakeTrial:
        def __init__(self) -> None:
            self.user_attrs: dict[str, Any] = {}

        def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
            return choices[0]

        def suggest_float(self, name: str, low: float, high: float, *, log: bool) -> float:
            return low

        def suggest_int(self, name: str, low: int, high: int) -> int:
            return low

        def set_user_attr(self, name: str, value: Any) -> None:
            self.user_attrs[name] = value

    app_config = ApplicationConfig.model_validate(compose_payload("command=optim", "training=base"))
    trial_config = optim_trial_config(app_config, FakeTrial())

    assert trial_config.training.artifacts.names == ("none",)
    assert not trial_config.training.diagnostics.latent_embeddings.enabled
    assert not trial_config.training.swanlab.enabled
    assert trial_config.training.model.encoder.input.kind == "grud"
    assert trial_config.training.trainer.batch_size == app_config.optim.search.batch_size[0]
    assert app_config.training.model.n_clusters == 4


def test_summary_command_combines_metrics_and_writes_figures(tmp_path: Path) -> None:
    from trails_simulate import workflow
    from trails_simulate.config import ApplicationConfig

    train_root = tmp_path / "train"
    baseline_root = tmp_path / "baseline"
    run_id_a = "base/train_500_test_300/k3/0"
    run_id_b = "base/train_500_test_300/k3/1"
    write_metrics_csv(
        train_root / "train_metrics.csv",
        [
            {"run_id": run_id_a, "method": "trails", "cindex": 0.7, "ari": 0.2},
            {"run_id": run_id_b, "method": "trails", "cindex": 0.8, "ari": 0.4},
        ],
    )
    write_metrics_csv(
        baseline_root / "baseline_metrics.csv",
        [
            {"run_id": run_id_a, "method": "summary_kmeans", "cindex": 0.5, "ari": 0.1},
            {"run_id": run_id_b, "method": "summary_kmeans", "cindex": 0.6, "ari": 0.2},
        ],
    )
    payload = compose_payload("command=summary", "summary.metrics=[cindex,ari,nmi]")
    payload["summary"]["train_roots"] = [str(train_root)]
    payload["summary"]["baseline_roots"] = [str(baseline_root)]
    app_config = ApplicationConfig.model_validate(payload)

    result = workflow.run(app_config, hydra_run_dir=tmp_path / "run", project_root=ROOT)

    assert result["command"] == "summary"
    assert result["n_rows"] == 4
    assert "nmi" in result["metrics"]["skipped"]
    assert (tmp_path / "run" / "summary_metrics.csv").exists()
    assert (tmp_path / "run" / "summary_metrics_grouped.csv").exists()
    assert (tmp_path / "run" / "summary_summary.json").exists()
    assert (tmp_path / "run" / "figures" / "base_metrics_by_train_size.png").exists()
    assert (tmp_path / "run" / "figures" / "base_metrics_by_train_size.pdf").exists()

    with (tmp_path / "run" / "summary_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["scenario"] == "base"
    assert rows[0]["train_size"] == "500"
    assert rows[0]["test_size"] == "300"
    assert rows[0]["n_clusters"] == "3"
    assert rows[0]["repeat"] == "0"
    assert rows[0]["method_label"] == "trails"


def test_summary_command_merges_multiple_roots_with_distinct_method_labels(
    tmp_path: Path,
) -> None:
    from trails_simulate import workflow
    from trails_simulate.config import ApplicationConfig

    train_base = tmp_path / "train-base"
    train_mtan = tmp_path / "train-mtan"
    baseline_classic = tmp_path / "baseline-classic"
    baseline_fpca = tmp_path / "baseline-fpca"
    run_id_a = "base/train_500_test_300/k3/0"
    run_id_b = "base/train_500_test_300/k3/1"
    write_metrics_csv(
        train_base / "train_metrics.csv",
        [
            {"run_id": run_id_a, "method": "trails", "cindex": 0.7, "ari": 0.2},
            {"run_id": run_id_b, "method": "trails", "cindex": 0.8, "ari": 0.4},
        ],
    )
    write_metrics_csv(
        train_mtan / "train_metrics.csv",
        [
            {"run_id": run_id_a, "method": "trails", "cindex": 0.9, "ari": 0.5},
            {"run_id": run_id_b, "method": "trails", "cindex": 1.0, "ari": 0.7},
        ],
    )
    write_metrics_csv(
        baseline_classic / "baseline_metrics.csv",
        [{"run_id": run_id_a, "method": "summary_kmeans", "cindex": 0.55, "ari": 0.1}],
    )
    write_metrics_csv(
        baseline_fpca / "baseline_metrics.csv",
        [{"run_id": run_id_a, "method": "fpca_kmeans", "cindex": 0.6, "ari": 0.12}],
    )

    payload = compose_payload("command=summary", "summary.metrics=[cindex,ari]")
    payload["summary"]["train_roots"] = [str(train_base), str(train_mtan)]
    payload["summary"]["baseline_roots"] = [str(baseline_classic), str(baseline_fpca)]
    payload["summary"]["train_labels"] = ["base", "mtan"]
    payload["summary"]["baseline_labels"] = ["classic", "fpca"]
    app_config = ApplicationConfig.model_validate(payload)

    result = workflow.run(app_config, hydra_run_dir=tmp_path / "run", project_root=ROOT)

    assert result["n_rows"] == 6
    assert len(result["inputs"]) == 4
    assert (tmp_path / "run" / "figures" / "base_metrics_by_train_size.png").exists()
    with (tmp_path / "run" / "summary_metrics_grouped.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        grouped = list(csv.DictReader(handle))
    means = {
        row["method_label"]: float(row["cindex_mean"])
        for row in grouped
        if row["scenario"] == "base" and row["n_clusters"] == "3"
    }
    assert means["trails (base)"] == pytest.approx(0.75)
    assert means["trails (mtan)"] == pytest.approx(0.95)
    assert means["summary_kmeans"] == pytest.approx(0.55)
    assert means["fpca_kmeans"] == pytest.approx(0.6)


def test_summary_command_plots_scenarioless_train_run_ids(tmp_path: Path) -> None:
    from trails_simulate import workflow
    from trails_simulate.config import ApplicationConfig

    train_root = tmp_path / "train"
    baseline_root = tmp_path / "baseline"
    train_run_id = "train_500_test_300/k3/0"
    baseline_run_id = "base/train_500_test_300/k3/0"
    write_metrics_csv(
        train_root / "train_metrics.csv",
        [
            {
                "run_id": train_run_id,
                "method": "trails",
                "data_root": str(tmp_path / "data" / "base" / train_run_id),
                "cindex": 0.9,
                "ari": 0.5,
            },
        ],
    )
    write_metrics_csv(
        baseline_root / "baseline_metrics.csv",
        [{"run_id": baseline_run_id, "method": "summary_kmeans", "cindex": 0.6, "ari": 0.2}],
    )
    payload = compose_payload("command=summary", "summary.metrics=[cindex,ari]")
    payload["summary"]["train_roots"] = [str(train_root)]
    payload["summary"]["baseline_roots"] = [str(baseline_root)]
    app_config = ApplicationConfig.model_validate(payload)

    result = workflow.run(app_config, hydra_run_dir=tmp_path / "run", project_root=ROOT)

    assert result["parse_warnings"] == []
    assert (tmp_path / "run" / "figures" / "base_metrics_by_train_size.png").exists()
    with (tmp_path / "run" / "summary_metrics_grouped.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        grouped = list(csv.DictReader(handle))
    assert {(row["scenario"], row["method_label"], row["n_clusters"]) for row in grouped} == {
        ("base", "summary_kmeans", "3"),
        ("base", "trails", "3"),
    }


def test_optim_command_filters_configured_run_ids_with_shared_storage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from trails_simulate import optim
    from trails_simulate.config import ApplicationConfig

    class FakeTrial:
        def __init__(self, number: int) -> None:
            self.datetime_complete = None
            self.datetime_start = None
            self.number = number
            self.params: dict[str, Any] = {}
            self.state = SimpleNamespace(name="RUNNING")
            self.user_attrs: dict[str, Any] = {}
            self.values: list[float] | None = None

        def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
            value = choices[0]
            self.params[name] = value
            return value

        def suggest_float(self, name: str, low: float, high: float, *, log: bool) -> float:
            del high, log
            self.params[name] = low
            return low

        def suggest_int(self, name: str, low: int, high: int) -> int:
            del high
            self.params[name] = low
            return low

        def set_user_attr(self, name: str, value: Any) -> None:
            self.user_attrs[name] = value

    class FakeStudy:
        def __init__(self, *, storage: str, study_name: str) -> None:
            self.best_trials: list[FakeTrial] = []
            self.storage = storage
            self.study_name = study_name
            self.trials: list[FakeTrial] = []

        def ask(self) -> FakeTrial:
            trial = FakeTrial(len(self.trials))
            self.trials.append(trial)
            return trial

        def tell(self, trial: FakeTrial, *, values: tuple[float, float]) -> None:
            trial.values = list(values)
            trial.state = SimpleNamespace(name="COMPLETE")
            self.best_trials = list(self.trials)

    class FakeSamplers:
        class TPESampler:
            def __init__(self, *, seed: int) -> None:
                self.seed = seed

    class FakeOptuna:
        def __init__(self) -> None:
            self.created: list[FakeStudy] = []
            self.samplers = FakeSamplers

        def create_study(self, **kwargs: Any) -> FakeStudy:
            study = FakeStudy(storage=str(kwargs["storage"]), study_name=str(kwargs["study_name"]))
            self.created.append(study)
            return study

    data_root = tmp_path / "data"
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k2" / "0", n_clusters=2, seed=43)
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k3" / "0", n_clusters=3, seed=44)
    app_config = ApplicationConfig.model_validate(
        compose_payload(
            "command=optim",
            "optim.n_trials=1",
            f"paths.data_root={data_root}",
            "optim.run_ids=[base/train_6_test_2/k3/0]",
            f"optim.storage={tmp_path / 'shared.db'}",
        )
    )
    fake_optuna = FakeOptuna()
    monkeypatch.setattr(optim, "load_optuna", lambda: fake_optuna)

    def fake_run_optim_split_job(job: optim.OptimSplitJob) -> optim.OptimSplitResult:
        return optim.OptimSplitResult(
            metrics={"ari": 0.2, "cindex": 0.7},
            run_id=job.run_paths.run_id,
            seed=job.seed,
            split_index=job.split_index,
            trial_number=job.trial_number,
        )

    monkeypatch.setattr(optim, "run_optim_split_job", fake_run_optim_split_job)

    result = optim.run_optim_command(app_config, tmp_path / "run", ROOT)

    assert result["selected_run_ids"] == ["base/train_6_test_2/k3/0"]
    assert result["selection"]["source"] == "configured"
    assert fake_optuna.created[0].study_name == "optim"
    assert fake_optuna.created[0].storage.endswith("shared.db")
    assert (tmp_path / "run" / "trials.csv").exists()
    assert (tmp_path / "run" / "optim_summary.json").exists()
    assert (tmp_path / "run" / "dataset_fingerprint.json").exists()
    assert (tmp_path / "run" / "figures" / "pareto_front.png").exists()


def test_optim_command_runs_all_splits_and_aggregates_mean_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from trails_simulate import optim
    from trails_simulate.config import ApplicationConfig

    class FakeTrial:
        def __init__(self, number: int) -> None:
            self.datetime_complete = None
            self.datetime_start = None
            self.number = number
            self.params: dict[str, Any] = {}
            self.state = SimpleNamespace(name="RUNNING")
            self.user_attrs: dict[str, Any] = {}
            self.values: list[float] | None = None

        def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
            value = choices[0]
            self.params[name] = value
            return value

        def suggest_float(self, name: str, low: float, high: float, *, log: bool) -> float:
            del high, log
            self.params[name] = low
            return low

        def suggest_int(self, name: str, low: int, high: int) -> int:
            del high
            self.params[name] = low
            return low

        def set_user_attr(self, name: str, value: Any) -> None:
            self.user_attrs[name] = value

    class FakeStudy:
        def __init__(self) -> None:
            self.best_trials: list[FakeTrial] = []
            self.study_name = "optim"
            self.trials: list[FakeTrial] = []

        def ask(self) -> FakeTrial:
            trial = FakeTrial(len(self.trials))
            self.trials.append(trial)
            return trial

        def tell(self, trial: FakeTrial, *, values: tuple[float, float]) -> None:
            trial.values = list(values)
            trial.state = SimpleNamespace(name="COMPLETE")
            self.best_trials = list(self.trials)

    class FakeSamplers:
        class TPESampler:
            def __init__(self, *, seed: int) -> None:
                self.seed = seed

    class FakeOptuna:
        samplers = FakeSamplers

        def create_study(self, **_kwargs: Any) -> FakeStudy:
            return FakeStudy()

    data_root = tmp_path / "data"
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k2" / "0", n_clusters=2, seed=43)
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k3" / "0", n_clusters=3, seed=44)
    app_config = ApplicationConfig.model_validate(
        compose_payload("command=optim", "optim.n_trials=1", f"paths.data_root={data_root}")
    )

    def fake_run_optim_split_job(job: optim.OptimSplitJob) -> optim.OptimSplitResult:
        metrics = (
            {"ari": 0.2, "cindex": 0.8}
            if "k2" in job.run_paths.run_id
            else {"ari": 0.4, "cindex": 0.6}
        )
        return optim.OptimSplitResult(
            metrics=metrics,
            run_id=job.run_paths.run_id,
            seed=job.seed,
            split_index=job.split_index,
            trial_number=job.trial_number,
        )

    monkeypatch.setattr(optim, "load_optuna", lambda: FakeOptuna())
    monkeypatch.setattr(optim, "run_optim_split_job", fake_run_optim_split_job)

    result = optim.run_optim_command(app_config, tmp_path / "run", ROOT)

    assert result["selected_run_ids"] == [
        "base/train_6_test_2/k2/0",
        "base/train_6_test_2/k3/0",
    ]
    assert result["selection"]["source"] == "all"
    assert result["trials"][0]["values"] == pytest.approx([0.7, 0.3])
    with (tmp_path / "run" / "trials.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["mean_cindex"]) == pytest.approx(0.7)
    assert float(rows[0]["mean_ari"]) == pytest.approx(0.3)
    assert float(rows[0]["std_cindex"]) == pytest.approx(0.1)
    assert float(rows[0]["std_ari"]) == pytest.approx(0.1)
    assert float(rows[0]["mean_objective"]) == pytest.approx(0.5)


def test_optim_parallel_uses_shared_pool_and_rotates_devices(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from concurrent.futures import Future

    from trails_simulate import optim
    from trails_simulate.config import ApplicationConfig

    class FakeTrial:
        def __init__(self, number: int) -> None:
            self.datetime_complete = None
            self.datetime_start = None
            self.number = number
            self.params: dict[str, Any] = {}
            self.state = SimpleNamespace(name="RUNNING")
            self.user_attrs: dict[str, Any] = {}
            self.values: list[float] | None = None

        def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
            value = choices[0]
            self.params[name] = value
            return value

        def suggest_float(self, name: str, low: float, high: float, *, log: bool) -> float:
            del high, log
            self.params[name] = low
            return low

        def suggest_int(self, name: str, low: int, high: int) -> int:
            del high
            self.params[name] = low
            return low

        def set_user_attr(self, name: str, value: Any) -> None:
            self.user_attrs[name] = value

    class FakeStudy:
        def __init__(self) -> None:
            self.best_trials: list[FakeTrial] = []
            self.study_name = "optim"
            self.trials: list[FakeTrial] = []

        def ask(self) -> FakeTrial:
            trial = FakeTrial(len(self.trials))
            self.trials.append(trial)
            return trial

        def tell(self, trial: FakeTrial, *, values: tuple[float, float]) -> None:
            trial.values = list(values)
            trial.state = SimpleNamespace(name="COMPLETE")
            self.best_trials = list(self.trials)

    class FakeSamplers:
        class TPESampler:
            def __init__(self, *, seed: int) -> None:
                self.seed = seed

    class FakeOptuna:
        samplers = FakeSamplers

        def create_study(self, **_kwargs: Any) -> FakeStudy:
            return FakeStudy()

    data_root = tmp_path / "data"
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k2" / "0", n_clusters=2, seed=43)
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k3" / "0", n_clusters=3, seed=44)
    app_config = ApplicationConfig.model_validate(
        compose_payload(
            "command=optim",
            "optim.n_trials=2",
            f"paths.data_root={data_root}",
            "optim.parallel.workers=2",
            "optim.parallel.max_active_trials=2",
            "optim.parallel.devices=[cuda:0,cuda:1]",
        )
    )
    seen_max_workers: list[int] = []
    submitted_devices: list[str] = []
    submitted_trials: list[int] = []

    class FakeExecutor:
        def __init__(self, max_workers: int, **_kwargs: Any) -> None:
            seen_max_workers.append(max_workers)

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def submit(self, fn: Any, job: optim.OptimSplitJob) -> Future[Any]:
            submitted_devices.append(job.device)
            submitted_trials.append(job.trial_number)
            future: Future[Any] = Future()
            try:
                future.set_result(fn(job))
            except Exception as error:
                future.set_exception(error)
            return future

    def fake_run_optim_split_job(job: optim.OptimSplitJob) -> optim.OptimSplitResult:
        return optim.OptimSplitResult(
            metrics={"ari": 0.2, "cindex": 0.7},
            run_id=job.run_paths.run_id,
            seed=job.seed,
            split_index=job.split_index,
            trial_number=job.trial_number,
        )

    monkeypatch.setattr(optim, "load_optuna", lambda: FakeOptuna())
    monkeypatch.setattr(optim, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(optim, "run_optim_split_job", fake_run_optim_split_job)

    result = optim.run_optim_command(app_config, tmp_path / "run", ROOT)

    assert seen_max_workers == [2]
    assert len(submitted_devices) == 4
    assert submitted_devices == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    assert sorted(set(submitted_trials)) == [0, 1]
    assert result["completed_after"] == 2


def test_optim_resume_rejects_changed_dataset_fingerprint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from trails_simulate import optim
    from trails_simulate.config import ApplicationConfig

    class FakeTrial:
        def __init__(self, number: int) -> None:
            self.datetime_complete = None
            self.datetime_start = None
            self.number = number
            self.params: dict[str, Any] = {}
            self.state = SimpleNamespace(name="RUNNING")
            self.user_attrs: dict[str, Any] = {}
            self.values: list[float] | None = None

        def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
            value = choices[0]
            self.params[name] = value
            return value

        def suggest_float(self, name: str, low: float, high: float, *, log: bool) -> float:
            del high, log
            self.params[name] = low
            return low

        def suggest_int(self, name: str, low: int, high: int) -> int:
            del high
            self.params[name] = low
            return low

        def set_user_attr(self, name: str, value: Any) -> None:
            self.user_attrs[name] = value

    class FakeStudy:
        def __init__(self) -> None:
            self.best_trials: list[FakeTrial] = []
            self.study_name = "optim"
            self.trials: list[FakeTrial] = []

        def ask(self) -> FakeTrial:
            trial = FakeTrial(len(self.trials))
            self.trials.append(trial)
            return trial

        def tell(self, trial: FakeTrial, *, values: tuple[float, float]) -> None:
            trial.values = list(values)
            trial.state = SimpleNamespace(name="COMPLETE")
            self.best_trials = list(self.trials)

    class FakeSamplers:
        class TPESampler:
            def __init__(self, *, seed: int) -> None:
                self.seed = seed

    class FakeOptuna:
        samplers = FakeSamplers

        def create_study(self, **_kwargs: Any) -> FakeStudy:
            return FakeStudy()

    data_root = tmp_path / "data"
    split_root = data_root / "base" / "train_6_test_2" / "k2" / "0"
    write_synthetic_split(split_root, n_clusters=2, seed=43)
    app_config = ApplicationConfig.model_validate(
        compose_payload("command=optim", "optim.n_trials=1", f"paths.data_root={data_root}")
    )

    def fake_run_optim_split_job(job: optim.OptimSplitJob) -> optim.OptimSplitResult:
        return optim.OptimSplitResult(
            metrics={"ari": 0.2, "cindex": 0.7},
            run_id=job.run_paths.run_id,
            seed=job.seed,
            split_index=job.split_index,
            trial_number=job.trial_number,
        )

    monkeypatch.setattr(optim, "load_optuna", lambda: FakeOptuna())
    monkeypatch.setattr(optim, "run_optim_split_job", fake_run_optim_split_job)
    optim.run_optim_command(app_config, tmp_path / "run", ROOT)

    write_synthetic_split(split_root, n_clusters=2, seed=99)
    resume_config = ApplicationConfig.model_validate(
        compose_payload(
            "command=optim",
            "optim.n_trials=1",
            "optim.resume=true",
            f"paths.data_root={data_root}",
        )
    )
    with pytest.raises(ValueError, match="fingerprint"):
        optim.run_optim_command(resume_config, tmp_path / "run", ROOT)


def test_paths_reject_removed_validation_data_field() -> None:
    from trails_simulate.config import ApplicationConfig

    payload = compose_payload()
    paths = dict(cast(dict[str, Any], payload["paths"]))
    paths["val_data"] = "data/val.pt"
    payload["paths"] = paths

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ApplicationConfig.model_validate(payload)


def test_simulation_rejects_removed_seed_stride_field() -> None:
    from trails_simulate.config import ApplicationConfig

    payload = compose_payload()
    simulation = dict(cast(dict[str, Any], payload["simulation"]))
    simulation["seed_stride"] = 100
    payload["simulation"] = simulation

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ApplicationConfig.model_validate(payload)


def test_generator_config_rejects_removed_patients_field() -> None:
    from trails_simulate.config import ApplicationConfig

    payload = compose_payload()
    simulation = dict(cast(dict[str, Any], payload["simulation"]))
    generator = dict(cast(dict[str, Any], simulation["generator"]))
    generator["patients"] = 10
    simulation["generator"] = generator
    payload["simulation"] = simulation

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ApplicationConfig.model_validate(payload)
