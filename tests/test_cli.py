import csv
import importlib.util
import json
import logging
import sys
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


def load_script(name: str) -> Any:
    module_name = f"{name}_script"
    script_path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def compose_payload(*overrides: str) -> dict[str, Any]:
    command, config_overrides = command_config_from_overrides(overrides)
    config_dir = str((ROOT / "configs").resolve())
    __import__("trails_simulate.config")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=f"{command}", overrides=config_overrides)
    payload = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def with_paths_dir(config: Any, run_dir: Path) -> Any:
    return config.model_copy(update={"paths": config.paths.model_copy(update={"dir": run_dir})})


def command_config_from_overrides(overrides: tuple[str, ...]) -> tuple[str, list[str]]:
    command = "simulate"
    config_overrides: list[str] = []
    for override in overrides:
        if override.startswith("command="):
            command = override.split("=", maxsplit=1)[1]
            continue
        config_overrides.append(override)

    if command != "simulate":
        return command, config_overrides
    baseline_fields = {
        "fallback_n_clusters",
        "fpca_components",
        "fpca_grid_size",
        "kmeans_iters",
        "methods",
        "n_clusters",
        "ridge_alpha",
        "risk_feature_weight",
    }
    summary_fields = {
        "baseline_labels",
        "baseline_roots",
        "metrics",
        "train_labels",
        "train_roots",
    }
    training_prefixes = (
        "artifacts.",
        "diagnostics.",
        "model.",
        "parallel.",
        "swanlab.",
        "trainer.",
    )
    if any(override.split("=", maxsplit=1)[0] in baseline_fields for override in overrides):
        return "baseline", config_overrides
    if any(
        override.startswith("optim.") or override.startswith("optim=") for override in overrides
    ):
        return "optim", config_overrides
    if any(override.split("=", maxsplit=1)[0] in summary_fields for override in overrides):
        return "summary", config_overrides
    if any(
        override.startswith(training_prefixes) or override.startswith("training=")
        for override in overrides
    ):
        return "train", config_overrides
    if any(
        override.startswith("paths.data_root") or override.startswith("paths.explicit_split")
        for override in overrides
    ):
        return "train", config_overrides
    return command, config_overrides


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


def test_legacy_single_entrypoint_is_removed() -> None:
    assert not (ROOT / "main.py").exists()
    assert not (ROOT / "configs" / "config.yaml").exists()
    assert not (ROOT / "configs" / "commands").exists()
    assert not (ROOT / "configs" / "baseline" / "default.yaml").exists()
    assert not (ROOT / "configs" / "summary" / "default.yaml").exists()
    assert not (ROOT / "configs" / "case" / "default.yaml").exists()
    assert not (ROOT / "configs" / "paths" / "default.yaml").exists()
    assert not (ROOT / "src" / "trails_simulate" / "workflow.py").exists()
    assert not (ROOT / "src" / "trails_case" / "workflow.py").exists()

    for command in ("simulate", "train", "baseline", "optim", "summary", "case"):
        assert (ROOT / "scripts" / f"{command}.py").exists()
        assert (ROOT / "configs" / f"{command}.yaml").exists()


def test_command_configs_flatten_non_optim_namespaces() -> None:
    forbidden_namespaces = {"baseline", "case", "simulation", "summary", "training"}

    for command in ("simulate", "train", "baseline", "summary", "case"):
        payload = compose_payload(f"command={command}")
        assert forbidden_namespaces.isdisjoint(payload)
        assert {"root", "prefix", "suffix", "dir"}.issubset(payload["paths"])

    optim_payload = compose_payload("command=optim")
    assert "optim" in optim_payload
    assert "training" not in optim_payload
    assert {"root", "prefix", "suffix", "dir"}.issubset(optim_payload["paths"])


def test_paths_defaults_are_inlined_into_data_commands() -> None:
    for command in ("train", "baseline", "optim"):
        root_config = (ROOT / "configs" / f"{command}.yaml").read_text(encoding="utf-8")
        payload = compose_payload(f"command={command}")
        assert "/paths: default" not in root_config
        assert payload["paths"]["data_root"] == "data/simulated"
        assert payload["paths"]["explicit_split"] == {
            "enabled": False,
            "train_data": "data/simulated/train.pt",
            "test_data": "data/simulated/test.pt",
        }

    for command in ("simulate", "summary", "case"):
        root_config = (ROOT / "configs" / f"{command}.yaml").read_text(encoding="utf-8")
        payload = compose_payload(f"command={command}")
        assert "/paths: default" not in root_config
        assert "data_root" not in payload["paths"]
        assert "explicit_split" not in payload["paths"]


def test_legacy_namespace_overrides_are_rejected() -> None:
    from hydra.errors import ConfigCompositionException

    invalid_overrides = [
        ("command=simulate", "simulation.train_size=[10]"),
        ("command=train", "training.trainer.device=cpu"),
        ("command=baseline", "baseline=default"),
        ("command=baseline", "baseline.methods=[summary_kmeans]"),
        ("command=summary", "summary=default"),
        ("command=summary", "summary.train_roots=[outputs/train/demo]"),
        ("command=case", "case=default"),
        ("command=case", "case.observations_csv=data/case/observations.csv"),
    ]

    for command_override, invalid_override in invalid_overrides:
        with pytest.raises(ConfigCompositionException):
            compose_payload(command_override, invalid_override)


def test_removed_paths_default_group_is_rejected_by_validation() -> None:
    from trails_simulate.config import TrainApplicationConfig

    payload = compose_payload("command=train", "paths=default")

    with pytest.raises(ValidationError, match="paths"):
        TrainApplicationConfig.model_validate(payload)


def test_summary_command_config_validates_required_roots(tmp_path: Path) -> None:
    from trails_simulate.config import SummaryApplicationConfig

    payload = compose_payload("command=summary")

    with pytest.raises(ValidationError, match="at least one"):
        SummaryApplicationConfig.model_validate(payload)

    roots_payload = compose_payload("command=summary")
    roots_payload["train_roots"] = [str(tmp_path / "train")]
    roots_payload["baseline_roots"] = [str(tmp_path / "baseline")]
    roots_config = SummaryApplicationConfig.model_validate(roots_payload)

    assert roots_config.train_roots == (tmp_path / "train",)
    assert roots_config.baseline_roots == (tmp_path / "baseline",)
    assert roots_config.metrics == ("acc", "ari", "nmi", "cindex")

    plural_payload = compose_payload("command=summary")
    plural_payload["train_roots"] = [
        str(tmp_path / "train-base"),
        str(tmp_path / "train-mtan"),
    ]
    plural_payload["train_labels"] = ["base", "mtan"]
    plural_config = SummaryApplicationConfig.model_validate(plural_payload)

    assert plural_config.train_roots == (
        tmp_path / "train-base",
        tmp_path / "train-mtan",
    )
    assert plural_config.train_labels == ("base", "mtan")

    bad_label_payload = compose_payload("command=summary")
    bad_label_payload["baseline_roots"] = [
        str(tmp_path / "baseline-a"),
        str(tmp_path / "baseline-b"),
    ]
    bad_label_payload["baseline_labels"] = ["only-one"]
    with pytest.raises(ValidationError, match="baseline_labels"):
        SummaryApplicationConfig.model_validate(bad_label_payload)


def test_simulation_configs_validate() -> None:
    from trails_simulate.config import SimulateApplicationConfig

    expected_features = {
        "quick": 10,
        "base": 10,
        "imbalance": 8,
        "censored": 14,
        "high_dimension": 24,
    }
    for simulation, n_features in expected_features.items():
        app_config = SimulateApplicationConfig.model_validate(
            compose_payload(f"simulation={simulation}")
        )

        assert app_config.name == simulation
        assert len(app_config.train_size) == len(app_config.test_size)
        assert all(value > 0 for value in app_config.train_size)
        assert all(value > 0 for value in app_config.test_size)
        assert all(k > 1 for k in app_config.generator.n_clusters_tuple_)
        assert len(app_config.generator.feature_names) == n_features
        assert "patients" not in cast(
            dict[str, Any], compose_payload(f"simulation={simulation}")["generator"]
        )

    base_config = SimulateApplicationConfig.model_validate(compose_payload("simulation=base"))
    assert base_config.train_size == (500, 1000, 2000, 3000, 5000)
    assert base_config.test_size == (300, 300, 300, 300, 300)
    assert base_config.generator.n_clusters_tuple_ == (2, 3, 4, 5)
    assert base_config.repeats == 5


def test_baseline_config_includes_fpca_and_rejects_duplicate_methods() -> None:
    from trails_simulate.config import BaselineApplicationConfig

    app_config = BaselineApplicationConfig.model_validate(compose_payload("command=baseline"))
    assert app_config.methods == (
        "summary_kmeans",
        "risk_stratified_kmeans",
        "fpca_kmeans",
    )
    assert app_config.seed == 20260517
    assert app_config.fallback_n_clusters == 3

    payload = compose_payload("methods=[summary_kmeans,summary_kmeans]")
    with pytest.raises(ValidationError, match="duplicates"):
        BaselineApplicationConfig.model_validate(payload)


def test_paths_config_defaults_use_composed_output_dir_and_keep_inputs_explicit() -> None:
    from trails_simulate.config import SimulateApplicationConfig, TrainApplicationConfig

    simulate_config = SimulateApplicationConfig.model_validate(compose_payload("simulation=base"))
    train_config = TrainApplicationConfig.model_validate(
        compose_payload("command=train", "paths.data_root=data/simulated/base")
    )
    override_config = TrainApplicationConfig.model_validate(
        compose_payload("command=train", "paths.dir=outputs/train/my-run")
    )

    assert {"root", "prefix", "suffix", "dir"}.issubset(type(simulate_config.paths).model_fields)
    assert simulate_config.paths.root == Path("outputs/simulate")
    assert simulate_config.paths.prefix == "base"
    assert simulate_config.paths.suffix
    assert simulate_config.paths.dir.as_posix().startswith("outputs/simulate/base-")
    assert train_config.paths.data_root == Path("data/simulated/base")
    assert not train_config.paths.explicit_split.enabled
    assert train_config.paths.explicit_split.train_data == Path("data/simulated/train.pt")
    assert train_config.paths.root == Path("outputs/train")
    assert train_config.paths.prefix == train_config.name
    assert train_config.paths.suffix
    assert train_config.paths.dir.as_posix().startswith(f"outputs/train/{train_config.name}-")
    assert override_config.paths.dir == Path("outputs/train/my-run")

    old_payload = compose_payload("simulation=base")
    old_payload["run"] = {"dir": "outputs/simulate/old"}
    with pytest.raises(ValidationError, match="run"):
        SimulateApplicationConfig.model_validate(old_payload)


def test_training_configs_validate() -> None:
    from trails_simulate.config import TrainApplicationConfig

    expected_input = {
        "small": "grud",
        "base": "grud",
        "large": "grud",
        "mtan": "mtan",
    }
    for training, input_kind in expected_input.items():
        app_config = TrainApplicationConfig.model_validate(compose_payload(f"training={training}"))
        assert app_config.model.encoder.input.kind == input_kind
        assert app_config.model.n_clusters > 1
        assert app_config.trainer.batch_size is None
        assert app_config.trainer.valid_size == 0.2
        assert app_config.parallel.workers == 1
        assert app_config.parallel.devices == ()
        assert app_config.parallel.torch_threads is None

    explicit_config = TrainApplicationConfig.model_validate(
        compose_payload(
            "training=base",
            "trainer.batch_size=64",
            "parallel.workers=4",
            "parallel.devices=[cuda:0,cuda:1]",
            "parallel.torch_threads=2",
        )
    )
    assert explicit_config.trainer.batch_size == 64
    assert explicit_config.parallel.workers == 4
    assert explicit_config.parallel.devices == ("cuda:0", "cuda:1")
    assert explicit_config.parallel.torch_threads == 2

    invalid_payload = compose_payload("training=base")
    invalid_payload["parallel"]["workers"] = 0
    with pytest.raises(ValidationError, match="greater than 0"):
        TrainApplicationConfig.model_validate(invalid_payload)


def test_simulation_list_overrides_are_validated() -> None:
    from trails_simulate.config import SimulateApplicationConfig

    app_config = SimulateApplicationConfig.model_validate(
        compose_payload(
            "simulation=quick",
            "train_size=[10]",
            "test_size=[5]",
            "generator.n_clusters=[2]",
        )
    )

    assert app_config.train_size == (10,)
    assert app_config.test_size == (5,)
    assert app_config.generator.n_clusters_tuple_ == (2,)


def test_simulation_rejects_mismatched_train_test_lists() -> None:
    from trails_simulate.config import SimulateApplicationConfig

    payload = compose_payload("simulation=quick")
    payload["train_size"] = [10, 20]
    payload["test_size"] = [5]

    with pytest.raises(ValidationError, match="equal length"):
        SimulateApplicationConfig.model_validate(payload)


def test_simulation_rejects_sample_size_not_larger_than_k() -> None:
    from trails_simulate.config import SimulateApplicationConfig

    payload = compose_payload(
        "simulation=quick",
        "train_size=[2]",
        "test_size=[1]",
        "generator.n_clusters=[3]",
    )

    with pytest.raises(ValidationError, match="greater than every requested K"):
        SimulateApplicationConfig.model_validate(payload)


def test_simulate_command_writes_grid_manifest_and_splits(tmp_path: Path) -> None:
    simulate_script = load_script("simulate")

    from trails.data import ClinicalTimeSeriesDataset
    from trails_simulate.config import SimulateApplicationConfig

    app_config = SimulateApplicationConfig.model_validate(
        compose_payload(
            "simulation=quick",
            "train_size=[6,8]",
            "test_size=[2,4]",
            "generator.n_clusters=[2,3]",
            "repeats=2",
        )
    )

    scenario_root = tmp_path / "run"
    app_config = with_paths_dir(app_config, scenario_root)
    result = simulate_script.run(app_config)
    train_path = scenario_root / "train_6_test_2" / "k2" / "0" / "train.pt"
    test_path = scenario_root / "train_6_test_2" / "k2" / "0" / "test.pt"
    manifest_path = scenario_root / "simulation_manifest.csv"
    summary_path = scenario_root / "simulation_summary.json"

    assert result["command"] == "simulate"
    assert result["data_root"] == str(scenario_root)
    assert result["run_dir"] == str(scenario_root)
    assert len(result["runs"]) == 8
    assert train_path.exists()
    assert test_path.exists()
    assert manifest_path.exists()
    assert summary_path.exists()
    assert result["outputs"] == {
        "manifest": str(manifest_path),
        "summary": str(summary_path),
    }

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
    from trails_simulate.config import TrainApplicationConfig
    from trails_simulate.path import discover_dataset_runs

    data_root = tmp_path / "data"
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k2" / "0", n_clusters=2, seed=43)
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k3" / "0", n_clusters=3, seed=44)
    app_config = TrainApplicationConfig.model_validate(
        compose_payload("command=train", f"paths.data_root={data_root}")
    )

    runs = discover_dataset_runs(app_config)

    assert [run.run_id for run in runs] == [
        "base/train_6_test_2/k2/0",
        "base/train_6_test_2/k3/0",
    ]


def test_dataset_discovery_resolves_relative_inputs_from_cwd(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from trails_simulate.config import TrainApplicationConfig
    from trails_simulate.path import discover_dataset_runs

    write_synthetic_split(
        tmp_path / "relative-data" / "base" / "train_6_test_2" / "k2" / "0",
        n_clusters=2,
        seed=43,
    )
    app_config = TrainApplicationConfig.model_validate(
        compose_payload("command=train", "paths.data_root=relative-data")
    )

    monkeypatch.chdir(tmp_path)
    runs = discover_dataset_runs(app_config)

    assert runs[0].train_data == (
        tmp_path / "relative-data" / "base" / "train_6_test_2" / "k2" / "0" / "train.pt"
    )
    assert runs[0].test_data == (
        tmp_path / "relative-data" / "base" / "train_6_test_2" / "k2" / "0" / "test.pt"
    )


def test_train_command_runs_all_discovered_splits_and_uses_dataset_k(
    monkeypatch,
    caplog,
    tmp_path: Path,
) -> None:
    train_script = load_script("train")

    from trails.data import ClinicalTimeSeriesDataset
    from trails_simulate import train_jobs
    from trails_simulate.config import TrainApplicationConfig
    from trails_simulate.evaluation import prediction_payload_from_dataset
    from trails_simulate.training import TrainResult

    data_root = tmp_path / "data"
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k2" / "0", n_clusters=2, seed=43)
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k3" / "0", n_clusters=3, seed=44)
    seen_clusters: list[int] = []

    app_config = TrainApplicationConfig.model_validate(
        compose_payload(
            "command=train",
            "training=small",
            f"paths.data_root={data_root}",
            "artifacts.names=[none]",
        )
    )

    def fake_fit_training_run(*args: Any, **kwargs: Any) -> TrainResult:
        run_config = args[0]
        train_paths = kwargs["train_paths"]
        seen_clusters.append(run_config.model.n_clusters)
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

    monkeypatch.setattr(train_jobs, "fit_training_run", fake_fit_training_run)
    caplog.set_level(logging.INFO)

    app_config = with_paths_dir(app_config, tmp_path / "run")
    result = train_script.run(app_config)

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

    train_script = load_script("train")

    from trails.data import ClinicalTimeSeriesDataset
    from trails_simulate import train_jobs
    from trails_simulate.config import TrainApplicationConfig
    from trails_simulate.evaluation import prediction_payload_from_dataset
    from trails_simulate.training import TrainResult

    data_root = tmp_path / "data"
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k2" / "0", n_clusters=2, seed=43)
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k3" / "0", n_clusters=3, seed=44)
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k4" / "0", n_clusters=4, seed=45)
    seen_devices: list[str] = []
    submitted_slots: list[int] = []
    seen_max_workers: list[int] = []

    app_config = TrainApplicationConfig.model_validate(
        compose_payload(
            "command=train",
            "training=small",
            f"paths.data_root={data_root}",
            "artifacts.names=[none]",
            "parallel.workers=2",
            "parallel.devices=[cuda:0,cuda:1]",
        )
    )

    def fake_fit_training_run(*args: Any, **kwargs: Any) -> TrainResult:
        run_config = args[0]
        train_paths = kwargs["train_paths"]
        seen_devices.append(run_config.trainer.device)
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

    monkeypatch.setattr(train_jobs, "fit_training_run", fake_fit_training_run)
    monkeypatch.setattr(train_jobs, "ProcessPoolExecutor", FakeExecutor)

    app_config = with_paths_dir(app_config, tmp_path / "run")
    result = train_script.run(app_config)

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
    from trails_simulate.config import TrainApplicationConfig
    from trails_simulate.train_jobs import config_with_training_device, device_for_worker_slot

    same_device_config = TrainApplicationConfig.model_validate(
        compose_payload("training=small", "trainer.device=cuda:0")
    )
    assert device_for_worker_slot(same_device_config, 0) is None
    assert same_device_config.trainer.device == "cuda:0"

    rotated_config = TrainApplicationConfig.model_validate(
        compose_payload("training=small", "parallel.devices=[cuda:0,cuda:1]")
    )
    assert device_for_worker_slot(rotated_config, 0) == "cuda:0"
    assert device_for_worker_slot(rotated_config, 1) == "cuda:1"
    assert device_for_worker_slot(rotated_config, 2) == "cuda:0"
    assert config_with_training_device(rotated_config, "cuda:1").trainer.device == "cuda:1"


def test_completed_train_run_log_omits_timing_fields() -> None:
    from trails_simulate.command_utils import format_completed_train_run

    completed_message = format_completed_train_run(
        run_id="base/train_500_test_300/k3/0",
        n_clusters=3,
        seed=7,
        prediction_path=Path("trails.pt"),
        metrics={"cindex": 0.9, "ari": 0.5},
    )
    assert "Completed train run: base/train_500_test_300/k3/0" in completed_message
    assert "duration=" not in completed_message
    assert "elapsed=" not in completed_message
    assert "remaining=" not in completed_message
    assert "cindex=0.9" in completed_message
    assert "ari=0.5" in completed_message
    assert "prediction=trails.pt" in completed_message


def test_fit_training_run_resolves_auto_batch_size_and_records_effective_value(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from trails_simulate import training
    from trails_simulate.config import TrainApplicationConfig
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
    app_config = TrainApplicationConfig.model_validate(
        compose_payload(
            "command=train",
            "training=small",
            "artifacts.names=[config]",
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


def install_fake_tqdm(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[Any]]:
    from trails import progress

    bars: list[Any] = []

    class FakeTqdm:
        _lock: Any = None

        def __init__(self, iterable: Any = None, **kwargs: Any) -> None:
            self.iterable = [] if iterable is None else iterable
            self.kwargs = kwargs
            self.updates: list[int | float] = []
            self.postfixes: list[dict[str, Any]] = []
            self.closed = False
            self.entered = False
            bars.append(self)

        def __iter__(self) -> Any:
            return iter(self.iterable)

        def __enter__(self) -> "FakeTqdm":
            self.entered = True
            return self

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
            self.closed = True

        def update(self, n: int | float = 1) -> bool:
            self.updates.append(n)
            return True

        def set_postfix(
            self,
            ordered_dict: Any = None,
            refresh: bool = True,
            **kwargs: Any,
        ) -> None:
            self.postfixes.append({"ordered_dict": ordered_dict, "refresh": refresh, **kwargs})

        def close(self) -> None:
            self.closed = True

        @classmethod
        def get_lock(cls) -> Any:
            return cls._lock

        @classmethod
        def set_lock(cls, lock: Any) -> None:
            cls._lock = lock

        @staticmethod
        def write(_message: str) -> None:
            return None

    monkeypatch.setattr(progress, "tqdm", FakeTqdm)
    return progress, bars


def test_progress_bar_infers_nested_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    progress, bars = install_fake_tqdm(monkeypatch)

    for _epoch in progress.ProgressBar(range(1), desc="Epoch"):
        list(progress.ProgressBar(range(1), desc="Train"))

    assert bars[0].kwargs["position"] == 0
    assert bars[0].kwargs["leave"] is True
    assert bars[0].kwargs["desc"] == "Epoch"
    assert bars[1].kwargs["position"] == 1
    assert bars[1].kwargs["leave"] is False
    assert bars[1].kwargs["desc"] == "Train"


def test_progress_manager_worker_scope_assigns_depth_grid_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress, bars = install_fake_tqdm(monkeypatch)

    with progress.ProgressManager.worker_scope(
        worker_slot=1,
        workers=2,
        description_prefix="run-a",
    ):
        for _epoch in progress.ProgressBar(range(1), desc="Epoch"):
            list(progress.ProgressBar(range(1), desc="Train"))

    assert bars[0].kwargs["position"] == 2
    assert bars[0].kwargs["leave"] is False
    assert bars[0].kwargs["desc"] == "run-a Epoch"
    assert bars[1].kwargs["position"] == 4
    assert bars[1].kwargs["leave"] is False
    assert bars[1].kwargs["desc"] == "run-a Train"


def test_progress_bar_supports_manual_context_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress, bars = install_fake_tqdm(monkeypatch)

    with progress.ProgressBar(desc="Train splits", total=2) as bar:
        bar.update()
        bar.set_postfix(completed="1/2")

    assert bars[0].kwargs["position"] == 0
    assert bars[0].kwargs["leave"] is True
    assert bars[0].kwargs["desc"] == "Train splits"
    assert bars[0].kwargs["total"] == 2
    assert bars[0].entered is True
    assert bars[0].closed is True
    assert bars[0].updates == [1]
    assert bars[0].postfixes == [{"ordered_dict": None, "refresh": True, "completed": "1/2"}]


def test_progress_logging_shortens_info_to_single_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from os import terminal_size

    from trails import progress

    messages: list[str] = []
    monkeypatch.setattr(progress.tqdm, "write", messages.append)
    monkeypatch.setattr(
        progress.shutil,
        "get_terminal_size",
        lambda fallback: terminal_size((48, 20)),
    )
    handler = progress.TqdmLoggingHandler()
    handler.setFormatter(progress.CompactTqdmFormatter("%(levelname)s:%(name)s:%(message)s"))
    record = logging.LogRecord(
        "tests.progress",
        logging.INFO,
        __file__,
        1,
        "Completed train run:\n%s",
        ("metric=value " * 20,),
        None,
    )

    handler.emit(record)

    assert len(messages) == 1
    assert "\n" not in messages[0]
    assert len(messages[0]) <= 47
    assert messages[0].endswith("...")


def test_progress_logging_prefixes_worker_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trails import progress

    messages: list[str] = []
    monkeypatch.setattr(progress.tqdm, "write", messages.append)
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    try:
        configure_logger = progress.configure_tqdm_logging
        configure_logger()
        with progress.ProgressManager.worker_scope(
            worker_slot=0,
            workers=1,
            description_prefix="run-a",
        ):
            logging.getLogger("tests.progress").info("Resolving\nbatch size to %s", 128)
    finally:
        root_logger.handlers[:] = original_handlers
        root_logger.setLevel(original_level)
        logging.captureWarnings(False)

    assert messages == ["run-a Resolving batch size to 128"]


def test_progress_manager_worker_initargs_include_log_queue() -> None:
    from trails import progress

    with progress.ProgressManager(workers=2) as manager:
        tqdm_lock, workers, log_queue = manager.worker_initargs()

    assert tqdm_lock is not None
    assert workers == 2
    assert log_queue is not None


def test_progress_manager_initialize_worker_uses_queue_logging() -> None:
    import logging.handlers
    import queue

    from trails import progress

    log_queue: queue.Queue[Any] = queue.Queue()
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    try:
        progress.ProgressManager.initialize_worker(progress.tqdm.get_lock(), 2, log_queue)
        handlers = list(root_logger.handlers)

        assert any(isinstance(handler, logging.handlers.QueueHandler) for handler in handlers)
        assert not any(isinstance(handler, progress.TqdmLoggingHandler) for handler in handlers)
        assert not any(
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
            for handler in handlers
        )

        with progress.ProgressManager.worker_scope(
            worker_slot=1,
            description_prefix="run-b",
        ):
            logging.getLogger("tests.progress.worker").info("hello")
        record = log_queue.get_nowait()
    finally:
        root_logger.handlers[:] = original_handlers
        root_logger.setLevel(original_level)
        logging.captureWarnings(False)

    assert record.progress_description_prefix == "run-b"
    assert record.getMessage() == "hello"


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
    baseline_script = load_script("baseline")

    from trails_simulate.config import BaselineApplicationConfig

    data_root = tmp_path / "data"
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k2" / "0", n_clusters=2, seed=43)
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k3" / "0", n_clusters=3, seed=44)

    app_config = BaselineApplicationConfig.model_validate(
        compose_payload("command=baseline", f"paths.data_root={data_root}")
    )

    app_config = with_paths_dir(app_config, tmp_path / "run")
    result = baseline_script.run(app_config)

    assert result["command"] == "baseline"
    assert [run["n_clusters"] for run in result["runs"]] == [2, 3]
    assert (
        tmp_path / "run" / "base" / "train_6_test_2" / "k2" / "0" / "summary_kmeans.pt"
    ).exists()
    assert (tmp_path / "run" / "baseline_summary.json").exists()
    assert (tmp_path / "run" / "baseline_metrics.csv").exists()


def test_optim_trial_config_updates_training_namespace() -> None:
    from trails_simulate.config import OptimApplicationConfig
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

    app_config = OptimApplicationConfig.model_validate(
        compose_payload("command=optim", "training=base")
    )
    trial_config = optim_trial_config(app_config, FakeTrial())

    assert trial_config.artifacts.names == ("none",)
    assert not trial_config.diagnostics.latent_embeddings.enabled
    assert not trial_config.swanlab.enabled
    assert trial_config.model.encoder.input.kind == "grud"
    assert trial_config.trainer.batch_size == app_config.optim.search.batch_size[0]
    assert app_config.model.n_clusters == 4


def test_summary_command_combines_metrics_and_writes_figures(tmp_path: Path) -> None:
    summary_script = load_script("summary")

    from trails_simulate.config import SummaryApplicationConfig

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
    payload = compose_payload("command=summary", "metrics=[cindex,ari,nmi]")
    payload["train_roots"] = [str(train_root)]
    payload["baseline_roots"] = [str(baseline_root)]
    app_config = SummaryApplicationConfig.model_validate(payload)

    app_config = with_paths_dir(app_config, tmp_path / "run")
    result = summary_script.run(app_config)

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
    summary_script = load_script("summary")

    from trails_simulate.config import SummaryApplicationConfig

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

    payload = compose_payload("command=summary", "metrics=[cindex,ari]")
    payload["train_roots"] = [str(train_base), str(train_mtan)]
    payload["baseline_roots"] = [str(baseline_classic), str(baseline_fpca)]
    payload["train_labels"] = ["base", "mtan"]
    payload["baseline_labels"] = ["classic", "fpca"]
    app_config = SummaryApplicationConfig.model_validate(payload)

    app_config = with_paths_dir(app_config, tmp_path / "run")
    result = summary_script.run(app_config)

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
    summary_script = load_script("summary")

    from trails_simulate.config import SummaryApplicationConfig

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
    payload = compose_payload("command=summary", "metrics=[cindex,ari]")
    payload["train_roots"] = [str(train_root)]
    payload["baseline_roots"] = [str(baseline_root)]
    app_config = SummaryApplicationConfig.model_validate(payload)

    app_config = with_paths_dir(app_config, tmp_path / "run")
    result = summary_script.run(app_config)

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
    optim_script = load_script("optim")

    from trails_simulate import optim
    from trails_simulate.config import OptimApplicationConfig

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
    app_config = OptimApplicationConfig.model_validate(
        compose_payload(
            "command=optim",
            "optim.n_trials=1",
            f"paths.data_root={data_root}",
            "optim.run_ids=[base/train_6_test_2/k3/0]",
            f"optim.storage={tmp_path / 'shared.db'}",
        )
    )
    fake_optuna = FakeOptuna()
    monkeypatch.setattr(optim_script, "load_optuna", lambda: fake_optuna)

    def fake_run_optim_split_job(job: optim.OptimSplitJob) -> optim.OptimSplitResult:
        return optim.OptimSplitResult(
            metrics={"ari": 0.2, "cindex": 0.7},
            run_id=job.run_paths.run_id,
            seed=job.seed,
            split_index=job.split_index,
            trial_number=job.trial_number,
        )

    monkeypatch.setattr(optim, "run_optim_split_job", fake_run_optim_split_job)

    app_config = with_paths_dir(app_config, tmp_path / "run")
    result = optim_script.run(app_config)

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
    optim_script = load_script("optim")

    from trails_simulate import optim
    from trails_simulate.config import OptimApplicationConfig

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
    _progress, bars = install_fake_tqdm(monkeypatch)
    app_config = OptimApplicationConfig.model_validate(
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

    monkeypatch.setattr(optim_script, "load_optuna", lambda: FakeOptuna())
    monkeypatch.setattr(optim, "run_optim_split_job", fake_run_optim_split_job)

    app_config = with_paths_dir(app_config, tmp_path / "run")
    result = optim_script.run(app_config)

    optim_bars = [bar for bar in bars if bar.kwargs["desc"] == "Optim splits"]
    assert len(optim_bars) == 1
    assert optim_bars[0].kwargs["total"] == 2
    assert optim_bars[0].updates == [1, 1]
    assert optim_bars[0].postfixes[-1] == {
        "ordered_dict": None,
        "refresh": True,
        "completed": "2/2",
        "trials": "1/1",
    }
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

    optim_script = load_script("optim")

    from trails_simulate import optim
    from trails_simulate.config import OptimApplicationConfig

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
    app_config = OptimApplicationConfig.model_validate(
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
    submitted_worker_slots: list[int] = []
    _progress, bars = install_fake_tqdm(monkeypatch)

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
            submitted_worker_slots.append(job.worker_slot)
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

    monkeypatch.setattr(optim_script, "load_optuna", lambda: FakeOptuna())
    monkeypatch.setattr(optim, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(optim, "run_optim_split_job", fake_run_optim_split_job)

    app_config = with_paths_dir(app_config, tmp_path / "run")
    result = optim_script.run(app_config)

    optim_bars = [bar for bar in bars if bar.kwargs["desc"] == "Optim splits"]
    assert len(optim_bars) == 1
    assert optim_bars[0].kwargs["total"] == 4
    assert optim_bars[0].updates == [1, 1, 1, 1]
    assert optim_bars[0].postfixes[-1] == {
        "ordered_dict": None,
        "refresh": True,
        "completed": "4/4",
        "trials": "2/2",
    }
    assert seen_max_workers == [2]
    assert len(submitted_devices) == 4
    assert submitted_devices == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    assert sorted(set(submitted_trials)) == [0, 1]
    assert submitted_worker_slots == [0, 1, 0, 1]
    assert result["completed_after"] == 2


def test_optim_resume_rejects_changed_dataset_fingerprint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    optim_script = load_script("optim")

    from trails_simulate import optim
    from trails_simulate.config import OptimApplicationConfig

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
    app_config = OptimApplicationConfig.model_validate(
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

    monkeypatch.setattr(optim_script, "load_optuna", lambda: FakeOptuna())
    monkeypatch.setattr(optim, "run_optim_split_job", fake_run_optim_split_job)
    app_config = with_paths_dir(app_config, tmp_path / "run")
    optim_script.run(app_config)

    write_synthetic_split(split_root, n_clusters=2, seed=99)
    resume_config = OptimApplicationConfig.model_validate(
        compose_payload(
            "command=optim",
            "optim.n_trials=1",
            "optim.resume=true",
            f"paths.data_root={data_root}",
        )
    )
    with pytest.raises(ValueError, match="fingerprint"):
        resume_config = with_paths_dir(resume_config, tmp_path / "run")
        optim_script.run(resume_config)


def test_paths_reject_removed_validation_data_field() -> None:
    from trails_simulate.config import TrainApplicationConfig

    payload = compose_payload("command=train")
    paths = dict(cast(dict[str, Any], payload["paths"]))
    paths["val_data"] = "data/val.pt"
    payload["paths"] = paths

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainApplicationConfig.model_validate(payload)


def test_simulation_rejects_removed_seed_stride_field() -> None:
    from trails_simulate.config import SimulateApplicationConfig

    payload = compose_payload()
    payload["seed_stride"] = 100

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SimulateApplicationConfig.model_validate(payload)


def test_generator_config_rejects_removed_patients_field() -> None:
    from trails_simulate.config import SimulateApplicationConfig

    payload = compose_payload()
    generator = dict(cast(dict[str, Any], payload["generator"]))
    generator["patients"] = 10
    payload["generator"] = generator

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SimulateApplicationConfig.model_validate(payload)
