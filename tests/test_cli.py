import csv
from pathlib import Path
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


def test_command_enum_rejects_removed_paper_grid() -> None:
    from trails_simulate.config import ApplicationConfig

    payload = compose_payload("command=paper_grid")

    with pytest.raises(ValidationError, match="Input should be"):
        ApplicationConfig.model_validate(payload)


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
    assert base_config.simulation.train_size == (375, 750, 1500, 3750, 7500)
    assert base_config.simulation.test_size == (125, 250, 500, 1250, 2500)
    assert base_config.simulation.generator.n_clusters_tuple_ == (2, 3, 4, 5)
    assert base_config.simulation.repeats == 5


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
        assert app_config.training.trainer.valid_size == 0.2


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

    assert result["command"] == "simulate"
    assert result["data_root"] == str(scenario_root)
    assert len(result["runs"]) == 8
    assert train_path.exists()
    assert test_path.exists()
    assert manifest_path.exists()

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
            metrics={"cindex": 1.0, "cluster_entropy": 0.0},
            prediction=prediction,
            run_dir=None,
        )

    monkeypatch.setattr(workflow, "fit_training_run", fake_fit_training_run)

    result = workflow.run_train_command(app_config, tmp_path / "run", ROOT)

    assert [run["run_id"] for run in result["runs"]] == [
        "base/train_6_test_2/k2/0",
        "base/train_6_test_2/k3/0",
    ]
    assert seen_clusters == [2, 3]
    assert (tmp_path / "run" / "base" / "train_6_test_2" / "k2" / "0" / "trails.pt").exists()
    assert (tmp_path / "run" / "train_summary.json").exists()
    assert (tmp_path / "run" / "train_metrics.csv").exists()


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


def test_optim_command_rejects_shared_storage_for_multiple_splits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from trails_simulate import optim
    from trails_simulate.config import ApplicationConfig

    data_root = tmp_path / "data"
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k2" / "0", n_clusters=2, seed=43)
    write_synthetic_split(data_root / "base" / "train_6_test_2" / "k3" / "0", n_clusters=3, seed=44)
    app_config = ApplicationConfig.model_validate(
        compose_payload(
            "command=optim",
            f"paths.data_root={data_root}",
            f"optim.storage={tmp_path / 'shared.db'}",
        )
    )

    monkeypatch.setattr(optim, "load_optuna", lambda: object())

    with pytest.raises(ValueError, match="one dataset"):
        optim.run_optim_command(app_config, tmp_path / "run", ROOT)


def test_optim_command_runs_independent_studies_for_multiple_splits(
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
            self.state = type("TrialState", (), {"name": "COMPLETE"})()
            self.user_attrs: dict[str, Any] = {}
            self.values: list[float] | None = None

        def set_user_attr(self, name: str, value: Any) -> None:
            self.user_attrs[name] = value

    class FakeStudy:
        def __init__(self, *, storage: str, study_name: str) -> None:
            self.best_trials: list[FakeTrial] = []
            self.storage = storage
            self.study_name = study_name
            self.trials: list[FakeTrial] = []

        def optimize(self, objective: Any, n_trials: int) -> None:
            for number in range(n_trials):
                trial = FakeTrial(number)
                trial.values = list(objective(trial))
                self.trials.append(trial)
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
        )
    )
    fake_optuna = FakeOptuna()
    monkeypatch.setattr(optim, "load_optuna", lambda: fake_optuna)

    def fake_run_optim_trial(trial: FakeTrial, **kwargs: Any) -> tuple[float, float]:
        trial.set_user_attr("seed", kwargs["seed"])
        return 0.7, 0.2

    monkeypatch.setattr(optim, "run_optim_trial", fake_run_optim_trial)

    result = optim.run_optim_command(app_config, tmp_path / "run", ROOT)

    assert len(result["runs"]) == 2
    assert [study.study_name for study in fake_optuna.created] == [
        "optim-base-train_6_test_2-k2-0",
        "optim-base-train_6_test_2-k3-0",
    ]
    assert fake_optuna.created[0].storage != fake_optuna.created[1].storage
    assert fake_optuna.created[0].storage.endswith("base/train_6_test_2/k2/0/study.db")
    assert (tmp_path / "run" / "base" / "train_6_test_2" / "k2" / "0" / "trials.csv").exists()
    assert (tmp_path / "run" / "optim_summary.json").exists()


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
