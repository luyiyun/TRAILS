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


def test_scenario_configs_validate() -> None:
    from trails_simulate.config import ApplicationConfig

    for scenario in ("quick", "debug", "formal_5x", "optim"):
        payload = compose_payload(f"scenario={scenario}")
        app_config = ApplicationConfig.model_validate(payload)

        assert "experiment" not in payload
        for removed_key in (
            "artifacts",
            "diagnostics",
            "model",
            "simulator",
            "swanlab",
            "trainer",
        ):
            assert removed_key not in payload
        assert app_config.simulation.name
        assert app_config.simulation.train_size > 0
        assert app_config.simulation.test_size > 0
        assert app_config.simulation.mechanism_seed == app_config.simulation.seed
        assert app_config.simulation.generator.n_clusters > 1
        assert "patients" not in cast(dict[str, Any], payload["simulation"])["generator"]
        assert app_config.training.model.n_clusters > 1
        assert app_config.training.trainer.valid_size == 0.2
        assert app_config.baseline.methods == ("summary_kmeans", "risk_stratified_kmeans")
        assert app_config.baseline.n_clusters is None
        assert app_config.baseline.kmeans_iters == 50
        assert app_config.training.diagnostics.latent_embeddings.enabled == (scenario == "debug")
        assert "seed_stride" not in payload["simulation"]
        assert "val_data" not in payload["paths"]

        if scenario == "optim":
            assert app_config.command == "optim"
            assert app_config.training.artifacts.names == ("none",)
            assert app_config.optim.search.encoder_input_kind == ("grud", "mtan")
            assert app_config.optim.search.encoder_mapping_kind == (
                "gru",
                "lstm",
                "transformer",
            )
            assert app_config.optim.search.decoder_kind == ("gru", "lstm", "transformer")
            assert app_config.optim.search.decoder_conditioning == (
                "initial_state",
                "concat_time",
            )
            assert app_config.optim.search.hidden_dim == (32, 64, 128)
            assert app_config.optim.search.learning_rate.log
            assert app_config.optim.search.warmup_epochs.high == 5
            assert not app_config.training.swanlab.enabled
        else:
            assert app_config.command == "simulate"


def test_quick_config_uses_small_simulation_sizes() -> None:
    from trails_simulate.config import ApplicationConfig

    app_config = ApplicationConfig.model_validate(compose_payload("scenario=quick"))

    assert app_config.simulation.train_size == 64
    assert app_config.simulation.test_size == 24
    assert app_config.training.trainer.valid_size == 0.2


def test_simulation_size_overrides_are_validated() -> None:
    from trails_simulate.config import ApplicationConfig

    app_config = ApplicationConfig.model_validate(
        compose_payload(
            "scenario=quick",
            "simulation.train_size=10",
            "simulation.test_size=5",
        )
    )

    assert app_config.simulation.train_size == 10
    assert app_config.simulation.test_size == 5


def test_simulate_command_writes_single_train_test_split(tmp_path: Path) -> None:
    from trails.data import ClinicalTimeSeriesDataset
    from trails_simulate import workflow
    from trails_simulate.config import ApplicationConfig

    app_config = ApplicationConfig.model_validate(
        compose_payload(
            "scenario=quick",
            "simulation.train_size=6",
            "simulation.test_size=4",
            f"paths.data_root={tmp_path / 'data'}",
        )
    )

    result = workflow.run_simulate_command(app_config, tmp_path / "run", ROOT)
    data_root = tmp_path / "data"
    train_data = ClinicalTimeSeriesDataset.load(data_root / "train.pt")
    test_data = ClinicalTimeSeriesDataset.load(data_root / "test.pt")

    assert result["command"] == "simulate"
    assert len(train_data) == 6
    assert len(test_data) == 4
    assert (data_root / "simulation_summary.json").exists()
    assert result["repeats"][0]["run_id"] == "single"
    assert result["repeats"][0]["seed"] == app_config.simulation.seed
    assert train_data.metadata["generation_params"]["n_patients"] == 10
    assert train_data.metadata["generation_params"]["mechanism_seed"] == app_config.simulation.seed
    assert train_data.metadata["generation_params"]["sample_seed"] == app_config.simulation.seed
    assert torch.allclose(train_data.metadata["cluster_means"], test_data.metadata["cluster_means"])


def test_simulate_command_writes_repeat_train_test_splits(tmp_path: Path) -> None:
    from trails.data import ClinicalTimeSeriesDataset
    from trails_simulate import workflow
    from trails_simulate.config import ApplicationConfig

    app_config = ApplicationConfig.model_validate(
        compose_payload(
            "scenario=quick",
            "simulation.repeats=2",
            "simulation.train_size=5",
            "simulation.test_size=3",
            f"paths.data_root={tmp_path / 'data'}",
        )
    )

    result = workflow.run_simulate_command(app_config, tmp_path / "run", ROOT)

    assert [repeat["run_id"] for repeat in result["repeats"]] == ["repeat_000", "repeat_001"]
    assert [repeat["seed"] for repeat in result["repeats"]] == [
        app_config.simulation.seed,
        app_config.simulation.seed + 1,
    ]
    repeat_cluster_means = []
    for repeat in ("repeat_000", "repeat_001"):
        train_data = ClinicalTimeSeriesDataset.load(tmp_path / "data" / repeat / "train.pt")
        test_data = ClinicalTimeSeriesDataset.load(tmp_path / "data" / repeat / "test.pt")
        assert len(train_data) == 5
        assert len(test_data) == 3
        assert torch.allclose(
            train_data.metadata["cluster_means"],
            test_data.metadata["cluster_means"],
        )
        repeat_cluster_means.append(train_data.metadata["cluster_means"])
    assert torch.allclose(repeat_cluster_means[0], repeat_cluster_means[1])


def test_train_command_writes_summary_metrics_and_predictions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from trails.data import ClinicalTimeSeriesDataset
    from trails_simulate import workflow
    from trails_simulate.config import ApplicationConfig
    from trails_simulate.evaluation import prediction_payload_from_dataset
    from trails_simulate.generators import (
        ClinicalTimeSeriesDatasetGenerator,
        ClinicalTimeSeriesDatasetGeneratorConfig,
    )
    from trails_simulate.training import TrainResult

    source = ClinicalTimeSeriesDatasetGenerator(
        ClinicalTimeSeriesDatasetGeneratorConfig(
            n_clusters=2,
            min_visits=3,
            max_visits=4,
            hidden_size=12,
            latent_dim=4,
            attention_layers=1,
        )
    ).simulate(n_patients=10, seed=43)
    train_data, test_data = source.split_counts([6, 4], seed=43)
    data_root = tmp_path / "data"
    train_data.save(data_root / "train.pt")
    test_data.save(data_root / "test.pt")

    app_config = ApplicationConfig.model_validate(
        compose_payload(
            "scenario=quick",
            "command=train",
            f"paths.data_root={data_root}",
            "training.model.n_clusters=2",
            "training.artifacts.names=[none]",
        )
    )

    def fake_fit_training_run(*args: Any, **kwargs: Any) -> TrainResult:
        train_paths = kwargs["train_paths"]
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

    assert result["command"] == "train"
    assert (tmp_path / "run" / "train_summary.json").exists()
    assert (tmp_path / "run" / "train_metrics.csv").exists()
    assert (tmp_path / "run" / "predictions" / "single" / "trails.pt").exists()
    assert result["runs"][0]["run_id"] == "single"
    assert result["runs"][0]["metrics"]["cindex"] == 1.0


def test_baseline_command_writes_summary_metrics_and_predictions(tmp_path: Path) -> None:
    from trails_simulate import workflow
    from trails_simulate.config import ApplicationConfig
    from trails_simulate.generators import (
        ClinicalTimeSeriesDatasetGenerator,
        ClinicalTimeSeriesDatasetGeneratorConfig,
    )

    source = ClinicalTimeSeriesDatasetGenerator(
        ClinicalTimeSeriesDatasetGeneratorConfig(
            n_clusters=2,
            min_visits=3,
            max_visits=4,
            hidden_size=12,
            latent_dim=4,
            attention_layers=1,
        )
    ).simulate(n_patients=12, seed=43)
    train_data, test_data = source.split_counts([8, 4], seed=43)
    data_root = tmp_path / "data"
    train_data.save(data_root / "train.pt")
    test_data.save(data_root / "test.pt")

    app_config = ApplicationConfig.model_validate(
        compose_payload(
            "scenario=quick",
            "command=baseline",
            f"paths.data_root={data_root}",
            "simulation.generator.n_clusters=2",
        )
    )

    result = workflow.run_baseline_command(app_config, tmp_path / "run", ROOT)

    assert result["command"] == "baseline"
    assert result["baseline"]["n_clusters_resolved"] == 2
    assert (tmp_path / "run" / "baseline_summary.json").exists()
    assert (tmp_path / "run" / "baseline_metrics.csv").exists()
    methods = {method["method"]: method["metrics"] for method in result["runs"][0]["methods"]}
    assert set(methods) == {"summary_kmeans", "risk_stratified_kmeans"}
    for method in methods:
        assert (tmp_path / "run" / "predictions" / "single" / f"{method}.pt").exists()
    for metrics in methods.values():
        assert "cindex" in metrics
        assert "ari" in metrics
        assert "nmi" in metrics
        assert "cluster_entropy" in metrics


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

    app_config = ApplicationConfig.model_validate(compose_payload("scenario=optim"))
    trial_config = optim_trial_config(app_config, FakeTrial())

    assert trial_config.training.artifacts.names == ("none",)
    assert not trial_config.training.diagnostics.latent_embeddings.enabled
    assert not trial_config.training.swanlab.enabled
    assert trial_config.training.model.encoder.input.kind == "grud"
    assert trial_config.training.trainer.batch_size == app_config.optim.search.batch_size[0]
    assert app_config.training.model.n_clusters == 4


def test_paths_reject_removed_validation_data_field() -> None:
    from trails_simulate.config import ApplicationConfig

    payload = compose_payload("scenario=quick")
    paths = dict(cast(dict[str, Any], payload["paths"]))
    paths["val_data"] = "data/val.pt"
    payload["paths"] = paths

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ApplicationConfig.model_validate(payload)


def test_simulation_rejects_removed_seed_stride_field() -> None:
    from trails_simulate.config import ApplicationConfig

    payload = compose_payload("scenario=quick")
    simulation = dict(cast(dict[str, Any], payload["simulation"]))
    simulation["seed_stride"] = 100
    payload["simulation"] = simulation

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ApplicationConfig.model_validate(payload)


def test_generator_config_rejects_removed_patients_field() -> None:
    from trails_simulate.config import ApplicationConfig

    payload = compose_payload("scenario=quick")
    simulation = dict(cast(dict[str, Any], payload["simulation"]))
    generator = dict(cast(dict[str, Any], simulation["generator"]))
    generator["patients"] = 10
    simulation["generator"] = generator
    payload["simulation"] = simulation

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ApplicationConfig.model_validate(payload)
