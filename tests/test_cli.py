from pathlib import Path
from typing import Any, cast

import pytest
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

        assert app_config.experiment.name
        assert app_config.experiment.train_size > 0
        assert app_config.experiment.test_size > 0
        assert app_config.simulator.patients == app_config.experiment.train_size
        assert app_config.diagnostics.latent_embeddings.enabled == (scenario == "debug")
        assert "seed_stride" not in payload["experiment"]
        assert "val_data" not in payload["paths"]

        if scenario == "optim":
            assert app_config.command == "optim"
            assert app_config.artifacts.names == ("none",)
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
            assert not app_config.swanlab.enabled


def test_quick_config_uses_small_experiment_sizes() -> None:
    from trails_simulate.config import ApplicationConfig

    app_config = ApplicationConfig.model_validate(compose_payload("scenario=quick"))

    assert app_config.experiment.train_size == 64
    assert app_config.experiment.test_size == 24
    assert app_config.simulator.patients == 64
    assert app_config.trainer.valid_size == 0.2


def test_experiment_size_overrides_drive_simulator_patient_count() -> None:
    from trails_simulate.config import ApplicationConfig

    app_config = ApplicationConfig.model_validate(
        compose_payload(
            "scenario=quick",
            "experiment.train_size=10",
            "experiment.test_size=5",
        )
    )

    assert app_config.experiment.train_size == 10
    assert app_config.experiment.test_size == 5
    assert app_config.simulator.patients == 10


def test_paths_reject_removed_validation_data_field() -> None:
    from trails_simulate.config import ApplicationConfig

    payload = compose_payload("scenario=quick")
    paths = dict(cast(dict[str, Any], payload["paths"]))
    paths["val_data"] = "data/val.pt"
    payload["paths"] = paths

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ApplicationConfig.model_validate(payload)


def test_experiment_rejects_removed_seed_stride_field() -> None:
    from trails_simulate.config import ApplicationConfig

    payload = compose_payload("scenario=quick")
    experiment = dict(cast(dict[str, Any], payload["experiment"]))
    experiment["seed_stride"] = 100
    payload["experiment"] = experiment

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ApplicationConfig.model_validate(payload)
