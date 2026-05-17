import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TINY_OVERRIDES = [
    "scenario=quick",
    "simulator.split_patients=[8,6,4]",
    "simulator.n_clusters=2",
    "simulator.min_visits=3",
    "simulator.max_visits=4",
    "simulator.hidden_size=12",
    "simulator.latent_dim=4",
    "simulator.attention_layers=2",
    "simulator.attention_heads=2",
    "model.n_clusters=2",
    "model.encoder_hidden_dim=8",
    "model.decoder_hidden_dim=8",
    "model.latent_dim=4",
    "trainer.max_epochs=1",
    "trainer.warmup_epochs=0",
    "trainer.batch_size=4",
    "trainer.gmm_init_iters=1",
    "swanlab.enabled=false",
]


def test_main_help() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "powered by Hydra" in result.stdout
    assert "scenario: formal_5x, normal_swanlab, quick" in result.stdout


def test_scenario_configs_validate() -> None:
    from main import ApplicationConfig

    config_dir = str((ROOT / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        for scenario in ("quick", "normal_swanlab", "formal_5x"):
            cfg = compose(config_name="config", overrides=[f"scenario={scenario}"])
            payload = OmegaConf.to_container(cfg, resolve=True)
            app_config = ApplicationConfig.model_validate(payload)
            assert app_config.experiment.name
            assert app_config.simulator.split_patients is not None


def test_simulate_command_generates_train_val_test_splits(tmp_path: Path) -> None:
    data_root = tmp_path / "splits"
    run_dir = tmp_path / "simulate-run"
    payload = run_main(
        "command=simulate",
        f"paths.data_root={data_root}",
        f"hydra.run.dir={run_dir}",
        *TINY_OVERRIDES,
    )

    assert (data_root / "train.pt").exists()
    assert (data_root / "val.pt").exists()
    assert (data_root / "test.pt").exists()
    assert payload["command"] == "simulate"
    assert payload["out_dir"] == str(data_root)
    assert payload["split_patients"] == {"test": 4, "train": 8, "val": 6}
    assert payload["splits"]["train"]["seed"] == 20260517
    assert payload["splits"]["val"]["seed"] == 20260518
    assert payload["splits"]["test"]["seed"] == 20260519
    assert payload["splits"]["train"]["n_patients"] == 8


def test_train_command_uses_generated_splits(tmp_path: Path) -> None:
    data_root = tmp_path / "splits"
    run_dir = tmp_path / "train-run"
    run_main(
        "command=simulate",
        f"paths.data_root={data_root}",
        f"hydra.run.dir={tmp_path / 'simulate-run'}",
        *TINY_OVERRIDES,
    )

    payload = run_main(
        "command=train",
        f"paths.data_root={data_root}",
        f"hydra.run.dir={run_dir}",
        "artifacts.names=[config,history,test,model,plot]",
        *TINY_OVERRIDES,
    )
    artifact_run = Path(payload["run_dir"])

    assert payload["command"] == "train"
    assert payload["paths"]["data"] == str(data_root / "train.pt")
    assert payload["paths"]["val_data"] == str(data_root / "val.pt")
    assert payload["paths"]["test_data"] == str(data_root / "test.pt")
    assert payload["paths"]["test_data_used"] == str(data_root / "test.pt")
    assert "history" in payload
    assert "test" in payload
    assert "ari" in payload["test"]
    assert (artifact_run / "config.json").exists()
    assert (artifact_run / "history.json").exists()
    assert (artifact_run / "history.csv").exists()
    assert (artifact_run / "test_metrics.json").exists()
    assert (artifact_run / "model.pt").exists()
    assert (artifact_run / "history.png").stat().st_size > 0


def test_train_artifacts_none_skips_train_artifact_dir(tmp_path: Path) -> None:
    data_root = tmp_path / "splits"
    run_dir = tmp_path / "train-none-run"
    run_main(
        "command=simulate",
        f"paths.data_root={data_root}",
        f"hydra.run.dir={tmp_path / 'simulate-none-run'}",
        *TINY_OVERRIDES,
    )

    payload = run_main(
        "command=train",
        f"paths.data_root={data_root}",
        f"hydra.run.dir={run_dir}",
        "artifacts.names=[none]",
        *TINY_OVERRIDES,
    )

    assert payload["run_dir"] is None
    assert not (run_dir / "train").exists()


def test_experiment_repeats_generate_data_train_and_metric_summaries(tmp_path: Path) -> None:
    run_dir = tmp_path / "experiment-run"
    payload = run_main(
        "command=experiment",
        "experiment.repeats=2",
        "experiment.seed=101",
        "experiment.seed_stride=10",
        "artifacts.names=[config,test]",
        f"hydra.run.dir={run_dir}",
        *TINY_OVERRIDES,
    )

    assert payload["command"] == "experiment"
    assert payload["hydra_run_dir"] == str(run_dir)
    assert [repeat["seed"] for repeat in payload["repeats"]] == [101, 111]
    assert payload["repeats"][0]["splits"]["train"]["seed"] == 101
    assert payload["repeats"][0]["splits"]["val"]["seed"] == 102
    assert payload["repeats"][0]["splits"]["test"]["seed"] == 103
    assert payload["repeats"][1]["splits"]["train"]["seed"] == 111

    for index in range(2):
        repeat_dir = run_dir / f"repeat_{index:03d}"
        assert (repeat_dir / "data" / "train.pt").exists()
        assert (repeat_dir / "data" / "val.pt").exists()
        assert (repeat_dir / "data" / "test.pt").exists()
        train_run_dir = Path(payload["repeats"][index]["train_run_dir"])
        assert train_run_dir.exists()
        assert (train_run_dir / "config.json").exists()
        assert (train_run_dir / "test_metrics.json").exists()

    assert (run_dir / "experiment_summary.json").exists()
    assert (run_dir / "test_metrics.csv").exists()
    assert (run_dir / "test_metrics_summary.json").exists()
    assert "loss" in payload["metrics_summary"]


def run_main(*overrides: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "main.py", *overrides],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)
