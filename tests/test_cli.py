import json
import os
import subprocess
import sys
from pathlib import Path

import torch
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
    assert "scenario: debug, formal_5x, quick" in result.stdout


def test_scenario_configs_validate() -> None:
    from main import ApplicationConfig

    config_dir = str((ROOT / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        for scenario in ("quick", "debug", "formal_5x"):
            cfg = compose(config_name="config", overrides=[f"scenario={scenario}"])
            payload = OmegaConf.to_container(cfg, resolve=True)
            app_config = ApplicationConfig.model_validate(payload)
            assert app_config.experiment.name
            assert app_config.simulator.split_patients is not None
            assert app_config.diagnostics.latent_embeddings.enabled == (scenario == "debug")


def test_simulate_command_generates_train_val_test_splits(tmp_path: Path) -> None:
    data_root = tmp_path / "splits"
    run_dir = tmp_path / "simulate-run"
    stdout = run_main(
        "command=simulate",
        f"paths.data_root={data_root}",
        f"hydra.run.dir={run_dir}",
        *TINY_OVERRIDES,
    )

    assert (data_root / "train.pt").exists()
    assert (data_root / "val.pt").exists()
    assert (data_root / "test.pt").exists()
    assert "TRAILS simulate complete" in stdout
    assert f"Data root: {data_root}" in stdout
    assert "train patients=8 seed=20260517" in stdout
    assert "val   patients=6 seed=20260518" in stdout
    assert "test  patients=4 seed=20260519" in stdout


def test_train_command_uses_generated_splits(tmp_path: Path) -> None:
    data_root = tmp_path / "splits"
    run_dir = tmp_path / "train-run"
    run_main(
        "command=simulate",
        f"paths.data_root={data_root}",
        f"hydra.run.dir={tmp_path / 'simulate-run'}",
        *TINY_OVERRIDES,
    )

    stdout = run_main(
        "command=train",
        f"paths.data_root={data_root}",
        f"hydra.run.dir={run_dir}",
        "artifacts.names=[config,history,test,model,plot]",
        *TINY_OVERRIDES,
    )
    artifact_run = next((run_dir / "train").iterdir())

    assert "TRAILS train complete" in stdout
    assert f"Train data: {data_root / 'train.pt'}" in stdout
    assert f"Validation data: {data_root / 'val.pt'}" in stdout
    assert f"Test data: {data_root / 'test.pt'}" in stdout
    assert "Test metrics:" in stdout
    assert "loss" in stdout
    assert "ari" in stdout
    assert (artifact_run / "config.json").exists()
    assert (artifact_run / "history.json").exists()
    assert (artifact_run / "history.csv").exists()
    assert (artifact_run / "test_metrics.json").exists()
    assert (artifact_run / "model.pt").exists()
    assert (artifact_run / "history.png").stat().st_size > 0
    config = json.loads((artifact_run / "config.json").read_text(encoding="utf-8"))
    assert config["paths"]["data"] == str(data_root / "train.pt")
    assert config["paths"]["val_data"] == str(data_root / "val.pt")
    assert config["paths"]["test_data"] == str(data_root / "test.pt")
    assert config["paths"]["test_data_used"] == str(data_root / "test.pt")


def test_train_command_saves_latent_embedding_diagnostics(tmp_path: Path) -> None:
    data_root = tmp_path / "splits"
    run_dir = tmp_path / "train-diagnostics-run"
    fake_umap_root = write_fake_umap_module(tmp_path / "fake-umap")
    run_main(
        "command=simulate",
        f"paths.data_root={data_root}",
        f"hydra.run.dir={tmp_path / 'simulate-diagnostics-run'}",
        *TINY_OVERRIDES,
    )

    stdout = run_main(
        "command=train",
        f"paths.data_root={data_root}",
        f"hydra.run.dir={run_dir}",
        "artifacts.names=[none]",
        "diagnostics.latent_embeddings.enabled=true",
        *TINY_OVERRIDES,
        env=pythonpath_env(fake_umap_root),
    )
    artifact_run = next((run_dir / "train").iterdir())
    embedding_dir = artifact_run / "latent_embeddings"

    assert "TRAILS train complete" in stdout
    assert f"Artifacts: {artifact_run}" in stdout
    for split, n_samples in {"train": 8, "val": 6, "test": 4}.items():
        data_path = embedding_dir / f"{split}_embeddings.pt"
        plot_path = embedding_dir / f"{split}_pca_umap.png"

        assert data_path.exists()
        assert plot_path.stat().st_size > 0
        payload = torch.load(data_path, map_location="cpu", weights_only=False)
        assert payload["split"] == split
        assert payload["z"].shape == (n_samples, 4)
        assert payload["pred_cluster"].shape == (n_samples,)
        assert payload["true_cluster"].shape == (n_samples,)


def test_train_artifacts_none_skips_train_artifact_dir(tmp_path: Path) -> None:
    data_root = tmp_path / "splits"
    run_dir = tmp_path / "train-none-run"
    run_main(
        "command=simulate",
        f"paths.data_root={data_root}",
        f"hydra.run.dir={tmp_path / 'simulate-none-run'}",
        *TINY_OVERRIDES,
    )

    stdout = run_main(
        "command=train",
        f"paths.data_root={data_root}",
        f"hydra.run.dir={run_dir}",
        "artifacts.names=[none]",
        *TINY_OVERRIDES,
    )

    assert "TRAILS train complete" in stdout
    assert "Artifacts: not saved" in stdout
    assert not (run_dir / "train").exists()


def test_experiment_repeats_generate_data_train_and_metric_summaries(tmp_path: Path) -> None:
    run_dir = tmp_path / "experiment-run"
    stdout = run_main(
        "command=experiment",
        "experiment.repeats=2",
        "experiment.seed=101",
        "experiment.seed_stride=10",
        "artifacts.names=[config,test]",
        f"hydra.run.dir={run_dir}",
        *TINY_OVERRIDES,
    )
    payload = json.loads((run_dir / "experiment_summary.json").read_text(encoding="utf-8"))

    assert "TRAILS experiment complete" in stdout
    assert f"Hydra run: {run_dir}" in stdout
    assert "Repeats: 2" in stdout
    assert "Seeds: 101, 111" in stdout
    assert "Metric summary:" in stdout
    assert "Repeat results:" in stdout
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


def write_fake_umap_module(root: Path) -> Path:
    package = root / "umap"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "\n".join(
            [
                "import numpy as np",
                "",
                "class UMAP:",
                "    def __init__(self, n_components=2, n_neighbors=15, random_state=None):",
                "        self.n_components = n_components",
                "",
                "    def fit_transform(self, x):",
                "        array = np.asarray(x, dtype=float)",
                "        if array.shape[1] >= 2:",
                "            return array[:, :2]",
                "        if array.shape[1] == 1:",
                "            return np.column_stack([array[:, 0], np.zeros(array.shape[0])])",
                "        return np.zeros((array.shape[0], 2))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return root


def pythonpath_env(path: Path) -> dict[str, str]:
    existing = os.environ.get("PYTHONPATH")
    paths = [str(path)]
    if existing:
        paths.append(existing)
    return {"PYTHONPATH": os.pathsep.join(paths)}


def run_main(*overrides: str, env: dict[str, str] | None = None) -> str:
    subprocess_env = None if env is None else {**os.environ, **env}
    result = subprocess.run(
        [sys.executable, "main.py", *overrides],
        check=True,
        capture_output=True,
        text=True,
        env=subprocess_env,
    )
    return result.stdout
