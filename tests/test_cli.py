import json
import subprocess
import sys
from pathlib import Path


def test_main_help() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "TRAILS research commands" in result.stdout


def test_simulate_and_train_cli(tmp_path: Path) -> None:
    data_path = simulate_dataset(tmp_path)
    run_dir = tmp_path / "runs"

    train = subprocess.run(
        [
            sys.executable,
            "main.py",
            "train",
            "--data",
            str(data_path),
            "--val-data",
            str(data_path),
            "--test-data",
            str(data_path),
            "--epochs",
            "1",
            "--warmup-epochs",
            "1",
            "--batch-size",
            "4",
            "--clusters",
            "2",
            "--encoder-hidden-dim",
            "8",
            "--decoder-hidden-dim",
            "8",
            "--latent-dim",
            "4",
            "--save-dir",
            str(run_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(train.stdout)
    timestamped_runs = list(run_dir.iterdir())
    assert len(timestamped_runs) == 1
    run_path = timestamped_runs[0]

    assert '"history"' in train.stdout
    assert '"test"' in train.stdout
    assert '"val_ari"' in train.stdout
    assert payload["run_dir"] == str(run_path)
    assert (run_path / "config.json").exists()
    assert (run_path / "history.json").exists()
    assert (run_path / "history.csv").exists()
    assert (run_path / "test_metrics.json").exists()
    assert (run_path / "model.pt").exists()
    assert (run_path / "history.png").stat().st_size > 0

    config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
    history = json.loads((run_path / "history.json").read_text(encoding="utf-8"))
    test_metrics = json.loads((run_path / "test_metrics.json").read_text(encoding="utf-8"))
    assert config["paths"]["test_data"] == str(data_path)
    assert config["paths"]["test_data_used"] == str(data_path)
    assert config["artifacts"] == ["config", "history", "model", "plot", "test"]
    assert "global_epoch" in history[-1]
    assert "loss" in test_metrics


def test_train_accepts_explicit_artifact_list(tmp_path: Path) -> None:
    data_path = simulate_dataset(tmp_path)
    run_dir = tmp_path / "explicit-runs"

    subprocess.run(
        [
            sys.executable,
            "main.py",
            "train",
            "--data",
            str(data_path),
            "--epochs",
            "1",
            "--warmup-epochs",
            "1",
            "--batch-size",
            "4",
            "--clusters",
            "2",
            "--encoder-hidden-dim",
            "8",
            "--decoder-hidden-dim",
            "8",
            "--latent-dim",
            "4",
            "--save-dir",
            str(run_dir),
            "--save-artifacts",
            "config",
            "history",
            "test",
            "model",
            "plot",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run_path = next(run_dir.iterdir())
    assert (run_path / "config.json").exists()
    assert (run_path / "history.json").exists()
    assert (run_path / "history.csv").exists()
    assert (run_path / "test_metrics.json").exists()
    assert (run_path / "model.pt").exists()
    assert (run_path / "history.png").exists()


def test_train_save_artifacts_none_skips_run_dir(tmp_path: Path) -> None:
    data_path = simulate_dataset(tmp_path)
    run_dir = tmp_path / "runs-none"

    train = subprocess.run(
        [
            sys.executable,
            "main.py",
            "train",
            "--data",
            str(data_path),
            "--epochs",
            "1",
            "--warmup-epochs",
            "1",
            "--batch-size",
            "4",
            "--clusters",
            "2",
            "--encoder-hidden-dim",
            "8",
            "--decoder-hidden-dim",
            "8",
            "--latent-dim",
            "4",
            "--save-dir",
            str(run_dir),
            "--save-artifacts",
            "none",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(train.stdout)
    assert payload["run_dir"] is None
    assert not run_dir.exists()


def simulate_dataset(tmp_path: Path) -> Path:
    data_path = tmp_path / "demo.pt"
    simulate = subprocess.run(
        [
            sys.executable,
            "main.py",
            "simulate",
            "--out",
            str(data_path),
            "--patients",
            "8",
            "--clusters",
            "2",
            "--min-visits",
            "3",
            "--max-visits",
            "4",
            "--hidden-size",
            "12",
            "--latent-dim",
            "4",
            "--attention-layers",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert data_path.exists()
    assert '"censoring_rate"' in simulate.stdout
    assert '"n_patients": 8' in simulate.stdout
    return data_path
