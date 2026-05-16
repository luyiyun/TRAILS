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

    train = subprocess.run(
        [
            sys.executable,
            "main.py",
            "train",
            "--data",
            str(data_path),
            "--val-data",
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"history"' in train.stdout
    assert '"test"' in train.stdout
    assert '"val_ari"' in train.stdout
