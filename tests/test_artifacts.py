import csv
from pathlib import Path

from trails.artifacts import flatten_history, plot_history, save_history_csv
from trails.trainer import HistoryEntry


def nested_history() -> list[HistoryEntry]:
    return [
        {
            "epoch": 1,
            "global_epoch": 1,
            "stage": "warmup",
            "train": {
                "loss": 3.0,
                "reconstruction_loss": 2.0,
                "survival_loss": 1.0,
                "vade_kl_loss": 0.0,
                "reconstruction_loss_weight": 0.9,
            },
            "valid": {
                "loss": 2.5,
                "reconstruction_loss": 1.6,
                "survival_loss": 0.9,
                "vade_kl_loss": 0.0,
                "cindex": 0.55,
            },
        },
        {
            "epoch": 1,
            "global_epoch": 2,
            "stage": "vade",
            "best_global_epoch": 2,
            "best_monitor": "valid/cindex",
            "best_monitor_value": 0.62,
            "train": {
                "loss": 2.0,
                "reconstruction_loss": 1.1,
                "survival_loss": 0.7,
                "vade_kl_loss": 0.2,
                "cindex": 0.6,
                "ari": 0.1,
            },
            "valid": {
                "loss": 1.8,
                "reconstruction_loss": 1.0,
                "survival_loss": 0.6,
                "vade_kl_loss": 0.2,
                "cindex": 0.62,
                "ari": 0.2,
            },
        },
    ]


def test_flatten_history_expands_train_and_valid_metrics() -> None:
    rows = flatten_history(nested_history())

    assert rows[0]["loss"] == 3.0
    assert rows[0]["val_loss"] == 2.5
    assert rows[0]["reconstruction_loss"] == 2.0
    assert rows[0]["val_reconstruction_loss"] == 1.6
    assert rows[0]["val_cindex"] == 0.55
    assert rows[1]["cindex"] == 0.6
    assert rows[1]["val_cindex"] == 0.62
    assert "train" not in rows[0]
    assert "valid" not in rows[0]


def test_save_history_csv_writes_analysis_ready_columns(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"

    save_history_csv(path, nested_history())

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["loss"] == "3.0"
    assert rows[0]["val_loss"] == "2.5"
    assert rows[0]["reconstruction_loss"] == "2.0"
    assert rows[0]["val_reconstruction_loss"] == "1.6"
    assert rows[0]["reconstruction_loss_weight"] == "0.9"
    assert rows[0]["val_cindex"] == "0.55"
    assert "train" not in rows[0]
    assert "valid" not in rows[0]


def test_plot_history_uses_flattened_nested_metrics(tmp_path: Path) -> None:
    path = tmp_path / "history.png"

    plot_history(path, nested_history())

    assert path.exists()
    assert path.stat().st_size > 0
