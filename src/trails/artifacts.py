from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

from .trainer import HistoryEntry

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ARTIFACT_NAMES = frozenset({"config", "history", "test", "model", "plot"})
ARTIFACT_TOKENS = tuple(sorted((*ARTIFACT_NAMES, "all", "none")))


def resolve_artifact_names(tokens: Sequence[str] | None) -> frozenset[str]:
    if tokens is None or len(tokens) == 0 or "all" in tokens:
        if "none" in (tokens or ()):
            raise ValueError("--save-artifacts cannot combine 'all' and 'none'.")
        return ARTIFACT_NAMES
    if "none" in tokens:
        if len(tokens) > 1:
            raise ValueError("--save-artifacts cannot combine 'none' with other artifacts.")
        return frozenset()

    unknown = sorted(set(tokens) - ARTIFACT_NAMES)
    if unknown:
        raise ValueError(f"Unknown artifact name(s): {', '.join(unknown)}.")
    return frozenset(tokens)


def create_timestamped_run_dir(base_dir: str | Path, created_at: datetime | None = None) -> Path:
    timestamp = (created_at or datetime.now().astimezone()).strftime("%Y%m%d-%H%M%S")
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)

    for suffix in range(1000):
        directory_name = timestamp if suffix == 0 else f"{timestamp}-{suffix:02d}"
        candidate = root / directory_name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique run directory under {root}.")


def save_json(path: str | Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_history_csv(path: str | Path, history: Sequence[HistoryEntry]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _history_fieldnames(history)

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in history:
            writer.writerow({name: entry.get(name, "") for name in fieldnames})


def plot_history(path: str | Path, history: Sequence[HistoryEntry]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    panels = _available_panels(history)
    if not panels:
        panels = [("History", [])]

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(10.0, max(3.2, 2.8 * len(panels))),
        sharex=True,
        constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]

    x_values = _x_values(history)
    for ax, (title, names) in zip(axes, panels, strict=True):
        if names:
            for name in names:
                y_values = _metric_values(history, name)
                ax.plot(x_values, y_values, marker="o", linewidth=1.8, markersize=3.5, label=name)
            ax.legend(frameon=False, ncols=min(3, len(names)))
        else:
            ax.text(0.5, 0.5, "No numeric history metrics", ha="center", va="center")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _mark_stage_boundary(ax, history, x_values)

    axes[-1].set_xlabel("Global epoch")
    fig.suptitle("TRAILS training history", fontsize=13, fontweight="bold")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _history_fieldnames(history: Sequence[HistoryEntry]) -> list[str]:
    preferred = [
        "global_epoch",
        "epoch",
        "stage",
        "loss",
        "reconstruction_loss",
        "survival_loss",
        "vade_kl_loss",
        "val_loss",
        "val_reconstruction_loss",
        "val_survival_loss",
        "val_vade_kl_loss",
        "val_c_index",
        "val_ari",
        "val_nmi",
    ]
    present = {key for entry in history for key in entry}
    ordered = [name for name in preferred if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _available_panels(history: Sequence[HistoryEntry]) -> list[tuple[str, list[str]]]:
    panel_candidates = [
        ("Total loss", ["loss", "val_loss"]),
        (
            "Loss components",
            [
                "reconstruction_loss",
                "survival_loss",
                "vade_kl_loss",
                "val_reconstruction_loss",
                "val_survival_loss",
                "val_vade_kl_loss",
            ],
        ),
        ("Validation concordance", ["val_c_index"]),
        ("Cluster recovery", ["val_ari", "val_nmi"]),
    ]
    return [
        (title, [name for name in names if _has_numeric_values(history, name)])
        for title, names in panel_candidates
        if any(_has_numeric_values(history, name) for name in names)
    ]


def _x_values(history: Sequence[HistoryEntry]) -> list[float]:
    if all(isinstance(entry.get("global_epoch"), int | float) for entry in history):
        return [float(entry["global_epoch"]) for entry in history]
    return [float(index + 1) for index in range(len(history))]


def _metric_values(history: Sequence[HistoryEntry], name: str) -> list[float]:
    values: list[float] = []
    for entry in history:
        value = entry.get(name)
        values.append(float(value) if isinstance(value, int | float) else float("nan"))
    return values


def _has_numeric_values(history: Sequence[HistoryEntry], name: str) -> bool:
    return any(isinstance(entry.get(name), int | float) for entry in history)


def _mark_stage_boundary(
    ax: Any,
    history: Sequence[HistoryEntry],
    x_values: Sequence[float],
) -> None:
    stages = [entry.get("stage") for entry in history]
    if not stages or not x_values:
        return

    first_stage = stages[0]
    for index, stage in enumerate(stages[1:], start=1):
        if stage != first_stage:
            boundary = (x_values[index - 1] + x_values[index]) / 2.0
            ax.axvline(boundary, color="0.45", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.text(
                boundary,
                0.98,
                f"{first_stage} -> {stage}",
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=8,
                color="0.35",
            )
            return
