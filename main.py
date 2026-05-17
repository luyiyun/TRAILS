from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from trails.artifacts import (
    ARTIFACT_TOKENS,
    create_timestamped_run_dir,
    plot_history,
    resolve_artifact_names,
    save_history_csv,
    save_json,
)
from trails.config import DataConfig, ModelConfig, TrailsConfig, TrainerConfig
from trails.data import ClinicalTimeSeriesDataset
from trails.estimator import TrailsEstimator
from trails_simulate import generate_clinical_time_series_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="TRAILS research commands for simulation and model training.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Generate a synthetic clinical dataset.")
    simulate.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .pt dataset path, or output directory when --split-patients is set.",
    )
    simulate.add_argument("--patients", type=int, default=128, help="Number of patients.")
    simulate.add_argument(
        "--split-patients",
        nargs=3,
        type=int,
        metavar=("TRAIN", "VAL", "TEST"),
        default=None,
        help=(
            "Generate train.pt, val.pt, and test.pt under --out with these patient counts. "
            "Uses seed, seed+1, and seed+2."
        ),
    )
    simulate.add_argument("--clusters", type=int, default=3, help="Number of latent subtypes.")
    simulate.add_argument("--min-visits", type=int, default=4, help="Minimum visits per patient.")
    simulate.add_argument("--max-visits", type=int, default=8, help="Maximum visits per patient.")
    simulate.add_argument("--followup-days", type=float, default=365.0, help="Follow-up horizon.")
    simulate.add_argument(
        "--hidden-size", type=int, default=100, help="Latent profile hidden size."
    )
    simulate.add_argument("--latent-dim", type=int, default=5, help="Latent cluster dimension.")
    simulate.add_argument(
        "--attention-layers", type=int, default=3, help="Number of pseudo-attention layers."
    )
    simulate.add_argument(
        "--attention-heads",
        type=int,
        default=None,
        help="Number of pseudo-attention heads. Defaults to a divisor of hidden size.",
    )
    simulate.add_argument(
        "--censoring-rate", type=float, default=0.3, help="Target random censoring rate."
    )
    simulate.add_argument("--weibull-shape", type=float, default=1.0, help="Weibull shape.")
    simulate.add_argument("--x-low", type=float, default=-10.0, help="Latent mean lower bound.")
    simulate.add_argument("--x-high", type=float, default=10.0, help="Latent mean upper bound.")
    simulate.add_argument("--beta-low", type=float, default=-2.5, help="Survival beta lower bound.")
    simulate.add_argument("--beta-high", type=float, default=2.5, help="Survival beta upper bound.")
    simulate.add_argument("--seed", type=int, default=2026, help="Random seed.")

    train = subparsers.add_parser("train", help="Train the phase-one GRU-D Surv-VaDER model.")
    train.add_argument("--data", type=Path, required=True, help="Input .pt dataset path.")
    train.add_argument(
        "--val-data",
        type=Path,
        default=None,
        help="Optional validation .pt dataset path for per-epoch metrics.",
    )
    train.add_argument(
        "--test-data",
        type=Path,
        default=None,
        help="Optional held-out .pt dataset path for final test metrics.",
    )
    train.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    train.add_argument(
        "--warmup-epochs",
        type=int,
        default=1,
        help="Pretrain epochs before deterministic VaDE mixture initialization.",
    )
    train.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    train.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    train.add_argument("--clusters", type=int, default=3, help="Number of latent subtypes.")
    train.add_argument("--encoder-hidden-dim", type=int, default=32, help="GRU-D hidden size.")
    train.add_argument(
        "--decoder-hidden-dim", type=int, default=32, help="GRU decoder hidden size."
    )
    train.add_argument("--latent-dim", type=int, default=8, help="Patient latent dimension.")
    train.add_argument("--n-layers", type=int, default=1, help="GRU decoder layer count.")
    train.add_argument(
        "--dropout", type=float, default=0.0, help="Decoder dropout for multi-layer GRU."
    )
    train.add_argument("--seed", type=int, default=2026, help="Random seed.")
    train.add_argument("--save", type=Path, default=None, help="Optional model checkpoint path.")
    train.add_argument(
        "--save-dir",
        type=Path,
        default=Path("runs"),
        help="Directory where timestamped training run artifacts are saved.",
    )
    train.add_argument(
        "--save-artifacts",
        nargs="+",
        choices=ARTIFACT_TOKENS,
        default=["all"],
        metavar="ARTIFACT",
        help=("Artifacts to save: config history test model plot, or all/none. Defaults to all."),
    )
    train.add_argument(
        "--swanlab",
        action="store_true",
        help="Enable SwanLab live logging for per-epoch training and validation metrics.",
    )
    train.add_argument(
        "--swanlab-project",
        default="TRAILS",
        help="SwanLab project name used when --swanlab is enabled.",
    )
    train.add_argument(
        "--swanlab-experiment",
        default=None,
        help="Optional SwanLab experiment name. Defaults to a timestamped TRAILS run name.",
    )
    train.add_argument(
        "--swanlab-mode",
        default=None,
        help="Optional SwanLab mode, such as cloud, local, or disabled.",
    )

    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "simulate":
        if args.split_patients is not None and any(count <= 0 for count in args.split_patients):
            parser.error("--split-patients values must be positive.")
        return _run_simulate(args)
    if args.command == "train":
        try:
            args.save_artifacts = resolve_artifact_names(args.save_artifacts)
        except ValueError as error:
            parser.error(str(error))
        return _run_train(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))


def _run_simulate(args: argparse.Namespace) -> int:
    if args.split_patients is not None:
        return _run_simulate_splits(args)

    dataset = _generate_simulated_dataset(args, n_patients=args.patients, seed=args.seed)
    dataset.save(args.out)
    print(
        json.dumps(
            _simulation_summary(
                dataset,
                clusters=args.clusters,
                out=args.out,
                seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_simulate_splits(args: argparse.Namespace) -> int:
    split_names = ("train", "val", "test")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    for offset, (name, patient_count) in enumerate(
        zip(split_names, args.split_patients, strict=True)
    ):
        seed = args.seed + offset
        path = out_dir / f"{name}.pt"
        dataset = _generate_simulated_dataset(args, n_patients=patient_count, seed=seed)
        dataset.save(path)
        summaries[name] = _simulation_summary(
            dataset,
            clusters=args.clusters,
            out=path,
            seed=seed,
        )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "split_patients": {
                    name: count
                    for name, count in zip(split_names, args.split_patients, strict=True)
                },
                "splits": summaries,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _generate_simulated_dataset(
    args: argparse.Namespace,
    *,
    n_patients: int,
    seed: int,
) -> ClinicalTimeSeriesDataset:
    return generate_clinical_time_series_dataset(
        n_patients=n_patients,
        n_clusters=args.clusters,
        min_visits=args.min_visits,
        max_visits=args.max_visits,
        followup_days=args.followup_days,
        latent_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        attention_layers=args.attention_layers,
        attention_heads=args.attention_heads,
        censoring_rate=args.censoring_rate,
        weibull_shape=args.weibull_shape,
        x_low=args.x_low,
        x_high=args.x_high,
        beta_low=args.beta_low,
        beta_high=args.beta_high,
        seed=seed,
    )


def _simulation_summary(
    dataset: ClinicalTimeSeriesDataset,
    *,
    clusters: int,
    out: Path,
    seed: int,
) -> dict[str, Any]:
    event_rate = sum(float(sample.event) for sample in dataset) / len(dataset)
    return {
        "censoring_rate": 1.0 - event_rate,
        "clusters": clusters,
        "features": dataset.feature_names,
        "n_features": dataset.n_features,
        "n_patients": len(dataset),
        "out": str(out),
        "seed": seed,
    }


def _run_train(args: argparse.Namespace) -> int:
    dataset = ClinicalTimeSeriesDataset.load(args.data)
    validation_dataset = (
        None if args.val_data is None else ClinicalTimeSeriesDataset.load(args.val_data)
    )
    test_dataset = (
        dataset if args.test_data is None else ClinicalTimeSeriesDataset.load(args.test_data)
    )
    config = TrailsConfig(
        data=DataConfig(n_features=dataset.n_features),
        model=ModelConfig(
            encoder_hidden_dim=args.encoder_hidden_dim,
            decoder_hidden_dim=args.decoder_hidden_dim,
            latent_dim=args.latent_dim,
            n_clusters=args.clusters,
            n_layers=args.n_layers,
            dropout=args.dropout,
        ),
        trainer=TrainerConfig(
            max_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        ),
        seed=args.seed,
    )
    _start_swanlab_run(args, config)
    try:
        estimator = TrailsEstimator(config).fit(
            dataset,
            validation_data=validation_dataset,
            history_callback=_swanlab_history_logger() if args.swanlab else None,
        )
        metrics = estimator.test(test_dataset)
        if args.swanlab:
            _log_swanlab_test_metrics(metrics, estimator.history)
    finally:
        if args.swanlab:
            _finish_swanlab_run()

    run_dir = _save_training_artifacts(args, config, estimator, metrics)

    if args.save is not None:
        estimator.save(args.save)

    output = {
        "history": estimator.history,
        "run_dir": None if run_dir is None else str(run_dir),
        "test": metrics,
    }
    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _save_training_artifacts(
    args: argparse.Namespace,
    config: TrailsConfig,
    estimator: TrailsEstimator,
    metrics: dict[str, float],
) -> Path | None:
    artifacts: frozenset[str] = args.save_artifacts
    if not artifacts:
        return None

    created_at = datetime.now().astimezone()
    run_dir = create_timestamped_run_dir(args.save_dir, created_at)

    if "config" in artifacts:
        save_json(
            run_dir / "config.json",
            _training_run_config(args, config, artifacts, created_at, run_dir),
        )
    if "history" in artifacts:
        save_json(run_dir / "history.json", estimator.history)
        save_history_csv(run_dir / "history.csv", estimator.history)
    if "test" in artifacts:
        save_json(run_dir / "test_metrics.json", metrics)
    if "model" in artifacts:
        estimator.save(run_dir / "model.pt")
    if "plot" in artifacts:
        plot_history(run_dir / "history.png", estimator.history)

    return run_dir


def _start_swanlab_run(args: argparse.Namespace, config: TrailsConfig) -> None:
    if not args.swanlab:
        return

    import swanlab

    experiment_name = args.swanlab_experiment or datetime.now().astimezone().strftime(
        "trails-%Y%m%d-%H%M%S"
    )
    init_kwargs: dict[str, Any] = {
        "project": args.swanlab_project,
        "experiment_name": experiment_name,
        "config": _swanlab_config(args, config),
    }
    if args.swanlab_mode is not None:
        init_kwargs["mode"] = args.swanlab_mode
    swanlab.init(**init_kwargs)


def _finish_swanlab_run() -> None:
    import swanlab

    swanlab.finish()


def _swanlab_history_logger() -> Any:
    import swanlab

    def log_history(entry: dict[str, float | str]) -> None:
        metrics: dict[str, float] = {}
        for name, value in entry.items():
            if not isinstance(value, int | float):
                continue
            if name == "global_epoch":
                metrics["epoch/global"] = float(value)
            elif name == "epoch":
                metrics["epoch/local"] = float(value)
            elif name.startswith("val_"):
                metrics[f"val/{name.removeprefix('val_')}"] = float(value)
            else:
                metrics[f"train/{name}"] = float(value)

        stage = str(entry["stage"])
        metrics["stage/warmup"] = 1.0 if stage == "warmup" else 0.0
        metrics["stage/vade"] = 1.0 if stage == "vade" else 0.0
        step = int(float(entry["global_epoch"]))
        swanlab.log(metrics, step=step)

    return log_history


def _log_swanlab_test_metrics(
    metrics: dict[str, float], history: list[dict[str, float | str]]
) -> None:
    import swanlab

    step = int(float(history[-1]["global_epoch"])) if history else 0
    swanlab.log({f"test/{name}": value for name, value in metrics.items()}, step=step)


def _swanlab_config(args: argparse.Namespace, config: TrailsConfig) -> dict[str, Any]:
    return {
        "config": config.model_dump(mode="json"),
        "paths": {
            "data": str(args.data),
            "test_data": None if args.test_data is None else str(args.test_data),
            "val_data": None if args.val_data is None else str(args.val_data),
        },
        "save_artifacts": sorted(args.save_artifacts),
    }


def _training_run_config(
    args: argparse.Namespace,
    config: TrailsConfig,
    artifacts: frozenset[str],
    created_at: datetime,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "artifacts": sorted(artifacts),
        "config": config.model_dump(mode="json"),
        "created_at": created_at.isoformat(timespec="seconds"),
        "paths": {
            "data": str(args.data),
            "run_dir": str(run_dir),
            "save": None if args.save is None else str(args.save),
            "save_dir": str(args.save_dir),
            "test_data": None if args.test_data is None else str(args.test_data),
            "test_data_used": str(args.data if args.test_data is None else args.test_data),
            "val_data": None if args.val_data is None else str(args.val_data),
        },
        "train_args": {
            "batch_size": args.batch_size,
            "clusters": args.clusters,
            "decoder_hidden_dim": args.decoder_hidden_dim,
            "dropout": args.dropout,
            "encoder_hidden_dim": args.encoder_hidden_dim,
            "epochs": args.epochs,
            "latent_dim": args.latent_dim,
            "learning_rate": args.learning_rate,
            "n_layers": args.n_layers,
            "seed": args.seed,
            "warmup_epochs": args.warmup_epochs,
        },
        "swanlab": {
            "enabled": args.swanlab,
            "experiment": args.swanlab_experiment,
            "mode": args.swanlab_mode,
            "project": args.swanlab_project,
        },
    }


if __name__ == "__main__":
    main()
