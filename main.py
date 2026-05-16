from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from trails.config import DataConfig, EstimatorConfig, ModelConfig, TrainerConfig
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
    simulate.add_argument("--out", type=Path, required=True, help="Output .pt dataset path.")
    simulate.add_argument("--patients", type=int, default=128, help="Number of patients.")
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
    train.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
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

    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "simulate":
        return _run_simulate(args)
    if args.command == "train":
        return _run_train(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))


def _run_simulate(args: argparse.Namespace) -> int:
    dataset = generate_clinical_time_series_dataset(
        n_patients=args.patients,
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
        seed=args.seed,
    )
    dataset.save(args.out)
    event_rate = sum(float(sample.event) for sample in dataset) / len(dataset)
    print(
        json.dumps(
            {
                "censoring_rate": 1.0 - event_rate,
                "clusters": args.clusters,
                "features": dataset.feature_names,
                "n_features": dataset.n_features,
                "n_patients": len(dataset),
                "out": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_train(args: argparse.Namespace) -> int:
    dataset = ClinicalTimeSeriesDataset.load(args.data)
    config = EstimatorConfig(
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
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        ),
        seed=args.seed,
    )
    estimator = TrailsEstimator(config).fit(dataset)
    metrics = estimator.test(dataset)

    if args.save is not None:
        estimator.save(args.save)

    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
