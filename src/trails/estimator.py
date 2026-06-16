from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from .artifacts import save_json
from .config import TrailsConfig
from .data import ClinicalTimeSeriesDataset, infer_data_config
from .diagnostics import LatentDiagnostics
from .metrics import cluster_assignment_diagnostics, concordance_index, gaussian_log_prob
from .model import TrailsSurvVaderModel
from .trainer import HistoryEntry, TrailsTrainer

HistoryCallback = Callable[[HistoryEntry], None]
type SelectionMetricValue = int | float
type SelectionMetricRow = dict[str, SelectionMetricValue]
type KSelectionMetrics = dict[str, dict[str, float]]


class TrailsEstimator:
    def __init__(self, config: TrailsConfig | None = None) -> None:
        self.config = config or TrailsConfig()
        torch.manual_seed(self.config.seed)
        self.model = TrailsSurvVaderModel(self.config.data, self.config.model)
        trainer_config = self.config.trainer.model_copy(update={"seed": self.config.seed})
        self.trainer = TrailsTrainer(self.model, trainer_config)
        self.history: list[HistoryEntry] = []

    def fit(
        self,
        data: ClinicalTimeSeriesDataset,
        history_callback: HistoryCallback | None = None,
        validation_data: ClinicalTimeSeriesDataset | None = None,
    ) -> TrailsEstimator:
        self._validate_data_config(data)
        if validation_data is not None:
            self._validate_data_config(validation_data)
        self.model.set_feature_means(data.feature_means)
        if self.config.model.encoder.input.kind in {"mtan", "mtan2"}:
            min_time, max_time = observed_time_range(data)
            self.model.set_reference_time_range(min_time, max_time)
        self.history = self.trainer.fit(
            data,
            history_callback=history_callback,
            validation_data=validation_data,
        )
        return self

    def predict(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        self._validate_data_config(data)
        return self.trainer.predict(data)

    def predict_proba(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        self._validate_data_config(data)
        return self.trainer.predict_proba(data)

    def predict_risk(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        self._validate_data_config(data)
        return self.trainer.predict_risk(data)

    def test(self, data: ClinicalTimeSeriesDataset) -> dict[str, float]:
        self._validate_data_config(data)
        return self.trainer.test(data)

    def selection_metrics(self, data: ClinicalTimeSeriesDataset) -> dict[str, float]:
        self._validate_data_config(data)
        outputs, batch = self.trainer._collect_outputs(data)
        latent = outputs.latent_mean.detach()
        n_samples = int(latent.shape[0])
        if n_samples == 0:
            raise ValueError("Selection metrics require at least one sample.")

        # 用确定性的 latent_mean 评估 VaDE MoG prior，避免采样噪声影响 K 选择。
        log_prior = torch.log_softmax(self.model.mixture_logits.detach(), dim=-1).unsqueeze(0)
        component_log_prob = gaussian_log_prob(
            latent.unsqueeze(1),
            self.model.mixture_means.detach().unsqueeze(0),
            self.model.mixture_log_variances.detach().unsqueeze(0),
        ).sum(dim=-1)
        log_likelihood = torch.logsumexp(log_prior + component_log_prob, dim=-1)
        total_log_likelihood = float(log_likelihood.sum().item())
        mean_nll = float((-log_likelihood).mean().item())
        n_clusters = self.config.model.n_clusters
        latent_dim = self.config.model.latent_dim
        n_parameters = n_clusters * (2 * latent_dim) + (n_clusters - 1)
        bic = -2.0 * total_log_likelihood + math.log(float(n_samples)) * float(n_parameters)

        pred_cluster = torch.argmax(outputs.cluster_probabilities.detach().cpu(), dim=-1).long()
        metrics = {
            "cindex": float(
                concordance_index(
                    self.trainer._risk_score(outputs).detach().cpu().float(),
                    batch["survival_time"].detach().cpu().float(),
                    batch["event"].detach().cpu().float(),
                )
            ),
            "bic": float(bic),
            "mean_nll": mean_nll,
            "n_parameters": float(n_parameters),
            **cluster_assignment_diagnostics(pred_cluster, n_clusters=n_clusters),
        }
        return metrics

    def select_n_clusters(
        self,
        data: ClinicalTimeSeriesDataset,
        *,
        candidate_clusters: Sequence[int] | None = None,
        valid_fraction: float | None = None,
        inherit_best: bool = False,
        result_dir: str | Path | None = None,
    ) -> KSelectionMetrics:
        self._validate_data_config(data)
        candidates = resolve_candidate_clusters(candidate_clusters, self.config.model.n_clusters)
        valid_fraction = (
            self.config.trainer.valid_size if valid_fraction is None else valid_fraction
        )
        if valid_fraction <= 0.0 or valid_fraction >= 1.0:
            raise ValueError("valid_fraction must be greater than 0 and less than 1.")

        train_data, valid_data = data.split(
            [1.0 - valid_fraction, valid_fraction], seed=self.config.seed
        )
        rows: list[SelectionMetricRow] = []
        estimators: dict[int, TrailsEstimator] = {}
        for n_clusters in candidates:
            candidate_config = self.config.model_copy(
                update={
                    "model": self.config.model.model_copy(update={"n_clusters": n_clusters}),
                    "trainer": self.config.trainer.model_copy(
                        update={"seed": self.config.seed, "valid_size": 0.0}
                    ),
                }
            )
            estimator = self.__class__(candidate_config).fit(
                train_data,
                validation_data=valid_data,
            )
            metrics = estimator.selection_metrics(valid_data)
            rows.append({"n_clusters": n_clusters, **metrics})
            estimators[n_clusters] = estimator

        ranked_rows = score_k_selection_rows(rows)
        metrics_by_name = selection_rows_to_metrics(ranked_rows)
        if result_dir is not None:
            save_k_selection_runs(Path(result_dir), ranked_rows, metrics_by_name, estimators)
        if inherit_best:
            best_k = int(ranked_rows[0]["n_clusters"])
            self._inherit_estimator(estimators[best_k])
        return metrics_by_name

    def latent_diagnostics(self, data: ClinicalTimeSeriesDataset) -> LatentDiagnostics:
        self._validate_data_config(data)
        outputs, batch = self.trainer._collect_outputs(data)
        cluster_probabilities = outputs.cluster_probabilities.detach().cpu()
        diagnostics: LatentDiagnostics = {
            "z": outputs.latent_mean.detach().cpu(),
            "cluster_probabilities": cluster_probabilities,
            "pred_cluster": torch.argmax(cluster_probabilities, dim=-1).long(),
            "sample_index": torch.arange(len(data), dtype=torch.long),
        }
        if "cluster_label" in batch:
            diagnostics["true_cluster"] = batch["cluster_label"].detach().cpu().long()
        return diagnostics

    def save(self, path: str | Path) -> None:
        checkpoint = {
            "config": self.config.model_dump(mode="json"),
            "history": self.history,
            "model_state": self.model.state_dict(),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, destination)

    @classmethod
    def load(cls, path: str | Path) -> TrailsEstimator:
        checkpoint: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=False)
        estimator = cls(TrailsConfig.model_validate(checkpoint["config"]))
        estimator.model.load_state_dict(checkpoint["model_state"])
        estimator.history = list(checkpoint.get("history", []))
        return estimator

    def _validate_data_config(self, data: ClinicalTimeSeriesDataset) -> None:
        inferred = infer_data_config(data)
        if inferred != self.config.data:
            raise ValueError(
                "Data shape does not match estimator config: "
                f"expected {self.config.data}, got {inferred}."
            )

    def _inherit_estimator(self, estimator: TrailsEstimator) -> None:
        self.config = estimator.config
        self.model = estimator.model
        self.trainer = estimator.trainer
        self.history = list(estimator.history)


def resolve_candidate_clusters(
    candidate_clusters: Sequence[int] | None,
    max_clusters: int,
) -> tuple[int, ...]:
    if candidate_clusters is None:
        return validate_candidate_clusters(tuple(range(2, max_clusters + 1)))
    return validate_candidate_clusters(candidate_clusters)


def validate_candidate_clusters(candidate_clusters: Sequence[int]) -> tuple[int, ...]:
    candidates = tuple(int(value) for value in candidate_clusters)
    if not candidates:
        raise ValueError("candidate_clusters must contain at least one K value.")
    if any(value <= 1 for value in candidates):
        raise ValueError("candidate_clusters values must be greater than 1.")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_clusters values must be unique.")
    return candidates


def score_k_selection_rows(rows: list[SelectionMetricRow]) -> list[SelectionMetricRow]:
    if not rows:
        raise ValueError("K selection requires at least one candidate row.")

    bics = [float(require_selection_value(row, "bic")) for row in rows]
    min_bic = min(bics)
    max_bic = max(bics)
    bic_range = max_bic - min_bic

    scored_rows: list[SelectionMetricRow] = []
    for row, bic in zip(rows, bics, strict=True):
        cindex = float(require_selection_value(row, "cindex"))
        bic_norm = 0.0 if bic_range == 0.0 else (bic - min_bic) / bic_range
        selection_score = math.sqrt(cindex**2 + (1.0 - bic_norm) ** 2)
        scored_rows.append(
            {
                **row,
                "bic_norm": float(bic_norm),
                "selection_score": float(selection_score),
            }
        )

    ranked_rows = sorted(
        scored_rows,
        key=lambda row: (
            -float(require_selection_value(row, "selection_score")),
            -float(require_selection_value(row, "cindex")),
            float(require_selection_value(row, "bic")),
            int(require_selection_value(row, "n_clusters")),
        ),
    )
    return [{**row, "rank": rank} for rank, row in enumerate(ranked_rows, start=1)]


def selection_rows_to_metrics(rows: Sequence[SelectionMetricRow]) -> KSelectionMetrics:
    metrics: KSelectionMetrics = {}
    for row in rows:
        n_clusters = int(require_selection_value(row, "n_clusters"))
        cluster_key = str(n_clusters)
        for name, value in row.items():
            if name == "n_clusters":
                continue
            metrics.setdefault(name, {})[cluster_key] = float(value)
    return metrics


def selection_metrics_to_rows(metrics: KSelectionMetrics) -> list[SelectionMetricRow]:
    cluster_keys = sorted({cluster for values in metrics.values() for cluster in values}, key=int)
    rows: list[SelectionMetricRow] = []
    for cluster_key in cluster_keys:
        row: SelectionMetricRow = {"n_clusters": int(cluster_key)}
        for name, values in metrics.items():
            if cluster_key in values:
                row[name] = float(values[cluster_key])
        rows.append(row)
    if "rank" in metrics:
        rows.sort(key=lambda row: float(require_selection_value(row, "rank")))
    return rows


def selected_k_from_selection_metrics(metrics: KSelectionMetrics) -> int:
    ranks = metrics.get("rank")
    if not ranks:
        raise ValueError("K selection metrics must include a rank metric.")
    best = min(ranks.items(), key=lambda item: (float(item[1]), int(item[0])))
    if float(best[1]) != 1.0:
        raise ValueError("K selection metrics must include a rank value of 1.")
    return int(best[0])


def best_selection_metrics(metrics: KSelectionMetrics) -> dict[str, float]:
    best_key = str(selected_k_from_selection_metrics(metrics))
    return {name: values[best_key] for name, values in metrics.items() if best_key in values}


def save_k_selection_runs(
    result_dir: Path,
    rows: Sequence[SelectionMetricRow],
    metrics_by_name: KSelectionMetrics,
    estimators: dict[int, TrailsEstimator],
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(result_dir / "selection_metrics.csv", index=False)
    save_json(result_dir / "selection_metrics.json", metrics_by_name)
    for row in rows:
        n_clusters = int(require_selection_value(row, "n_clusters"))
        estimator = estimators[n_clusters]
        candidate_dir = result_dir / f"k{n_clusters}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        estimator.save(candidate_dir / "model.pt")
        save_json(candidate_dir / "history.json", estimator.history)
        save_json(candidate_dir / "metrics.json", dict(row))
        save_json(candidate_dir / "config.json", estimator.config.model_dump(mode="json"))


def require_selection_value(row: SelectionMetricRow, name: str) -> SelectionMetricValue:
    value = row.get(name)
    if not isinstance(value, int | float):
        raise ValueError(f"K selection row is missing numeric field {name!r}.")
    return value


def observed_time_range(data: ClinicalTimeSeriesDataset) -> tuple[float, float]:
    compact_data = data.with_return_kind("compact")
    min_time: float | None = None
    max_time: float | None = None
    for sample in compact_data.samples:
        observed_times = sample.times[sample.mask > 0].float()
        if observed_times.numel() == 0:
            continue
        sample_min = float(observed_times.min().item())
        sample_max = float(observed_times.max().item())
        min_time = sample_min if min_time is None else min(min_time, sample_min)
        max_time = sample_max if max_time is None else max(max_time, sample_max)

    if min_time is None or max_time is None:
        raise ValueError("mTAN reference time grid requires at least one observed time.")
    return min_time, max_time
