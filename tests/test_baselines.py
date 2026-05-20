from __future__ import annotations

import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

from trails.data import ClinicalTimeSeriesDataset, make_clinical_sample
from trails_simulate.baselines import (
    RiskStratifiedKMeansBaseline,
    SummaryKMeansBaseline,
    make_baseline,
)
from trails_simulate.evaluation import evaluate_predictions
from trails_simulate.generators import (
    ClinicalTimeSeriesDatasetGenerator,
    ClinicalTimeSeriesDatasetGeneratorConfig,
)


def simulate_split() -> tuple[ClinicalTimeSeriesDataset, ClinicalTimeSeriesDataset]:
    source = ClinicalTimeSeriesDatasetGenerator(
        ClinicalTimeSeriesDatasetGeneratorConfig(
            n_clusters=2,
            min_visits=3,
            max_visits=5,
            hidden_size=12,
            latent_dim=4,
            attention_layers=1,
        )
    ).simulate(n_patients=16, seed=29)
    train_data, test_data = source.split_counts([10, 6], seed=31)
    return train_data, test_data


def with_replaced_test_survival(data: ClinicalTimeSeriesDataset) -> ClinicalTimeSeriesDataset:
    samples = []
    for index in range(len(data)):
        sample = data[index]
        samples.append(
            make_clinical_sample(
                times=sample.times,
                x=sample.x,
                mask=sample.mask,
                delta_time=sample.delta_time,
                survival_time=float(index + 1) * 100.0,
                event=0.0,
                cluster_label=sample.cluster_label,
            )
        )
    return ClinicalTimeSeriesDataset(
        samples,
        feature_names=data.feature_names,
        description=data.description,
        metadata=data.metadata,
    )


def test_summary_baseline_class_api_and_sklearn_fit() -> None:
    train_data, test_data = simulate_split()
    baseline = SummaryKMeansBaseline(n_clusters=2, random_state=101)

    fitted = baseline.fit(train_data)
    prediction = fitted.predict(test_data)
    metrics = evaluate_predictions(prediction, n_clusters=2)

    assert fitted is baseline
    assert isinstance(baseline.cluster_model, KMeans)
    assert prediction["pred_cluster"].shape == (len(test_data),)
    assert metrics.keys() >= {"cindex", "ari", "nmi", "cluster_entropy"}


def test_risk_stratified_baseline_class_api_and_sklearn_fit() -> None:
    train_data, test_data = simulate_split()
    baseline = RiskStratifiedKMeansBaseline(
        n_clusters=2,
        random_state=101,
        ridge_alpha=1.0,
        risk_feature_weight=1.0,
    )

    prediction = baseline.fit_predict(train_data)
    test_prediction = baseline.predict(test_data)

    assert isinstance(baseline.cluster_model, KMeans)
    assert isinstance(baseline.risk_model, Ridge)
    assert prediction["pred_cluster"].shape == (len(train_data),)
    assert test_prediction["pred_cluster"].shape == (len(test_data),)


def test_baseline_factory_builds_registered_methods() -> None:
    summary = make_baseline(
        "summary_kmeans",
        n_clusters=2,
        random_state=17,
        kmeans_iters=5,
        ridge_alpha=1.0,
        risk_feature_weight=1.0,
    )
    risk = make_baseline(
        "risk_stratified_kmeans",
        n_clusters=2,
        random_state=17,
        kmeans_iters=5,
        ridge_alpha=1.0,
        risk_feature_weight=1.0,
    )

    assert isinstance(summary, SummaryKMeansBaseline)
    assert isinstance(risk, RiskStratifiedKMeansBaseline)


def test_baseline_feature_extraction_keeps_missing_values_for_imputer() -> None:
    sample_a = make_clinical_sample(
        times=torch.tensor([0.0, 2.0]),
        x=torch.tensor([[1.0, 0.0], [3.0, 0.0]]),
        mask=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        delta_time=torch.zeros(2, 2),
        survival_time=4.0,
        event=1.0,
        cluster_label=0,
    )
    sample_b = make_clinical_sample(
        times=torch.tensor([0.0, 1.0, 3.0]),
        x=torch.tensor([[0.0, 2.0], [0.0, 5.0], [0.0, 8.0]]),
        mask=torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        delta_time=torch.zeros(3, 2),
        survival_time=5.0,
        event=0.0,
        cluster_label=1,
    )
    data = ClinicalTimeSeriesDataset(
        [sample_a, sample_b],
        feature_names=["a", "b"],
    )

    raw_features = SummaryKMeansBaseline(n_clusters=2, random_state=13).extract_features(data)

    assert raw_features.shape == (2, 10)
    assert bool(torch.isnan(torch.as_tensor(raw_features)).any())


def test_baseline_assignments_do_not_use_test_survival_labels() -> None:
    train_data, test_data = simulate_split()
    altered_test_data = with_replaced_test_survival(test_data)

    for baseline in (
        SummaryKMeansBaseline(n_clusters=2, random_state=202),
        RiskStratifiedKMeansBaseline(
            n_clusters=2,
            random_state=202,
            ridge_alpha=1.0,
            risk_feature_weight=1.0,
        ),
    ):
        baseline.fit(train_data)
        original = baseline.predict(test_data)
        altered = baseline.predict(altered_test_data)

        assert torch.equal(original["pred_cluster"], altered["pred_cluster"])
        assert torch.allclose(original["risk_score"], altered["risk_score"])
