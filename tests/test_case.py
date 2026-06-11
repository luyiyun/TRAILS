import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def compose_payload(*overrides: str) -> dict[str, Any]:
    config_dir = str((ROOT / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(overrides))
    payload = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def write_case_csvs(root: Path, *, include_labels: bool = True) -> tuple[Path, Path]:
    patients = root / "patients.csv"
    observations = root / "observations.csv"
    root.mkdir(parents=True, exist_ok=True)
    patient_rows: list[dict[str, Any]] = [
        {"patient_id": "p1", "survival_time": 5, "event": 1},
        {"patient_id": "p2", "survival_time": 4, "event": 0},
    ]
    if include_labels:
        patient_rows[0]["cluster_label"] = 0
        patient_rows[1]["cluster_label"] = 1
    pd.DataFrame(patient_rows).to_csv(patients, index=False)
    pd.DataFrame(
        [
            {"patient_id": "p1", "time": 0, "feature": "crp", "value": 1.0},
            {"patient_id": "p1", "time": 2, "feature": "albumin", "value": 3.0},
            {"patient_id": "p2", "time": 1, "feature": "crp", "value": 2.0},
            {"patient_id": "p2", "time": 1, "feature": "albumin", "value": 4.0},
        ]
    ).to_csv(observations, index=False)
    return patients, observations


def test_case_config_defaults_validate() -> None:
    from trails_case.config import CaseApplicationConfig

    config = CaseApplicationConfig.model_validate(compose_payload("case=default"))

    assert config.command == "case"
    assert config.run.prefix == "case"
    assert config.training.swanlab.enabled
    assert config.training.swanlab.experiment == "case"
    assert config.training.artifacts.names == ("all",)
    assert config.training.diagnostics.latent_embeddings.enabled
    assert config.case.outputs.patient_clusters == Path("patient_clusters.csv")


def test_case_importer_builds_aligned_dataset(tmp_path: Path) -> None:
    from trails.data import ClinicalTimeSeriesDataset
    from trails_case.data import patient_summaries_from_metadata

    patients, observations = write_case_csvs(tmp_path)

    dataset = ClinicalTimeSeriesDataset.load_from_csv(
        patients_csv=patients,
        observations_csv=observations,
        description="case test",
        metadata={"source": "case_csv"},
    )
    patient_summaries = patient_summaries_from_metadata(dataset.metadata)
    first = dataset[0].to_aligned()

    assert len(dataset) == 2
    assert dataset.feature_names == ["crp", "albumin"]
    assert dataset.metadata["patient_ids"] == ["p1", "p2"]
    assert dataset.has_cluster_labels
    assert torch.allclose(first.times, torch.tensor([0.0, 2.0]))
    assert torch.allclose(first.x, torch.tensor([[1.0, 0.0], [0.0, 3.0]]))
    assert torch.allclose(first.mask, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.allclose(first.delta_time, torch.tensor([[0.0, 0.0], [2.0, 2.0]]))
    assert patient_summaries[0].missing_fraction == 0.5
    assert patient_summaries[1].n_visits == 1


def test_case_importer_respects_configured_feature_order(tmp_path: Path) -> None:
    from trails.data import ClinicalTimeSeriesDataset

    patients, observations = write_case_csvs(tmp_path)

    dataset = ClinicalTimeSeriesDataset.load_from_csv(
        patients_csv=patients,
        observations_csv=observations,
        description="case test",
        use_features=["albumin", "crp"],
    )

    assert dataset.feature_names == ["albumin", "crp"]
    assert torch.allclose(dataset[0].to_aligned().x, torch.tensor([[0.0, 1.0], [3.0, 0.0]]))


@pytest.mark.parametrize(
    ("patients_text", "observations_text", "match"),
    [
        (
            "patient_id,survival_time\np1,5\n",
            "patient_id,time,feature,value\np1,0,crp,1\n",
            "must contain column 'event'",
        ),
        (
            "patient_id,survival_time,event\np1,5,1\n",
            "patient_id,time,feature,value\np2,0,crp,1\n",
            "contains unknown patient_ids",
        ),
        (
            "patient_id,survival_time,event\np1,5,1\n",
            "patient_id,time,feature,value\np1,0,crp,1\np1,0.0,crp,2\n",
            "contains duplicate observations",
        ),
        (
            "patient_id,survival_time,event\np1,5,1\n",
            "patient_id,time,feature,value\np1,0,crp,abc\n",
            "could not convert string to float",
        ),
        (
            "patient_id,survival_time,event\np1,5,2\n",
            "patient_id,time,feature,value\np1,0,crp,1\n",
            "must contain 0 or 1 event values",
        ),
        (
            "patient_id,survival_time,event\np1,5,1\np2,4,0\n",
            "patient_id,time,feature,value\np1,0,crp,1\n",
            "Every patient must have at least one observation",
        ),
    ],
)
def test_case_importer_validation_errors(
    tmp_path: Path,
    patients_text: str,
    observations_text: str,
    match: str,
) -> None:
    from trails.data import ClinicalTimeSeriesDataset

    patients = tmp_path / "patients.csv"
    observations = tmp_path / "observations.csv"
    patients.write_text(patients_text, encoding="utf-8")
    observations.write_text(observations_text, encoding="utf-8")

    with pytest.raises((AssertionError, ValueError), match=match):
        ClinicalTimeSeriesDataset.load_from_csv(
            patients_csv=patients,
            observations_csv=observations,
            description="bad case",
        )


def test_case_command_writes_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from trails_case import workflow
    from trails_case.config import CaseApplicationConfig

    patients, observations = write_case_csvs(tmp_path / "data", include_labels=False)

    class FakeEstimator:
        def __init__(self, config: Any) -> None:
            self.config = config
            self.history = [
                {
                    "epoch": 1,
                    "global_epoch": 1,
                    "stage": "vade",
                    "train": {"loss": 1.0, "cindex": 0.5},
                }
            ]

        def fit(self, *_args: Any, history_callback: Any = None, **_kwargs: Any) -> Any:
            if history_callback is not None:
                history_callback(self.history[0])
            return self

        def predict(self, data: Any) -> torch.Tensor:
            return torch.arange(len(data), dtype=torch.long) % self.config.model.n_clusters

        def predict_risk(self, data: Any) -> torch.Tensor:
            return torch.arange(len(data), dtype=torch.float32)

        def predict_proba(self, data: Any) -> torch.Tensor:
            probabilities = torch.zeros(len(data), self.config.model.n_clusters)
            for index in range(len(data)):
                probabilities[index, index % self.config.model.n_clusters] = 1.0
            return probabilities

        def save(self, path: str | Path) -> None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": self.config.model_dump(mode="json")}, destination)

    monkeypatch.setattr(workflow, "TrailsEstimator", FakeEstimator)
    config = CaseApplicationConfig.model_validate(
        compose_payload(
            "case=default",
            f"case.patients_csv={patients}",
            f"case.observations_csv={observations}",
            "training.swanlab.enabled=false",
            "training.diagnostics.latent_embeddings.enabled=false",
            "training.trainer.device=cpu",
        )
    )

    result = workflow.run_case_command(config, tmp_path / "run", ROOT)

    assert result["command"] == "case"
    assert (tmp_path / "run" / "case_dataset.pt").exists()
    assert (tmp_path / "run" / "case_dataset_summary.json").exists()
    assert (tmp_path / "run" / "config.json").exists()
    assert (tmp_path / "run" / "history.json").exists()
    assert (tmp_path / "run" / "history.csv").exists()
    assert (tmp_path / "run" / "history.png").exists()
    assert (tmp_path / "run" / "model.pt").exists()
    assert (tmp_path / "run" / "predictions.pt").exists()
    assert (tmp_path / "run" / "patient_clusters.csv").exists()
    assert (tmp_path / "run" / "cluster_summary.csv").exists()
    assert (tmp_path / "run" / "cluster_feature_summary.csv").exists()
    assert (tmp_path / "run" / "case_summary.json").exists()

    rows = pd.read_csv(
        tmp_path / "run" / "patient_clusters.csv",
        keep_default_na=False,
        dtype=str,
    ).to_dict("records")
    assert rows[0]["patient_id"] == "p1"
    assert rows[0]["pred_cluster"] == "0"
    assert "cluster_prob_0" in rows[0]
    assert rows[0]["n_observations"] == "2"

    summary = json.loads((tmp_path / "run" / "case_summary.json").read_text(encoding="utf-8"))
    assert summary["data"]["n_patients"] == 2
    assert summary["outputs"]["patient_clusters"] == str(tmp_path / "run" / "patient_clusters.csv")
