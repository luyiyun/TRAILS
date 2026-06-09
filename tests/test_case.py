import csv
import json
from pathlib import Path
from typing import Any, cast

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
    label_header = ",cluster_label" if include_labels else ""
    label_rows = ["p1,5,1,0", "p2,4,0,1"] if include_labels else ["p1,5,1", "p2,4,0"]
    patients.write_text(
        "\n".join([f"patient_id,survival_time,event{label_header}", *label_rows, ""]),
        encoding="utf-8",
    )
    observations.write_text(
        "\n".join(
            [
                "patient_id,time,feature,value",
                "p1,0,crp,1.0",
                "p1,2,albumin,3.0",
                "p2,1,crp,2.0",
                "p2,1,albumin,4.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
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
    from trails_case.config import CaseColumnsConfig
    from trails_case.data import load_case_dataset_from_csv

    patients, observations = write_case_csvs(tmp_path)

    imported = load_case_dataset_from_csv(
        patients_csv=patients,
        observations_csv=observations,
        columns=CaseColumnsConfig(),
        description="case test",
        feature_order=[],
    )
    dataset = imported.dataset
    first = dataset[0].to_aligned()

    assert len(dataset) == 2
    assert dataset.feature_names == ["crp", "albumin"]
    assert dataset.metadata["patient_ids"] == ["p1", "p2"]
    assert dataset.has_cluster_labels
    assert torch.allclose(first.times, torch.tensor([0.0, 2.0]))
    assert torch.allclose(first.x, torch.tensor([[1.0, 0.0], [0.0, 3.0]]))
    assert torch.allclose(first.mask, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.allclose(first.delta_time, torch.tensor([[0.0, 0.0], [2.0, 2.0]]))
    assert imported.patient_summaries[0].missing_fraction == 0.5
    assert imported.patient_summaries[1].n_visits == 1


def test_case_importer_respects_configured_feature_order(tmp_path: Path) -> None:
    from trails_case.config import CaseColumnsConfig
    from trails_case.data import load_case_dataset_from_csv

    patients, observations = write_case_csvs(tmp_path)

    imported = load_case_dataset_from_csv(
        patients_csv=patients,
        observations_csv=observations,
        columns=CaseColumnsConfig(),
        description="case test",
        feature_order=["albumin", "crp"],
    )

    assert imported.dataset.feature_names == ["albumin", "crp"]
    assert torch.allclose(
        imported.dataset[0].to_aligned().x, torch.tensor([[0.0, 1.0], [3.0, 0.0]])
    )


@pytest.mark.parametrize(
    ("patients_text", "observations_text", "match"),
    [
        (
            "patient_id,survival_time\np1,5\n",
            "patient_id,time,feature,value\np1,0,crp,1\n",
            "missing required column",
        ),
        (
            "patient_id,survival_time,event\np1,5,1\n",
            "patient_id,time,feature,value\np2,0,crp,1\n",
            "unknown patient_id",
        ),
        (
            "patient_id,survival_time,event\np1,5,1\n",
            "patient_id,time,feature,value\np1,0,crp,1\np1,0.0,crp,2\n",
            "duplicate observation",
        ),
        (
            "patient_id,survival_time,event\np1,5,1\n",
            "patient_id,time,feature,value\np1,0,crp,abc\n",
            "must be numeric",
        ),
        (
            "patient_id,survival_time,event\np1,5,2\n",
            "patient_id,time,feature,value\np1,0,crp,1\n",
            "must be 0 or 1",
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
    from trails_case.config import CaseColumnsConfig
    from trails_case.data import load_case_dataset_from_csv

    patients = tmp_path / "patients.csv"
    observations = tmp_path / "observations.csv"
    patients.write_text(patients_text, encoding="utf-8")
    observations.write_text(observations_text, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_case_dataset_from_csv(
            patients_csv=patients,
            observations_csv=observations,
            columns=CaseColumnsConfig(),
            description="bad case",
            feature_order=[],
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

    with (tmp_path / "run" / "patient_clusters.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["patient_id"] == "p1"
    assert rows[0]["pred_cluster"] == "0"
    assert "cluster_prob_0" in rows[0]
    assert rows[0]["n_observations"] == "2"

    summary = json.loads((tmp_path / "run" / "case_summary.json").read_text(encoding="utf-8"))
    assert summary["data"]["n_patients"] == 2
    assert summary["outputs"]["patient_clusters"] == str(tmp_path / "run" / "patient_clusters.csv")
