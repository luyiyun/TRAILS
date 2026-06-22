import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Literal

import pytest
import torch

from trails.config import (
    DataConfig,
    DecoderConfig,
    EncoderConfig,
    EncoderInputConfig,
    EncoderMappingConfig,
    ModelConfig,
    TrailsConfig,
    TrainerConfig,
)
from trails.data import ClinicalTimeSeriesDataset, make_clinical_sample
from trails.estimator import TrailsEstimator
from trails_simulate import (
    ClinicalTimeSeriesDatasetGenerator,
    ClinicalTimeSeriesDatasetGeneratorConfig,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "cluster_attribution.py"
SPEC = importlib.util.spec_from_file_location("cluster_attribution_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
cluster_attribution: Any = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cluster_attribution
SPEC.loader.exec_module(cluster_attribution)


def tiny_config(n_features: int) -> TrailsConfig:
    return TrailsConfig(
        data=DataConfig(n_features=n_features),
        model=ModelConfig(
            n_clusters=2,
            latent_dim=4,
            encoder=EncoderConfig(
                input=EncoderInputConfig(hidden_dim=8),
                mapping=EncoderMappingConfig(hidden_dim=8),
            ),
            decoder=DecoderConfig(hidden_dim=8),
        ),
        trainer=TrainerConfig(
            max_epochs=1,
            warmup_epochs=1,
            batch_size=None,
            gmm_init_iters=2,
            valid_size=0.0,
        ),
        seed=13,
    )


def tiny_mtan_config(n_features: int, kind: Literal["mtan", "mtan2"]) -> TrailsConfig:
    return TrailsConfig(
        data=DataConfig(n_features=n_features),
        model=ModelConfig(
            n_clusters=2,
            latent_dim=4,
            encoder=EncoderConfig(
                input=EncoderInputConfig(
                    kind=kind,
                    hidden_dim=4,
                    n_heads=2,
                    num_ref_points=5,
                ),
                mapping=EncoderMappingConfig(kind="gru", hidden_dim=8),
            ),
            decoder=DecoderConfig(hidden_dim=8),
        ),
        trainer=TrainerConfig(
            max_epochs=1,
            warmup_epochs=1,
            batch_size=None,
            gmm_init_iters=2,
            valid_size=0.0,
        ),
        seed=13,
    )


def simulate_dataset(seed: int) -> ClinicalTimeSeriesDataset:
    config = ClinicalTimeSeriesDatasetGeneratorConfig(
        n_clusters=2,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
    )
    return ClinicalTimeSeriesDatasetGenerator(config, mechanism_seed=seed).simulate(
        n_patients=8,
        seed=seed,
    )


def sparse_time_dataset() -> ClinicalTimeSeriesDataset:
    samples = []
    for index in range(4):
        samples.append(
            make_clinical_sample(
                times=torch.tensor([0.0, 10.0]),
                x=torch.tensor(
                    [
                        [1.0 + float(index), 0.0],
                        [2.0 + float(index), 3.0 + float(index)],
                    ]
                ),
                mask=torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
                delta_time=torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
                survival_time=5.0 + float(index),
                event=float(index % 2),
                cluster_label=index % 2,
            )
        )
    return ClinicalTimeSeriesDataset(samples, feature_names=["feature_a", "feature_b"])


def save_model_and_data(
    tmp_path: Path,
    data: ClinicalTimeSeriesDataset,
    config: TrailsConfig,
) -> tuple[Path, Path]:
    estimator = TrailsEstimator(config).fit(data)
    model_path = tmp_path / "model.pt"
    data_path = tmp_path / "dataset.pt"
    estimator.save(model_path)
    data.save(data_path)
    return model_path, data_path


def run_attribution(
    model_path: Path,
    data_path: Path,
    output_dir: Path,
    *extra_args: str,
) -> dict[str, Any]:
    return cluster_attribution.main(
        [
            "--model-path",
            str(model_path),
            "--data-path",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--n-time-bins",
            "4",
            "--ig-steps",
            "2",
            "--device",
            "cpu",
            "--no-progress",
            *extra_args,
        ]
    )


@pytest.mark.parametrize("kind", ["grud", "mtan", "mtan2"])
def test_script_attributions_have_expected_shape(
    tmp_path: Path,
    kind: Literal["grud", "mtan", "mtan2"],
) -> None:
    data = simulate_dataset(seed=43)
    config = (
        tiny_config(data.n_features) if kind == "grud" else tiny_mtan_config(data.n_features, kind)
    )
    model_path, data_path = save_model_and_data(tmp_path, data, config)
    output_dir = tmp_path / f"attr_{kind}"

    run_attribution(model_path, data_path, output_dir, "--plot-features", "2")
    payload = torch.load(output_dir / "cluster_attributions.pt", weights_only=False)
    attribution = payload["attribution"]

    assert attribution["mean_attribution"].shape == (2, data.n_features, 4)
    assert attribution["mean_abs_attribution"].shape == (2, data.n_features, 4)
    assert attribution["sem_attribution"].shape == (2, data.n_features, 4)
    assert attribution["sem_abs_attribution"].shape == (2, data.n_features, 4)
    assert attribution["sample_count"].shape == (2, data.n_features, 4)
    assert attribution["target_clusters"].tolist() == [0, 1]


def test_script_attributions_respect_mask_and_empty_time_bins(tmp_path: Path) -> None:
    data = sparse_time_dataset()
    model_path, data_path = save_model_and_data(tmp_path, data, tiny_config(data.n_features))
    output_dir = tmp_path / "attr_sparse"

    run_attribution(model_path, data_path, output_dir, "--n-time-bins", "5", "--plot-features", "2")
    attribution = torch.load(output_dir / "cluster_attributions.pt", weights_only=False)[
        "attribution"
    ]

    assert torch.isnan(attribution["mean_attribution"][:, :, 1:4]).all()
    assert torch.isnan(attribution["sem_attribution"][:, :, 1:4]).all()
    assert torch.all(attribution["sample_count"][:, :, 1:4] == 0)
    assert torch.all(attribution["observation_count"][:, 0, 0] == len(data))
    assert torch.all(attribution["observation_count"][:, 0, 4] == len(data))
    assert torch.all(attribution["observation_count"][:, 1, 0] == 0)
    assert torch.all(attribution["observation_count"][:, 1, 4] == len(data))


def test_script_feature_selection_top_n_names_and_unknown_feature(tmp_path: Path) -> None:
    data = simulate_dataset(seed=53)
    model_path, data_path = save_model_and_data(tmp_path, data, tiny_config(data.n_features))

    top_summary = run_attribution(
        model_path,
        data_path,
        tmp_path / "attr_top",
        "--plot-features",
        "2",
    )
    named_summary = run_attribution(
        model_path,
        data_path,
        tmp_path / "attr_named",
        "--plot-features",
        data.feature_names[0],
        data.feature_names[1],
    )

    assert top_summary["feature_selection"]["mode"] == "top_n"
    assert top_summary["feature_selection"]["top_n"] == 2
    assert len(top_summary["feature_selection"]["features"]) == 2
    assert named_summary["feature_selection"]["mode"] == "names"
    assert named_summary["feature_selection"]["features"] == data.feature_names[:2]
    with pytest.raises(ValueError, match="unknown feature names"):
        run_attribution(
            model_path,
            data_path,
            tmp_path / "attr_missing",
            "--plot-features",
            "missing_feature",
        )


def test_cluster_attribution_script_saves_outputs(tmp_path: Path) -> None:
    data = simulate_dataset(seed=61)
    model_path, data_path = save_model_and_data(tmp_path, data, tiny_config(data.n_features))
    output_dir = tmp_path / "attribution"

    summary = run_attribution(model_path, data_path, output_dir, "--plot-features", "2")

    assert summary["output_dir"] == str(output_dir.resolve())
    assert (output_dir / "args.json").exists()
    assert (output_dir / "cluster_attributions.csv").exists()
    assert (output_dir / "cluster_attributions.pt").exists()
    assert (output_dir / "attribution_summary.json").exists()
    assert (output_dir / "attribution_lines.png").exists()
    assert (output_dir / "attribution_lines.pdf").exists()

    args_payload = json.loads((output_dir / "args.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output_dir / "cluster_attributions.csv").open()))
    assert args_payload["feature_selection"]["features"] == summary["feature_selection"]["features"]
    assert args_payload["raw_argv"]
    assert len(rows) == 2 * data.n_features * 4
    assert {
        "cluster",
        "feature",
        "time_center",
        "mean_attribution",
        "sem_abs_attribution",
        "sample_count",
    } <= set(rows[0])
