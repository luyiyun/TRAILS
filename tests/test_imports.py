from pathlib import Path


def test_imports() -> None:
    import trails
    import trails_case
    import trails_simulate

    assert "ClinicalTimeSeriesDataset" in trails.__all__
    assert trails_case.__doc__
    assert trails_simulate.__all__ == [
        "DEFAULT_FEATURE_NAMES",
        "ClinicalTimeSeriesDatasetGenerator",
        "ClinicalTimeSeriesDatasetGeneratorConfig",
    ]


def test_trails_package_does_not_import_experiment_packages() -> None:
    for path in Path("src/trails").glob("*.py"):
        text = path.read_text()
        assert "trails_simulate" not in text
        assert "trails_case" not in text
        assert "scripts" not in text


def test_case_package_does_not_import_simulation_package() -> None:
    for path in Path("src/trails_case").glob("*.py"):
        text = path.read_text()
        assert "trails_simulate" not in text


def test_case_package_does_not_use_stdlib_csv() -> None:
    for path in Path("src/trails_case").glob("*.py"):
        text = path.read_text()
        assert "import csv" not in text
        assert "csv." not in text
