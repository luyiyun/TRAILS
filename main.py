from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from trails.progress import configure_tqdm_logging
from trails_simulate.config import ApplicationConfig
from trails_simulate.summary import format_run_summary
from trails_simulate.workflow import run

LOGGER = logging.getLogger(__name__)


def load_app_config(raw_config: DictConfig) -> ApplicationConfig:
    payload: Any = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Hydra config must resolve to a mapping.")
    try:
        return ApplicationConfig.model_validate(payload)
    except ValidationError as error:
        raise ValueError(str(error)) from error


def raw_command(raw_config: DictConfig) -> str:
    command = OmegaConf.select(raw_config, "command", default="simulate")
    return str(command)


def run_case_from_raw_config(
    raw_config: DictConfig,
    *,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    from trails_case.config import CaseApplicationConfig
    from trails_case.summary import format_case_summary
    from trails_case.workflow import run_case_command

    payload: Any = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Hydra config must resolve to a mapping.")
    try:
        config = CaseApplicationConfig.model_validate(payload)
    except ValidationError as error:
        raise ValueError(str(error)) from error

    result = run_case_command(config, hydra_run_dir=hydra_run_dir, project_root=project_root)
    LOGGER.info(format_case_summary(result))
    return result


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    configure_tqdm_logging()
    project_root = Path(get_original_cwd())
    hydra_run_dir = Path(HydraConfig.get().runtime.output_dir)
    if raw_command(raw_config) == "case":
        run_case_from_raw_config(
            raw_config,
            hydra_run_dir=hydra_run_dir,
            project_root=project_root,
        )
        return

    config = load_app_config(raw_config)
    result = run(config, hydra_run_dir=hydra_run_dir, project_root=project_root)
    LOGGER.info(format_run_summary(result))


if __name__ == "__main__":
    main()
