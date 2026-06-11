from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from trails.progress import configure_tqdm_logging
from trails_case.config import CaseApplicationConfig
from trails_case.summary import format_case_summary
from trails_case.workflow import run_case_command
from trails_simulate.config import ApplicationConfig
from trails_simulate.summary import format_run_summary
from trails_simulate.workflow import run as run_simulation

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    configure_tqdm_logging()

    project_root = Path(get_original_cwd())
    hydra_run_dir = Path(HydraConfig.get().runtime.output_dir)

    payload: Any = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Hydra config must resolve to a mapping.")

    if raw_config.get("command", "simulate") == "case":
        config = CaseApplicationConfig.model_validate(payload)
        result = run_case_command(config, hydra_run_dir=hydra_run_dir, project_root=project_root)
        LOGGER.info(format_case_summary(result))
        return

    config = ApplicationConfig.model_validate(payload)
    result = run_simulation(config, hydra_run_dir=hydra_run_dir, project_root=project_root)
    LOGGER.info(format_run_summary(result))


if __name__ == "__main__":
    main()
