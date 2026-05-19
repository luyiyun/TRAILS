from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from trails_simulate.config import ApplicationConfig
from trails_simulate.summary import format_run_summary
from trails_simulate.workflow import run


def load_app_config(raw_config: DictConfig) -> ApplicationConfig:
    payload: Any = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Hydra config must resolve to a mapping.")
    try:
        return ApplicationConfig.model_validate(payload)
    except ValidationError as error:
        raise ValueError(str(error)) from error


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    config = load_app_config(raw_config)
    project_root = Path(get_original_cwd())
    hydra_run_dir = Path(HydraConfig.get().runtime.output_dir)
    result = run(config, hydra_run_dir=hydra_run_dir, project_root=project_root)
    print(format_run_summary(result))


if __name__ == "__main__":
    main()
