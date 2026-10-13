"""由显式指定的冻结候选生成MIMIC综合主图。"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from trails_simulate.config import resolved_payload

from .config import MimicFigureConfig
from .figure_plots import plot_main_figure, save_source_data
from .reporting import evaluate_runs


def run(config: MimicFigureConfig) -> None:
    ##################################################
    # 一、独立读取冻结结果，按方法能力计算共享统计。
    ##################################################
    report = evaluate_runs(config, supplementary=False)

    ##################################################
    # 二、绘制综合主图并保存同源聚合数据。
    ##################################################
    tables = plot_main_figure(report, config)
    save_source_data(report.output, tables)


@hydra.main(config_path="../../configs", config_name="mimic/figure", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    run(MimicFigureConfig.model_validate(resolved_payload(raw_config)))


if __name__ == "__main__":
    main()
