"""独立读取MIMIC冻结结果，生成全部补充统计与图件。"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from trails_simulate.config import resolved_payload

from .config import MimicSupplementaryConfig
from .figure_plots import aggregate_tables, configure_plotting, save_source_data
from .reporting import evaluate_runs
from .supplementary_plots import candidate_diagnostics, plot_agreement, plot_umap


def run(config: MimicSupplementaryConfig) -> None:
    ##################################################
    # 一、完整各集统计、临床特征、23项轨迹及治疗连续指标。
    ##################################################
    configure_plotting()
    report = evaluate_runs(config, supplementary=True)
    tables = aggregate_tables(report)

    ##################################################
    # 二、候选诊断、训练历史、标签一致性及训练拟合的UMAP。
    ##################################################
    tables.update(candidate_diagnostics(config, report.output))
    plot_agreement(report)
    tables["umap_aggregate_bins"] = plot_umap(report, config)
    save_source_data(report.output, tables)


@hydra.main(config_path="../../configs", config_name="mimic/supplementary", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    run(MimicSupplementaryConfig.model_validate(resolved_payload(raw_config)))


if __name__ == "__main__":
    main()
