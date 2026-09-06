"""器官支持与治疗措施的簇间描述和总体检验。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .intervention_extraction import (
    INTERVENTION_BINARY_COLUMNS,
    INTERVENTION_CONTINUOUS_COLUMNS,
)

CONTINUOUS_EXPOSURES = {
    "invasive_ventilation_hours": "invasive_ventilation_any",
    "invasive_ventilation_fraction": "invasive_ventilation_any",
    "noninvasive_ventilation_hours": "noninvasive_ventilation_any",
    "noninvasive_ventilation_fraction": "noninvasive_ventilation_any",
    "hfnc_hours": "hfnc_any",
    "hfnc_fraction": "hfnc_any",
    "vasopressor_hours": "vasopressor_any",
    "vasopressor_fraction": "vasopressor_any",
    "vasopressor_ned_peak": "vasopressor_any",
    "vasopressor_ned_twa": "vasopressor_any",
    "antibiotic_agent_count": "antibiotic_any",
    "antibiotic_first_start_hours": "antibiotic_any",
}
DISPLAY_LABELS = {
    "invasive_ventilation_any": "Invasive ventilation",
    "invasive_ventilation_hours": "Invasive ventilation duration",
    "invasive_ventilation_fraction": "Invasive ventilation proportion",
    "noninvasive_ventilation_any": "Noninvasive ventilation",
    "noninvasive_ventilation_hours": "Noninvasive ventilation duration",
    "noninvasive_ventilation_fraction": "Noninvasive ventilation proportion",
    "hfnc_any": "High-flow nasal cannula",
    "hfnc_hours": "High-flow nasal cannula duration",
    "hfnc_fraction": "High-flow nasal cannula proportion",
    "vasopressor_any": "Vasopressor exposure",
    "vasopressor_hours": "Vasopressor duration",
    "vasopressor_fraction": "Vasopressor proportion",
    "vasopressor_ned_peak": "Peak total NED",
    "vasopressor_ned_twa": "Time-weighted total NED",
    "rrt_any": "Renal replacement therapy",
    "crrt_any": "Continuous renal replacement therapy",
    "antibiotic_any": "Antibiotic exposure",
    "antibiotic_agent_count": "Antibiotic agent count",
    "antibiotic_first_start_hours": "First antibiotic start",
}
DISPLAY_UNITS = {
    "invasive_ventilation_hours": "Hours",
    "invasive_ventilation_fraction": "Observed-window proportion",
    "noninvasive_ventilation_hours": "Hours",
    "noninvasive_ventilation_fraction": "Observed-window proportion",
    "hfnc_hours": "Hours",
    "hfnc_fraction": "Observed-window proportion",
    "vasopressor_hours": "Hours",
    "vasopressor_fraction": "Observed-window proportion",
    "vasopressor_ned_peak": "mcg/kg/min",
    "vasopressor_ned_twa": "mcg/kg/min",
    "antibiotic_agent_count": "Distinct prescription labels",
    "antibiotic_first_start_hours": "Hours after ICU admission",
}


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    """对当前方法和划分内可估计的总体检验作Benjamini-Hochberg校正。"""
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    observed = p_values.dropna().sort_values(ascending=False)
    if observed.empty:
        return adjusted
    ranks = np.arange(len(observed), 0, -1, dtype=float)
    values = np.minimum.accumulate(observed.to_numpy() * len(observed) / ranks)
    adjusted.loc[observed.index] = np.minimum(values, 1.0)
    return adjusted


class ClusterInterventionAnalysis:
    def __init__(
        self, assignments: pd.DataFrame, interventions: pd.DataFrame, n_clusters: int
    ) -> None:
        required_assignments = {"patient_id", "pred_cluster"}
        required_interventions = {
            "patient_id",
            *INTERVENTION_BINARY_COLUMNS,
            *CONTINUOUS_EXPOSURES,
        }
        if missing := required_assignments - set(assignments):
            raise ValueError(f"聚类分配缺少字段：{sorted(missing)}")
        if missing := required_interventions - set(interventions):
            raise ValueError(f"治疗数据缺少字段：{sorted(missing)}")
        if assignments["patient_id"].duplicated().any():
            raise ValueError("聚类分配的patient_id必须唯一")
        if interventions["patient_id"].duplicated().any():
            raise ValueError("治疗数据的patient_id必须唯一")
        self.data = assignments.loc[:, ["patient_id", "pred_cluster"]].merge(
            interventions.loc[
                :, ["patient_id", *INTERVENTION_BINARY_COLUMNS, *CONTINUOUS_EXPOSURES]
            ],
            on="patient_id",
            how="left",
            validate="one_to_one",
        )
        if self.data[list(required_interventions - {"patient_id"})].isna().all(axis=1).any():
            raise ValueError("至少一个评价患者在治疗数据中不存在")
        labels = self.data["pred_cluster"].to_numpy(dtype=np.int64)
        if n_clusters < 1 or not np.isin(labels, np.arange(n_clusters)).all():
            raise ValueError("pred_cluster超出配置的簇标签范围")
        flags = self.data.loc[:, INTERVENTION_BINARY_COLUMNS]
        if not flags.isin((0, 1)).to_numpy().all():
            raise ValueError("治疗暴露标志必须为0或1且不能缺失")
        self.n_clusters = n_clusters

    @staticmethod
    def _categorical_test(data: pd.DataFrame, variable: str) -> dict[str, Any]:
        table = pd.crosstab(data["pred_cluster"], data[variable]).to_numpy(dtype=np.int64)
        table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
        if min(table.shape, default=0) < 2:
            return {"test": "Unavailable", "statistic": None, "p_value": None}
        chi_square, chi_p, _, expected = cast(
            tuple[float, float, float, np.ndarray],
            stats.chi2_contingency(table, correction=False),
        )
        if np.all(expected >= 5):
            test, statistic, p_value = "Pearson chi-square", chi_square, chi_p
        elif table.shape == (2, 2):
            statistic, p_value = cast(tuple[float, float], stats.fisher_exact(table))
            test = "Fisher exact"
        else:
            method = stats.MonteCarloMethod(n_resamples=9999, rng=np.random.default_rng(20260517))
            statistic, p_value, _, _ = cast(
                tuple[float, float, float, np.ndarray],
                stats.chi2_contingency(table, correction=False, method=method),
            )
            test = "Pearson chi-square (Monte Carlo)"
        denominator = table.sum() * (min(table.shape) - 1)
        return {
            "test": test,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "effect_name": "Cramer's V",
            "effect_size": float(np.sqrt(chi_square / denominator)),
        }

    @staticmethod
    def _continuous_test(data: pd.DataFrame, variable: str) -> dict[str, Any]:
        groups = [
            cast(pd.Series, pd.to_numeric(group[variable], errors="coerce"))
            .dropna()
            .to_numpy(dtype=float)
            for _, group in data.groupby("pred_cluster", observed=True)
        ]
        groups = [group for group in groups if len(group)]
        n_observed = sum(map(len, groups))
        if len(groups) < 2 or np.unique(np.concatenate(groups)).size < 2:
            return {"test": "Unavailable", "statistic": None, "p_value": None}
        result = stats.kruskal(*groups)
        return {
            "test": "Kruskal-Wallis",
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "effect_name": "epsilon-squared",
            "effect_size": (
                None
                if n_observed == len(groups)
                else max(
                    0.0,
                    (result.statistic - len(groups) + 1) / (n_observed - len(groups)),
                )
            ),
        }

    @staticmethod
    def _summaries(
        data: pd.DataFrame, variable: str, summary_type: str, n_clusters: int
    ) -> list[dict[str, Any]]:
        groups = [("overall", data)] + [
            (f"cluster_{cluster}", data.loc[data["pred_cluster"] == cluster])
            for cluster in range(n_clusters)
        ]
        rows: list[dict[str, Any]] = []
        for group_name, group in groups:
            exposure = CONTINUOUS_EXPOSURES.get(variable)
            values = cast(pd.Series, pd.to_numeric(group[variable], errors="coerce"))
            observed = values.dropna()
            row: dict[str, Any] = {
                "domain": (exposure or variable).removesuffix("_any"),
                "variable": variable,
                "population": "all" if exposure is None else f"{exposure}=1",
                "group": group_name,
                "n_eligible": len(group),
                "n_observed": len(observed),
                "n_missing": int(values.isna().sum()),
                "summary_type": summary_type,
            }
            if summary_type == "n (%)":
                row.update(count=int(observed.sum()), percent=float(100.0 * observed.mean()))
            elif len(observed):
                q25, median, q75 = observed.quantile([0.25, 0.5, 0.75])
                row.update(q25=float(q25), median=float(median), q75=float(q75))
            rows.append(row)
        return rows

    def calculate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        descriptive: list[dict[str, Any]] = []
        tests: list[dict[str, Any]] = []
        for variable in INTERVENTION_BINARY_COLUMNS:
            descriptive.extend(self._summaries(self.data, variable, "n (%)", self.n_clusters))
            tests.append(
                {
                    "domain": variable.removesuffix("_any"),
                    "variable": variable,
                    "population": "all",
                    **self._categorical_test(self.data, variable),
                }
            )
        for variable, exposure in CONTINUOUS_EXPOSURES.items():
            exposed = self.data.loc[self.data[exposure] == 1]
            descriptive.extend(self._summaries(exposed, variable, "median [IQR]", self.n_clusters))
            tests.append(
                {
                    "domain": exposure.removesuffix("_any"),
                    "variable": variable,
                    "population": f"{exposure}=1",
                    **self._continuous_test(exposed, variable),
                }
            )
        descriptive_frame = pd.DataFrame(descriptive)
        test_frame = pd.DataFrame(tests)
        test_frame["q_value_bh"] = _bh_adjust(cast(pd.Series, test_frame["p_value"]))
        return descriptive_frame, test_frame

    @staticmethod
    def _panel_title(tests: pd.DataFrame, variable: str) -> str:
        q_value = tests.loc[tests["variable"] == variable, "q_value_bh"].iloc[0]
        q_label = "NA" if pd.isna(q_value) else f"{float(q_value):.3g}"
        return f"{DISPLAY_LABELS[variable]}\nBH q={q_label}"

    def plot(self, descriptive: pd.DataFrame, tests: pd.DataFrame, output: Path) -> dict[str, str]:
        """绘制每簇暴露率，以及暴露患者的连续终点中位数和IQR。"""
        outputs: dict[str, str] = {}
        cluster_groups = [f"cluster_{cluster}" for cluster in range(self.n_clusters)]
        colors = [plt.get_cmap("tab10")(cluster % 10) for cluster in range(self.n_clusters)]

        figure, raw_axes = plt.subplots(2, 4, figsize=(14.0, 7.0), layout="constrained")
        for axis, variable in zip(raw_axes.ravel(), INTERVENTION_BINARY_COLUMNS, strict=False):
            rows = (
                descriptive.loc[
                    (descriptive["variable"] == variable)
                    & descriptive["group"].isin(cluster_groups)
                ]
                .set_index("group")
                .reindex(cluster_groups)
            )
            axis.bar(range(self.n_clusters), rows["percent"], color=colors)
            axis.set(
                title=self._panel_title(tests, variable),
                xlabel="Predicted cluster",
                ylabel="Exposed patients (%)",
                ylim=(0.0, 100.0),
                xticks=range(self.n_clusters),
            )
            axis.grid(axis="y", alpha=0.2)
        raw_axes.ravel()[-1].set_visible(False)
        for suffix in ("png", "pdf"):
            path = output / f"organ_support_treatment_binary.{suffix}"
            figure.savefig(path, dpi=220)
            outputs[f"binary_plot_{suffix}"] = str(path)
        plt.close(figure)

        figure, raw_axes = plt.subplots(3, 4, figsize=(14.0, 10.0), layout="constrained")
        for axis, variable in zip(raw_axes.ravel(), CONTINUOUS_EXPOSURES, strict=True):
            rows = (
                descriptive.loc[
                    (descriptive["variable"] == variable)
                    & descriptive["group"].isin(cluster_groups)
                ]
                .set_index("group")
                .reindex(cluster_groups)
            )
            intervals = rows.reindex(columns=["median", "q25", "q75"])
            observed = intervals["median"].notna()
            x_values = np.arange(self.n_clusters)[observed]
            medians = intervals.loc[observed, "median"].to_numpy(dtype=float)
            q25 = intervals.loc[observed, "q25"].to_numpy(dtype=float)
            q75 = intervals.loc[observed, "q75"].to_numpy(dtype=float)
            if len(x_values):
                axis.errorbar(
                    x_values,
                    medians,
                    yerr=np.vstack((medians - q25, q75 - medians)),
                    fmt="o",
                    color="#0072B2",
                    capsize=3,
                )
            else:
                axis.text(0.5, 0.5, "No observed data", ha="center", va="center")
            axis.set(
                title=self._panel_title(tests, variable),
                xlabel="Predicted cluster",
                ylabel=DISPLAY_UNITS[variable],
                xticks=range(self.n_clusters),
            )
            axis.grid(axis="y", alpha=0.2)
        for suffix in ("png", "pdf"):
            path = output / f"organ_support_treatment_continuous.{suffix}"
            figure.savefig(path, dpi=220)
            outputs[f"continuous_plot_{suffix}"] = str(path)
        plt.close(figure)
        return outputs

    def save(self, output: Path, *, include_plots: bool = True) -> dict[str, str]:
        """保存可直接汇入09评价摘要的聚合表和可选图形。"""
        output.mkdir(parents=True, exist_ok=True)
        descriptive, tests = self.calculate()
        descriptive_path = output / "organ_support_treatment_descriptive.csv"
        tests_path = output / "organ_support_treatment_tests.csv"
        descriptive.to_csv(descriptive_path, index=False)
        tests.to_csv(tests_path, index=False)
        outputs = {
            "descriptive_csv": str(descriptive_path),
            "tests_csv": str(tests_path),
        }
        if include_plots:
            outputs.update(self.plot(descriptive, tests, output))
        return outputs


assert set(CONTINUOUS_EXPOSURES) == set(INTERVENTION_CONTINUOUS_COLUMNS) - {"observed_window_hours"}
