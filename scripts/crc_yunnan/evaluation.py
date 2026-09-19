"""CRC各评价集复用的统计；患者级中间结果仅留在进程内。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy import stats
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

from trails import TrailsPrediction

from .config import BASELINE_LABELS, CRCEvaluationConfig

SPLIT_LABELS = {"train": "训练集", "validation": "验证集", "test": "测试集"}


def cluster_statistics(frame: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """空簇保留为零人数，分配不确定性按最终模型后验计算。"""
    grouped = frame.groupby("pred_cluster", observed=True)
    table: pd.DataFrame = grouped.agg(  # type: ignore[reportAssignmentType]
        人数=("event", "size"),
        事件数=("event", "sum"),
        平均最大后验概率=("confidence", "mean"),
        平均归一化熵=("entropy", "mean"),
        平均预测风险=("risk_score", "mean"),
    ).reindex(range(n_clusters))
    table[["人数", "事件数"]] = table[["人数", "事件数"]].fillna(0).astype(int)
    table["占比（%）"] = table["人数"] / len(frame) * 100
    table["事件比例（%）"] = table["事件数"] / table["人数"].replace(0, np.nan) * 100
    return table.rename_axis("簇").reset_index()


def cluster_survival(
    frame: pd.DataFrame, n_clusters: int, months: list[int]
) -> tuple[pd.DataFrame, dict[int, KaplanMeierFitter], dict[str, Any]]:
    """KM对象供风险表绘图复用，时间点统计与曲线采用同一拟合。"""
    models: dict[int, KaplanMeierFitter] = {}
    records: list[dict[str, Any]] = []
    for _, group in frame.groupby("pred_cluster", observed=True):
        label = int(group["pred_cluster"].iloc[0])
        km = KaplanMeierFitter(label=f"簇{label}").fit(group["survival_time"] / 30, group["event"])
        models[label] = km
        for month in months:
            supported = month <= group["survival_time"].max() / 30
            bounds = km.confidence_interval_.reindex([float(month)], method="ffill").iloc[0]
            records.append(
                {
                    "簇": label,
                    "月份": month,
                    "人数": len(group),
                    "风险人数": int((group["survival_time"].to_numpy() >= month * 30).sum()),
                    "生存率": float(km.predict(month)) if supported else np.nan,
                    "95%CI下限": float(bounds.iloc[0]) if supported else np.nan,
                    "95%CI上限": float(bounds.iloc[1]) if supported else np.nan,
                    "不可计算原因": "" if supported else "超过该簇观察随访范围",
                }
            )
    reason = (
        "" if len(models) >= 2 and frame["event"].to_numpy().any() else "不足两个有患者的簇或无事件"
    )
    p_value = (
        float(
            multivariate_logrank_test(
                frame["survival_time"], frame["pred_cluster"], frame["event"]
            ).p_value
        )
        if not reason
        else None
    )
    return (
        pd.DataFrame(records),
        models,
        {
            "总体log-rank_p值": p_value,
            "不可计算原因": reason,
            "空簇数": n_clusters - len(models),
        },
    )


def adjusted_cox(frame: pd.DataFrame, reference: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """固定四项调整变量；收敛失败不改模型、不自动添加惩罚。"""
    columns = [
        "pred_cluster",
        "survival_time",
        "event",
        "Age",
        "Sex",
        "AJCC_8th_ed_Stage",
        "Primary_site",
    ]
    analysis: pd.DataFrame = frame.loc[:, columns].dropna().copy()
    summary: dict[str, Any] = {
        "参考簇": reference,
        "输入人数": len(frame),
        "完整病例人数": len(analysis),
        "缺失排除人数": len(frame) - len(analysis),
        "调整变量": ["年龄（每10岁）", "性别", "AJCC第8版分期", "原发部位"],
    }
    reason = ""
    if analysis.empty or not analysis["event"].to_numpy().any():
        reason = "完整病例无事件"
    elif analysis["pred_cluster"].nunique() < 2:
        reason = "完整病例不足两个有患者的簇"
    elif reference not in analysis["pred_cluster"].unique():
        reason = "当前数据集没有训练集确定的参考簇"
    result_columns = ["簇", "HR", "95%CI下限", "95%CI上限", "p值", "人数", "参考簇"]
    if reason:
        summary["不可计算原因"] = reason
        return pd.DataFrame(columns=result_columns), summary
    analysis["age_per_10"] = analysis["Age"] / 10
    formula = (
        f"C(pred_cluster, Treatment(reference={reference})) + age_per_10"
        " + C(Sex) + C(AJCC_8th_ed_Stage) + C(Primary_site)"
    )
    fitter = CoxPHFitter().fit(analysis, "survival_time", "event", formula=formula)
    coefficients = fitter.summary
    rows: list[dict[str, Any]] = []
    counts = analysis["pred_cluster"].value_counts().sort_index()
    for cluster, count in zip(counts.index.to_numpy(), counts.to_numpy(), strict=True):
        if int(cluster) == reference:
            hr, lower, upper, p_value = 1.0, 1.0, 1.0, np.nan
        else:
            row = coefficients.loc[
                f"C(pred_cluster, Treatment(reference={reference}))[T.{int(cluster)}]"
            ]
            hr, lower, upper, p_value = row[
                ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
            ].to_numpy(dtype=float)
        rows.append(
            {
                "簇": int(cluster),
                "HR": hr,
                "95%CI下限": lower,
                "95%CI上限": upper,
                "p值": p_value,
                "人数": int(count),
                "参考簇": int(cluster) == reference,
            }
        )
    summary["不可计算原因"] = ""
    return pd.DataFrame(rows, columns=result_columns), summary


def clinical_characteristics(
    frame: pd.DataFrame, n_clusters: int, config: CRCEvaluationConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """基线描述和总体检验；分类比例的分母包括缺失患者。"""
    numeric, categorical = config.columns.numeric_baseline, config.columns.categorical_baseline
    groups = [
        ("总体", frame),
        *[(f"簇{k}", frame.loc[frame["pred_cluster"].eq(k)]) for k in range(n_clusters)],
    ]
    rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed)
    for variable in [*numeric, *categorical]:
        label = BASELINE_LABELS.get(variable, variable)
        p_value, method, reason = np.nan, "", ""
        observed = frame.dropna(subset=[variable])
        if variable in numeric:
            samples = [
                group[variable].to_numpy(dtype=float)
                for _, group in observed.groupby("pred_cluster", observed=True)
            ]
            if len(samples) < 2:
                reason = "不足两个非空组"
            elif observed[variable].nunique() <= 1:
                reason = "所有有效值相同"
            else:
                p_value, method = float(stats.kruskal(*samples).pvalue), "Kruskal–Wallis"
            for group_name, group in groups:
                values = group[variable].dropna()
                rows.append(
                    {
                        "变量": label,
                        "类型": "连续",
                        "类别": "",
                        "组别": group_name,
                        "人数": len(group),
                        "有效人数": len(values),
                        "缺失人数": int(group[variable].isna().sum()),
                        "中位数": values.median(),
                        "下四分位数": values.quantile(0.25),
                        "上四分位数": values.quantile(0.75),
                        "类别人数": np.nan,
                        "比例（%）": np.nan,
                    }
                )
        else:
            contingency = pd.crosstab(observed["pred_cluster"], observed[variable]).to_numpy()
            if min(contingency.shape, default=0) < 2:
                reason = "有效簇或类别不足两个"
            else:
                chi: Any = stats.chi2_contingency(contingency, correction=False)
                if (chi.expected_freq < 5).any():
                    fisher: Any = stats.fisher_exact(
                        contingency,
                        method=stats.MonteCarloMethod(n_resamples=config.fisher_resamples, rng=rng),
                    )
                    p_value = float(fisher.pvalue)
                    method = "Monte Carlo Fisher"
                else:
                    p_value, method = float(chi.pvalue), "χ²"
            levels = sorted(observed[variable].astype(str).unique()) + ["缺失"]
            for group_name, group in groups:
                counts = group[variable].fillna("缺失").astype(str).value_counts()
                for level in levels:
                    count = int(counts.get(level, 0))
                    rows.append(
                        {
                            "变量": label,
                            "类型": "分类",
                            "类别": level,
                            "组别": group_name,
                            "人数": len(group),
                            "有效人数": int(group[variable].notna().sum()),
                            "缺失人数": int(group[variable].isna().sum()),
                            "中位数": np.nan,
                            "下四分位数": np.nan,
                            "上四分位数": np.nan,
                            "类别人数": count,
                            "比例（%）": count / len(group) * 100 if len(group) else np.nan,
                        }
                    )
        tests.append({"变量": label, "检验": method, "p值": p_value, "不可计算原因": reason})
    test_table = pd.DataFrame(tests)
    valid = test_table["p值"].notna()
    test_table["BH_q值"] = np.nan
    if valid.to_numpy().any():
        test_table.loc[valid, "BH_q值"] = stats.false_discovery_control(
            test_table.loc[valid, "p值"].to_numpy(dtype=float)
        )
    return pd.DataFrame(rows), test_table


def trajectory_statistics(
    observations: pd.DataFrame,
    patients: pd.DataFrame,
    parameters: pd.DataFrame,
    metadata: pd.DataFrame,
    n_clusters: int,
    landmark: float,
    bin_days: int,
) -> pd.DataFrame:
    """先还原原始尺度，再按患者分箱汇总；患者等权且不插值。"""
    features = parameters["feature"].tolist()
    params = parameters.set_index("feature").loc[features]
    values = observations[features].mul(params["scale"]).add(params["center"])
    logged = params.index[params["transform"].eq("log1p")].tolist()
    values[logged] = np.expm1(values[logged])
    restored = pd.concat([observations[["patient_id", "time_days"]], values], axis=1)
    restored = restored.merge(patients[["patient_id", "pred_cluster"]], on="patient_id")
    n_bins = int(np.ceil(landmark / bin_days))
    restored["时间箱"] = np.minimum((restored["time_days"] // bin_days).astype(int), n_bins - 1)
    long = restored.melt(
        id_vars=["patient_id", "pred_cluster", "时间箱"],
        value_vars=features,
        var_name="特征",
        value_name="数值",
    ).dropna(subset=["数值"])
    per_patient = (
        long.groupby(["patient_id", "pred_cluster", "时间箱", "特征"], observed=True)["数值"]
        .agg(患者中位数="median", 观测数="size")
        .reset_index()
    )
    grouped = per_patient.groupby(["pred_cluster", "时间箱", "特征"], observed=True)
    table: pd.DataFrame = grouped.agg(  # type: ignore[reportAssignmentType]
        观测人数=("patient_id", "size"),
        有效观测数=("观测数", "sum"),
        中位数=("患者中位数", "median"),
    )
    quantiles = grouped["患者中位数"].quantile(np.array([0.25, 0.75])).unstack()
    quantiles.columns = ["下四分位数", "上四分位数"]
    full = pd.MultiIndex.from_product(
        [range(n_clusters), range(n_bins), features], names=["pred_cluster", "时间箱", "特征"]
    )
    table = table.join(quantiles).reindex(full).reset_index()  # type: ignore[reportAssignmentType]
    table = table.rename(columns={"pred_cluster": "簇"})
    table[["观测人数", "有效观测数"]] = table[["观测人数", "有效观测数"]].fillna(0).astype(int)
    denominators = patients["pred_cluster"].value_counts().reindex(range(n_clusters), fill_value=0)
    table["簇总人数"] = table["簇"].map(denominators.to_dict())  # type: ignore[reportArgumentType]
    table["覆盖率（%）"] = table["观测人数"] / table["簇总人数"].replace(0, np.nan) * 100
    table["起始天"] = table["时间箱"] * bin_days
    table["结束天"] = np.minimum(table["起始天"] + bin_days, landmark)
    table["中点月份"] = (table["起始天"] + table["结束天"]) / 60
    feature_groups = dict(zip(metadata["feature"], metadata["feature_group"], strict=True))
    table["特征组"] = table["特征"].map(feature_groups)  # type: ignore[reportArgumentType]
    return table


def survival_evaluation(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    prediction: TrailsPrediction,
    config: CRCEvaluationConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """预测指标只使用训练删失分布；校准和KM均按当前评价集观察结局计算。"""
    event = frame["event"].to_numpy(dtype=bool)
    # 05从float32张量计算指标，沿用同一精度以保留相同的并列时间定义。
    time = frame["survival_time"].to_numpy(dtype=np.float32).astype(float)
    risk = frame["risk_score"].to_numpy(dtype=float)
    y_train = Surv.from_arrays(
        train["event"].to_numpy(dtype=bool),
        train["survival_time"].to_numpy(dtype=np.float32).astype(float),
    )
    target = Surv.from_arrays(event, time)
    months = np.arange(
        config.survival.curve_step_months,
        config.survival.months[-1] + 1,
        config.survival.curve_step_months,
    )
    times = months.astype(float) * 30
    probabilities = prediction.survival(times.tolist()).numpy().astype(float)
    summary: dict[str, Any] = {
        "人数": len(frame),
        "事件数": int(event.sum()),
        "Harrell_C-index": None,
        "IPCW_C-index": None,
        "IPCW截断月份": config.survival.months[-1],
        "IBS": None,
        "IBS起止月份": [int(months[0]), int(months[-1])],
        "删失分布来源": "训练集",
        "不可计算原因": {},
    }
    reasons: dict[str, str] = summary["不可计算原因"]
    comparable = event.any() and (
        time[event].min() < time.max()
        or ((~event).any() and time[event].min() <= time[~event].max())
    )
    if comparable:
        summary["Harrell_C-index"] = float(concordance_index_censored(event, time, risk)[0])
    else:
        reasons["Harrell_C-index"] = "无有效生存比较对"
    train_has_events = bool(train["event"].to_numpy().any())
    censoring = CensoringDistributionEstimator().fit(y_train) if train_has_events else None
    train_max = float(y_train["time"].max())
    support = (times >= time.min()) & (times < time.max()) & (times < train_max)
    censor_positive = np.zeros(len(times), dtype=bool)
    if censoring is not None and support.any():
        censor_positive[support] = np.asarray(censoring.predict_proba(times[support])) > 0
    support &= censor_positive
    # AUC的库实现会为全部事件计算IPCW；不隐式截断不受支持的事件。
    event_support = censoring is not None and event.any() and time[event].max() <= train_max
    if event_support and censoring is not None:
        event_support = bool((np.asarray(censoring.predict_proba(time[event])) > 0).all())
    in_tau = event & (time < times[-1])
    tau_comparable = in_tau.any() and (
        time[in_tau].min() < time.max()
        or ((~event).any() and time[in_tau].min() <= time[~event].max())
    )
    if tau_comparable and support[-1] and censoring is not None:
        summary["IPCW_C-index"] = float(
            concordance_index_ipcw(y_train, target, risk, tau=times[-1])[0]
        )
    else:
        reasons["IPCW_C-index"] = "截断内无可比较事件或训练删失分布不支持"
    curve = pd.DataFrame(
        {
            "月份": months,
            "风险人数": [int((time >= t).sum()) for t in times],
            "累积事件数": [int((event & (time <= t)).sum()) for t in times],
            "动态AUC": np.nan,
            "Brier": np.nan,
            "AUC不可计算原因": "",
            "Brier不可计算原因": "",
        }
    )
    for index, t in enumerate(times):
        if support[index] and event_support and (event & (time <= t)).any() and (time > t).any():
            auc, _ = cumulative_dynamic_auc(y_train, target, 1 - probabilities[:, index], [t])
            curve.loc[index, "动态AUC"] = float(auc[0])
        else:
            curve.loc[index, "AUC不可计算原因"] = "缺少病例/对照或训练事件及随访删失支持不足"
    brier_supported = support & (time.max() <= train_max) & event.any()
    if brier_supported.any():
        _, scores = brier_score(
            y_train, target, probabilities[:, brier_supported], times[brier_supported]
        )
        curve.loc[brier_supported, "Brier"] = scores
    curve.loc[~brier_supported, "Brier不可计算原因"] = "当前集无事件或训练事件及随访删失支持不足"
    for metric in ("AUC", "Brier"):
        unavailable = curve.loc[curve[metric + "不可计算原因"].ne("")]
        if not unavailable.empty:
            reasons[metric] = "；".join(unavailable[metric + "不可计算原因"].unique())
            summary[metric + "不可计算月份"] = unavailable["月份"].tolist()
    if brier_supported.all():
        summary["IBS"] = float(integrated_brier_score(y_train, target, probabilities, times))
    else:
        reasons["IBS"] = "预设积分网格未全部得到支持，不缩短积分区间"
    calibration: list[dict[str, Any]] = []
    for month in config.survival.months:
        column = int(np.flatnonzero(months == month)[0])
        predicted = probabilities[:, column]
        bins = np.asarray(
            pd.qcut(
                predicted,
                q=min(config.survival.calibration_bins, len(frame)),
                labels=False,
                duplicates="drop",
            ),
            dtype=float,
        )
        if np.isnan(bins).all():
            bins = np.zeros(len(frame))
        for group in np.unique(bins):
            selected = bins == group
            km = KaplanMeierFitter().fit(time[selected] / 30, event[selected])
            supported = month <= time[selected].max() / 30
            bounds = km.confidence_interval_.reindex([float(month)], method="ffill").iloc[0]
            calibration.append(
                {
                    "月份": month,
                    "分位组": int(group) + 1,
                    "人数": int(selected.sum()),
                    "实际分组数": len(np.unique(bins)),
                    "平均预测生存率": float(predicted[selected].mean()),
                    "KM生存率": float(km.predict(month)) if supported else np.nan,
                    "95%CI下限": float(bounds.iloc[0]) if supported else np.nan,
                    "95%CI上限": float(bounds.iloc[1]) if supported else np.nan,
                    "不可计算原因": "" if supported else "超过该组观察随访范围",
                }
            )
    km = KaplanMeierFitter().fit(time / 30, event)
    bounds = km.confidence_interval_.reindex(months.astype(float), method="ffill")
    overall = pd.DataFrame(
        {
            "月份": months,
            "平均预测生存率": probabilities.mean(axis=0),
            "KM生存率": km.predict(months).to_numpy(),
            "95%CI下限": bounds.iloc[:, 0].to_numpy(),
            "95%CI上限": bounds.iloc[:, 1].to_numpy(),
        }
    )
    beyond = months > time.max() / 30
    overall.loc[beyond, ["KM生存率", "95%CI下限", "95%CI上限"]] = np.nan
    overall["不可计算原因"] = np.where(beyond, "超过观察随访范围", "")
    return summary, curve, pd.DataFrame(calibration), overall
