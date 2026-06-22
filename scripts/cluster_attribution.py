from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from torch import Tensor, nn

from trails.data import ClinicalTimeSeriesDataset, infer_data_config, make_data_loader
from trails.estimator import TrailsEstimator
from trails.model import TrailsSurvVaderModel
from trails.progress import ProgressBar

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

Assignment = Literal["posterior", "hard"]
DEFAULT_TOP_N_FEATURES = 6
TARGET_SCORE = "cluster_logit"
CJK_FONT_CANDIDATES = (
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Songti SC",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "SimHei",
    "Microsoft YaHei",
    "WenQuanYi Zen Hei",
)


class ClusterLogitWrapper(nn.Module):
    def __init__(self, model: TrailsSurvVaderModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: Tensor,
        times: Tensor,
        mask: Tensor,
        delta_time: Tensor | None = None,
        sequence_lengths: Tensor | None = None,
        feature_lengths: Tensor | None = None,
    ) -> Tensor:
        output = self.model(
            times=times,
            x=x,
            mask=mask,
            delta_time=delta_time,
            sequence_lengths=sequence_lengths,
            feature_lengths=feature_lengths,
        )
        return output.cluster_logits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Captum integrated-gradient cluster attribution for TRAILS models."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <model-path parent>/attribution.",
    )
    parser.add_argument("--target-clusters", type=int, nargs="*", default=())
    parser.add_argument("--n-time-bins", type=int, default=16)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--assignment", choices=("posterior", "hard"), default="posterior")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--plot-features",
        nargs="*",
        default=None,
        help=(
            "Feature names to plot, comma/space separated; a single positive integer selects "
            "the top-N features. Default: top 6."
        ),
    )
    parser.add_argument("--font-family", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    summary = run(args, raw_argv=raw_argv)
    print(f"Saved attribution outputs to {summary['output_dir']}")
    return summary


def run(args: argparse.Namespace, *, raw_argv: Sequence[str] = ()) -> dict[str, Any]:
    # 参数与路径
    model_path = Path(args.model_path).expanduser().resolve()
    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else model_path.parent / "attribution"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    n_time_bins = int(args.n_time_bins)
    ig_steps = int(args.ig_steps)
    if n_time_bins <= 0:
        raise ValueError("--n-time-bins must be greater than 0.")
    if ig_steps <= 0:
        raise ValueError("--ig-steps must be greater than 0.")
    assignment: Assignment = args.assignment

    # 载入模型和数据
    estimator = TrailsEstimator.load(model_path, device=args.device)
    dataset = ClinicalTimeSeriesDataset.load(data_path)
    inferred_data_config = infer_data_config(dataset)
    if inferred_data_config != estimator.config.data:
        raise ValueError(
            "Data shape does not match estimator config: "
            f"expected {estimator.config.data}, got {inferred_data_config}."
        )

    model = estimator.model
    model.eval()
    device = model.feature_means.device
    model_data = dataset.with_return_kind(
        "compact" if estimator.config.model.encoder.input.kind == "mtan2" else "aligned"
    )
    loader = make_data_loader(model_data, estimator.trainer.config, shuffle=False)

    target_clusters = tuple(int(value) for value in args.target_clusters)
    if not target_clusters:
        target_clusters = tuple(range(estimator.config.model.n_clusters))
    if any(value < 0 for value in target_clusters):
        raise ValueError("--target-clusters values must be non-negative.")
    invalid_clusters = [
        value for value in target_clusters if value >= estimator.config.model.n_clusters
    ]
    if invalid_clusters:
        raise ValueError(
            "--target-clusters must be valid cluster ids in "
            f"[0, {estimator.config.model.n_clusters}); got {invalid_clusters}."
        )
    if len(set(target_clusters)) != len(target_clusters):
        raise ValueError("--target-clusters values must be unique.")

    # 准备 IG 和累计张量
    min_time: float | None = None
    max_time: float | None = None
    for sample in dataset.with_return_kind("compact").samples:
        observed_times = sample.times[sample.mask > 0].float()
        if observed_times.numel() == 0:
            continue
        sample_min = float(observed_times.min().item())
        sample_max = float(observed_times.max().item())
        min_time = sample_min if min_time is None else min(min_time, sample_min)
        max_time = sample_max if max_time is None else max(max_time, sample_max)
    if min_time is None or max_time is None:
        raise ValueError("Attribution requires at least one observed time.")
    if min_time == max_time:
        min_time -= 0.5
        max_time += 0.5
    time_edges = torch.linspace(min_time, max_time, n_time_bins + 1, dtype=torch.float64)
    time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

    n_clusters = len(target_clusters)
    n_features = dataset.n_features
    feature_names = tuple(dataset.feature_names)
    attribution_sum = torch.zeros(n_clusters, n_features, n_time_bins, dtype=torch.float64)
    attribution_square_sum = torch.zeros_like(attribution_sum)
    abs_attribution_sum = torch.zeros_like(attribution_sum)
    abs_attribution_square_sum = torch.zeros_like(attribution_sum)
    weight_sum = torch.zeros_like(attribution_sum)
    observation_count = torch.zeros_like(attribution_sum)
    sample_count = torch.zeros_like(attribution_sum)

    integrated_gradients = IntegratedGradients(ClusterLogitWrapper(model))
    total_progress = len(loader) * n_clusters * ig_steps
    progress_bar: ProgressBar[Any] | None = (
        ProgressBar(desc="Attribution", total=total_progress) if not args.no_progress else None
    )

    # Integrated Gradients 主循环
    try:
        for batch_index, batch in enumerate(loader, start=1):
            device_batch = {name: value.to(device) for name, value in batch.items()}
            x = device_batch["x"].detach()
            times = device_batch["times"].detach()
            mask = device_batch["mask"].detach()
            delta_time = device_batch.get("delta_time")
            sequence_lengths = device_batch.get("sequence_lengths")
            feature_lengths = device_batch.get("feature_lengths")
            baseline = model.feature_means.to(device=device, dtype=x.dtype).view(1, 1, -1)
            baseline = baseline.expand_as(x)

            with torch.no_grad():
                model_output = model(
                    times=times,
                    x=x,
                    mask=mask,
                    delta_time=delta_time,
                    sequence_lengths=sequence_lengths,
                    feature_lengths=feature_lengths,
                )
                probabilities = model_output.cluster_probabilities.detach()
                target_index = torch.as_tensor(
                    target_clusters,
                    dtype=torch.long,
                    device=probabilities.device,
                )
                if assignment == "posterior":
                    assignment_weights = probabilities.index_select(dim=1, index=target_index)
                else:
                    predicted = torch.argmax(probabilities, dim=-1)
                    assignment_weights = torch.stack(
                        [
                            (predicted == int(cluster)).to(dtype=probabilities.dtype)
                            for cluster in target_clusters
                        ],
                        dim=1,
                    )

            expanded_times = times.unsqueeze(-1).expand_as(mask) if times.ndim == 2 else times
            observed = torch.nonzero(mask.detach().cpu() > 0, as_tuple=False)
            if observed.numel() == 0:
                continue
            observed_times = expanded_times.detach().cpu()[
                observed[:, 0],
                observed[:, 1],
                observed[:, 2],
            ]
            bin_indices_all = torch.bucketize(observed_times, time_edges[1:-1]).long()
            observed_sample_indices_all = observed[:, 0].long()
            observed_feature_indices_all = observed[:, 2].long()

            for target_position, target_cluster in enumerate(target_clusters):
                if progress_bar is not None:
                    progress_bar.set_postfix(
                        cluster=int(target_cluster),
                        batch=f"{batch_index}/{len(loader)}",
                    )
                attribution = integrated_gradients.attribute(
                    inputs=x,
                    baselines=baseline,
                    additional_forward_args=(
                        times,
                        mask,
                        delta_time,
                        sequence_lengths,
                        feature_lengths,
                    ),
                    target=int(target_cluster),
                    n_steps=ig_steps,
                )
                if isinstance(attribution, tuple):
                    attribution = attribution[0]
                if not isinstance(attribution, Tensor):
                    raise RuntimeError("Captum IntegratedGradients returned an unexpected payload.")
                if progress_bar is not None:
                    progress_bar.update(ig_steps)

                values = (attribution * (mask > 0).to(dtype=attribution.dtype)).detach()
                values_cpu = values.cpu().double()
                weights_cpu = assignment_weights[:, target_position].detach().cpu().double()
                observed_weights = weights_cpu[observed_sample_indices_all]
                positive = observed_weights > 0
                if not bool(positive.any()):
                    continue

                sample_indices = observed_sample_indices_all[positive]
                feature_indices = observed_feature_indices_all[positive]
                bin_indices = bin_indices_all[positive]
                flat_feature_bin = feature_indices * n_time_bins + bin_indices
                flat_values = values_cpu[
                    observed[positive, 0],
                    observed[positive, 1],
                    observed[positive, 2],
                ]
                flat_weights = weights_cpu[sample_indices]
                observation_count[target_position].view(-1).index_add_(
                    0,
                    flat_feature_bin,
                    torch.ones_like(flat_weights, dtype=observation_count.dtype),
                )

                # 先把同一病人、同一指标、同一时间 bin 的多个观测合并为样本级贡献。
                flat_sample_feature_bin = (
                    sample_indices * (n_features * n_time_bins)
                    + feature_indices * n_time_bins
                    + bin_indices
                )
                batch_size = int(mask.shape[0])
                sample_value_sum = torch.zeros(
                    batch_size * n_features * n_time_bins,
                    dtype=torch.float64,
                )
                sample_abs_sum = torch.zeros_like(sample_value_sum)
                sample_observation_count = torch.zeros_like(sample_value_sum)
                sample_value_sum.index_add_(0, flat_sample_feature_bin, flat_values)
                sample_abs_sum.index_add_(0, flat_sample_feature_bin, flat_values.abs())
                sample_observation_count.index_add_(
                    0,
                    flat_sample_feature_bin,
                    torch.ones_like(flat_values, dtype=torch.float64),
                )

                valid_sample_bins = torch.nonzero(
                    sample_observation_count > 0,
                    as_tuple=False,
                ).flatten()
                sample_bin = valid_sample_bins % (n_features * n_time_bins)
                contributing_samples = valid_sample_bins // (n_features * n_time_bins)
                sample_bin_weights = weights_cpu[contributing_samples]
                sample_means = (
                    sample_value_sum[valid_sample_bins]
                    / sample_observation_count[valid_sample_bins]
                )
                sample_abs_means = (
                    sample_abs_sum[valid_sample_bins] / sample_observation_count[valid_sample_bins]
                )

                attribution_sum[target_position].view(-1).index_add_(
                    0,
                    sample_bin,
                    sample_means * sample_bin_weights,
                )
                attribution_square_sum[target_position].view(-1).index_add_(
                    0,
                    sample_bin,
                    sample_means.square() * sample_bin_weights,
                )
                abs_attribution_sum[target_position].view(-1).index_add_(
                    0,
                    sample_bin,
                    sample_abs_means * sample_bin_weights,
                )
                abs_attribution_square_sum[target_position].view(-1).index_add_(
                    0,
                    sample_bin,
                    sample_abs_means.square() * sample_bin_weights,
                )
                weight_sum[target_position].view(-1).index_add_(
                    0,
                    sample_bin,
                    sample_bin_weights,
                )
                sample_count[target_position].view(-1).index_add_(
                    0,
                    sample_bin,
                    torch.ones_like(sample_bin_weights, dtype=sample_count.dtype),
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    # 汇总均值、SEM 和长表
    mean_attribution = torch.full_like(attribution_sum, float("nan"))
    mean_abs_attribution = torch.full_like(abs_attribution_sum, float("nan"))
    observed_bins = weight_sum > 0
    mean_attribution[observed_bins] = attribution_sum[observed_bins] / weight_sum[observed_bins]
    mean_abs_attribution[observed_bins] = (
        abs_attribution_sum[observed_bins] / weight_sum[observed_bins]
    )

    sem_attribution = torch.full_like(attribution_sum, float("nan"))
    sem_abs_attribution = torch.full_like(abs_attribution_sum, float("nan"))
    sem_bins = (weight_sum > 0) & (sample_count > 1)
    variance = torch.zeros_like(attribution_sum)
    abs_variance = torch.zeros_like(abs_attribution_sum)
    variance[sem_bins] = (
        attribution_square_sum[sem_bins] / weight_sum[sem_bins]
        - mean_attribution[sem_bins].square()
    )
    abs_variance[sem_bins] = (
        abs_attribution_square_sum[sem_bins] / weight_sum[sem_bins]
        - mean_abs_attribution[sem_bins].square()
    )
    sem_attribution[sem_bins] = torch.sqrt(
        variance[sem_bins].clamp_min(0.0) / sample_count[sem_bins]
    )
    sem_abs_attribution[sem_bins] = torch.sqrt(
        abs_variance[sem_bins].clamp_min(0.0) / sample_count[sem_bins]
    )

    records: list[dict[str, Any]] = []
    for target_position, target_cluster in enumerate(target_clusters):
        for feature_index, feature_name in enumerate(feature_names):
            for bin_index, time_center in enumerate(time_centers.tolist()):
                records.append(
                    {
                        "cluster": int(target_cluster),
                        "feature": feature_name,
                        "feature_index": feature_index,
                        "time_bin": bin_index,
                        "time_start": float(time_edges[bin_index].item()),
                        "time_end": float(time_edges[bin_index + 1].item()),
                        "time_center": float(time_center),
                        "mean_attribution": float(
                            mean_attribution[target_position, feature_index, bin_index].item()
                        ),
                        "mean_abs_attribution": float(
                            mean_abs_attribution[target_position, feature_index, bin_index].item()
                        ),
                        "sem_attribution": float(
                            sem_attribution[target_position, feature_index, bin_index].item()
                        ),
                        "sem_abs_attribution": float(
                            sem_abs_attribution[target_position, feature_index, bin_index].item()
                        ),
                        "observation_count": int(
                            observation_count[target_position, feature_index, bin_index].item()
                        ),
                        "sample_count": int(
                            sample_count[target_position, feature_index, bin_index].item()
                        ),
                        "weight_sum": float(
                            weight_sum[target_position, feature_index, bin_index].item()
                        ),
                    }
                )
    attribution_table = pd.DataFrame.from_records(records)

    # 选择绘图指标
    plot_tokens: list[str] = []
    for value in args.plot_features or []:
        plot_tokens.extend(token.strip() for token in str(value).split(",") if token.strip())
    feature_scores: dict[str, float] = {}
    for feature_index, feature_name in enumerate(feature_names):
        values = mean_abs_attribution[:, feature_index, :]
        finite = values[torch.isfinite(values)]
        feature_scores[feature_name] = (
            float(finite.mean().item()) if finite.numel() else float("-inf")
        )

    if not plot_tokens:
        selection_mode = "top_n"
        top_n = min(DEFAULT_TOP_N_FEATURES, n_features)
    elif len(plot_tokens) == 1 and plot_tokens[0].isdigit():
        selection_mode = "top_n"
        top_n = int(plot_tokens[0])
        if top_n <= 0:
            raise ValueError("--plot-features as a number must be a positive integer.")
        top_n = min(top_n, n_features)
    else:
        selection_mode = "names"
        top_n = None

    if selection_mode == "top_n":
        ranked_features = sorted(feature_scores.items(), key=lambda item: (-item[1], item[0]))
        plot_features = [feature_name for feature_name, _score in ranked_features[:top_n]]
    else:
        plot_features = plot_tokens

    feature_to_index = {feature_name: index for index, feature_name in enumerate(feature_names)}
    missing_features = [
        feature_name for feature_name in plot_features if feature_name not in feature_to_index
    ]
    if missing_features:
        preview = ", ".join(feature_names[:8])
        raise ValueError(
            "plot_features contains unknown feature names: "
            f"{', '.join(missing_features)}. Available features include: {preview}."
        )
    plot_feature_indices = [feature_to_index[feature_name] for feature_name in plot_features]
    selected_feature_scores = {
        feature_name: feature_scores[feature_name] for feature_name in plot_features
    }
    feature_selection = {
        "feature_indices": plot_feature_indices,
        "features": plot_features,
        "mode": selection_mode,
        "raw": list(args.plot_features or []),
        "scores": selected_feature_scores,
        "top_n": top_n,
    }

    # 保存结果和图形
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    plot_warnings: list[str] = []
    selected_font: str | None = None
    if args.font_family:
        if args.font_family in available_fonts:
            selected_font = args.font_family
        else:
            plot_warnings.append(
                f"Requested font_family '{args.font_family}' was not found; "
                "trying CJK fallback fonts."
            )
    if selected_font is None:
        selected_font = next(
            (candidate for candidate in CJK_FONT_CANDIDATES if candidate in available_fonts),
            None,
        )
    if selected_font is None:
        plot_warnings.append(
            "No common CJK font was found by matplotlib; Chinese feature names may render as boxes."
        )
    else:
        matplotlib.rcParams["font.family"] = [selected_font]

    ncols = 1 if n_clusters == 1 else 2
    nrows = (n_clusters + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharex=True,
        figsize=(max(7.2, 5.2 * ncols), max(3.6, 3.1 * nrows)),
        squeeze=False,
    )
    axis_list = list(axes.flat)
    x_values = time_centers.detach().cpu().float().numpy()
    for target_position, target_cluster in enumerate(target_clusters):
        ax = axis_list[target_position]
        for feature_index in plot_feature_indices:
            y_values = (
                mean_abs_attribution[target_position, feature_index].detach().cpu().float().numpy()
            )
            y_errors = (
                sem_abs_attribution[target_position, feature_index].detach().cpu().float().numpy()
            )
            ax.errorbar(
                x_values,
                y_values,
                yerr=y_errors,
                marker="o",
                markersize=3.2,
                linewidth=1.4,
                capsize=2.5,
                label=feature_names[feature_index],
            )
        ax.set_title(f"Cluster {target_cluster}", loc="left", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
        ax.grid(True, axis="x", alpha=0.12, linewidth=0.6)
    for ax in axis_list[n_clusters:]:
        ax.set_visible(False)
    handles, labels = axis_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 4),
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
    fig.suptitle("Cluster attribution over time", fontsize=13, fontweight="bold", y=0.975)
    fig.supxlabel("Time")
    fig.supylabel("Mean absolute attribution")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91 if handles else 0.95))

    csv_path = output_dir / "cluster_attributions.csv"
    tensor_path = output_dir / "cluster_attributions.pt"
    summary_path = output_dir / "attribution_summary.json"
    args_path = output_dir / "args.json"
    png_path = output_dir / "attribution_lines.png"
    pdf_path = output_dir / "attribution_lines.pdf"
    attribution_table.to_csv(csv_path, index=False)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    argparse_payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            argparse_payload[key] = str(value)
        elif isinstance(value, tuple):
            argparse_payload[key] = list(value)
        else:
            argparse_payload[key] = value
    args_payload = {
        "argparse": argparse_payload,
        "data_path": str(data_path),
        "feature_selection": feature_selection,
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "raw_argv": list(raw_argv),
    }
    tensor_payload = {
        "abs_attribution_square_sum": abs_attribution_square_sum,
        "abs_attribution_sum": abs_attribution_sum,
        "assignment": assignment,
        "attribution_square_sum": attribution_square_sum,
        "attribution_sum": attribution_sum,
        "feature_names": list(feature_names),
        "feature_selection": feature_selection,
        "mean_abs_attribution": mean_abs_attribution,
        "mean_attribution": mean_attribution,
        "observation_count": observation_count,
        "sample_count": sample_count,
        "sem_abs_attribution": sem_abs_attribution,
        "sem_attribution": sem_attribution,
        "target_clusters": torch.as_tensor(target_clusters, dtype=torch.long),
        "target_score": TARGET_SCORE,
        "time_centers": time_centers,
        "time_edges": time_edges,
        "weight_sum": weight_sum,
    }
    torch.save(
        {
            "args": args_payload,
            "attribution": tensor_payload,
            "data_path": str(data_path),
            "model_path": str(model_path),
        },
        tensor_path,
    )

    summary = {
        "args": args_payload,
        "attribution": {
            "assignment": assignment,
            "ig_steps": ig_steps,
            "n_time_bins": n_time_bins,
            "target_clusters": [int(value) for value in target_clusters],
            "target_score": TARGET_SCORE,
        },
        "data": {
            "data_path": str(data_path),
            "description": dataset.description,
            "n_features": dataset.n_features,
            "n_patients": len(dataset),
        },
        "feature_selection": feature_selection,
        "model": {
            "model_path": str(model_path),
            "n_clusters": estimator.config.model.n_clusters,
            "trails_config": estimator.config.model_dump(mode="json"),
        },
        "output_dir": str(output_dir),
        "outputs": {
            "args": str(args_path),
            "csv": str(csv_path),
            "line_plot": {
                "error_bar": "sem",
                "features": plot_features,
                "font_family": selected_font,
                "pdf": str(pdf_path),
                "png": str(png_path),
                "value": "abs",
                "warnings": plot_warnings,
            },
            "summary": str(summary_path),
            "tensor": str(tensor_path),
        },
    }
    args_path.write_text(
        json.dumps(args_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    main()
