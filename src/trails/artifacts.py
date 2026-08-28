"""保存训练配置、历史、潜空间诊断和可视化产物。"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import matplotlib
import torch
from torch import Tensor

from .diagnostics import LatentDiagnostics
from .trainer import HistoryEntry

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ARTIFACT_NAMES = frozenset({"config", "history", "test", "model", "plot"})
ARTIFACT_TOKENS = tuple(sorted((*ARTIFACT_NAMES, "all", "none")))


def resolve_artifact_names(tokens: Sequence[str] | None) -> frozenset[str]:
    """将 CLI 产物选择标记解析为规范名称集合。

    空输入和 ``"all"`` 选择全部产物，单独的 ``"none"`` 关闭全部产物。

    参数：
        tokens: 用户提供的产物名称、``"all"`` 或 ``"none"``。

    返回：
        去重后的规范产物名称集合。

    异常：
        ValueError: 当名称未知或 ``"none"`` 与其他选择组合时抛出。
    """
    if tokens is None or len(tokens) == 0 or "all" in tokens:
        if "none" in (tokens or ()):
            raise ValueError("--save-artifacts cannot combine 'all' and 'none'.")
        return ARTIFACT_NAMES
    if "none" in tokens:
        if len(tokens) > 1:
            raise ValueError("--save-artifacts cannot combine 'none' with other artifacts.")
        return frozenset()

    unknown = sorted(set(tokens) - ARTIFACT_NAMES)
    if unknown:
        raise ValueError(f"Unknown artifact name(s): {', '.join(unknown)}.")
    return frozenset(tokens)


def create_timestamped_run_dir(base_dir: str | Path, created_at: datetime | None = None) -> Path:
    """在基础目录下创建唯一的时间戳运行目录。

    同一秒内目录重名时依次添加两位数字后缀，最多尝试 1000 个候选名称。

    参数：
        base_dir: 保存运行目录的根路径。
        created_at: 可选的指定时间；默认使用当前本地时区时间。

    返回：
        已成功创建的运行目录路径。

    异常：
        RuntimeError: 当全部候选目录名均已存在时抛出。
    """
    timestamp = (created_at or datetime.now().astimezone()).strftime("%Y%m%d-%H%M%S")
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)

    for suffix in range(1000):
        directory_name = timestamp if suffix == 0 else f"{timestamp}-{suffix:02d}"
        candidate = root / directory_name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique run directory under {root}.")


def save_json(path: str | Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    """以 UTF-8 编码和缩进格式保存 JSON 载荷，并自动创建父目录。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_history_csv(path: str | Path, history: Sequence[HistoryEntry]) -> None:
    """将分阶段训练历史展开为统一字段并保存为 CSV 文件。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = flatten_history(history)
    fieldnames = _history_fieldnames(rows)

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def plot_history(path: str | Path, history: Sequence[HistoryEntry]) -> None:
    """绘制可用的损失、C-index 和聚类恢复训练曲线。

    面板仅包含历史中实际存在数值的指标，并在 warmup 与正式训练阶段发生变化
    时标记边界。无数值指标时仍会生成带说明的占位图。

    参数：
        path: 输出图片路径，格式由扩展名决定。
        history: 每轮训练与验证指标组成的历史记录。
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    rows = flatten_history(history)
    panels = _available_panels(rows)
    if not panels:
        panels = [("History", [])]

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(10.0, max(3.2, 2.8 * len(panels))),
        sharex=True,
        constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]

    x_values = _x_values(rows)
    for ax, (title, names) in zip(axes, panels, strict=True):
        if names:
            for name in names:
                y_values = _metric_values(rows, name)
                ax.plot(x_values, y_values, marker="o", linewidth=1.8, markersize=3.5, label=name)
            ax.legend(frameon=False, ncols=min(3, len(names)))
        else:
            ax.text(0.5, 0.5, "No numeric history metrics", ha="center", va="center")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _mark_stage_boundary(ax, rows, x_values)

    axes[-1].set_xlabel("Global epoch")
    fig.suptitle("TRAILS training history", fontsize=13, fontweight="bold")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_latent_embedding_artifacts(
    root_dir: str | Path,
    split_name: str,
    diagnostics: LatentDiagnostics,
    *,
    random_state: int,
) -> dict[str, str]:
    """保存指定数据划分的潜空间张量及 PCA/UMAP 投影图。

    参数：
        root_dir: 运行产物根目录。
        split_name: 数据划分名称，用于文件名和图标题。
        diagnostics: 患者潜空间嵌入、簇概率、预测及可选真值标签。
        random_state: UMAP 降维使用的随机种子。

    返回：
        包含 ``"data"`` 和 ``"plot"`` 文件路径的字典。
    """
    destination = Path(root_dir) / "latent_embeddings"
    destination.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {"split": split_name}
    payload.update(diagnostics)
    data_path = destination / f"{split_name}_embeddings.pt"
    torch.save(payload, data_path)

    plot_path = destination / f"{split_name}_pca_umap.png"
    plot_latent_embedding_projection(
        plot_path,
        z=diagnostics["z"],
        pred_cluster=diagnostics["pred_cluster"],
        true_cluster=diagnostics.get("true_cluster"),
        split_name=split_name,
        random_state=random_state,
    )
    return {"data": str(data_path), "plot": str(plot_path)}


def plot_latent_embedding_projection(
    path: str | Path,
    *,
    z: Tensor,
    pred_cluster: Tensor,
    true_cluster: Tensor | None,
    split_name: str,
    random_state: int,
) -> None:
    """绘制按预测簇和可选真值标签着色的 PCA/UMAP 潜空间投影。

    参数：
        path: 输出图片路径。
        z: 形状为 ``(n_samples, latent_dim)`` 的潜空间嵌入。
        pred_cluster: 每个样本的预测簇标签。
        true_cluster: 可选的参考簇标签。
        split_name: 显示在图标题中的数据划分名称。
        random_state: UMAP 降维随机种子。
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    pca = _pca_projection(z)
    umap_projection = _umap_projection(z, random_state=random_state)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), constrained_layout=True)
    panels = [
        (axes[0, 0], pca, true_cluster, "PCA by true label", "No true labels"),
        (axes[0, 1], pca, pred_cluster, "PCA by predicted cluster", "No predictions"),
        (axes[1, 0], umap_projection, true_cluster, "UMAP by true label", "No true labels"),
        (
            axes[1, 1],
            umap_projection,
            pred_cluster,
            "UMAP by predicted cluster",
            "No predictions",
        ),
    ]
    for ax, coordinates, labels, title, unavailable_text in panels:
        _plot_labeled_scatter(
            ax,
            coordinates,
            labels,
            title=title,
            unavailable_text=unavailable_text,
        )

    fig.suptitle(f"TRAILS latent embeddings: {split_name}", fontsize=13, fontweight="bold")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def flatten_history(history: Sequence[HistoryEntry]) -> list[dict[str, Any]]:
    """将嵌套的训练/验证历史转换为适合 CSV 和绘图的扁平记录。"""
    rows: list[dict[str, Any]] = []
    for entry in history:
        row: dict[str, Any] = {}
        for name in (
            "global_epoch",
            "epoch",
            "stage",
            "best_global_epoch",
            "best_monitor",
            "best_monitor_value",
            "early_stopped",
        ):
            value = entry.get(name)
            if value is not None:
                row[name] = value
        row.update(_history_metrics(entry.get("train")))
        row.update(
            {f"val_{name}": value for name, value in _history_metrics(entry.get("valid")).items()}
        )
        rows.append(row)
    return rows


def _history_metrics(value: object) -> dict[str, int | float]:
    """从历史子记录中提取数值指标。"""
    if not isinstance(value, Mapping):
        return {}
    return {str(name): metric for name, metric in value.items() if isinstance(metric, int | float)}


def _history_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """按首选顺序生成历史 CSV 字段名。"""
    preferred = [
        "global_epoch",
        "epoch",
        "stage",
        "best_global_epoch",
        "best_monitor",
        "best_monitor_value",
        "early_stopped",
        "loss",
        "reconstruction_loss",
        "survival_loss",
        "vade_kl_loss",
        "reconstruction_loss_weight",
        "survival_loss_weight",
        "vade_kl_loss_weight",
        "reconstruction_log_variance",
        "survival_log_variance",
        "cindex",
        "acc",
        "ari",
        "nmi",
        "val_loss",
        "val_reconstruction_loss",
        "val_survival_loss",
        "val_vade_kl_loss",
        "val_reconstruction_loss_weight",
        "val_survival_loss_weight",
        "val_vade_kl_loss_weight",
        "val_reconstruction_log_variance",
        "val_survival_log_variance",
        "val_cindex",
        "val_ari",
        "val_nmi",
    ]
    present = {key for row in rows for key in row}
    ordered = [name for name in preferred if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _available_panels(history: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[str]]]:
    """返回历史中具有数值的绘图面板及指标。"""
    panel_candidates = [
        ("Total loss", ["loss", "val_loss"]),
        (
            "Loss components",
            [
                "reconstruction_loss",
                "survival_loss",
                "vade_kl_loss",
                "val_reconstruction_loss",
                "val_survival_loss",
                "val_vade_kl_loss",
            ],
        ),
        ("Concordance", ["cindex", "val_cindex"]),
        ("Cluster recovery", ["acc", "ari", "nmi", "val_acc", "val_ari", "val_nmi"]),
    ]
    return [
        (title, [name for name in names if _has_numeric_values(history, name)])
        for title, names in panel_candidates
        if any(_has_numeric_values(history, name) for name in names)
    ]


def _x_values(history: Sequence[Mapping[str, Any]]) -> list[float]:
    """优先使用全局轮次生成历史曲线横坐标。"""
    if all(isinstance(entry.get("global_epoch"), int | float) for entry in history):
        return [float(entry["global_epoch"]) for entry in history]
    return [float(index + 1) for index in range(len(history))]


def _metric_values(history: Sequence[Mapping[str, Any]], name: str) -> list[float]:
    """提取指定指标，并以 NaN 填充缺失轮次。"""
    values: list[float] = []
    for entry in history:
        value = entry.get(name)
        values.append(float(value) if isinstance(value, int | float) else float("nan"))
    return values


def _has_numeric_values(history: Sequence[Mapping[str, Any]], name: str) -> bool:
    """判断历史中是否存在指定数值指标。"""
    return any(isinstance(entry.get(name), int | float) for entry in history)


def _mark_stage_boundary(
    ax: Any,
    history: Sequence[Mapping[str, Any]],
    x_values: Sequence[float],
) -> None:
    """在曲线上标记第一次训练阶段切换的位置。"""
    stages = [entry.get("stage") for entry in history]
    if not stages or not x_values:
        return

    first_stage = stages[0]
    for index, stage in enumerate(stages[1:], start=1):
        if stage != first_stage:
            boundary = (x_values[index - 1] + x_values[index]) / 2.0
            ax.axvline(boundary, color="0.45", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.text(
                boundary,
                0.98,
                f"{first_stage} -> {stage}",
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=8,
                color="0.35",
            )
            return


def _pca_projection(z: Tensor) -> Tensor:
    """通过 SVD 计算二维 PCA 投影，并处理小样本或单分量情形。"""
    embeddings = z.detach().cpu().float()
    n_samples = int(embeddings.shape[0])
    if n_samples <= 1:
        return embeddings.new_zeros((n_samples, 2))

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    n_components = min(2, int(vh.shape[0]))
    projection = centered @ vh[:n_components].T
    if n_components == 2:
        return projection
    return torch.cat([projection, projection.new_zeros((n_samples, 2 - n_components))], dim=1)


def _umap_projection(z: Tensor, *, random_state: int) -> Tensor:
    """计算二维 UMAP 投影，小于四个样本时退化为 PCA。"""
    embeddings = z.detach().cpu().float()
    n_samples = int(embeddings.shape[0])
    if n_samples <= 3:
        return _pca_projection(embeddings)

    try:
        umap_module: Any = import_module("umap")
    except ImportError as error:
        raise RuntimeError(
            "diagnostics.latent_embeddings.enabled requires umap-learn. "
            "Add umap-learn to pyproject.toml and run uv sync before using debug diagnostics."
        ) from error

    n_neighbors = min(15, max(2, n_samples - 1))
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    return torch.as_tensor(reducer.fit_transform(embeddings.numpy()), dtype=torch.float32)


def _plot_labeled_scatter(
    ax: Any,
    coordinates: Tensor,
    labels: Tensor | None,
    *,
    title: str,
    unavailable_text: str,
) -> None:
    """绘制按离散标签着色的二维散点面板。"""
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if labels is None:
        ax.text(0.5, 0.5, unavailable_text, ha="center", va="center", transform=ax.transAxes)
        return

    label_values = labels.detach().cpu().long()
    points = coordinates.detach().cpu().float()
    for label in torch.unique(label_values, sorted=True):
        mask = label_values == label
        ax.scatter(
            points[mask, 0].tolist(),
            points[mask, 1].tolist(),
            s=18,
            alpha=0.78,
            label=str(int(label.item())),
        )
    ax.legend(title="label", frameon=False, markerscale=1.2)
