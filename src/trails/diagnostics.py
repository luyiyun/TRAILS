"""TRAILS 返回的潜空间表示所使用的类型定义。"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from torch import Tensor


class LatentDiagnostics(TypedDict):
    """用于诊断和可视化的患者级潜空间输出。

    属性：
        z: 按 ``sample_index`` 排列的患者潜空间嵌入。
        cluster_probabilities: 每个混合分量的后验概率。
        pred_cluster: 每位患者后验概率最大的簇分配。
        sample_index: 每行输出所对应的原始数据集索引。
        true_cluster: 数据集提供参考簇标签时包含的可选字段。
    """

    z: Tensor
    cluster_probabilities: Tensor
    pred_cluster: Tensor
    sample_index: Tensor
    true_cluster: NotRequired[Tensor]
