from __future__ import annotations

from typing import NotRequired, TypedDict

from torch import Tensor


class LatentDiagnostics(TypedDict):
    z: Tensor
    cluster_probabilities: Tensor
    pred_cluster: Tensor
    sample_index: Tensor
    true_cluster: NotRequired[Tensor]
