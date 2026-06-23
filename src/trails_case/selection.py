from __future__ import annotations

from .config import CaseApplicationConfig


def case_k_selection_candidates(config: CaseApplicationConfig) -> tuple[int, ...]:
    if config.k_selection.candidate_clusters:
        return config.k_selection.candidate_clusters
    return tuple(range(2, config.model.n_clusters + 1))


def case_k_selection_valid_fraction(config: CaseApplicationConfig) -> float:
    valid_fraction = (
        config.trainer.valid_size
        if config.k_selection.valid_size is None
        else config.k_selection.valid_size
    )
    if valid_fraction <= 0.0 or valid_fraction >= 1.0:
        raise ValueError(
            "case K selection requires k_selection.valid_size or "
            "trainer.valid_size to be greater than 0 and less than 1."
        )
    return valid_fraction
