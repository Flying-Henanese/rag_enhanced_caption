"""Rank fusion utilities shared by retrieval integrations."""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, constant: int = 60
) -> dict[str, float]:
    """Fuse independent rankings without comparing their raw scores.

    Args:
        rankings: Ordered identifier lists from independent retrievers.
        constant: RRF smoothing constant.

    Returns:
        An insertion-ordered mapping sorted by descending fused score.
    """
    if constant < 0:
        raise ValueError("RRF constant must be non-negative")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (constant + rank)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
