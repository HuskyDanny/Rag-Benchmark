"""Evaluation metrics for search quality benchmarking."""

from __future__ import annotations

from src.judge import any_match, facts_match, find_matches


async def compute_precision_at_k(
    returned: list[str],
    expected: list[str],
    k: int = 5,
) -> float:
    """Precision@K = matched / k."""
    top_k = returned[:k]
    matches = await find_matches(top_k, expected)
    return len(matches) / k


async def compute_recall_at_k(
    returned: list[str],
    expected: list[str],
    k: int = 5,
) -> float:
    """Recall@K = matched / total_expected."""
    if not expected:
        return 1.0
    top_k = returned[:k]
    matches = await find_matches(top_k, expected)
    return len(matches) / len(expected)


async def compute_mrr(
    returned: list[str],
    expected: list[str],
) -> float:
    """Mean Reciprocal Rank: 1/(rank of first relevant result)."""
    for i, ret in enumerate(returned):
        for exp in expected:
            if await facts_match(ret, exp):
                return 1.0 / (i + 1)
    return 0.0


async def compute_temporal_accuracy(
    returned: list[str],
    expected_not: list[str],
) -> bool:
    """True if none of the unwanted facts appear in results."""
    found = await any_match(returned, expected_not)
    return len(found) == 0
