"""Shared search utilities used by multiple experiments."""

from __future__ import annotations

from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
)


def build_temporal_filter(query_time: datetime) -> SearchFilters:
    """Build a SearchFilter that only returns edges valid at query_time."""
    return SearchFilters(
        valid_at=[
            [
                DateFilter(
                    date=query_time,
                    comparison_operator=ComparisonOperator.less_than_equal,
                )
            ]
        ],
        invalid_at=[
            [DateFilter(comparison_operator=ComparisonOperator.is_null)],
            [
                DateFilter(
                    date=query_time,
                    comparison_operator=ComparisonOperator.greater_than,
                )
            ],
        ],
    )


async def run_search(
    graphiti: Graphiti,
    query: str,
    config: SearchConfig,
    group_id: str,
    query_time: datetime | None = None,
) -> list[str]:
    """Execute a search and return fact strings."""
    search_filter = build_temporal_filter(query_time) if query_time else None
    results = await graphiti.search_(
        query=query, config=config, group_ids=[group_id], search_filter=search_filter
    )
    return [edge.fact for edge in results.edges]
