"""Search strategy configurations for Graphiti benchmark."""

from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    SearchConfig,
)
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF


def get_search_strategies() -> dict[str, SearchConfig]:
    """Return a dict of named search strategies for benchmarking."""
    hybrid = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    hybrid.limit = 10

    bm25_only = SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25],
            reranker=EdgeReranker.rrf,
        ),
        limit=10,
    )

    cosine_only = SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.rrf,
        ),
        limit=10,
    )

    return {"hybrid": hybrid, "bm25_only": bm25_only, "cosine_only": cosine_only}
