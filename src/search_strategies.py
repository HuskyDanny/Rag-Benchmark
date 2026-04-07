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


# ── Method/reranker name → enum mappings ──

_METHOD_MAP = {
    "bm25": EdgeSearchMethod.bm25,
    "cosine_similarity": EdgeSearchMethod.cosine_similarity,
    "bfs": EdgeSearchMethod.bfs,
}

_RERANKER_MAP = {
    "rrf": EdgeReranker.rrf,
    "mmr": EdgeReranker.mmr,
    "cross_encoder": EdgeReranker.cross_encoder,
    "node_distance": EdgeReranker.node_distance,
    "episode_mentions": EdgeReranker.episode_mentions,
}

# Defaults matching Graphiti SDK
DEFAULT_SIM_MIN_SCORE = 0.6
DEFAULT_MMR_LAMBDA = 0.5
DEFAULT_BFS_MAX_DEPTH = 3


def build_search_config(params: dict) -> SearchConfig:
    """Build a Graphiti SearchConfig from a parameter dictionary.

    Supported keys:
      search_methods: list[str] (default ["bm25", "cosine_similarity"])
      reranker: str (default "rrf")
      sim_min_score: float (default 0.6)
      mmr_lambda: float (default 0.5)
      bfs_max_depth: int (default 3)
      reranker_min_score: float (default 0)
      limit: int (default 10)
    """
    method_names = params.get("search_methods", ["bm25", "cosine_similarity"])
    if isinstance(method_names, str):
        method_names = [method_names]
    methods = [_METHOD_MAP[m] for m in method_names]

    reranker_name = params.get("reranker", "rrf")
    reranker = _RERANKER_MAP[reranker_name]

    return SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=methods,
            reranker=reranker,
            sim_min_score=params.get("sim_min_score", DEFAULT_SIM_MIN_SCORE),
            mmr_lambda=params.get("mmr_lambda", DEFAULT_MMR_LAMBDA),
            bfs_max_depth=params.get("bfs_max_depth", DEFAULT_BFS_MAX_DEPTH),
        ),
        limit=params.get("limit", 10),
        reranker_min_score=params.get("reranker_min_score", 0),
    )
