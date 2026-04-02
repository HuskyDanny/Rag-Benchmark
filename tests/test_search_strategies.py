import pytest


def test_get_strategies_returns_three():
    from src.search_strategies import get_search_strategies

    strategies = get_search_strategies()
    assert set(strategies.keys()) == {"hybrid", "bm25_only", "cosine_only"}


def test_hybrid_uses_multiple_methods():
    from src.search_strategies import get_search_strategies

    strategies = get_search_strategies()
    hybrid = strategies["hybrid"]
    assert hybrid.edge_config is not None
    assert len(hybrid.edge_config.search_methods) >= 2


def test_bm25_only_uses_bm25():
    from src.search_strategies import get_search_strategies
    from graphiti_core.search.search_config import EdgeSearchMethod

    strategies = get_search_strategies()
    bm25 = strategies["bm25_only"]
    assert bm25.edge_config is not None
    assert bm25.edge_config.search_methods == [EdgeSearchMethod.bm25]


def test_cosine_only_uses_cosine():
    from src.search_strategies import get_search_strategies
    from graphiti_core.search.search_config import EdgeSearchMethod

    strategies = get_search_strategies()
    cosine = strategies["cosine_only"]
    assert cosine.edge_config is not None
    assert cosine.edge_config.search_methods == [EdgeSearchMethod.cosine_similarity]
