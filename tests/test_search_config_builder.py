"""Tests for the parameterized search config builder."""

from __future__ import annotations

from src.search_strategies import build_search_config


def test_build_default_config():
    config = build_search_config({})
    assert config.limit == 10
    assert config.edge_config is not None
    assert len(config.edge_config.search_methods) == 2


def test_build_with_custom_mmr():
    config = build_search_config({"reranker": "mmr", "mmr_lambda": 0.3})
    assert config.edge_config.mmr_lambda == 0.3


def test_build_with_bfs():
    config = build_search_config(
        {
            "search_methods": ["bm25", "cosine_similarity", "bfs"],
            "bfs_max_depth": 2,
        }
    )
    assert len(config.edge_config.search_methods) == 3
    assert config.edge_config.bfs_max_depth == 2


def test_build_with_custom_limit():
    config = build_search_config({"limit": 20})
    assert config.limit == 20


def test_build_with_sim_min_score():
    config = build_search_config({"sim_min_score": 0.8})
    assert config.edge_config.sim_min_score == 0.8


def test_build_with_reranker_min_score():
    config = build_search_config({"reranker_min_score": 0.1})
    assert config.reranker_min_score == 0.1


def test_build_single_method_string():
    """Support passing a single method as string instead of list."""
    config = build_search_config({"search_methods": "bm25"})
    assert len(config.edge_config.search_methods) == 1
