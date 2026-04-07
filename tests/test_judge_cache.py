"""Tests for the LLM judge result cache."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.judge_cache import _cache_key, cache_stats


def test_cache_key_order_independent():
    """Same fact pair in different order → same key."""
    k1 = _cache_key("Alice works at Google", "Alice is employed by Google")
    k2 = _cache_key("Alice is employed by Google", "Alice works at Google")
    assert k1 == k2


def test_cache_key_case_insensitive():
    """Same fact with different case → same key."""
    k1 = _cache_key("Alice works at Google", "alice works at google")
    k2 = _cache_key("ALICE WORKS AT GOOGLE", "Alice Works At Google")
    assert k1 == k2


def test_cache_key_strips_whitespace():
    k1 = _cache_key("  hello  ", "world")
    k2 = _cache_key("hello", "  world  ")
    assert k1 == k2


def test_cache_key_different_facts_different_keys():
    k1 = _cache_key("Alice works at Google", "Bob works at Meta")
    k2 = _cache_key("Alice works at Google", "Alice joined Facebook")
    assert k1 != k2


@pytest.mark.asyncio
async def test_cached_facts_match_caches_result(tmp_path):
    """Second call returns cached result without calling judge."""
    # Reset module state for isolation
    import src.judge_cache as jc

    jc._memory_cache = {}
    jc._disk_loaded = True  # skip disk load
    jc.CACHE_DIR = tmp_path

    with patch("src.judge_cache.facts_match", new_callable=AsyncMock) as mock_judge:
        mock_judge.return_value = True

        # First call → hits judge
        result1 = await jc.cached_facts_match("A works at B", "A employed by B")
        assert result1 is True
        assert mock_judge.call_count == 1

        # Second call → cache hit, no judge call
        result2 = await jc.cached_facts_match("A works at B", "A employed by B")
        assert result2 is True
        assert mock_judge.call_count == 1  # still 1

        # Verify persisted to disk
        cache_file = tmp_path / "judgments.json"
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert len(data) == 1
        assert list(data.values())[0] is True
