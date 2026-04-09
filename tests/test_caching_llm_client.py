"""Tests for the caching LLM client wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_cache_hit_skips_llm_call(tmp_path):
    """Second call with same messages returns cached response without LLM call."""
    from graphiti_core.prompts.models import Message
    from graphiti_core.llm_client.config import ModelSize

    from src.caching_llm_client import CachingLLMClient

    client = CachingLLMClient.__new__(CachingLLMClient)
    client._memory_cache = {}
    client._cache_dir = tmp_path
    client._hits = 0
    client._misses = 0

    messages = [
        Message(role="system", content="Extract entities"),
        Message(role="user", content="Alice works at Google"),
    ]
    expected_response = {"entities": [{"name": "Alice"}, {"name": "Google"}]}

    # Mock the parent's _generate_response
    with patch.object(
        CachingLLMClient.__bases__[0],
        "_generate_response",
        new_callable=AsyncMock,
        return_value=expected_response,
    ) as mock_llm:
        # First call → miss → calls LLM
        result1 = await client._generate_response(messages, model_size=ModelSize.medium)
        assert result1 == expected_response
        assert mock_llm.call_count == 1
        assert client._misses == 1

        # Second call → hit → skips LLM
        result2 = await client._generate_response(messages, model_size=ModelSize.medium)
        assert result2 == expected_response
        assert mock_llm.call_count == 1  # still 1
        assert client._hits == 1

        # Verify disk cache exists
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 1


def test_cache_key_deterministic():
    """Same messages produce same key."""
    from graphiti_core.prompts.models import Message
    from graphiti_core.llm_client.config import ModelSize

    from src.caching_llm_client import CachingLLMClient

    client = CachingLLMClient.__new__(CachingLLMClient)

    msgs = [Message(role="user", content="hello")]
    k1 = client._cache_key(msgs, None, ModelSize.medium)
    k2 = client._cache_key(msgs, None, ModelSize.medium)
    assert k1 == k2


def test_cache_key_differs_for_different_content():
    """Different messages produce different keys."""
    from graphiti_core.prompts.models import Message
    from graphiti_core.llm_client.config import ModelSize

    from src.caching_llm_client import CachingLLMClient

    client = CachingLLMClient.__new__(CachingLLMClient)

    msgs1 = [Message(role="user", content="Alice works at Google")]
    msgs2 = [Message(role="user", content="Bob lives in NYC")]
    k1 = client._cache_key(msgs1, None, ModelSize.medium)
    k2 = client._cache_key(msgs2, None, ModelSize.medium)
    assert k1 != k2


def test_cache_stats():
    """Stats report correct values."""
    from src.caching_llm_client import CachingLLMClient

    import types

    client = CachingLLMClient.__new__(CachingLLMClient)
    client._memory_cache = {"a": {}, "b": {}}
    client._hits = 8
    client._misses = 2
    client._cache_dir = types.SimpleNamespace(glob=lambda pattern: [1, 2, 3])
    stats = client.cache_stats()
    assert stats["memory_entries"] == 2
    assert stats["hits"] == 8
    assert stats["misses"] == 2
    assert stats["hit_rate"] == 0.8
