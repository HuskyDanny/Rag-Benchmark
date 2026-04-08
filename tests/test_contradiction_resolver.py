"""Tests for the post-ingestion contradiction resolver."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_facts_contradict_same_attribute():
    """Same entity, same attribute, different values → contradiction."""
    from src.contradiction_resolver import _facts_contradict

    with patch("src.contradiction_resolver._get_client") as mock_client:
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="YES"))]
        mock_client.return_value.chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        result = await _facts_contradict("Alice works at Google", "Alice works at Meta")
        assert result is True


@pytest.mark.asyncio
async def test_facts_dont_contradict_different_attributes():
    """Same entity, different attributes → not a contradiction."""
    from src.contradiction_resolver import _facts_contradict

    with patch("src.contradiction_resolver._get_client") as mock_client:
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="NO"))]
        mock_client.return_value.chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        result = await _facts_contradict("Alice works at Google", "Alice lives in NYC")
        assert result is False


@pytest.mark.asyncio
async def test_resolve_contradictions_invalidates_older_edge():
    """When two CURRENT edges contradict, older one gets invalid_at set."""
    from src.contradiction_resolver import resolve_contradictions

    mock_driver = AsyncMock()

    # First call: get all CURRENT edges — two contradicting edges for same entity
    mock_driver.execute_query.side_effect = [
        # Query 1: get CURRENT edges
        (
            [
                {
                    "source": "Alice",
                    "target": "Google",
                    "relation": "WORKS_AT",
                    "fact": "Alice works at Google",
                    "uuid": "edge-old",
                    "valid_at": "2023-01-01",
                    "created_at": "2023-01-01",
                },
                {
                    "source": "Alice",
                    "target": "Meta",
                    "relation": "EMPLOYED_BY",
                    "fact": "Alice works at Meta",
                    "uuid": "edge-new",
                    "valid_at": "2024-06-01",
                    "created_at": "2024-06-01",
                },
            ],
            None,
            None,
        ),
        # Query 2: UPDATE to set invalid_at on older edge
        ([], None, None),
    ]

    with patch(
        "src.contradiction_resolver._facts_contradict", new_callable=AsyncMock
    ) as mock_check:
        mock_check.return_value = True

        count = await resolve_contradictions(mock_driver, "test_group")

        assert count == 1
        # Verify the UPDATE query was called with the older edge's UUID
        update_call = mock_driver.execute_query.call_args_list[1]
        assert "edge-old" in str(update_call)


@pytest.mark.asyncio
async def test_resolve_no_contradictions():
    """No contradicting edges → nothing invalidated."""
    from src.contradiction_resolver import resolve_contradictions

    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = (
        [
            {
                "source": "Alice",
                "target": "Google",
                "relation": "WORKS_AT",
                "fact": "Alice works at Google",
                "uuid": "e1",
                "valid_at": None,
                "created_at": None,
            },
        ],
        None,
        None,
    )

    count = await resolve_contradictions(mock_driver, "test_group")
    assert count == 0  # Only 1 edge per entity, nothing to compare
