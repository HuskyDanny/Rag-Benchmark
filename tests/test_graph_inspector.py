"""Tests for Neo4j graph inspector with mocked driver."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.graph_inspector import inspect_edges, inspect_node_duplicates, inspect_nodes


def _mock_driver(records: list[dict]) -> AsyncMock:
    """Create a mock driver that returns the given records."""
    driver = AsyncMock()
    # Records should behave like dicts
    mock_records = [
        type("Record", (), {"__iter__": lambda s: iter(r.items()), **r})()
        for r in records
    ]
    # Simpler: just return dicts directly since we call dict(r)
    driver.execute_query.return_value = (records, None, None)
    return driver


@pytest.mark.asyncio
async def test_inspect_nodes_returns_entities():
    records = [
        {"name": "Alice", "uuid": "u1", "group_id": "g1"},
        {"name": "Google", "uuid": "u2", "group_id": "g1"},
    ]
    driver = _mock_driver(records)
    result = await inspect_nodes(driver, "g1")
    assert len(result) == 2
    assert result[0]["name"] == "Alice"
    driver.execute_query.assert_called_once()


@pytest.mark.asyncio
async def test_inspect_edges_returns_relations():
    records = [
        {
            "source": "Alice",
            "target": "Google",
            "relation": "WORKS_AT",
            "fact": "Alice works at Google",
            "valid_at": "2023-01-01",
            "invalid_at": None,
        }
    ]
    driver = _mock_driver(records)
    result = await inspect_edges(driver, "g1")
    assert len(result) == 1
    assert result[0]["fact"] == "Alice works at Google"
    assert result[0]["invalid_at"] is None


@pytest.mark.asyncio
async def test_inspect_node_duplicates_empty_when_no_dupes():
    driver = _mock_driver([])
    result = await inspect_node_duplicates(driver, "g1")
    assert result == []


@pytest.mark.asyncio
async def test_inspect_node_duplicates_finds_dupes():
    records = [{"name": "alice", "count": 3}]
    driver = _mock_driver(records)
    result = await inspect_node_duplicates(driver, "g1")
    assert len(result) == 1
    assert result[0]["count"] == 3
