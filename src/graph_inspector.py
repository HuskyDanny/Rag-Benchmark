"""Neo4j graph inspection — direct Cypher queries for ingestion quality validation."""

from __future__ import annotations

from typing import Any


async def inspect_nodes(driver: Any, group_id: str) -> list[dict]:
    """Query all entity nodes for a group_id.

    Returns: [{"name": str, "uuid": str, "group_id": str}]
    """
    records, _, _ = await driver.execute_query(
        """
        MATCH (n:Entity)
        WHERE n.group_id = $group_id
        RETURN n.name AS name, n.uuid AS uuid, n.group_id AS group_id
        """,
        group_id=group_id,
    )
    return [dict(r) for r in records]


async def inspect_edges(driver: Any, group_id: str) -> list[dict]:
    """Query all RELATES_TO edges for a group_id with temporal bounds.

    Returns: [{
        "source": str, "target": str, "relation": str, "fact": str,
        "valid_at": datetime|None, "invalid_at": datetime|None
    }]
    """
    records, _, _ = await driver.execute_query(
        """
        MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
        WHERE r.group_id = $group_id
        RETURN s.name AS source, t.name AS target,
               r.name AS relation, r.fact AS fact,
               r.valid_at AS valid_at, r.invalid_at AS invalid_at
        """,
        group_id=group_id,
    )
    return [dict(r) for r in records]


async def inspect_node_duplicates(driver: Any, group_id: str) -> list[dict]:
    """Find potential duplicate entities (same lowercase name, multiple nodes).

    Returns: [{"name": str, "count": int}]
    """
    records, _, _ = await driver.execute_query(
        """
        MATCH (n:Entity)
        WHERE n.group_id = $group_id
        WITH toLower(n.name) AS norm_name, count(*) AS cnt
        WHERE cnt > 1
        RETURN norm_name AS name, cnt AS count
        """,
        group_id=group_id,
    )
    return [dict(r) for r in records]
