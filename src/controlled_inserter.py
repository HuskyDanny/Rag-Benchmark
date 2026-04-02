"""Phase 1: Insert known-good triplets directly into Neo4j.

Uses EntityNode.save() and EntityEdge.save() to bypass Graphiti's
LLM-based entity resolution. This is correct for Phase 1 because
the data is known-good — no deduplication needed.
"""

from graphiti_core import Graphiti
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge

from src.models import TestCase

GROUP_ID = "benchmark_controlled"


async def insert_test_case(graphiti: Graphiti, test_case: TestCase) -> None:
    """Insert all triplets from a test case into the graph."""
    driver = graphiti.clients.driver
    embedder = graphiti.embedder
    node_cache: dict[str, EntityNode] = {}

    for triplet in test_case.triplets:
        source_node = await _get_or_create_node(
            node_cache, triplet.source.name, triplet.source.labels, embedder, driver
        )
        target_node = await _get_or_create_node(
            node_cache, triplet.target.name, triplet.target.labels, embedder, driver
        )

        edge = EntityEdge(
            group_id=GROUP_ID,
            source_node_uuid=source_node.uuid,
            target_node_uuid=target_node.uuid,
            created_at=triplet.edge.valid_at,
            name=triplet.edge.name,
            fact=triplet.edge.fact,
            valid_at=triplet.edge.valid_at,
            invalid_at=triplet.edge.invalid_at,
        )
        await edge.generate_embedding(embedder)
        await edge.save(driver)


async def _get_or_create_node(
    cache: dict[str, EntityNode],
    name: str,
    labels: list[str],
    embedder,
    driver,
) -> EntityNode:
    """Get existing node from cache or create, embed, and save a new one."""
    if name in cache:
        return cache[name]
    node = EntityNode(name=name, group_id=GROUP_ID, labels=labels)
    await node.generate_name_embedding(embedder)
    await node.save(driver)
    cache[name] = node
    return node


async def insert_all(graphiti: Graphiti, test_cases: list[TestCase]) -> None:
    """Insert all test cases into the graph."""
    for i, tc in enumerate(test_cases):
        print(f"  Inserting test case {i + 1}/{len(test_cases)}: {tc.id}")
        await insert_test_case(graphiti, tc)
