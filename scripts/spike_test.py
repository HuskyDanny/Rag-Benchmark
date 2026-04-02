"""Verify Neo4j connection + Graphiti add_triplet + search work."""

import asyncio
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

load_dotenv()

GROUP_ID = "spike_test"


async def main():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    llm_config = LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
        small_model=os.getenv("LLM_SMALL_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )
    llm_client = OpenAIGenericClient(config=llm_config)

    embedder_config = OpenAIEmbedderConfig(
        embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        api_key=api_key,
        base_url=base_url,
    )
    embedder = OpenAIEmbedder(config=embedder_config)

    print(f"Connecting to Neo4j at {uri}...")
    graphiti = Graphiti(uri, user, password, llm_client=llm_client, embedder=embedder)

    try:
        await graphiti.build_indices_and_constraints()
        print("OK: Connected and indices built")

        await clear_data(graphiti.clients.driver, group_ids=[GROUP_ID])
        print("OK: Cleared spike data")

        alice = EntityNode(name="Alice", group_id=GROUP_ID, labels=["Person"])
        google = EntityNode(name="Google", group_id=GROUP_ID, labels=["Organization"])
        edge = EntityEdge(
            group_id=GROUP_ID,
            source_node_uuid=alice.uuid,
            target_node_uuid=google.uuid,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            name="WORKS_AT",
            fact="Alice works at Google as a software engineer",
            valid_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        await graphiti.add_triplet(source_node=alice, edge=edge, target_node=google)
        print("OK: Inserted triplet: Alice --WORKS_AT--> Google")

        config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = 5
        results = await graphiti.search_(
            query="Where does Alice work?",
            config=config,
            group_ids=[GROUP_ID],
        )
        print(f"OK: Search returned {len(results.edges)} edges")
        for e in results.edges:
            print(f"  -> {e.fact}")

        if results.edges:
            print("\nSPIKE PASSED: Neo4j + Graphiti + search all working")
        else:
            print("\nSPIKE FAILED: Search returned no results")

        await clear_data(graphiti.clients.driver, group_ids=[GROUP_ID])
    finally:
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())
