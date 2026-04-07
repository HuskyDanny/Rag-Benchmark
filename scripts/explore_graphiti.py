"""Explore Graphiti's core operations with full parameter and behavior tracing."""

import asyncio
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EntityNode, EpisodeType
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config import (
    SearchConfig,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EdgeReranker,
)
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_RRF,
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
)
from graphiti_core.search.search_filters import (
    SearchFilters,
    DateFilter,
    ComparisonOperator,
)
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

GID = "explore"


def header(title):
    print(f"\n{'█' * 70}")
    print(f"  {title}")
    print(f"{'█' * 70}\n")


def section(title):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}\n")


async def dump_graph(g, label="Current"):
    driver = g.clients.driver
    nodes = await driver.execute_query(
        "MATCH (n:Entity) WHERE n.group_id = $gid "
        "RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary, labels(n) AS labels",
        gid=GID,
    )
    edges = await driver.execute_query(
        "MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity) WHERE e.group_id = $gid "
        "RETURN s.name AS src, e.name AS rel, t.name AS tgt, e.fact AS fact, "
        "e.valid_at AS valid_at, e.invalid_at AS invalid_at, e.expired_at AS expired_at, "
        "e.uuid AS uuid",
        gid=GID,
    )
    episodes = await driver.execute_query(
        "MATCH (ep:Episodic) WHERE ep.group_id = $gid "
        "RETURN ep.name AS name, ep.content AS content, ep.valid_at AS valid_at",
        gid=GID,
    )
    print(f"  📊 Graph State: {label}")
    print(
        f"  Nodes: {len(nodes.records)}  Edges: {len(edges.records)}  Episodes: {len(episodes.records)}"
    )

    if nodes.records:
        print(f"\n  Nodes:")
        for r in nodes.records:
            summary = f' summary="{r["summary"][:60]}..."' if r.get("summary") else ""
            print(f'    • {r["name"]}  labels={r["labels"]}{summary}')

    if edges.records:
        print(f"\n  Edges:")
        for r in edges.records:
            v = str(r["valid_at"])[:10] if r.get("valid_at") else "NULL"
            i = str(r["invalid_at"])[:10] if r.get("invalid_at") else "NULL"
            e = str(r["expired_at"])[:10] if r.get("expired_at") else "NULL"
            status = "EXPIRED" if r.get("invalid_at") else "CURRENT"
            print(f'    • {r["src"]} ──{r["rel"]}──► {r["tgt"]}  [{status}]')
            print(f'      fact="{r["fact"]}"')
            print(f"      valid_at={v}  invalid_at={i}  expired_at={e}")

    if episodes.records:
        print(f"\n  Episodes:")
        for r in episodes.records:
            v = str(r["valid_at"])[:10] if r.get("valid_at") else "?"
            content = (r["content"] or "")[:80]
            print(f'    • {r["name"]}  ({v})  "{content}"')

    print()


async def create_graphiti():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-397B-A17B")
    small = os.getenv("LLM_SMALL_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    print(f"  Config:")
    print(f"    LLM model:       {model}")
    print(f"    LLM small model: {small}")
    print(
        f"    Embedding model: {os.getenv('EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B')}"
    )
    print(f"    Base URL:        {base_url}")
    print(f"    Neo4j:           bolt://localhost:7687")
    print()

    llm_client = OpenAIGenericClient(
        config=LLMConfig(
            api_key=api_key, base_url=base_url, model=model, small_model=small
        )
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
            api_key=api_key,
            base_url=base_url,
        )
    )
    g = Graphiti(
        "bolt://localhost:7687",
        "neo4j",
        "benchmark123",
        llm_client=llm_client,
        embedder=embedder,
    )
    await g.build_indices_and_constraints()
    return g


# ═══════════════════════════════════════════════════════════════════════════
# OPERATION 1: add_triplet — Manual graph construction
# ═══════════════════════════════════════════════════════════════════════════


async def op1_add_triplet(g):
    header("OPERATION 1: add_triplet(source_node, edge, target_node)")
    print("  Inserts (source)──[edge]──►(target). Does LLM entity resolution.")
    print(
        "  Steps: embed nodes → embed edge → resolve nodes (LLM) → resolve edges (LLM) → save"
    )
    print(
        "  Required: EntityNode(name, group_id), EntityEdge(group_id, source_node_uuid,"
    )
    print("            target_node_uuid, created_at, name, fact)")
    print("  Optional: labels, summary, valid_at, invalid_at")

    await clear_data(g.clients.driver, group_ids=[GID])

    section("1a. Creating EntityNode objects")
    alice = EntityNode(name="Alice", group_id=GID, labels=["Person"])
    google = EntityNode(name="Google", group_id=GID, labels=["Company"])
    print(f"  alice = EntityNode(name='Alice', group_id='{GID}', labels=['Person'])")
    print(f"    → uuid={alice.uuid[:12]}...")
    print(f"    → name_embedding=None (not yet generated)")
    print(f"  google = EntityNode(name='Google', group_id='{GID}', labels=['Company'])")
    print(f"    → uuid={google.uuid[:12]}...")

    section("1b. Creating EntityEdge object")
    edge = EntityEdge(
        group_id=GID,
        source_node_uuid=alice.uuid,
        target_node_uuid=google.uuid,
        created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        name="WORKS_AT",
        fact="Alice works at Google as a software engineer",
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        invalid_at=None,
    )
    print(f"  edge = EntityEdge(")
    print(f"    group_id='{GID}',")
    print(f"    source_node_uuid='{alice.uuid[:12]}...',  # Alice")
    print(f"    target_node_uuid='{google.uuid[:12]}...',  # Google")
    print(f"    created_at=2024-01-15,")
    print(f"    name='WORKS_AT',")
    print(f"    fact='Alice works at Google as a software engineer',")
    print(f"    valid_at=2024-01-15,  # when this became true")
    print(f"    invalid_at=None,       # still true (None = no expiry)")
    print(f"  )")
    print(f"    → uuid={edge.uuid[:12]}...")
    print(f"    → fact_embedding=None (not yet generated)")

    section("1c. Calling add_triplet (SLOW — does LLM entity resolution)")
    t0 = time.time()
    print(
        f"  await graphiti.add_triplet(source_node=alice, edge=edge, target_node=google)"
    )
    print(f"  ... calling embedding API for nodes + edge ...")
    print(f"  ... resolving nodes against existing graph ...")
    print(f"  ... resolving edge conflicts ...")
    result = await g.add_triplet(source_node=alice, edge=edge, target_node=google)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"\n  Return value (AddTripletResults):")
    print(f"    nodes: {[n.name for n in result.nodes]}")
    print(f"    edges: {[e.fact for e in result.edges]}")

    await dump_graph(g, "After add_triplet")


# ═══════════════════════════════════════════════════════════════════════════
# OPERATION 2: Direct save — Bypassing LLM resolution
# ═══════════════════════════════════════════════════════════════════════════


async def op2_direct_save(g):
    header("OPERATION 2: EntityNode.save() + EntityEdge.save() — Direct insertion")
    print(
        """  What it does:
    Saves nodes and edges directly to Neo4j WITHOUT LLM entity resolution.
    Much faster (~0.4s vs ~5-160s for add_triplet).

  When to use:
    - Controlled benchmarks with known-good data
    - Bulk imports where you trust the data
    - When you want to avoid LLM deduplication

  Caveat:
    - No duplicate detection — same entity inserted twice = two nodes
    - You must generate embeddings yourself before saving
    - No edge conflict resolution
  """
    )

    await clear_data(g.clients.driver, group_ids=[GID])

    section("2a. Create + embed + save nodes")
    bob = EntityNode(name="Bob", group_id=GID, labels=["Person"])
    meta = EntityNode(name="Meta", group_id=GID, labels=["Company"])

    t0 = time.time()
    print(f"  # Generate embedding (API call)")
    await bob.generate_name_embedding(g.embedder)
    print(
        f"  bob.name_embedding = float[{len(bob.name_embedding)}]  (first 5: {bob.name_embedding[:5]})"
    )
    await bob.save(g.clients.driver)
    print(f"  bob.save(driver)  → saved to Neo4j")

    await meta.generate_name_embedding(g.embedder)
    await meta.save(g.clients.driver)
    print(f"  meta saved too")

    section("2b. Create + embed + save edge")
    edge = EntityEdge(
        group_id=GID,
        source_node_uuid=bob.uuid,
        target_node_uuid=meta.uuid,
        created_at=datetime(2024, 3, 1, tzinfo=timezone.utc),
        name="WORKS_AT",
        fact="Bob is a product manager at Meta",
        valid_at=datetime(2024, 3, 1, tzinfo=timezone.utc),
    )
    await edge.generate_embedding(g.embedder)
    print(f"  edge.fact_embedding = float[{len(edge.fact_embedding)}]")
    await edge.save(g.clients.driver)
    elapsed = time.time() - t0
    print(f"  edge.save(driver)  → saved to Neo4j")
    print(f"\n  Total time: {elapsed:.1f}s (vs ~5s for add_triplet)")

    await dump_graph(g, "After direct save")


# ═══════════════════════════════════════════════════════════════════════════
# OPERATION 3: add_episode — Full LLM extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════


async def op3_add_episode(g):
    header("OPERATION 3: add_episode(name, episode_body, ...) — LLM extraction")
    print(
        """  What it does:
    Takes raw text, extracts entities and relationships using LLM, and
    inserts them into the graph. This is Graphiti's core ingestion method.

  Under the hood (the full pipeline):
    1. Creates an Episodic node (stores the raw text)
    2. LLM call: extract_nodes — identifies entities in the text
    3. resolve_extracted_nodes — deduplicates against existing graph (LLM)
    4. LLM call: extract_edges — identifies relationships
    5. resolve_extracted_edges — detects contradictions, sets invalid_at (LLM)
    6. LLM call: extract_attributes — enriches entity summaries
    7. Saves everything to Neo4j

  Required parameters:
    name: str                    — unique identifier for this episode
    episode_body: str            — the raw text to process
    source: EpisodeType          — .text, .message, .json, etc.
    source_description: str      — describes the source
    reference_time: datetime     — when this information was valid

  Optional:
    group_id: str                — scopes the data (like a namespace)
    update_communities: bool     — whether to update community nodes (default False)
  """
    )

    await clear_data(g.clients.driver, group_ids=[GID])

    section("3a. Episode 1 — Simple fact extraction")
    t0 = time.time()
    print(f'  Text: "Alice is a senior engineer at Google working on AI research."')
    print(f"  reference_time: 2024-01-15")
    print(f"\n  Calling add_episode...")
    print(f"    → LLM extracts entities: Alice, Google, AI research")
    print(f"    → LLM extracts relationships: Alice WORKS_AT Google, etc.")
    print(f"    → Embeddings generated for each node and edge")

    result1 = await g.add_episode(
        name="ep1_alice_google",
        episode_body="Alice is a senior engineer at Google working on AI research.",
        source=EpisodeType.text,
        source_description="HR record",
        reference_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
        group_id=GID,
    )
    elapsed1 = time.time() - t0
    print(f"\n  Done in {elapsed1:.1f}s")
    print(f"\n  Return value (AddEpisodeResults):")
    print(f"    episode: {result1.episode.name}")
    print(f"    nodes created: {[n.name for n in result1.nodes]}")
    print(f"    edges created: {[e.fact for e in result1.edges]}")

    await dump_graph(g, "After Episode 1")

    section("3b. Episode 2 — Contradicting fact (temporal invalidation)")
    t0 = time.time()
    print(f'  Text: "Alice left Google and joined Meta as a tech lead."')
    print(f"  reference_time: 2024-06-01")
    print(f"\n  Calling add_episode...")
    print(f"    → LLM detects: Alice WORKS_AT Google is contradicted")
    print(f"    → Should set invalid_at on the old 'Alice at Google' edge")
    print(f"    → Creates new 'Alice at Meta' edge")

    result2 = await g.add_episode(
        name="ep2_alice_meta",
        episode_body="Alice left Google and joined Meta as a tech lead.",
        source=EpisodeType.text,
        source_description="HR record",
        reference_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        group_id=GID,
    )
    elapsed2 = time.time() - t0
    print(f"\n  Done in {elapsed2:.1f}s")
    print(f"\n  Return value:")
    print(f"    nodes: {[n.name for n in result2.nodes]}")
    print(f"    edges: {[e.fact for e in result2.edges]}")
    print(f"    invalidated edges: check graph state below")

    await dump_graph(g, "After Episode 2 (look for invalid_at changes!)")


# ═══════════════════════════════════════════════════════════════════════════
# OPERATION 4: search_ — The full search API
# ═══════════════════════════════════════════════════════════════════════════


async def op4_search(g):
    header("OPERATION 4: search_(query, config, ...) — Search the graph")
    print(
        """  What it does:
    Searches the knowledge graph using configurable strategies.

  Parameters:
    query: str                    — natural language question
    config: SearchConfig          — which search methods to use
    group_ids: list[str]          — scope to specific groups
    search_filter: SearchFilters  — temporal/property filters
    center_node_uuid: str         — for BFS-based search

  SearchConfig controls:
    edge_config.search_methods: list of EdgeSearchMethod
      - bm25: keyword matching (fulltext index)
      - cosine_similarity: embedding vector similarity
      - bfs: breadth-first search from a center node
    edge_config.reranker: how to combine results
      - rrf: Reciprocal Rank Fusion (math only, fast)
      - cross_encoder: LLM reranking (better quality, slower)
      - node_distance: graph distance based
    limit: int — max results (default 10)

  SearchFilters for temporal queries:
    valid_at: list[list[DateFilter]]    — filter by valid_at
    invalid_at: list[list[DateFilter]]  — filter by invalid_at
    Structure: outer list = OR, inner list = AND
    Example: [[is_null], [> date]] = "IS NULL OR > date"
  """
    )

    section("4a. Search WITHOUT filter — returns everything")
    config_hybrid = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    config_hybrid.limit = 5
    print(f"  config = EDGE_HYBRID_SEARCH_RRF (BM25 + cosine + RRF reranking)")
    print(f"  config.limit = 5")
    print(
        f"  config.edge_config.search_methods = {[str(m) for m in config_hybrid.edge_config.search_methods]}"
    )
    print(f"  config.edge_config.reranker = {config_hybrid.edge_config.reranker}")

    query = "Where does Alice work?"
    print(f'\n  Query: "{query}"')
    results = await g.search_(query=query, config=config_hybrid, group_ids=[GID])

    print(f"\n  Results ({len(results.edges)} edges):")
    for i, edge in enumerate(results.edges):
        inv = str(edge.invalid_at)[:10] if edge.invalid_at else "NULL"
        status = "EXPIRED" if edge.invalid_at else "CURRENT"
        print(f'    #{i+1}: "{edge.fact}"')
        print(
            f"         invalid_at={inv} [{status}]  score={results.edge_reranker_scores[i]:.4f}"
        )

    section("4b. Search WITH temporal filter — only current facts")
    qt = datetime(2024, 7, 1, tzinfo=timezone.utc)
    sf = SearchFilters(
        invalid_at=[
            [DateFilter(comparison_operator=ComparisonOperator.is_null)],
            [DateFilter(date=qt, comparison_operator=ComparisonOperator.greater_than)],
        ],
    )
    print(f"  SearchFilters(invalid_at=[[IS_NULL], [> 2024-07-01]])")
    print(f"  Meaning: only edges where invalid_at IS NULL OR invalid_at > 2024-07-01")

    results_filtered = await g.search_(
        query=query,
        config=config_hybrid,
        group_ids=[GID],
        search_filter=sf,
    )
    print(f"\n  Results ({len(results_filtered.edges)} edges):")
    for i, edge in enumerate(results_filtered.edges):
        inv = str(edge.invalid_at)[:10] if edge.invalid_at else "NULL"
        print(f'    #{i+1}: "{edge.fact}"  invalid_at={inv}')

    section("4c. BM25-only search")
    config_bm25 = SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25],
            reranker=EdgeReranker.rrf,
        ),
        limit=5,
    )
    print(f"  config.search_methods = [bm25]  (keyword only)")
    results_bm25 = await g.search_(query=query, config=config_bm25, group_ids=[GID])
    print(f"  Results ({len(results_bm25.edges)} edges):")
    for i, edge in enumerate(results_bm25.edges):
        print(f'    #{i+1}: "{edge.fact}"')

    section("4d. Cosine-only search")
    config_cosine = SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.rrf,
        ),
        limit=5,
    )
    print(f"  config.search_methods = [cosine_similarity]  (embedding only)")
    results_cosine = await g.search_(query=query, config=config_cosine, group_ids=[GID])
    print(f"  Results ({len(results_cosine.edges)} edges):")
    for i, edge in enumerate(results_cosine.edges):
        print(f'    #{i+1}: "{edge.fact}"')


async def main():
    header("GRAPHITI CORE API EXPLORATION")
    g = await create_graphiti()
    try:
        await op1_add_triplet(g)
        await clear_data(g.clients.driver, group_ids=[GID])
        await op2_direct_save(g)
        await clear_data(g.clients.driver, group_ids=[GID])
        await op3_add_episode(g)
        await op4_search(g)
        await clear_data(g.clients.driver, group_ids=[GID])
    finally:
        await g.close()
    print("\n  Done. Graph cleaned up.")


if __name__ == "__main__":
    asyncio.run(main())
