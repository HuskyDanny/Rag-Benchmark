"""Full trace of benchmark scenarios — captures graph state, LLM responses, and search results."""

import asyncio
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EntityNode, EpisodeType
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import (
    SearchFilters,
    DateFilter,
    ComparisonOperator,
)
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from src.models import TestCase
from src.judge import facts_match


# ── Helpers ──────────────────────────────────────────────────────────────────


async def create_graphiti():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-397B-A17B")
    small_model = os.getenv("LLM_SMALL_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    llm_client = OpenAIGenericClient(
        config=LLMConfig(
            api_key=api_key, base_url=base_url, model=model, small_model=small_model
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


async def dump_graph_state(g: Graphiti, group_id: str, label: str):
    """Print all nodes and edges in the graph for a group."""
    print(f"\n{'='*70}")
    print(f"  GRAPH STATE: {label}")
    print(f"{'='*70}")

    driver = g.clients.driver
    # Query all entity nodes
    nodes_result = await driver.execute_query(
        "MATCH (n:Entity) WHERE n.group_id = $gid RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels",
        gid=group_id,
    )
    print(f"\n  Nodes ({len(nodes_result.records)}):")
    for row in nodes_result.records:
        print(f"    [{row['uuid'][:8]}] {row['name']}  labels={row['labels']}")

    # Query all edges with temporal fields
    edges_result = await driver.execute_query(
        """MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
           WHERE e.group_id = $gid
           RETURN s.name AS source, e.name AS rel, t.name AS target,
                  e.fact AS fact, e.valid_at AS valid_at,
                  e.invalid_at AS invalid_at, e.expired_at AS expired_at,
                  e.uuid AS uuid""",
        gid=group_id,
    )
    print(f"\n  Edges ({len(edges_result.records)}):")
    for row in edges_result.records:
        valid = str(row.get("valid_at", "?"))[:19] if row.get("valid_at") else "NULL"
        invalid = (
            str(row.get("invalid_at", "?"))[:19] if row.get("invalid_at") else "NULL"
        )
        expired = (
            str(row.get("expired_at", "?"))[:19] if row.get("expired_at") else "NULL"
        )
        status = "CURRENT" if row.get("invalid_at") is None else "EXPIRED"
        print(
            f"    [{row['uuid'][:8]}] {row['source']} ──{row['rel']}──► {row['target']}"
        )
        print(f"             fact: \"{row['fact']}\"")
        print(
            f"             valid_at={valid}  invalid_at={invalid}  expired_at={expired}  [{status}]"
        )

    print()


async def run_search_trace(
    g: Graphiti, query: str, group_id: str, with_filter: bool, query_time=None
):
    """Run a search and print raw results."""
    config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    config.limit = 5

    search_filter = None
    if with_filter and query_time:
        search_filter = SearchFilters(
            valid_at=[
                [
                    DateFilter(
                        date=query_time,
                        comparison_operator=ComparisonOperator.less_than_equal,
                    )
                ]
            ],
            invalid_at=[
                [DateFilter(comparison_operator=ComparisonOperator.is_null)],
                [
                    DateFilter(
                        date=query_time,
                        comparison_operator=ComparisonOperator.greater_than,
                    )
                ],
            ],
        )

    filter_label = "WITH temporal filter" if with_filter else "NO filter (raw)"
    print(f'\n  Search: "{query}"  [{filter_label}]')
    if search_filter:
        print(f"    Filter: invalid_at IS NULL OR invalid_at > {query_time}")

    results = await g.search_(
        query=query,
        config=config,
        group_ids=[group_id],
        search_filter=search_filter,
    )

    print(f"    Results ({len(results.edges)} edges):")
    for i, edge in enumerate(results.edges):
        invalid = str(edge.invalid_at)[:19] if edge.invalid_at else "NULL"
        status = "CURRENT" if edge.invalid_at is None else "EXPIRED"
        print(f'      #{i+1}: "{edge.fact}"')
        print(f"           invalid_at={invalid} [{status}]")

    return [edge.fact for edge in results.edges]


# ── Scenario 1: Static Fact ─────────────────────────────────────────────────


async def trace_static_fact(g: Graphiti):
    print("\n" + "█" * 70)
    print("  SCENARIO 1: STATIC FACT (static_001)")
    print("  Facts that never change — the baseline")
    print("█" * 70)

    GID = "trace_static"
    await clear_data(g.clients.driver, group_ids=[GID])

    # ── Phase 1: Controlled Insertion ──
    print("\n" + "─" * 70)
    print("  PHASE 1: Controlled Insertion (direct save, known-good data)")
    print("─" * 70)

    await dump_graph_state(g, GID, "BEFORE insertion")

    print("  Inserting triplet: Paris ──CAPITAL_OF──► France")
    print("    valid_at = 2024-01-01, invalid_at = NULL (never expires)")

    paris = EntityNode(name="Paris", group_id=GID, labels=["City"])
    france = EntityNode(name="France", group_id=GID, labels=["Country"])
    await paris.generate_name_embedding(g.embedder)
    await paris.save(g.clients.driver)
    await france.generate_name_embedding(g.embedder)
    await france.save(g.clients.driver)

    edge = EntityEdge(
        group_id=GID,
        source_node_uuid=paris.uuid,
        target_node_uuid=france.uuid,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        name="CAPITAL_OF",
        fact="Paris is the capital of France",
        valid_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        invalid_at=None,
    )
    await edge.generate_embedding(g.embedder)
    await edge.save(g.clients.driver)

    await dump_graph_state(g, GID, "AFTER insertion")

    # ── Search ──
    query = "What is the capital of France?"
    qt = datetime(2024, 7, 1, tzinfo=timezone.utc)
    print(f'\n  Query: "{query}" at query_time={qt.date()}')
    print(f'  Expected: ["Paris is the capital of France"]')
    print(f"  Expected NOT: [] (no stale facts)")

    r_hybrid = await run_search_trace(g, query, GID, with_filter=True, query_time=qt)
    r_bm25 = await run_search_trace(g, query, GID, with_filter=False)

    # ── Scoring ──
    print(f"\n  Scoring:")
    print(f"    hybrid returned: {r_hybrid[:3]}")
    print(f"    bm25 returned:   {r_bm25[:3]}")
    print(f"    → Both find the fact. No stale facts exist. TempAcc = 100% for both.")
    print(f"    → This is why static facts don't differentiate strategies.")

    await clear_data(g.clients.driver, group_ids=[GID])


# ── Scenario 2: Evolving Fact ────────────────────────────────────────────────


async def trace_evolving_fact(g: Graphiti):
    print("\n" + "█" * 70)
    print("  SCENARIO 2: EVOLVING FACT (evolving_001 — Amy's job change)")
    print("  The CORE test — facts change over time")
    print("█" * 70)

    GID = "trace_evolving"
    await clear_data(g.clients.driver, group_ids=[GID])

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Controlled Insertion
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  PHASE 1: Controlled Insertion")
    print("  We manually set valid_at/invalid_at — this is 'perfect' data")
    print("─" * 70)

    await dump_graph_state(g, GID, "BEFORE insertion (empty)")

    # Insert OLD fact (expired)
    print("\n  Step 1: Insert OLD fact (Amy at Google, now EXPIRED)")
    amy1 = EntityNode(name="Amy", group_id=GID, labels=["Person"])
    google = EntityNode(name="Google", group_id=GID, labels=["Organization"])
    await amy1.generate_name_embedding(g.embedder)
    await amy1.save(g.clients.driver)
    await google.generate_name_embedding(g.embedder)
    await google.save(g.clients.driver)

    edge_old = EntityEdge(
        group_id=GID,
        source_node_uuid=amy1.uuid,
        target_node_uuid=google.uuid,
        created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        name="WORKS_AT",
        fact="Amy works at Google as a software engineer",
        valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        invalid_at=datetime(2024, 6, 1, tzinfo=timezone.utc),  # ◄── EXPIRED
    )
    await edge_old.generate_embedding(g.embedder)
    await edge_old.save(g.clients.driver)
    print(f"    Edge: Amy ──WORKS_AT──► Google")
    print(f"    valid_at=2024-01-15, invalid_at=2024-06-01 [EXPIRED]")

    # Insert NEW fact (current)
    print("\n  Step 2: Insert NEW fact (Amy at Facebook, CURRENT)")
    facebook = EntityNode(name="Facebook", group_id=GID, labels=["Organization"])
    await facebook.generate_name_embedding(g.embedder)
    await facebook.save(g.clients.driver)

    edge_new = EntityEdge(
        group_id=GID,
        source_node_uuid=amy1.uuid,
        target_node_uuid=facebook.uuid,
        created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        name="WORKS_AT",
        fact="Amy works at Facebook",
        valid_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        invalid_at=None,  # ◄── CURRENT
    )
    await edge_new.generate_embedding(g.embedder)
    await edge_new.save(g.clients.driver)
    print(f"    Edge: Amy ──WORKS_AT──► Facebook")
    print(f"    valid_at=2024-06-01, invalid_at=NULL [CURRENT]")

    await dump_graph_state(g, GID, "AFTER Phase 1 insertion (2 edges, 1 expired)")

    # ── Search in Phase 1 ──
    query = "Where does Amy work?"
    qt = datetime(2024, 7, 1, tzinfo=timezone.utc)
    print(f'\n  Query: "{query}" at query_time={qt.date()}')
    print(f'  Expected: ["Amy works at Facebook"]')
    print(f'  Expected NOT: ["Amy works at Google"]')

    print(f"\n  ── HYBRID (with temporal filter) ──")
    print(f"  Filter logic: only return edges where:")
    print(
        f"    valid_at <= 2024-07-01 AND (invalid_at IS NULL OR invalid_at > 2024-07-01)"
    )
    print(
        f"  → Edge 'Amy at Google': invalid_at=2024-06-01, which is NOT > 2024-07-01 → FILTERED OUT"
    )
    print(f"  → Edge 'Amy at Facebook': invalid_at=NULL → INCLUDED")
    r_hybrid = await run_search_trace(g, query, GID, with_filter=True, query_time=qt)

    print(f"\n  ── BM25 (no filter — sees everything) ──")
    print(f"  No temporal filter applied. Both edges match 'Amy' + 'work'.")
    r_bm25 = await run_search_trace(g, query, GID, with_filter=False)

    # ── TempAcc explained ──
    print(f"\n  ── TEMPORAL ACCURACY (TempAcc) EXPLAINED ──")
    print(f"  TempAcc = True if NO facts from expected_not appear in results")
    print(f'  expected_not = ["Amy works at Google"]')

    print(f"\n  Hybrid results: {r_hybrid}")
    has_stale_hybrid = False
    for fact in r_hybrid:
        m = await facts_match(fact, "Amy works at Google")
        if m:
            has_stale_hybrid = True
            print(
                f"    LLM judge: \"{fact}\" matches 'Amy works at Google'? YES → stale fact found!"
            )
    if not has_stale_hybrid:
        print(f"    No stale facts found → TempAcc = TRUE ✅")

    print(f"\n  BM25 results: {r_bm25}")
    has_stale_bm25 = False
    for fact in r_bm25:
        m = await facts_match(fact, "Amy works at Google")
        if m:
            has_stale_bm25 = True
            print(
                f"    LLM judge: \"{fact}\" matches 'Amy works at Google'? YES → stale fact found!"
            )
    if has_stale_bm25:
        print(f"    Stale fact found → TempAcc = FALSE ❌")
    else:
        print(f"    No stale facts → TempAcc = TRUE ✅")

    await clear_data(g.clients.driver, group_ids=[GID])

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Pipeline Insertion (add_episode with LLM extraction)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  PHASE 2: Pipeline Insertion (add_episode — LLM extracts entities)")
    print("  This is where Graphiti's LLM pipeline processes raw text")
    print("─" * 70)

    GID2 = "trace_evolving_p2"
    await clear_data(g.clients.driver, group_ids=[GID2])
    await dump_graph_state(g, GID2, "BEFORE any episodes (empty)")

    # Episode 1: Amy joins Google
    print("\n  Step 1: add_episode('Amy joined Google as a software engineer')")
    print("    reference_time = 2024-01-15")
    print("    Graphiti will: extract entities, create nodes, create edges")
    await g.add_episode(
        name="evolving_001_ep1",
        episode_body="Amy joined Google as a software engineer.",
        source=EpisodeType.text,
        source_description="benchmark test data",
        reference_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
        group_id=GID2,
    )
    await dump_graph_state(g, GID2, "AFTER Episode 1 (Amy joins Google)")

    # Episode 2: Amy leaves Google, joins Facebook
    print("\n  Step 2: add_episode('Amy left Google and started working at Facebook')")
    print("    reference_time = 2024-06-01")
    print("    Graphiti should: detect contradiction, set invalid_at on old edge")
    await g.add_episode(
        name="evolving_001_ep2",
        episode_body="Amy left Google and started working at Facebook.",
        source=EpisodeType.text,
        source_description="benchmark test data",
        reference_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        group_id=GID2,
    )
    await dump_graph_state(g, GID2, "AFTER Episode 2 (Amy moves to Facebook)")

    print("\n  ── KEY OBSERVATION ──")
    print("  Look at the graph state above.")
    print("  If add_episode correctly detected the contradiction:")
    print("    → The 'Amy works at Google' edge should have invalid_at set")
    print("    → The 'Amy works at Facebook' edge should have invalid_at = NULL")
    print("  If the LLM MISSED the contradiction:")
    print("    → Both edges will have invalid_at = NULL (both appear 'current')")
    print("    → Temporal filter can't help — it sees both as valid")
    print("    → TempAcc will FAIL even for hybrid")

    # Search Phase 2
    print(f"\n  Searching Phase 2 graph...")
    r_hybrid_p2 = await run_search_trace(
        g, query, GID2, with_filter=True, query_time=qt
    )
    r_bm25_p2 = await run_search_trace(g, query, GID2, with_filter=False)

    await clear_data(g.clients.driver, group_ids=[GID2])


# ── Scenario 3: Multi-Entity Evolution ───────────────────────────────────────


async def trace_multi_entity(g: Graphiti):
    print("\n" + "█" * 70)
    print("  SCENARIO 3: MULTI-ENTITY EVOLUTION (multi_entity_001)")
    print("  Two things change at once — CEO change + HQ move")
    print("█" * 70)

    GID = "trace_multi"
    await clear_data(g.clients.driver, group_ids=[GID])

    print("\n" + "─" * 70)
    print("  PHASE 2 ONLY (Pipeline — this is where it gets interesting)")
    print("─" * 70)

    await dump_graph_state(g, GID, "BEFORE (empty)")

    episodes = [
        ("Acme Corp appointed Bob as CEO.", datetime(2024, 1, 1, tzinfo=timezone.utc)),
        (
            "Acme Corp headquarters are in New York.",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
        (
            "Acme Corp replaced Bob with Jane as CEO.",
            datetime(2024, 6, 1, tzinfo=timezone.utc),
        ),
        (
            "Acme Corp moved headquarters to San Francisco.",
            datetime(2024, 6, 1, tzinfo=timezone.utc),
        ),
    ]

    for i, (text, ref_time) in enumerate(episodes):
        print(f'\n  Episode {i+1}: "{text}"')
        print(f"    reference_time = {ref_time.date()}")
        await g.add_episode(
            name=f"multi_001_ep{i+1}",
            episode_body=text,
            source=EpisodeType.text,
            source_description="benchmark",
            reference_time=ref_time,
            group_id=GID,
        )
        await dump_graph_state(g, GID, f"AFTER Episode {i+1}")

    query = "Who is the CEO of Acme Corp?"
    qt = datetime(2024, 7, 1, tzinfo=timezone.utc)
    print(f'\n  Query: "{query}"')
    print(f"  Expected: ['Jane is CEO of Acme Corp']")
    print(f"  Expected NOT: ['Bob is CEO']")

    await run_search_trace(g, query, GID, with_filter=True, query_time=qt)
    await run_search_trace(g, query, GID, with_filter=False)

    await clear_data(g.clients.driver, group_ids=[GID])


# ── Main ─────────────────────────────────────────────────────────────────────


async def main():
    g = await create_graphiti()
    try:
        await trace_static_fact(g)
        await trace_evolving_fact(g)
        await trace_multi_entity(g)
    finally:
        await g.close()


if __name__ == "__main__":
    asyncio.run(main())
