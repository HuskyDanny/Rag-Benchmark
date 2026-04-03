"""Benchmark runner with checkpoint/resume, per-phase port config, and CLI."""

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_filters import (
    SearchFilters,
    DateFilter,
    ComparisonOperator,
)
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from src.models import TestCase, QueryResult, CategoryReport
from src.search_strategies import get_search_strategies
from src.evaluator import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_temporal_accuracy,
)
from src.reporter import aggregate_results, print_full_report
from src.checkpoint import Checkpoint
from src import controlled_inserter, pipeline_inserter, presplit_inserter

load_dotenv()

GROUP_IDS = {
    "controlled": controlled_inserter.GROUP_ID,
    "pipeline": pipeline_inserter.GROUP_ID,
    "pipeline_presplit": presplit_inserter.GROUP_ID,
}

PHASE_PORTS = {
    "controlled": 7687,
    "pipeline": 7688,
    "pipeline_presplit": 7689,
}

INSERTERS = {
    "controlled": controlled_inserter,
    "pipeline": pipeline_inserter,
    "pipeline_presplit": presplit_inserter,
}


def load_test_cases(path: str = "data/test_cases.json") -> list[TestCase]:
    with open(path) as f:
        return [TestCase.model_validate(d) for d in json.load(f)]


async def create_graphiti(neo4j_port: int = 7687) -> Graphiti:
    """Create and initialize a Graphiti instance with configurable Neo4j port."""
    uri = f"bolt://localhost:{neo4j_port}"
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

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

    graphiti = Graphiti(uri, user, password, llm_client=llm_client, embedder=embedder)
    await graphiti.build_indices_and_constraints()
    return graphiti


async def wipe_graph(graphiti: Graphiti, group_id: str | None = None) -> None:
    if group_id:
        await clear_data(graphiti.clients.driver, group_ids=[group_id])
    else:
        await clear_data(graphiti.clients.driver)
    await graphiti.build_indices_and_constraints()


def build_temporal_filter(query_time: datetime) -> SearchFilters:
    """Build a SearchFilter that only returns edges valid at query_time."""
    return SearchFilters(
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


async def run_search(
    graphiti: Graphiti,
    query: str,
    config: SearchConfig,
    group_id: str,
    query_time: datetime | None = None,
) -> list[str]:
    search_filter = build_temporal_filter(query_time) if query_time else None
    results = await graphiti.search_(
        query=query, config=config, group_ids=[group_id], search_filter=search_filter
    )
    return [edge.fact for edge in results.edges]


async def evaluate_query(
    graphiti: Graphiti,
    test_case: TestCase,
    query_idx: int,
    strategy_name: str,
    config: SearchConfig,
    group_id: str,
) -> QueryResult:
    query = test_case.queries[query_idx]
    # Only hybrid gets temporal filter — baselines run raw
    query_time = query.query_time if strategy_name == "hybrid" else None
    returned_facts = await run_search(
        graphiti, query.query, config, group_id, query_time=query_time
    )

    p_at_5 = await compute_precision_at_k(returned_facts, query.expected_facts, k=5)
    r_at_5 = await compute_recall_at_k(returned_facts, query.expected_facts, k=5)
    mrr = await compute_mrr(returned_facts, query.expected_facts)
    temp_acc = await compute_temporal_accuracy(returned_facts, query.expected_not)

    return QueryResult(
        test_case_id=test_case.id,
        query=query.query,
        strategy=strategy_name,
        returned_facts=returned_facts[:5],
        expected_facts=query.expected_facts,
        expected_not=query.expected_not,
        precision_at_5=p_at_5,
        recall_at_5=r_at_5,
        mrr=mrr,
        temporal_accuracy=temp_acc,
    )


# ---------------------------------------------------------------------------
# Stage: insert
# ---------------------------------------------------------------------------


async def run_insert(
    graphiti: Graphiti, phase: str, test_cases: list[TestCase]
) -> None:
    """Insert test data with checkpoint/resume support."""
    ckpt = Checkpoint(phase, "insert")
    inserter = INSERTERS[phase]

    print(f"\n--- [{phase}] INSERT stage ---")
    for tc in test_cases:
        if ckpt.is_done(tc.id):
            print(f"  {tc.id} — skipped (checkpoint)")
            continue
        print(f"  {tc.id} — inserting...")
        await inserter.insert_test_case(graphiti, tc)
        ckpt.mark_done(tc.id)

    ckpt.mark_stage_complete()
    print(f"  INSERT complete for {phase}")


# ---------------------------------------------------------------------------
# Stage: evaluate
# ---------------------------------------------------------------------------


async def run_evaluate(
    graphiti: Graphiti, phase: str, test_cases: list[TestCase]
) -> list[QueryResult]:
    """Evaluate all queries with checkpoint/resume. Returns results in memory."""
    ckpt = Checkpoint(phase, "evaluate")
    group_id = GROUP_IDS[phase]
    strategies = get_search_strategies()
    all_results: list[QueryResult] = []

    print(f"\n--- [{phase}] EVALUATE stage ---")
    for tc in test_cases:
        for q_idx in range(len(tc.queries)):
            for strategy_name, config in strategies.items():
                key = f"{tc.id}|{q_idx}|{strategy_name}"
                if ckpt.is_done(key):
                    print(f"  {key} — skipped (checkpoint)")
                    continue
                print(f"  {key}")
                result = await evaluate_query(
                    graphiti, tc, q_idx, strategy_name, config, group_id
                )
                all_results.append(result)
                ckpt.mark_done(key)

    ckpt.mark_stage_complete()
    print(f"  EVALUATE complete for {phase} ({len(all_results)} new results)")
    return all_results


# ---------------------------------------------------------------------------
# Stage: report
# ---------------------------------------------------------------------------


def run_report(
    phase: str, results: list[QueryResult], test_cases: list[TestCase]
) -> list[CategoryReport]:
    """Aggregate and save results."""
    print(f"\n--- [{phase}] REPORT stage ---")
    categories = sorted({tc.category for tc in test_cases})
    all_reports: list[CategoryReport] = []

    for cat in categories:
        cat_ids = {tc.id for tc in test_cases if tc.category == cat}
        cat_results = [r for r in results if r.test_case_id in cat_ids]
        if cat_results:
            all_reports.extend(
                aggregate_results(cat_results, phase=phase, category=cat)
            )

    print_full_report(all_reports)

    Path("results").mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_path = f"results/{phase}_{ts}.json"
    with open(results_path, "w") as f:
        json.dump([r.model_dump(mode="json") for r in all_reports], f, indent=2)
    print(f"  Report saved to {results_path}")
    return all_reports


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

STAGES = ("insert", "evaluate", "report")


async def run_benchmark(
    phase: str,
    port: int | None = None,
    stage: str | None = None,
    clean: bool = False,
) -> list[CategoryReport]:
    """Run a single benchmark phase with optional stage selection."""
    if port is None:
        port = PHASE_PORTS.get(phase, 7687)

    stages_to_run = (stage,) if stage else STAGES
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases | phase={phase} port={port}")

    graphiti = await create_graphiti(neo4j_port=port)
    group_id = GROUP_IDS[phase]

    try:
        if clean:
            print("Cleaning checkpoint + graph data...")
            for s in STAGES[:2]:  # clear insert + evaluate checkpoints
                Checkpoint(phase, s).clear()
            await wipe_graph(graphiti, group_id)

        results: list[QueryResult] = []

        if "insert" in stages_to_run:
            await run_insert(graphiti, phase, test_cases)

        if "evaluate" in stages_to_run:
            results = await run_evaluate(graphiti, phase, test_cases)

        if "report" in stages_to_run:
            if not results:
                print("  No evaluation results — skipping report")
                return []
            return run_report(phase, results, test_cases)

    finally:
        await graphiti.close()

    return []


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Graphiti benchmark runner with checkpoint/resume"
    )
    parser.add_argument(
        "phase",
        choices=list(GROUP_IDS.keys()),
        help="Benchmark phase to run",
    )
    parser.add_argument(
        "--stage",
        choices=list(STAGES),
        default=None,
        help="Run only this stage (default: all stages)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Neo4j bolt port (default: phase-specific from PHASE_PORTS)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe checkpoint + graph data before running",
    )
    return parser


def main():
    args = build_cli().parse_args()
    asyncio.run(
        run_benchmark(
            phase=args.phase,
            port=args.port,
            stage=args.stage,
            clean=args.clean,
        )
    )


if __name__ == "__main__":
    main()
