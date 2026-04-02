"""Main orchestrator: runs both phases, all strategies, scores everything."""

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
from src import controlled_inserter, pipeline_inserter

load_dotenv()

GROUP_IDS = {
    "controlled": controlled_inserter.GROUP_ID,
    "pipeline": pipeline_inserter.GROUP_ID,
}


def load_test_cases(path: str = "data/test_cases.json") -> list[TestCase]:
    """Load test cases from JSON file."""
    with open(path) as f:
        return [TestCase.model_validate(d) for d in json.load(f)]


async def create_graphiti() -> Graphiti:
    """Create and initialize a Graphiti instance.

    Uses OpenAIGenericClient for SiliconFlow compatibility (JSON mode
    instead of structured output).
    """
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    small_model = os.getenv("LLM_SMALL_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    llm_config = LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        small_model=small_model,
    )
    llm_client = OpenAIGenericClient(config=llm_config)

    embedder_config = OpenAIEmbedderConfig(
        embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        api_key=api_key,
        base_url=base_url,
    )
    embedder = OpenAIEmbedder(config=embedder_config)

    graphiti = Graphiti(uri, user, password, llm_client=llm_client, embedder=embedder)
    await graphiti.build_indices_and_constraints()
    return graphiti


async def wipe_graph(graphiti: Graphiti, group_id: str | None = None) -> None:
    """Clear graph data, optionally scoped to a group."""
    if group_id:
        await clear_data(graphiti.clients.driver, group_ids=[group_id])
    else:
        await clear_data(graphiti.clients.driver)
    await graphiti.build_indices_and_constraints()


def build_temporal_filter(query_time: datetime) -> SearchFilters:
    """Build a SearchFilter that only returns edges valid at query_time.

    Filters:
    - valid_at <= query_time (edge was valid at or before query time)
    - invalid_at IS NULL OR invalid_at > query_time (edge hasn't expired)
    """
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
                    date=query_time, comparison_operator=ComparisonOperator.greater_than
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
    """Execute a search and return facts from matching edges."""
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
    """Run a single query and compute all metrics."""
    query = test_case.queries[query_idx]
    # Only apply temporal filter to hybrid — baselines run unfiltered
    # to test whether the temporal graph adds value over flat search
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


async def run_phase(
    graphiti: Graphiti, phase: str, test_cases: list[TestCase]
) -> list[QueryResult]:
    """Run one benchmark phase (controlled or pipeline)."""
    group_id = GROUP_IDS[phase]

    print(f"\n--- Phase: {phase.upper()} ---")
    print("Wiping graph...")
    await wipe_graph(graphiti, group_id)

    print("Inserting test data...")
    if phase == "controlled":
        await controlled_inserter.insert_all(graphiti, test_cases)
    else:
        await pipeline_inserter.insert_all(graphiti, test_cases)

    strategies = get_search_strategies()
    all_results: list[QueryResult] = []

    for tc in test_cases:
        for q_idx in range(len(tc.queries)):
            for strategy_name, config in strategies.items():
                print(f"  {tc.id} | query {q_idx + 1} | {strategy_name}")
                result = await evaluate_query(
                    graphiti, tc, q_idx, strategy_name, config, group_id
                )
                all_results.append(result)

    return all_results


async def run_benchmark(
    test_cases_path: str = "data/test_cases.json",
    phases: list[str] | None = None,
) -> list[CategoryReport]:
    """Run the full benchmark across specified phases."""
    if phases is None:
        phases = ["controlled", "pipeline"]

    test_cases = load_test_cases(test_cases_path)
    print(f"Loaded {len(test_cases)} test cases")

    graphiti = await create_graphiti()
    all_reports: list[CategoryReport] = []

    try:
        for phase in phases:
            results = await run_phase(graphiti, phase, test_cases)
            categories = sorted(set(tc.category for tc in test_cases))
            for cat in categories:
                cat_ids = {tc.id for tc in test_cases if tc.category == cat}
                cat_results = [r for r in results if r.test_case_id in cat_ids]
                reports = aggregate_results(cat_results, phase=phase, category=cat)
                all_reports.extend(reports)

        print_full_report(all_reports)

        Path("results").mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_path = f"results/benchmark_{ts}.json"
        with open(results_path, "w") as f:
            json.dump([r.model_dump(mode="json") for r in all_reports], f, indent=2)
        print(f"\nResults saved to {results_path}")

    finally:
        await graphiti.close()

    return all_reports


def main():
    """CLI entry point."""
    import sys

    phases = sys.argv[1:] if len(sys.argv) > 1 else None
    asyncio.run(run_benchmark(phases=phases))


if __name__ == "__main__":
    main()
