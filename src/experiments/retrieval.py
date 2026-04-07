"""Retrieval quality experiment — search strategies x queries -> P@5, R@5, MRR, TempAcc."""

from __future__ import annotations

from typing import Any

from graphiti_core import Graphiti

from src.checkpoint import Checkpoint
from src.evaluator import (
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_temporal_accuracy,
)
from src.experiments import ExperimentBase, RunConfig, register_experiment
from src.models import CategoryReport, QueryResult, TestCase
from src.reporter import aggregate_results, print_full_report
from src.search_strategies import get_search_strategies
from src.search_utils import run_search


@register_experiment
class RetrievalExperiment(ExperimentBase):
    """Measures retrieval quality across search strategies."""

    name = "retrieval"
    result_model = QueryResult
    report_model = CategoryReport

    def default_params(self) -> dict[str, Any]:
        return {"strategies": ["hybrid", "bm25_only", "cosine_only"]}

    async def measure(
        self,
        graphiti: Graphiti,
        test_cases: list[TestCase],
        run_config: RunConfig,
        checkpoint: Checkpoint,
    ) -> list[QueryResult]:
        strategy_names = run_config.params.get(
            "strategies", self.default_params()["strategies"]
        )
        all_strategies = get_search_strategies()
        strategies = {
            name: cfg for name, cfg in all_strategies.items() if name in strategy_names
        }
        results: list[QueryResult] = []

        for tc in test_cases:
            for q_idx in range(len(tc.queries)):
                for strategy_name, config in strategies.items():
                    key = f"{tc.id}|{q_idx}|{strategy_name}"
                    if checkpoint.is_done(key):
                        continue
                    print(f"  {key}")
                    result = await _evaluate_query(
                        graphiti,
                        tc,
                        q_idx,
                        strategy_name,
                        config,
                        run_config.group_id,
                    )
                    results.append(result)
                    checkpoint.mark_done(key)

        checkpoint.mark_stage_complete()
        return results

    def report(
        self,
        results: list[QueryResult],
        test_cases: list[TestCase],
        run_config: RunConfig,
    ) -> list[CategoryReport]:
        categories = sorted({tc.category for tc in test_cases})
        reports: list[CategoryReport] = []
        for cat in categories:
            cat_ids = {tc.id for tc in test_cases if tc.category == cat}
            cat_results = [r for r in results if r.test_case_id in cat_ids]
            if cat_results:
                reports.extend(
                    aggregate_results(cat_results, phase=run_config.phase, category=cat)
                )
        return reports

    def print_report(self, reports: list[CategoryReport]) -> None:
        print_full_report(reports)


async def _evaluate_query(
    graphiti: Graphiti,
    test_case: TestCase,
    query_idx: int,
    strategy_name: str,
    config: Any,
    group_id: str,
) -> QueryResult:
    """Evaluate a single query with a single strategy."""
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
