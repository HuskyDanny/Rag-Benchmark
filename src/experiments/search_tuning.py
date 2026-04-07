"""Search tuning experiment — parameterized search configs across multiple runs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.checkpoint import Checkpoint
from src.evaluator import (
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_temporal_accuracy,
)
from src.experiments import ExperimentBase, RunConfig, register_experiment
from src.models import QueryResult, TestCase
from src.search_strategies import (
    DEFAULT_BFS_MAX_DEPTH,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_SIM_MIN_SCORE,
    build_search_config,
)
from src.search_utils import run_search


class TuningRunSummary(BaseModel):
    """Aggregate summary for a single parameter configuration run."""

    run_id: str
    phase: str
    params: dict[str, Any]
    avg_precision_at_5: float
    avg_recall_at_5: float
    avg_mrr: float
    temporal_accuracy_pct: float
    num_queries: int


@register_experiment
class SearchTuningExperiment(ExperimentBase):
    """Runs search with parameterized config, enabling multi-run comparison."""

    name = "search_tuning"
    result_model = QueryResult
    report_model = TuningRunSummary

    def default_params(self) -> dict[str, Any]:
        return {
            "search_methods": ["bm25", "cosine_similarity"],
            "reranker": "rrf",
            "sim_min_score": DEFAULT_SIM_MIN_SCORE,
            "mmr_lambda": DEFAULT_MMR_LAMBDA,
            "bfs_max_depth": DEFAULT_BFS_MAX_DEPTH,
            "reranker_min_score": 0,
            "limit": 10,
            "use_temporal_filter": True,
        }

    def validate_params(self, params: dict[str, Any]) -> None:
        if "mmr_lambda" in params:
            val = params["mmr_lambda"]
            if not (0 <= val <= 1):
                raise ValueError(f"mmr_lambda must be 0-1, got {val}")
        if "sim_min_score" in params:
            val = params["sim_min_score"]
            if not (0 <= val <= 1):
                raise ValueError(f"sim_min_score must be 0-1, got {val}")
        if "bfs_max_depth" in params:
            val = params["bfs_max_depth"]
            if val < 1:
                raise ValueError(f"bfs_max_depth must be >= 1, got {val}")

    async def measure(
        self,
        graphiti: Any,
        test_cases: list[TestCase],
        run_config: RunConfig,
        checkpoint: Checkpoint,
    ) -> list[QueryResult]:
        config = build_search_config(run_config.params)
        use_temporal = run_config.params.get("use_temporal_filter", True)
        results: list[QueryResult] = []

        for tc in test_cases:
            for q_idx in range(len(tc.queries)):
                key = f"{tc.id}|{q_idx}"
                if checkpoint.is_done(key):
                    continue

                query = tc.queries[q_idx]
                query_time = query.query_time if use_temporal else None

                print(f"  {key}")
                returned_facts = await run_search(
                    graphiti,
                    query.query,
                    config,
                    run_config.group_id,
                    query_time=query_time,
                )

                p_at_5 = await compute_precision_at_k(
                    returned_facts, query.expected_facts, k=5
                )
                r_at_5 = await compute_recall_at_k(
                    returned_facts, query.expected_facts, k=5
                )
                mrr = await compute_mrr(returned_facts, query.expected_facts)
                temp_acc = await compute_temporal_accuracy(
                    returned_facts, query.expected_not
                )

                results.append(
                    QueryResult(
                        test_case_id=tc.id,
                        query=query.query,
                        strategy=f"tuning_{run_config.run_id}",
                        returned_facts=returned_facts[:5],
                        expected_facts=query.expected_facts,
                        expected_not=query.expected_not,
                        precision_at_5=p_at_5,
                        recall_at_5=r_at_5,
                        mrr=mrr,
                        temporal_accuracy=temp_acc,
                    )
                )
                checkpoint.mark_done(key)

        checkpoint.mark_stage_complete()
        return results

    def report(
        self,
        results: list[QueryResult],
        test_cases: list[TestCase],
        run_config: RunConfig,
    ) -> list[TuningRunSummary]:
        if not results:
            return []
        n = len(results)
        return [
            TuningRunSummary(
                run_id=run_config.run_id,
                phase=run_config.phase,
                params=run_config.params,
                avg_precision_at_5=round(sum(r.precision_at_5 for r in results) / n, 4),
                avg_recall_at_5=round(sum(r.recall_at_5 for r in results) / n, 4),
                avg_mrr=round(sum(r.mrr for r in results) / n, 4),
                temporal_accuracy_pct=round(
                    sum(1 for r in results if r.temporal_accuracy) / n * 100, 2
                ),
                num_queries=n,
            )
        ]

    def print_report(self, reports: list[TuningRunSummary]) -> None:
        if not reports:
            print("  No tuning results.")
            return
        for r in reports:
            print(f"\n  Run: {r.run_id}")
            print(f"  Phase: {r.phase}")
            print(f"  Params: {r.params}")
            print(
                f"  P@5={r.avg_precision_at_5:.4f}  R@5={r.avg_recall_at_5:.4f}  "
                f"MRR={r.avg_mrr:.4f}  TempAcc={r.temporal_accuracy_pct:.1f}%  "
                f"({r.num_queries} queries)"
            )
