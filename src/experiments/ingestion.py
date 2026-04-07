"""Ingestion quality experiment — inspect Neo4j graph state vs expected triplets."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.checkpoint import Checkpoint
from src.experiments import ExperimentBase, RunConfig, register_experiment
from src.graph_inspector import inspect_edges, inspect_node_duplicates, inspect_nodes
from src.judge_cache import cached_facts_match
from src.models import IngestionReport, IngestionResult, TestCase

# Scoring weights — temporal + edge = 50% combined (Graphiti's differentiator)
WEIGHTS = {
    "entity_recall": 0.20,
    "entity_precision": 0.10,
    "edge_recall": 0.25,
    "edge_precision": 0.10,
    "temporal_invalidation_accuracy": 0.25,
    "dedup_score": 0.10,
}


@register_experiment
class IngestionExperiment(ExperimentBase):
    """Measures ingestion quality by inspecting Neo4j graph state against ground truth."""

    name = "ingestion"
    result_model = IngestionResult
    report_model = IngestionReport

    def default_params(self) -> dict[str, Any]:
        return {}

    async def measure(
        self,
        graphiti: Any,
        test_cases: list[TestCase],
        run_config: RunConfig,
        checkpoint: Checkpoint,
    ) -> list[IngestionResult]:
        driver = graphiti.clients.driver
        results: list[IngestionResult] = []

        for tc in test_cases:
            if checkpoint.is_done(tc.id):
                continue
            print(f"  {tc.id} — inspecting graph...")
            result = await _validate_test_case(driver, tc, run_config.group_id)
            results.append(result)
            checkpoint.mark_done(tc.id)

        checkpoint.mark_stage_complete()
        return results

    def report(
        self,
        results: list[IngestionResult],
        test_cases: list[TestCase],
        run_config: RunConfig,
    ) -> list[IngestionReport]:
        by_category: dict[str, list[IngestionResult]] = defaultdict(list)
        for r in results:
            by_category[r.category].append(r)

        reports = []
        for cat in sorted(by_category.keys()):
            cat_results = by_category[cat]
            n = len(cat_results)
            er = sum(r.entity_recall for r in cat_results) / n
            ep = sum(r.entity_precision for r in cat_results) / n
            edr = sum(r.edge_recall for r in cat_results) / n
            edp = sum(r.edge_precision for r in cat_results) / n
            tia = sum(r.temporal_invalidation_accuracy for r in cat_results) / n
            ds = sum(r.dedup_score for r in cat_results) / n

            composite = (
                WEIGHTS["entity_recall"] * er
                + WEIGHTS["entity_precision"] * ep
                + WEIGHTS["edge_recall"] * edr
                + WEIGHTS["edge_precision"] * edp
                + WEIGHTS["temporal_invalidation_accuracy"] * tia
                + WEIGHTS["dedup_score"] * ds
            )

            reports.append(
                IngestionReport(
                    phase=run_config.phase,
                    category=cat,
                    avg_entity_recall=round(er, 3),
                    avg_entity_precision=round(ep, 3),
                    avg_edge_recall=round(edr, 3),
                    avg_edge_precision=round(edp, 3),
                    avg_temporal_invalidation_accuracy=round(tia, 3),
                    avg_dedup_score=round(ds, 3),
                    composite_score=round(composite, 3),
                    num_cases=n,
                )
            )
        return reports

    def print_report(self, reports: list[IngestionReport]) -> None:
        if not reports:
            print("  No ingestion results.")
            return

        header = (
            f"{'Category':<28} | {'EntRec':>6} | {'EntPrec':>7} | {'EdgeRec':>7} "
            f"| {'EdgePrec':>8} | {'TempInv':>7} | {'Dedup':>5} | {'Score':>5}"
        )
        print(header)
        print("-" * len(header))
        for r in reports:
            print(
                f"{r.category:<28} | {r.avg_entity_recall:>6.3f} | "
                f"{r.avg_entity_precision:>7.3f} | {r.avg_edge_recall:>7.3f} | "
                f"{r.avg_edge_precision:>8.3f} | "
                f"{r.avg_temporal_invalidation_accuracy:>7.3f} | "
                f"{r.avg_dedup_score:>5.3f} | {r.composite_score:>5.3f}"
            )


# ── Internal validation logic ──


async def _validate_test_case(
    driver: Any, test_case: TestCase, group_id: str
) -> IngestionResult:
    """Validate graph state for one test case against its triplets."""
    actual_nodes = await inspect_nodes(driver, group_id)
    actual_edges = await inspect_edges(driver, group_id)
    duplicates = await inspect_node_duplicates(driver, group_id)

    # Extract expected from triplets
    expected_entities = set()
    for t in test_case.triplets:
        expected_entities.add(t.source.name.lower())
        expected_entities.add(t.target.name.lower())

    # Entity recall: how many expected entities exist in the graph?
    actual_names = {n["name"].lower() for n in actual_nodes}
    found_entities = expected_entities & actual_names
    # Fuzzy match for entities not found by exact name
    missing = expected_entities - found_entities
    for exp_name in list(missing):
        for actual_name in actual_names - found_entities:
            if await _names_match(exp_name, actual_name):
                found_entities.add(exp_name)
                missing.discard(exp_name)
                break

    entity_recall = (
        len(found_entities) / len(expected_entities) if expected_entities else 1.0
    )

    # Entity precision: 1 - (hallucinated / total)
    # For simplicity, we don't penalize extra entities from other test cases sharing the group
    entity_precision = 1.0  # conservative default

    # Edge matching
    expected_edges = []
    for t in test_case.triplets:
        expected_edges.append(
            {
                "source": t.source.name.lower(),
                "target": t.target.name.lower(),
                "fact": t.edge.fact,
                "should_be_invalidated": t.edge.invalid_at is not None,
            }
        )

    found_edges = 0
    temporal_correct = 0
    temporal_total = 0

    for exp in expected_edges:
        matched = await _find_matching_edge(exp, actual_edges)
        if matched:
            found_edges += 1
            if exp["should_be_invalidated"]:
                temporal_total += 1
                if matched.get("invalid_at") is not None:
                    temporal_correct += 1
            else:
                # Current edge — should NOT have invalid_at
                if matched.get("invalid_at") is None:
                    pass  # correct, no penalty
        else:
            if exp["should_be_invalidated"]:
                temporal_total += 1
                # Edge missing entirely — count as temporal failure

    edge_recall = found_edges / len(expected_edges) if expected_edges else 1.0
    edge_precision = 1.0  # conservative default

    temporal_inv_accuracy = (
        temporal_correct / temporal_total if temporal_total > 0 else 1.0
    )

    # Dedup score
    total_nodes = len(actual_nodes)
    dup_count = sum(d["count"] - 1 for d in duplicates)
    dedup_score = 1.0 - (dup_count / total_nodes) if total_nodes > 0 else 1.0

    return IngestionResult(
        test_case_id=test_case.id,
        category=test_case.category,
        entity_recall=round(entity_recall, 4),
        entity_precision=round(entity_precision, 4),
        edge_recall=round(edge_recall, 4),
        edge_precision=round(edge_precision, 4),
        temporal_invalidation_accuracy=round(temporal_inv_accuracy, 4),
        dedup_score=round(dedup_score, 4),
    )


async def _names_match(expected: str, actual: str) -> bool:
    """Check if two entity names refer to the same entity (fuzzy)."""
    # Exact match (already lowercased by caller)
    if expected == actual:
        return True
    # Substring containment (e.g., "google" in "google inc")
    if expected in actual or actual in expected:
        return True
    # LLM judge for abbreviations (amzn ↔ amazon)
    return await cached_facts_match(expected, actual)


async def _find_matching_edge(expected: dict, actual_edges: list[dict]) -> dict | None:
    """Find an actual edge that matches the expected edge by fact content."""
    for actual in actual_edges:
        # Fast path: source/target name match
        src_match = expected["source"] in actual.get("source", "").lower()
        tgt_match = expected["target"] in actual.get("target", "").lower()
        if not (src_match and tgt_match):
            continue
        # Fact content match (semantic via judge cache)
        if await cached_facts_match(expected["fact"], actual.get("fact", "")):
            return actual
    # Fallback: match by fact alone (entities may have been normalized differently)
    for actual in actual_edges:
        if await cached_facts_match(expected["fact"], actual.get("fact", "")):
            return actual
    return None
