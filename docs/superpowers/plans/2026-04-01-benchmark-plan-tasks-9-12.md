# Benchmark Plan: Tasks 9-12

Parent: [[benchmark-plan-index]]

---

### Task 9: Reporter (Aggregate Results & Print Tables)

**Files:**
- Create: `src/reporter.py`
- Create: `tests/test_reporter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reporter.py
import pytest
from src.models import QueryResult, CategoryReport


def test_aggregate_results():
    from src.reporter import aggregate_results

    results = [
        QueryResult(
            test_case_id="evolving_001", query="q1", strategy="hybrid",
            returned_facts=[], expected_facts=[], expected_not=[],
            precision_at_5=0.8, recall_at_5=1.0, mrr=1.0, temporal_accuracy=True,
        ),
        QueryResult(
            test_case_id="evolving_002", query="q2", strategy="hybrid",
            returned_facts=[], expected_facts=[], expected_not=[],
            precision_at_5=0.6, recall_at_5=0.5, mrr=0.5, temporal_accuracy=False,
        ),
    ]
    reports = aggregate_results(results, phase="controlled", category="evolving_fact")
    assert len(reports) == 1
    r = reports[0]
    assert r.phase == "controlled"
    assert r.category == "evolving_fact"
    assert r.strategy == "hybrid"
    assert r.avg_precision_at_5 == pytest.approx(0.7)
    assert r.avg_recall_at_5 == pytest.approx(0.75)
    assert r.avg_mrr == pytest.approx(0.75)
    assert r.temporal_accuracy_pct == pytest.approx(50.0)
    assert r.num_queries == 2


def test_format_report_table():
    from src.reporter import format_report_table

    reports = [
        CategoryReport(
            phase="controlled", category="evolving_fact", strategy="hybrid",
            avg_precision_at_5=0.8, avg_recall_at_5=0.9, avg_mrr=0.85,
            temporal_accuracy_pct=90.0, num_queries=15,
        ),
    ]
    table = format_report_table(reports)
    assert "evolving_fact" in table
    assert "hybrid" in table
    assert "0.80" in table
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reporter.py -v`
Expected: FAIL

- [ ] **Step 3: Write the reporter**

```python
# src/reporter.py
"""Aggregate benchmark results and format report tables."""

from collections import defaultdict
from tabulate import tabulate
from src.models import QueryResult, CategoryReport


def aggregate_results(
    results: list[QueryResult], phase: str, category: str
) -> list[CategoryReport]:
    """Aggregate query results by strategy for a given phase and category."""
    by_strategy: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        by_strategy[r.strategy].append(r)

    reports = []
    for strategy, strategy_results in sorted(by_strategy.items()):
        n = len(strategy_results)
        reports.append(
            CategoryReport(
                phase=phase,
                category=category,
                strategy=strategy,
                avg_precision_at_5=sum(r.precision_at_5 for r in strategy_results) / n,
                avg_recall_at_5=sum(r.recall_at_5 for r in strategy_results) / n,
                avg_mrr=sum(r.mrr for r in strategy_results) / n,
                temporal_accuracy_pct=(
                    sum(1 for r in strategy_results if r.temporal_accuracy) / n * 100
                ),
                num_queries=n,
            )
        )
    return reports


def format_report_table(reports: list[CategoryReport]) -> str:
    """Format category reports as a table."""
    headers = ["Phase", "Category", "Strategy", "P@5", "R@5", "MRR", "TempAcc%", "N"]
    rows = [
        [
            r.phase, r.category, r.strategy,
            f"{r.avg_precision_at_5:.2f}", f"{r.avg_recall_at_5:.2f}",
            f"{r.avg_mrr:.2f}", f"{r.temporal_accuracy_pct:.1f}%", r.num_queries,
        ]
        for r in reports
    ]
    return tabulate(rows, headers=headers, tablefmt="grid")


def print_full_report(all_reports: list[CategoryReport]) -> None:
    """Print the full benchmark report grouped by phase."""
    for phase in ["controlled", "pipeline"]:
        phase_reports = [r for r in all_reports if r.phase == phase]
        if not phase_reports:
            continue
        print(f"\n{'='*80}")
        print(f"  PHASE: {phase.upper()}")
        print(f"{'='*80}")
        categories = sorted(set(r.category for r in phase_reports))
        for cat in categories:
            cat_reports = [r for r in phase_reports if r.category == cat]
            print(f"\n--- {cat} ---")
            print(format_report_table(cat_reports))

    # Summary
    print(f"\n{'='*80}")
    print("  SUMMARY: Hybrid vs Baselines (averaged across categories)")
    print(f"{'='*80}")
    for phase in ["controlled", "pipeline"]:
        phase_reports = [r for r in all_reports if r.phase == phase]
        if not phase_reports:
            continue
        print(f"\n  Phase: {phase}")
        for strategy in ["hybrid", "bm25_only", "cosine_only"]:
            sr = [r for r in phase_reports if r.strategy == strategy]
            if not sr:
                continue
            avg_p = sum(r.avg_precision_at_5 for r in sr) / len(sr)
            avg_r = sum(r.avg_recall_at_5 for r in sr) / len(sr)
            avg_m = sum(r.avg_mrr for r in sr) / len(sr)
            avg_t = sum(r.temporal_accuracy_pct for r in sr) / len(sr)
            print(
                f"    {strategy:15s}  P@5={avg_p:.2f}  R@5={avg_r:.2f}  "
                f"MRR={avg_m:.2f}  TempAcc={avg_t:.1f}%"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reporter.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/reporter.py tests/test_reporter.py
git commit -m "feat: reporter for aggregating and displaying benchmark results"
```

---

### Task 10: Benchmark Runner (Orchestrator)

**Files:**
- Create: `src/benchmark_runner.py`

**Depends on:** All previous tasks (3-9)

- [ ] **Step 1: Write the benchmark runner**

```python
# src/benchmark_runner.py
"""Main orchestrator: runs both phases, all strategies, scores everything."""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.search.search_config import SearchConfig, SearchResults

from src.models import TestCase, QueryResult
from src.search_strategies import get_search_strategies
from src.evaluator import (
    compute_precision_at_k, compute_recall_at_k,
    compute_mrr, compute_temporal_accuracy,
)
from src.reporter import aggregate_results, print_full_report, CategoryReport
from src import controlled_inserter, pipeline_inserter

load_dotenv()

GROUP_IDS = {
    "controlled": controlled_inserter.GROUP_ID,
    "pipeline": pipeline_inserter.GROUP_ID,
}


def load_test_cases(path: str = "data/test_cases.json") -> list[TestCase]:
    with open(path) as f:
        return [TestCase.model_validate(d) for d in json.load(f)]


async def create_graphiti() -> Graphiti:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    graphiti = Graphiti(uri, user, password)
    await graphiti.build_indices_and_constraints()
    return graphiti


async def wipe_graph(graphiti: Graphiti, group_id: str | None = None) -> None:
    if group_id:
        await graphiti.clients.driver.graph_ops.clear_data(
            graphiti.clients.driver, group_ids=[group_id]
        )
    else:
        await graphiti.clients.driver.graph_ops.clear_data(graphiti.clients.driver)
    await graphiti.build_indices_and_constraints()


async def run_search(
    graphiti: Graphiti, query: str, config: SearchConfig, group_id: str
) -> list[str]:
    results: SearchResults = await graphiti.search_(
        query=query, config=config, group_ids=[group_id]
    )
    return [edge.fact for edge in results.edges]


async def evaluate_query(
    graphiti: Graphiti, test_case: TestCase, query_idx: int,
    strategy_name: str, config: SearchConfig, group_id: str,
) -> QueryResult:
    query = test_case.queries[query_idx]
    returned_facts = await run_search(graphiti, query.query, config, group_id)

    p_at_5 = await compute_precision_at_k(returned_facts, query.expected_facts, k=5)
    r_at_5 = await compute_recall_at_k(returned_facts, query.expected_facts, k=5)
    mrr = await compute_mrr(returned_facts, query.expected_facts)
    temp_acc = await compute_temporal_accuracy(returned_facts, query.expected_not)

    return QueryResult(
        test_case_id=test_case.id, query=query.query, strategy=strategy_name,
        returned_facts=returned_facts[:5], expected_facts=query.expected_facts,
        expected_not=query.expected_not, precision_at_5=p_at_5,
        recall_at_5=r_at_5, mrr=mrr, temporal_accuracy=temp_acc,
    )


async def run_phase(
    graphiti: Graphiti, phase: str, test_cases: list[TestCase]
) -> list[QueryResult]:
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
                print(f"  {tc.id} | query {q_idx+1} | {strategy_name}")
                result = await evaluate_query(
                    graphiti, tc, q_idx, strategy_name, config, group_id
                )
                all_results.append(result)

    return all_results


async def run_benchmark(
    test_cases_path: str = "data/test_cases.json",
    phases: list[str] | None = None,
) -> list[CategoryReport]:
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
        with open(f"results/benchmark_{ts}.json", "w") as f:
            json.dump([r.model_dump(mode="json") for r in all_reports], f, indent=2)
        print(f"\nResults saved to results/benchmark_{ts}.json")

    finally:
        await graphiti.close()

    return all_reports


def main():
    import sys
    phases = sys.argv[1:] if len(sys.argv) > 1 else None
    asyncio.run(run_benchmark(phases=phases))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/benchmark_runner.py
git commit -m "feat: benchmark runner orchestrating both phases and all strategies"
```

---

### Task 11: Validation Spike (Verify Neo4j + Graphiti Work)

**Files:**
- Create: `scripts/spike_test.py`

- [ ] **Step 1: Write the validation spike**

```python
# scripts/spike_test.py
"""Verify Neo4j connection + Graphiti add_triplet + search work."""

import asyncio
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

load_dotenv()

GROUP_ID = "spike_test"


async def main():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    print(f"Connecting to Neo4j at {uri}...")
    graphiti = Graphiti(uri, user, password)

    try:
        await graphiti.build_indices_and_constraints()
        print("OK: Connected and indices built")

        await graphiti.clients.driver.graph_ops.clear_data(
            graphiti.clients.driver, group_ids=[GROUP_ID]
        )
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

        await graphiti.clients.driver.graph_ops.clear_data(
            graphiti.clients.driver, group_ids=[GROUP_ID]
        )
    finally:
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run the spike**

```bash
python3 scripts/spike_test.py
```

Expected: `SPIKE PASSED: Neo4j + Graphiti + search all working`

- [ ] **Step 3: Commit**

```bash
git add scripts/spike_test.py
git commit -m "feat: validation spike for Neo4j + Graphiti connectivity"
```

---

### Task 12: Run Full Benchmark & Capture Results

- [ ] **Step 1: Run unit tests**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Run validation spike**

Run: `python3 scripts/spike_test.py`
Expected: SPIKE PASSED

- [ ] **Step 3: Run Phase 1 only (controlled)**

Run: `python3 -m src.benchmark_runner controlled`
Expected: Results table for controlled phase

- [ ] **Step 4: Run Phase 2 (pipeline) — slower due to LLM extraction**

Run: `python3 -m src.benchmark_runner pipeline`
Expected: Results table for pipeline phase

- [ ] **Step 5: Run full benchmark (both phases)**

Run: `python3 -m src.benchmark_runner`
Expected: Full results comparing both phases across all categories and strategies

- [ ] **Step 6: Commit results**

```bash
git add results/
git commit -m "feat: benchmark results from full run"
```
