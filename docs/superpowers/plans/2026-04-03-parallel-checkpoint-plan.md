# Parallel & Resilient Benchmark Runner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add checkpoint/resume capability and parallel Docker container support so the benchmark survives crashes and runs 3x faster.

**Architecture:** New `checkpoint.py` handles per-case/per-query state persistence. `benchmark_runner.py` refactored into independent insert/evaluate/report stages with CLI flags (`--stage`, `--port`, `--clean`). Shell script orchestrates 3 Neo4j containers + 3 parallel Python processes.

**Tech Stack:** Python 3.11+, argparse, Docker, Neo4j 5 Community

**Spec:** `docs/superpowers/specs/2026-04-03-parallel-benchmark-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/checkpoint.py` | Create | Load/save/clear checkpoint state |
| `src/benchmark_runner.py` | Rewrite | Split into stages, add CLI, checkpoint integration |
| `scripts/run_parallel.sh` | Create | Docker orchestration + parallel launch |
| `tests/test_checkpoint.py` | Create | Unit tests for checkpoint module |

## Dependency Graph

```
Task 1 (checkpoint module + tests)
  → Task 2 (benchmark_runner refactor)
       → Task 3 (parallel shell script)
            → Task 4 (verify full run)
```

---

### Task 1: Checkpoint Module

**Files:**
- Create: `src/checkpoint.py`
- Create: `tests/test_checkpoint.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_checkpoint.py
import json
import pytest
from pathlib import Path


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    return tmp_path / "checkpoints"


def test_load_empty_returns_default(tmp_checkpoint_dir):
    from src.checkpoint import Checkpoint

    cp = Checkpoint("controlled", "insert", base_dir=str(tmp_checkpoint_dir))
    state = cp.load()
    assert state == {"completed": [], "status": "pending"}


def test_save_and_load_roundtrip(tmp_checkpoint_dir):
    from src.checkpoint import Checkpoint

    cp = Checkpoint("controlled", "insert", base_dir=str(tmp_checkpoint_dir))
    cp.mark_done("static_001")
    cp.mark_done("static_002")

    state = cp.load()
    assert "static_001" in state["completed"]
    assert "static_002" in state["completed"]
    assert state["status"] == "in_progress"


def test_is_done(tmp_checkpoint_dir):
    from src.checkpoint import Checkpoint

    cp = Checkpoint("controlled", "insert", base_dir=str(tmp_checkpoint_dir))
    cp.mark_done("static_001")

    assert cp.is_done("static_001") is True
    assert cp.is_done("static_002") is False


def test_clear(tmp_checkpoint_dir):
    from src.checkpoint import Checkpoint

    cp = Checkpoint("controlled", "insert", base_dir=str(tmp_checkpoint_dir))
    cp.mark_done("static_001")
    cp.clear()

    state = cp.load()
    assert state["completed"] == []


def test_mark_complete(tmp_checkpoint_dir):
    from src.checkpoint import Checkpoint

    cp = Checkpoint("controlled", "insert", base_dir=str(tmp_checkpoint_dir))
    cp.mark_done("static_001")
    cp.mark_stage_complete()

    state = cp.load()
    assert state["status"] == "completed"


def test_atomic_write_survives(tmp_checkpoint_dir):
    from src.checkpoint import Checkpoint

    cp = Checkpoint("controlled", "insert", base_dir=str(tmp_checkpoint_dir))
    cp.mark_done("case_1")
    # File should exist and be valid JSON
    assert cp.path.exists()
    data = json.loads(cp.path.read_text())
    assert "case_1" in data["completed"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the checkpoint module**

```python
# src/checkpoint.py
"""Checkpoint persistence for resumable benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_DIR = "results/checkpoints"


class Checkpoint:
    """Track completion state for a benchmark stage.

    Each (phase, stage) pair gets its own checkpoint file.
    State is saved after every mark_done() call for crash safety.
    """

    def __init__(self, phase: str, stage: str, base_dir: str = DEFAULT_DIR):
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.path = self._dir / f"{phase}_{stage}.json"
        self._state: dict | None = None

    def load(self) -> dict:
        """Load checkpoint state or return default."""
        if self._state is not None:
            return self._state
        if self.path.exists():
            self._state = json.loads(self.path.read_text())
        else:
            self._state = {"completed": [], "status": "pending"}
        return self._state

    def _save(self) -> None:
        """Write state atomically (tmp + rename)."""
        state = self.load()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.rename(self.path)

    def mark_done(self, key: str) -> None:
        """Mark a case/query as completed and persist."""
        state = self.load()
        if key not in state["completed"]:
            state["completed"].append(key)
        state["status"] = "in_progress"
        self._save()

    def is_done(self, key: str) -> bool:
        """Check if a key was already completed."""
        return key in self.load()["completed"]

    def mark_stage_complete(self) -> None:
        """Mark the entire stage as finished."""
        self.load()["status"] = "completed"
        self._save()

    def clear(self) -> None:
        """Delete checkpoint and reset state."""
        if self.path.exists():
            self.path.unlink()
        self._state = {"completed": [], "status": "pending"}
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_checkpoint.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: checkpoint module for resumable benchmark runs"
```

---

### Task 2: Refactor Benchmark Runner

**Files:**
- Rewrite: `src/benchmark_runner.py`

This is the main change. The runner splits into 3 stages (insert, evaluate, report), each checkpointed. CLI adds `--stage`, `--port`, `--clean` flags.

- [ ] **Step 1: Rewrite benchmark_runner.py**

```python
# src/benchmark_runner.py
"""Benchmark runner with checkpoint/resume and configurable Neo4j port."""

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
    ComparisonOperator,
    DateFilter,
    SearchFilters,
)
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from src.checkpoint import Checkpoint
from src.evaluator import (
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_temporal_accuracy,
)
from src.models import CategoryReport, QueryResult, TestCase
from src.reporter import aggregate_results, print_full_report
from src.search_strategies import get_search_strategies
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


def load_test_cases(path: str = "data/test_cases.json") -> list[TestCase]:
    with open(path) as f:
        return [TestCase.model_validate(d) for d in json.load(f)]


async def create_graphiti(neo4j_port: int = 7687) -> Graphiti:
    uri = f"bolt://localhost:{neo4j_port}"
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    llm_client = OpenAIGenericClient(config=LLMConfig(
        api_key=api_key, base_url=base_url,
        model=os.getenv("LLM_MODEL", "Qwen/Qwen3.5-397B-A17B"),
        small_model=os.getenv("LLM_SMALL_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    ))
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
        embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        api_key=api_key, base_url=base_url,
    ))

    graphiti = Graphiti(uri, user, password, llm_client=llm_client, embedder=embedder)
    await graphiti.build_indices_and_constraints()
    return graphiti


def build_temporal_filter(query_time: datetime) -> SearchFilters:
    return SearchFilters(
        valid_at=[[DateFilter(date=query_time, comparison_operator=ComparisonOperator.less_than_equal)]],
        invalid_at=[
            [DateFilter(comparison_operator=ComparisonOperator.is_null)],
            [DateFilter(date=query_time, comparison_operator=ComparisonOperator.greater_than)],
        ],
    )


# ── Stage 1: Insert ─────────────────────────────────────────────────

async def run_insert(graphiti: Graphiti, phase: str, test_cases: list[TestCase], clean: bool = False) -> None:
    group_id = GROUP_IDS[phase]
    cp = Checkpoint(phase, "insert")

    if clean:
        cp.clear()
        await clear_data(graphiti.clients.driver, group_ids=[group_id])
        await graphiti.build_indices_and_constraints()

    inserters = {
        "controlled": controlled_inserter,
        "pipeline": pipeline_inserter,
        "pipeline_presplit": presplit_inserter,
    }
    inserter = inserters[phase]

    for i, tc in enumerate(test_cases):
        if cp.is_done(tc.id):
            print(f"  [skip] {tc.id} (already inserted)")
            continue
        print(f"  Inserting {i + 1}/{len(test_cases)}: {tc.id}")
        await inserter.insert_test_case(graphiti, tc)
        cp.mark_done(tc.id)

    cp.mark_stage_complete()
    print(f"  Insert stage complete ({len(test_cases)} cases)")


# ── Stage 2: Evaluate ────────────────────────────────────────────────

async def run_evaluate(graphiti: Graphiti, phase: str, test_cases: list[TestCase]) -> list[QueryResult]:
    group_id = GROUP_IDS[phase]
    cp = Checkpoint(phase, "evaluate")
    strategies = get_search_strategies()
    all_results: list[QueryResult] = []

    # Load previously completed results
    state = cp.load()
    for entry in state["completed"]:
        all_results.append(QueryResult.model_validate_json(entry))

    for tc in test_cases:
        for q_idx in range(len(tc.queries)):
            for strategy_name, config in strategies.items():
                key = f"{tc.id}|{q_idx}|{strategy_name}"
                if cp.is_done(key):
                    continue

                print(f"  {tc.id} | query {q_idx + 1} | {strategy_name}")
                query = tc.queries[q_idx]
                query_time = query.query_time if strategy_name == "hybrid" else None

                search_filter = build_temporal_filter(query_time) if query_time else None
                results = await graphiti.search_(
                    query=query.query,
                    config=config,
                    group_ids=[group_id],
                    search_filter=search_filter,
                )
                returned_facts = [edge.fact for edge in results.edges]

                p5 = await compute_precision_at_k(returned_facts, query.expected_facts, k=5)
                r5 = await compute_recall_at_k(returned_facts, query.expected_facts, k=5)
                mrr = await compute_mrr(returned_facts, query.expected_facts)
                ta = await compute_temporal_accuracy(returned_facts, query.expected_not)

                qr = QueryResult(
                    test_case_id=tc.id, query=query.query, strategy=strategy_name,
                    returned_facts=returned_facts[:5], expected_facts=query.expected_facts,
                    expected_not=query.expected_not, precision_at_5=p5,
                    recall_at_5=r5, mrr=mrr, temporal_accuracy=ta,
                )
                all_results.append(qr)
                cp.mark_done(key)  # checkpoint stores the key; full results saved in report stage

    cp.mark_stage_complete()
    return all_results


# ── Stage 3: Report ──────────────────────────────────────────────────

def run_report(phase: str, test_cases: list[TestCase], results: list[QueryResult]) -> list[CategoryReport]:
    all_reports: list[CategoryReport] = []
    categories = sorted(set(tc.category for tc in test_cases))
    for cat in categories:
        cat_ids = {tc.id for tc in test_cases if tc.category == cat}
        cat_results = [r for r in results if r.test_case_id in cat_ids]
        if cat_results:
            reports = aggregate_results(cat_results, phase=phase, category=cat)
            all_reports.extend(reports)

    print_full_report(all_reports)

    Path("results").mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = f"results/benchmark_{phase}_{ts}.json"
    with open(path, "w") as f:
        json.dump([r.model_dump(mode="json") for r in all_reports], f, indent=2)
    print(f"Results saved to {path}")
    return all_reports


# ── Orchestrator ─────────────────────────────────────────────────────

async def run_benchmark(phase: str, port: int = 7687, stage: str | None = None, clean: bool = False) -> None:
    test_cases = load_test_cases()
    print(f"Phase: {phase} | Port: {port} | Stage: {stage or 'all'} | Cases: {len(test_cases)}")

    graphiti = await create_graphiti(neo4j_port=port)
    try:
        if stage in (None, "insert"):
            print(f"\n--- INSERT ({phase}) ---")
            await run_insert(graphiti, phase, test_cases, clean=clean)
            if stage == "insert":
                return

        if stage in (None, "evaluate"):
            print(f"\n--- EVALUATE ({phase}) ---")
            results = await run_evaluate(graphiti, phase, test_cases)
            if stage == "evaluate":
                return

        if stage in (None, "report"):
            print(f"\n--- REPORT ({phase}) ---")
            if stage == "report":
                # Re-run evaluate to collect results from checkpoint
                results = await run_evaluate(graphiti, phase, test_cases)
            run_report(phase, test_cases, results)

    finally:
        await graphiti.close()


def main():
    parser = argparse.ArgumentParser(description="Graphiti Temporal Benchmark Runner")
    parser.add_argument("phase", choices=list(GROUP_IDS.keys()), help="Benchmark phase")
    parser.add_argument("--stage", choices=["insert", "evaluate", "report"], help="Run specific stage only")
    parser.add_argument("--port", type=int, default=None, help="Neo4j port (default: phase-specific)")
    parser.add_argument("--clean", action="store_true", help="Wipe checkpoints and graph data")
    args = parser.parse_args()

    port = args.port or PHASE_PORTS.get(args.phase, 7687)
    asyncio.run(run_benchmark(phase=args.phase, port=port, stage=args.stage, clean=args.clean))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify imports**

Run: `.venv/bin/python -c "from src.benchmark_runner import GROUP_IDS, PHASE_PORTS; print(GROUP_IDS); print(PHASE_PORTS)"`
Expected: Both dicts printed

- [ ] **Step 3: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All pass (existing tests shouldn't break — CLI change is backward compatible)

- [ ] **Step 4: Commit**

```bash
git add src/benchmark_runner.py
git commit -m "refactor: benchmark runner with checkpoint/resume + port config + CLI"
```

---

### Task 3: Parallel Shell Script

**Files:**
- Create: `scripts/run_parallel.sh`

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# Run all 3 benchmark phases in parallel with separate Neo4j containers.
# Usage: ./scripts/run_parallel.sh [--clean]
set -e

CLEAN_FLAG=""
if [ "$1" = "--clean" ]; then
    CLEAN_FLAG="--clean"
    echo "Clean mode: will wipe checkpoints and graph data"
fi

NEO4J_IMAGE="neo4j:5"
NEO4J_PASS="benchmark123"

declare -A PHASES
PHASES[controlled]=7687
PHASES[pipeline]=7688
PHASES[pipeline_presplit]=7689

echo "=== Starting Neo4j containers ==="
for phase in "${!PHASES[@]}"; do
    port=${PHASES[$phase]}
    name="neo4j-$phase"
    if docker ps -q -f name="$name" 2>/dev/null | grep -q .; then
        echo "  $name already running on port $port"
    else
        docker rm -f "$name" 2>/dev/null || true
        docker run -d --name "$name" -p "$port":7687 \
            -e NEO4J_AUTH="neo4j/$NEO4J_PASS" "$NEO4J_IMAGE" >/dev/null
        echo "  Started $name on port $port"
    fi
done

echo "=== Waiting for Neo4j readiness ==="
for phase in "${!PHASES[@]}"; do
    port=${PHASES[$phase]}
    name="neo4j-$phase"
    for i in $(seq 1 30); do
        if docker exec "$name" cypher-shell -u neo4j -p "$NEO4J_PASS" "RETURN 1" 2>/dev/null | grep -q "1"; then
            echo "  $name ready"
            break
        fi
        sleep 2
    done
done

echo "=== Running benchmarks in parallel ==="
PIDS=()
for phase in "${!PHASES[@]}"; do
    port=${PHASES[$phase]}
    PYTHONUNBUFFERED=1 .venv/bin/python -m src.benchmark_runner "$phase" --port "$port" $CLEAN_FLAG \
        > "results/${phase}_output.log" 2>&1 &
    PIDS+=($!)
    echo "  Started $phase (PID $!, port $port)"
done

echo "=== Waiting for all phases to complete ==="
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    if wait "$pid"; then
        echo "  PID $pid completed successfully"
    else
        echo "  PID $pid FAILED (exit code $?)"
        FAILED=1
    fi
done

echo "=== Results ==="
ls -la results/*.json 2>/dev/null
echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "All phases complete!"
else
    echo "Some phases failed. Check results/*_output.log for details."
fi
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/run_parallel.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_parallel.sh
git commit -m "feat: parallel benchmark script with Docker orchestration"
```

---

### Task 4: Verify Full Run

- [ ] **Step 1: Run unit tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 2: Test single-phase with checkpoint**

```bash
# Run controlled insert only
.venv/bin/python -m src.benchmark_runner controlled --stage insert --port 7687 --clean

# Verify checkpoint exists
cat results/checkpoints/controlled_insert.json | python3 -m json.tool | head -5

# Run evaluate (resumes from checkpoint)
.venv/bin/python -m src.benchmark_runner controlled --stage evaluate --port 7687

# Ctrl+C during evaluate, then re-run — should skip completed queries
.venv/bin/python -m src.benchmark_runner controlled --stage evaluate --port 7687
```

- [ ] **Step 3: Test parallel run (if 3 containers available)**

```bash
./scripts/run_parallel.sh --clean
```

Expected: All 3 phases complete, results in `results/`

- [ ] **Step 4: Commit results**

```bash
git add -f results/
git commit -m "feat: benchmark results from parallel resilient runner"
```
