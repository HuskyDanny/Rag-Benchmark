# Parallel & Resilient Benchmark Runner — Design Spec

**Date:** 2026-04-03
**Goal:** Run benchmark phases in true parallel with zero Neo4j contention, and survive crashes/timeouts without losing progress.

---

## 1. Problems

**Contention:** Neo4j Community Edition doesn't support multiple databases. All phases share a single graph instance — index locks, fulltext rebuilds, and concurrent writes conflict.

**Fragility:** A crash at case 40/75 during insertion or query 100/225 during evaluation loses ALL progress. Must restart from scratch.

**Speed:** Phases take 15-60 min each. Sequential = 90+ min total.

---

## 2. Solution: Parallel Containers + Checkpoint/Resume

### 2a. Separate Neo4j Containers

Each phase gets its own Neo4j container with a dedicated port:

| Phase | Container | Port |
|-------|----------|------|
| controlled | neo4j-controlled | 7687 |
| pipeline | neo4j-pipeline | 7688 |
| pipeline_presplit | neo4j-presplit | 7689 |

### 2b. Checkpoint/Resume

Each phase has 3 independent stages. Each stage checkpoints after every completed unit of work:

```
insert → evaluate → report
```

**Checkpoint files** in `results/checkpoints/`:

```json
// results/checkpoints/controlled_insert.json
{"completed_cases": ["static_001", "static_002", ...], "status": "in_progress"}

// results/checkpoints/controlled_eval.json  
{"completed_queries": [
  {"case_id": "static_001", "query_idx": 0, "strategy": "hybrid", "result": {...}},
  ...
], "status": "in_progress"}
```

**Resume logic:**
- On startup, load checkpoint if exists
- Skip already-completed cases/queries
- Continue from where it stopped
- Max 1 case of lost work on crash (checkpoint written per case/query)

---

## 3. CLI Interface

```bash
# Run a single phase (all stages, with checkpointing)
python -m src.benchmark_runner controlled

# Run a specific stage only
python -m src.benchmark_runner controlled --stage insert
python -m src.benchmark_runner controlled --stage evaluate
python -m src.benchmark_runner controlled --stage report

# Fresh run (wipe checkpoints + graph)
python -m src.benchmark_runner controlled --clean

# Specify Neo4j port (for parallel containers)
python -m src.benchmark_runner controlled --port 7687

# Run all phases in parallel (shell script)
./scripts/run_parallel.sh
```

---

## 4. Changes

### `src/checkpoint.py` (new)

```python
class Checkpoint:
    def __init__(self, phase: str, stage: str):
        self.path = Path(f"results/checkpoints/{phase}_{stage}.json")
    
    def load(self) -> dict:
        """Load checkpoint or return empty state."""
    
    def save(self, state: dict):
        """Write checkpoint atomically (write to .tmp, rename)."""
    
    def clear(self):
        """Delete checkpoint file."""
    
    def is_completed(self, key: str) -> bool:
        """Check if a specific case/query was already done."""
```

### `src/benchmark_runner.py` (modify)

- `create_graphiti(neo4j_port)` — accept port parameter
- `run_phase()` split into `run_insert()`, `run_evaluate()`, `run_report()`
- Each sub-function loads checkpoint, skips completed work, saves after each unit
- `PHASE_PORTS` dict maps phases to default ports
- CLI parser with `--stage`, `--port`, `--clean` flags

### `scripts/run_parallel.sh` (new)

```bash
#!/bin/bash
set -e

# Start containers (skip if already running)
for phase_port in "controlled:7687" "pipeline:7688" "presplit:7689"; do
    IFS=: read phase port <<< "$phase_port"
    name="neo4j-$phase"
    if ! docker ps -q -f name=$name | grep -q .; then
        docker run -d --name $name -p $port:7687 \
            -e NEO4J_AUTH=neo4j/benchmark123 neo4j:5
    fi
done

# Wait for Neo4j readiness
for port in 7687 7688 7689; do
    echo "Waiting for Neo4j on port $port..."
    until docker exec neo4j-$([ $port = 7687 ] && echo controlled || [ $port = 7688 ] && echo pipeline || echo presplit) \
        cypher-shell -u neo4j -p benchmark123 "RETURN 1" 2>/dev/null; do
        sleep 5
    done
done

# Run 3 phases in parallel
PYTHONUNBUFFERED=1 .venv/bin/python -m src.benchmark_runner controlled --port 7687 &
PYTHONUNBUFFERED=1 .venv/bin/python -m src.benchmark_runner pipeline --port 7688 &
PYTHONUNBUFFERED=1 .venv/bin/python -m src.benchmark_runner pipeline_presplit --port 7689 &
wait

echo "All phases complete. Results in results/"
```

---

## 5. Checkpoint Behavior Matrix

| Scenario | What Happens |
|----------|-------------|
| Crash during insertion at case 40 | Resume skips cases 1-39, continues from 40 |
| Crash during evaluation at query 100 | Resume loads 99 saved results, continues from 100 |
| `--clean` flag | Deletes checkpoints + wipes graph, starts fresh |
| Evaluation without prior insertion | Works if graph data still in Neo4j from previous run |
| Insert stage run twice | Second run is a no-op (all cases already checkpointed) |

---

## 6. Resource Requirements

- ~3GB RAM (1GB per Neo4j container)
- 3 concurrent SiliconFlow API sessions (IO-bound, fine within rate limits)
- ~100KB checkpoint files per phase
- Total benchmark time: ~35 min parallel (limited by slowest phase) vs ~90 min sequential

---

## 7. File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/checkpoint.py` | Create | Checkpoint load/save/clear |
| `src/benchmark_runner.py` | Modify | Split stages, add port/checkpoint/CLI |
| `scripts/run_parallel.sh` | Create | Docker orchestration + parallel launch |
| `results/checkpoints/` | Created at runtime | Checkpoint JSON files |

---

## 8. Backward Compatibility

- Default port remains 7687. Single-phase `python -m src.benchmark_runner controlled` works unchanged
- No `--stage` flag = runs all stages sequentially (current behavior + checkpointing)
- No `--port` flag = uses default 7687
- Checkpointing is always on — no opt-out needed, it's transparent
