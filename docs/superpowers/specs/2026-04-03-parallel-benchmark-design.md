# Parallel Benchmark via Separate Neo4j Containers — Design Spec

**Date:** 2026-04-03
**Goal:** Run all 3 benchmark phases in true parallel with zero Neo4j contention by using separate Docker containers per phase.

---

## 1. Problem

Neo4j Community Edition doesn't support multiple databases. All phases share a single graph instance, causing:
- Index lock contention during concurrent writes
- Fulltext index rebuild interference
- Group-based isolation (`group_id`) still shares indices

Phases take 15-60 minutes each. Sequential = 90+ minutes total.

## 2. Solution

Each phase gets its own Neo4j container with a dedicated port:

| Phase | Container Name | Port | Group ID |
|-------|---------------|------|----------|
| controlled | neo4j-controlled | 7687 | benchmark_controlled |
| pipeline | neo4j-pipeline | 7688 | benchmark_pipeline |
| pipeline_presplit | neo4j-presplit | 7689 | benchmark_presplit |

## 3. Changes

### `src/benchmark_runner.py`

`create_graphiti()` accepts an optional port override:

```python
async def create_graphiti(neo4j_port: int = 7687) -> Graphiti:
    uri = f"bolt://localhost:{neo4j_port}"
    ...
```

`run_phase()` maps phases to ports:

```python
PHASE_PORTS = {
    "controlled": 7687,
    "pipeline": 7688,
    "pipeline_presplit": 7689,
}
```

### `scripts/run_parallel.sh`

```bash
#!/bin/bash
# Start 3 Neo4j containers
for port in 7687 7688 7689; do
    name="neo4j-$port"
    docker run -d --name $name -p $port:7687 \
        -e NEO4J_AUTH=neo4j/benchmark123 neo4j:5
done

# Wait for all to be ready
# Run 3 phases in parallel
PYTHONUNBUFFERED=1 .venv/bin/python -m src.benchmark_runner controlled &
PYTHONUNBUFFERED=1 .venv/bin/python -m src.benchmark_runner pipeline &
PYTHONUNBUFFERED=1 .venv/bin/python -m src.benchmark_runner pipeline_presplit &
wait

# Cleanup containers
```

### Port allocation

Each container maps its internal 7687 to a unique external port. The Python process connects to `bolt://localhost:{port}`.

## 4. Resource Requirements

- ~3GB RAM (1GB per Neo4j container)
- 3 concurrent SiliconFlow API sessions (IO-bound, fine within rate limits)
- Total benchmark time: ~35 min (limited by slowest phase) instead of ~90 min sequential

## 5. Backward Compatibility

Default port remains 7687. Single-phase runs work unchanged. The parallel script is optional — `python -m src.benchmark_runner controlled` still works standalone.
