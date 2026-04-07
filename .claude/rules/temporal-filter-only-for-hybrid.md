# Temporal Filter Should Only Apply to Hybrid Strategy

## The Trap
Applying `SearchFilters` with `DateFilter` to ALL search strategies including baselines. This eliminates the independent variable — baselines never see stale facts, so temporal accuracy is trivially 100% for everyone. The benchmark can't measure whether the temporal graph adds value.

## The Solution
Only apply temporal `SearchFilters` to the `hybrid` strategy (representing Graphiti's temporal awareness). Run `bm25_only` and `cosine_only` without temporal filtering — they represent "flat search" that doesn't know about temporal validity. The gap between filtered-hybrid and unfiltered-baselines IS the signal.

Update `benchmark_runner.py:evaluate_query()` to conditionally apply the filter:
```python
query_time = query.query_time if strategy_name == "hybrid" else None
returned_facts = await run_search(graphiti, query.query, config, group_id, query_time=query_time)
```

## Context
- **When this applies:** Any benchmark run comparing temporal-aware vs flat search
- **Related files:** `src/benchmark_runner.py`
- **Discovered:** 2026-04-03, quality reviewer flagged that uniform filtering makes all strategies score identically
