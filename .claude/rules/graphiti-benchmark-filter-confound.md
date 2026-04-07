# Graphiti Benchmark: Don't Apply Temporal Filter to Baseline Strategies

## The Trap
Applying `SearchFilters(invalid_at=IS_NULL OR > query_time)` to ALL search strategies (hybrid, bm25_only, cosine_only) in the benchmark. This eliminates the independent variable — all strategies pre-filter stale edges identically, so the benchmark cannot measure whether Graphiti's temporal graph adds value over flat retrieval.

## The Solution
Only apply temporal SearchFilters to the `hybrid` strategy (or a dedicated "graphiti_filtered" strategy). Run `bm25_only` and `cosine_only` without filters to simulate what a naive flat vector store would return. This creates the real comparison:

```python
# WRONG — all strategies get the same temporal filter
search_filter = build_temporal_filter(query_time)  # applied to all 3

# CORRECT — baselines run raw, hybrid uses filter
if strategy_name == "hybrid":
    search_filter = build_temporal_filter(query_time)
else:
    search_filter = None  # baselines must see stale facts to measure the gap
```

## Context
- **When this applies:** Any benchmark or evaluation that compares Graphiti strategies
- **Related files:** `src/benchmark_runner.py:run_search()`, `src/benchmark_runner.py:build_temporal_filter()`
- **Discovered:** 2026-04-03, quality review — Phase 1 showed R@5=1.0 for ALL strategies including baselines, 0pp delta, making M2 unanswerable
