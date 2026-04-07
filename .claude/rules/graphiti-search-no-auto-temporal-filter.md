# Graphiti Search Does Not Auto-Filter Invalidated Edges

## The Trap
Calling `graphiti.search()` or `graphiti.search_()` and assuming results only contain currently-valid facts. Both methods return ALL edges including ones with `invalid_at` set — i.e., facts that have been superseded by newer information.

## The Solution
Always pass an explicit `SearchFilters` to exclude invalidated edges when you want current facts:

```python
from graphiti_core.search.search_filters import SearchFilters, DateFilter, ComparisonOperator

current_only = SearchFilters(
    invalid_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]]
)
results = await graphiti.search_(query, search_filter=current_only)
```

For historical queries (e.g., "what was true at time T"), use `valid_at` range filters instead:

```python
historical = SearchFilters(
    valid_at=[[
        DateFilter(date=point_in_time, comparison_operator=ComparisonOperator.less_than_equal)
    ]],
    invalid_at=[[
        DateFilter(date=point_in_time, comparison_operator=ComparisonOperator.greater_than),
        DateFilter(comparison_operator=ComparisonOperator.is_null),  # never invalidated OR invalidated after T
    ]]
)
```

## Context
- **When this applies:** Every search call in the benchmark — temporal precision depends on this
- **Related files:** `graphiti_core/search/search_filters.py`, `benchmark/runner.py`
- **Discovered:** 2026-04-02, during SDK validation — confirmed by reading `search_filters.py` source
