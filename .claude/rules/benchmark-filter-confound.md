# Don't Apply the Same Filters to All Benchmark Strategies

## The Trap
Applying identical search filters (e.g., temporal DateFilter) to ALL strategies including baselines. This eliminates the independent variable — every strategy gets the same pre-filtered data, making them score identically. The benchmark can't measure whether the feature (temporal graph) adds value.

## The Solution
Only apply the feature-under-test to the strategy that represents it. Baselines must run without the feature so the gap IS the measurement. For this benchmark: temporal filter on `hybrid` only, baselines run raw.

## Context
- **When this applies:** Any A/B benchmark comparing strategies with vs without a feature
- **Related files:** `src/benchmark_runner.py`
- **Discovered:** 2026-04-03, quality reviewer caught that all strategies scored identically (R@5=1.0, TempAcc=100%)
