# LLM Cache Only Works for Stateless Extraction Calls

## The Trap
Assuming prompt-level caching will speed up the entire `add_episode` pipeline. In practice, only `extract_nodes` (text → entities) is cacheable. The resolution steps (`resolve_extracted_nodes`, `resolve_extracted_edges`) include the current graph state (candidate nodes/edges) in their prompts — so the cache key changes every time the graph changes, producing permanent cache misses.

## The Solution
When optimizing `add_episode` performance, distinguish:
- **Stateless calls** (extract_nodes, extract_edges) — input is episode text only → cacheable ✅
- **Stateful calls** (resolve_nodes, resolve_edges) — input includes graph-dependent candidate list → NOT cacheable ❌

The resolution step is the actual bottleneck (O(graph_size) candidates in prompt). To speed it up, reduce candidate count or batch across independent test cases — don't rely on caching.

## Context
- **When this applies:** Any performance optimization for Graphiti's add_episode pipeline
- **Related files:** `src/caching_llm_client.py`, `graphiti_core/utils/maintenance/node_operations.py:_collect_candidate_nodes`
- **Discovered:** 2026-04-09, E2E test showed 1/6 cache hits (0.8x speedup) — resolution calls always miss
