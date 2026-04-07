# Graphiti's LLM Contradiction Detection is Unreliable

## The Trap
Assuming `add_episode` will always set `invalid_at` on old edges when new contradicting info arrives. In practice, when the LLM creates edges with different relationship names (e.g., `WORKS_AT` vs `LEFT_COMPANY`), Graphiti's `resolve_extracted_edges` fails to detect the contradiction. The old edge stays CURRENT.

## The Solution
Don't rely solely on `add_episode` for temporal invalidation. After ingestion, verify the graph state — check for entities with multiple CURRENT edges of the same semantic type. Consider post-processing to detect and fix missed invalidations.

## Context
- **When this applies:** Any Graphiti pipeline ingestion with evolving facts
- **Related files:** `scripts/explore_graphiti.py` (live evidence at lines 226-241)
- **Discovered:** 2026-04-03, "Alice works at Google" stayed CURRENT after "Alice left Google and joined Meta" was ingested
