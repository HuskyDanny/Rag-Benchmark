# add_episode_bulk Skips Edge Invalidation

## The Trap
Using `add_episode_bulk` to ingest benchmark scenarios that involve conflicting/updating facts. The bulk method explicitly skips the edge invalidation step — old contradicted edges keep `invalid_at=None`, meaning temporal queries will return stale facts as if they were current.

## The Solution
Use `add_episode` (sequential, not bulk) for any benchmark scenario that relies on temporal reasoning or fact updates. The non-bulk method runs `resolve_extracted_edges` which sets `invalid_at` on superseded edges.

Only use `add_episode_bulk` for append-only ingestion where no fact ever contradicts a previous one.

## Context
- **When this applies:** Benchmark data ingestion in `scripts/ingest_episodes.py` or equivalent
- **Related files:** `graphiti_core/graphiti.py` line ~1097 — docstring explicitly states this
- **Discovered:** 2026-04-02, during SDK validation
