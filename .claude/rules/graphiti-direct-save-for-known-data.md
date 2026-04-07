# Graphiti: Use Direct Save for Known-Good Data

## The Trap
Using `graphiti.add_triplet()` for controlled insertion of known-good data. `add_triplet` calls `resolve_extracted_nodes` which makes multiple LLM calls for entity deduplication — takes 159s+ per triplet through SiliconFlow/Qwen. Similarly, `add_episode` hangs after the first case when entity resolution kicks in.

## The Solution
For Phase 1 (controlled insertion with known-good data), bypass Graphiti's resolution layer entirely:
```python
await node.generate_name_embedding(embedder)
await node.save(driver)
await edge.generate_embedding(embedder)
await edge.save(driver)
```
Only use `add_triplet`/`add_episode` when you actually need Graphiti's LLM entity resolution (Phase 2).

## Context
- **When this applies:** Any Graphiti benchmark insertion with clean/known data
- **Related files:** `src/controlled_inserter.py`
- **Discovered:** 2026-04-02, benchmark hung at case 2/55 for 159s per triplet
