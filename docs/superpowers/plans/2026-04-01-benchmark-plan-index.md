# Graphiti Temporal Retrieval Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark harness that measures whether Graphiti's temporal knowledge graph improves retrieval recall/precision over flat BM25 and cosine-only search, with 55 test cases across 5 categories.

**Architecture:** Two-phase benchmark (controlled insertion via `add_triplet` + full pipeline via `add_episode`), three search strategies (hybrid/bm25/cosine), LLM-as-judge scoring, temporal accuracy tracking. Results aggregated by category x strategy.

**Tech Stack:** Python 3.11+, graphiti-core[neo4j], OpenAI API, Neo4j, pytest, pytest-asyncio, tabulate

**Key API Discovery:**
- Graphiti does NOT auto-filter `invalid_at`/`expired_at` edges in search — we must apply `SearchFilters` with `DateFilter` explicitly
- Graph wipe: `await graphiti.clients.driver.graph_ops.clear_data(graphiti.clients.driver)`
- `add_triplet(source_node: EntityNode, edge: EntityEdge, target_node: EntityNode)`
- EntityNode requires: `name`, `group_id`
- EntityEdge requires: `group_id`, `source_node_uuid`, `target_node_uuid`, `created_at`, `name`, `fact`

---

## Task Files

- [[benchmark-plan-tasks-1-4]] — Scaffolding, Models, Data Generator, Search Strategies
- [[benchmark-plan-tasks-5-8]] — Judge, Evaluator, Controlled Inserter, Pipeline Inserter
- [[benchmark-plan-tasks-9-12]] — Reporter, Benchmark Runner, Validation Spike, Run Benchmark

## Dependency Graph

```
Task 1 (scaffolding)
  -> Task 2 (models)
       -> Tasks 3, 4, 5, 7, 8, 9 (all independent of each other)
            -> Task 6 (evaluator, depends on Task 5 judge)
                 -> Task 10 (benchmark runner, depends on all above)
                      -> Task 11 (spike, validates infra)
                           -> Task 12 (run benchmark)
```

**Parallelizable after Task 2:** Tasks 3, 4, 5, 7, 8, 9 are independent.
