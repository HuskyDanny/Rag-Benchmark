# Graphiti Temporal Retrieval Benchmark — Design Spec

**Date:** 2026-04-01
**Goal:** Benchmark whether Graphiti's temporally-aware knowledge graph improves retrieval recall/precision over flat BM25 and embedding-only search, especially for evolving facts.

---

## 1. Problem Statement

LLM agents need persistent memory that handles contradictions (e.g., "Alice works at Google" → "Alice works at Meta"). Traditional vector stores treat each chunk independently. Graphiti builds a temporal knowledge graph that invalidates old facts when new ones arrive.

**Core question:** Does this temporal graph actually improve retrieval quality compared to simpler approaches (BM25-only, cosine-only)?

**Secondary question:** How does end-to-end quality (LLM extraction + retrieval) compare to controlled insertion (known-good data + retrieval)?

---

## 2. Test Data Schema

Each test case is a JSON object:

```json
{
  "id": "evolving_001",
  "category": "evolving_fact",
  "episodes": [
    {
      "text": "Amy joined Google as a software engineer.",
      "reference_time": "2024-01-15T00:00:00Z",
      "order": 1
    },
    {
      "text": "Amy left Google and started working at Facebook.",
      "reference_time": "2024-06-01T00:00:00Z",
      "order": 2
    }
  ],
  "queries": [
    {
      "query": "Where does Amy work?",
      "expected_facts": ["Amy works at Facebook"],
      "expected_not": ["Amy works at Google"],
      "query_time": "2024-07-01T00:00:00Z"
    },
    {
      "query": "Where did Amy work before Facebook?",
      "expected_facts": ["Amy worked at Google"],
      "expected_not": [],
      "query_time": "2024-07-01T00:00:00Z"
    }
  ]
}
```

### For controlled insertion (Phase 1), each test case also includes:

```json
{
  "triplets": [
    {
      "source": {"name": "Amy", "labels": ["Person"]},
      "target": {"name": "Google", "labels": ["Organization"]},
      "edge": {
        "name": "WORKS_AT",
        "fact": "Amy works at Google as a software engineer",
        "valid_at": "2024-01-15T00:00:00Z",
        "invalid_at": "2024-06-01T00:00:00Z"
      }
    },
    {
      "source": {"name": "Amy", "labels": ["Person"]},
      "target": {"name": "Facebook", "labels": ["Organization"]},
      "edge": {
        "name": "WORKS_AT",
        "fact": "Amy works at Facebook",
        "valid_at": "2024-06-01T00:00:00Z",
        "invalid_at": null
      }
    }
  ]
}
```

---

## 3. Test Categories

| Category | Count | What It Tests |
|----------|-------|---------------|
| `static_fact` | 10 | Baseline retrieval — facts that don't change |
| `evolving_fact` | 15 | Temporal updates — same entity, new state overwrites old |
| `multi_entity_evolution` | 10 | Multiple entities evolving simultaneously |
| `contradiction_resolution` | 10 | Conflicting info from different sources, recency wins |
| `entity_disambiguation` | 10 | Same name, different entities (e.g., two "John Smith"s) |
| **Total** | **55** | |

### Category Details

**static_fact (10):** Simple entity-relationship pairs that remain constant. Tests that the system can retrieve basic facts reliably across all strategies.
- Examples: "Paris is the capital of France", "Python was created by Guido van Rossum"
- Expected: All strategies should score well. This is the baseline.

**evolving_fact (15):** The core test. An entity's relationship changes over time. Episodes are inserted chronologically. Queries ask for the *current* state.
- Examples: Employment changes, relationship status, role changes, address moves
- Expected: Hybrid/graph should excel. BM25/cosine may return stale facts.

**multi_entity_evolution (10):** Multiple entities evolve in the same timeline. Tests that the graph handles concurrent temporal updates without cross-contamination.
- Examples: "Bob was CEO, then Jane became CEO" + "Alice moved from NY to SF"
- Expected: Graph should correctly scope temporal invalidation per entity pair.

**contradiction_resolution (10):** Two episodes provide conflicting information about the same fact. The later one should win.
- Examples: Source A says "revenue is $10M", Source B (later) says "revenue is $12M"
- Expected: Graph with temporal ordering should return the newer fact.

**entity_disambiguation (10):** Entities with identical or similar names but different contexts. Tests that the graph correctly separates them.
- Examples: "John Smith the engineer at Google" vs "John Smith the doctor at Mayo Clinic"
- Expected: Graph structure (different edges) should help disambiguate. Flat search may conflate.

---

## 4. Benchmark Architecture

### Phase 1: Controlled Insertion

1. Wipe Neo4j graph
2. For each test case: insert triplets via `add_triplet()` with exact temporal bounds
3. For each query in each test case: run 3 search strategies
4. Score results against ground truth

### Phase 2: Full Pipeline

1. Wipe Neo4j graph
2. For each test case: insert episodes via `add_episode()` in chronological order
3. For each query in each test case: run 3 search strategies
4. Score results against ground truth

### Search Strategies

| Strategy | Graphiti Config | What It Tests |
|----------|----------------|---------------|
| `hybrid` | `EDGE_HYBRID_SEARCH_RRF` (BM25 + cosine + RRF) | Full Graphiti retrieval power |
| `bm25_only` | Custom config: `EdgeSearchMethod.bm25` only | Keyword-only baseline |
| `cosine_only` | Custom config: `EdgeSearchMethod.cosine_similarity` only | Embedding-only baseline |

All strategies use `limit=10` (retrieve top 10 edges).

### Graph Wipe Between Phases

```python
# Clear all nodes and relationships
async with driver.session() as session:
    await session.run("MATCH (n) DETACH DELETE n")
```

Rebuild indices after wipe: `await graphiti.build_indices_and_constraints()`

---

## 5. Evaluation Metrics

### Per-Query Metrics

- **Precision@5** — Of the top 5 returned edges, how many match `expected_facts`?
- **Recall@5** — Of all `expected_facts`, how many appear in the top 5 results?
- **MRR** — 1/rank of the first correct result (0 if no correct result in top 10)
- **Temporal Accuracy** — Binary pass/fail: does the system return current facts AND exclude `expected_not` facts from top 5?

### Matching Logic

String matching is too brittle ("Amy works at Facebook" vs "Amy is employed by Facebook"). Use LLM-as-judge:

```python
async def facts_match(returned_fact: str, expected_fact: str) -> bool:
    """Ask GPT-4o-mini if two facts express the same information."""
    prompt = f"""Do these two statements express the same factual claim?
    Statement A: {returned_fact}
    Statement B: {expected_fact}
    Answer YES or NO only."""
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3
    )
    return "YES" in response.choices[0].message.content.upper()
```

### Aggregated Report

Results grouped by `(phase, category, strategy)`:

```
Phase: controlled
Category: evolving_fact (15 cases, 30 queries)
┌──────────────┬──────────┬──────────┬───────┬──────────────┐
│ Strategy     │ P@5      │ R@5      │ MRR   │ Temporal Acc  │
├──────────────┼──────────┼──────────┼───────┼──────────────┤
│ hybrid       │ 0.XX     │ 0.XX     │ 0.XX  │ XX%          │
│ bm25_only    │ 0.XX     │ 0.XX     │ 0.XX  │ XX%          │
│ cosine_only  │ 0.XX     │ 0.XX     │ 0.XX  │ XX%          │
└──────────────┴──────────┴──────────┴───────┴──────────────┘
```

Final summary: delta between hybrid and baselines, per category. This directly answers "is the temporal graph worth it?"

---

## 6. Project Structure

```
experiemnt_zep_rag/
├── pyproject.toml
├── .env                         # NEO4J_URI, OPENAI_API_KEY
├── data/
│   └── test_cases.json          # 55 test cases (generated)
├── src/
│   ├── __init__.py
│   ├── models.py                # Pydantic models for test cases, results
│   ├── data_generator.py        # Generates test_cases.json programmatically
│   ├── benchmark_runner.py      # Orchestrates Phase 1 + Phase 2 + reporting
│   ├── controlled_inserter.py   # Phase 1: add_triplet with known-good data
│   ├── pipeline_inserter.py     # Phase 2: add_episode with raw text
│   ├── search_strategies.py     # Configures hybrid/bm25/cosine SearchConfigs
│   ├── evaluator.py             # Precision/recall/MRR/temporal scoring
│   ├── judge.py                 # LLM-as-judge fact matching
│   └── reporter.py              # Aggregates scores, prints tables
├── tests/
│   ├── test_evaluator.py        # Unit tests for scoring logic
│   └── test_judge.py            # Unit tests for LLM matching
└── results/
    └── (generated benchmark CSVs and summary)
```

---

## 7. Dependencies

```toml
[project]
name = "graphiti-temporal-benchmark"
requires-python = ">=3.11"
dependencies = [
    "graphiti-core[neo4j]",
    "openai>=1.0.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "tabulate>=0.9.0",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]
```

---

## 8. Agent Team Responsibilities

| Agent | Role | Deliverables |
|-------|------|-------------|
| **Validation Intern** | Verify Neo4j connection, Graphiti `add_triplet` + `search` round-trip works | Spike script proving API works |
| **Backend Agent** | Build all `src/` modules, generate test data, unit tests | All source files + `test_cases.json` |
| **Functional Agent** | Run the full benchmark end-to-end | Raw results in `results/` |
| **Quality Agent** | Analyze results, produce final verdict | Summary report with recommendations |

---

## 9. Success Criteria

The benchmark is successful if it produces a clear, quantitative answer to:

1. **Does Graphiti hybrid search beat BM25-only and cosine-only on static facts?** (Expected: marginally)
2. **Does Graphiti hybrid search beat baselines on evolving facts?** (Expected: significantly, due to temporal invalidation)
3. **How much does LLM extraction quality affect retrieval?** (Compare Phase 1 vs Phase 2 scores)
4. **What is the temporal accuracy across categories?** (The key metric for evolving knowledge)

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Neo4j not running | Validate connection before benchmark starts |
| LLM extraction produces unexpected entities | Phase 1 (controlled) isolates this; Phase 2 measures it |
| LLM-as-judge is unreliable | Unit test the judge with known positive/negative pairs |
| `add_triplet` doesn't set temporal bounds correctly | Validation Intern spike tests this explicitly |
| BM25 returns expired edges (Graphiti may not filter by temporal validity in search) | Check if Graphiti search respects `invalid_at`; if not, add post-filter |
