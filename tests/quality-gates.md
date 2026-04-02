# Quality Gates — Graphiti Temporal Retrieval Benchmark

> **These are HARD GATES.** The swarm cannot claim "done" until every gate passes.
> Each gate is binary pass/fail with explicit evidence requirements.

---

## 1. Benchmark Goals (What Must Be Answered)

The benchmark must produce a clear, quantitative answer to each of these questions:

| # | Question | Evidence Required |
|---|----------|-------------------|
| Q1 | Does Graphiti hybrid search beat BM25-only and cosine-only on **static facts**? | Recall@5 comparison table for `static_fact` category |
| Q2 | Does Graphiti hybrid search beat baselines on **evolving facts**? | Recall@5 + Temporal Accuracy for `evolving_fact` and `multi_entity_evolution` |
| Q3 | How much does LLM extraction quality affect retrieval? | Phase 1 vs Phase 2 delta per category |
| Q4 | What is the temporal accuracy across categories? | Temporal Accuracy column in the report for all 5 categories |

**Pass criteria:** All four questions have numeric answers in the final report. No "N/A" or missing cells.

---

## 2. Minimum Metrics Thresholds

### Gate M1: Static Fact Baseline

| Metric | Threshold | Applies To |
|--------|-----------|------------|
| Recall@5 | >= 0.80 | ALL strategies (`hybrid`, `bm25_only`, `cosine_only`) on `static_fact` category |

**Why:** Static facts are the easiest case — no temporal complexity. If any strategy can't hit 0.80 recall on unchanging facts, the benchmark infrastructure is broken.

**Evidence:** `results/` output showing Recall@5 >= 0.80 for all 3 strategies on `static_fact`.

### Gate M2: Hybrid Advantage on Evolving Facts

| Metric | Threshold | Applies To |
|--------|-----------|------------|
| Recall@5 delta | hybrid beats both `bm25_only` AND `cosine_only` by >= 10 percentage points | `evolving_fact` category |

**Why:** This is the core hypothesis. If hybrid doesn't meaningfully outperform flat strategies on evolving facts, the temporal graph isn't adding value.

**Evidence:** Recall@5 for `evolving_fact`: `hybrid` minus `bm25_only` >= 0.10, AND `hybrid` minus `cosine_only` >= 0.10.

### Gate M3: Temporal Accuracy

| Metric | Threshold | Applies To |
|--------|-----------|------------|
| Temporal Accuracy | >= 70% | `hybrid` strategy on `evolving_fact` AND `contradiction_resolution` categories |

**Why:** Temporal accuracy measures whether the system returns current facts AND excludes stale/invalidated facts. 70% is the minimum for the graph to demonstrate temporal awareness.

**Evidence:** Temporal Accuracy column in the report for `hybrid` on both categories.

### Gate M4: MRR Floor

| Metric | Threshold | Applies To |
|--------|-----------|------------|
| MRR | >= 0.50 | `hybrid` strategy, averaged across ALL categories |

**Why:** MRR >= 0.50 means the first correct result is, on average, in the top 2 positions. Below this, retrieval quality is too noisy for practical use.

**Evidence:** MRR column for `hybrid` across all 5 categories.

---

## 3. No-Workaround Policy

These are anti-patterns that **automatically fail** the quality gate review:

| Anti-Pattern | Why It's Banned |
|--------------|-----------------|
| Hardcoded expected results in test logic | Defeats the purpose of measuring — the benchmark must discover results, not assert predetermined ones |
| Skipped categories (e.g., excluding `entity_disambiguation` from scoring) | All 5 categories must be evaluated; selective reporting hides weaknesses |
| Mock data in benchmark runs | Benchmark must run against a real Neo4j instance with real Graphiti API calls |
| Hardcoded LLM judge responses | The judge must actually call the LLM; cached/mocked responses are only allowed in unit tests |
| Post-hoc result filtering (removing "bad" queries from the dataset) | All 55 test cases must be scored; no cherry-picking |
| Ignoring `expected_not` in temporal accuracy | Temporal accuracy requires both positive AND negative checks |

---

## 4. Code Quality Gates

### Gate C1: Unit Test Coverage

| Requirement | Pass Criteria |
|-------------|---------------|
| `src/evaluator.py` has unit tests | `tests/test_evaluator.py` exists, tests precision/recall/MRR/temporal accuracy with known inputs |
| `src/judge.py` has unit tests | `tests/test_judge.py` exists, tests fact matching with known positive/negative pairs |
| `src/models.py` has unit tests | `tests/test_models.py` exists, tests serialization/deserialization |
| `src/data_generator.py` has unit tests | Tests verify 55 cases generated, correct category distribution |
| `src/search_strategies.py` has unit tests | Tests verify SearchConfig construction for all 3 strategies |
| `src/reporter.py` has unit tests | Tests verify table formatting with sample data |
| All unit tests pass | `pytest tests/ -v` exits with code 0 |

**Exceptions:** `controlled_inserter.py` and `pipeline_inserter.py` require Neo4j — they are covered by functional gates (integration tests), not unit tests.

### Gate C2: No Dead Code

| Requirement | Pass Criteria |
|-------------|---------------|
| No unused imports | `grep` for unused imports finds none |
| No commented-out code blocks | No `# TODO: uncomment` or large commented sections |
| No placeholder/stub functions | Every function has a real implementation, not `pass` or `raise NotImplementedError` |

### Gate C3: Separation of Concerns

| Module | Single Responsibility |
|--------|-----------------------|
| `models.py` | Data structures only — no business logic |
| `data_generator.py` | Generates test cases only — no insertion or scoring |
| `judge.py` | LLM fact matching only — no metric calculation |
| `evaluator.py` | Metric calculation only — no LLM calls |
| `search_strategies.py` | SearchConfig construction only — no execution |
| `controlled_inserter.py` | Phase 1 insertion only |
| `pipeline_inserter.py` | Phase 2 insertion only |
| `reporter.py` | Formatting and display only — no computation |
| `benchmark_runner.py` | Orchestration only — delegates to all above |

**Evidence:** Code review agent confirms each module stays within its responsibility boundary.

---

## 5. Gate Summary Checklist

```
[ ] M1: Static fact recall@5 >= 0.80 for ALL strategies
[ ] M2: Hybrid recall@5 beats both baselines by >= 10pp on evolving_fact
[ ] M3: Hybrid temporal accuracy >= 70% on evolving_fact + contradiction_resolution
[ ] M4: Hybrid MRR >= 0.50 averaged across all categories
[ ] Q1-Q4: All four benchmark questions answered with numeric evidence
[ ] NW: No workarounds (no hardcoded results, no skipped categories, no mocks in benchmarks)
[ ] C1: Unit tests exist and pass for all modules (except inserters)
[ ] C2: No dead code
[ ] C3: Clear separation of concerns
```

**The benchmark is DONE only when every box is checked.**
