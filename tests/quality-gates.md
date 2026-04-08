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
| Q4 | What is the temporal accuracy across categories? | Temporal Accuracy column in the report for all 7 categories |
| Q5 | Does pre-split preprocessing improve temporal accuracy on compound and noisy input? | `pipeline` vs `pipeline_presplit` delta for `evolving_compound` and `noisy_fact` categories |

**Pass criteria:** All five questions have numeric answers in the final report. No "N/A" or missing cells.

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

**Evidence:** MRR column for `hybrid` across all 7 categories.

### Gate M5: Pre-Split Advantage on Compound Facts

| Metric | Threshold | Applies To |
|--------|-----------|------------|
| Temporal Accuracy delta | `pipeline_presplit` beats `pipeline` by >= 20 percentage points | `evolving_compound` category |

**Why:** The core hypothesis for the pre-split phase is that decomposing compound sentences into atomic facts fixes Graphiti's contradiction detection failure. A 20pp gap is the minimum signal to distinguish pre-split advantage from noise.

**Evidence:** Temporal Accuracy for `evolving_compound`: `pipeline_presplit` minus `pipeline` >= 0.20.

### Gate M6: Pre-Split Handles Noisy Input

| Metric | Threshold | Applies To |
|--------|-----------|------------|
| Recall@5 | >= 0.60 | `pipeline_presplit` strategy on `noisy_fact` category |

**Why:** The pre-split normalizes typos and abbreviations before calling `add_episode`. R@5 >= 0.60 on noisy input proves the normalization is effective (raw `pipeline` is expected to be significantly lower on this category).

**Evidence:** Recall@5 for `pipeline_presplit` on `noisy_fact` >= 0.60.

---

## 3. No-Workaround Policy

These are anti-patterns that **automatically fail** the quality gate review:

| Anti-Pattern | Why It's Banned |
|--------------|-----------------|
| Hardcoded expected results in test logic | Defeats the purpose of measuring — the benchmark must discover results, not assert predetermined ones |
| Skipped categories (e.g., excluding `entity_disambiguation` from scoring) | All 7 categories must be evaluated; selective reporting hides weaknesses |
| Mock data in benchmark runs | Benchmark must run against a real Neo4j instance with real Graphiti API calls |
| Hardcoded LLM judge responses | The judge must actually call the LLM; cached/mocked responses are only allowed in unit tests |
| Post-hoc result filtering (removing "bad" queries from the dataset) | All 75 test cases must be scored; no cherry-picking |
| Ignoring `expected_not` in temporal accuracy | Temporal accuracy requires both positive AND negative checks |

---

## 4. Code Quality Gates

### Gate C1: Unit Test Coverage

| Requirement | Pass Criteria |
|-------------|---------------|
| `src/evaluator.py` has unit tests | `tests/test_evaluator.py` exists, tests precision/recall/MRR/temporal accuracy with known inputs |
| `src/judge.py` has unit tests | `tests/test_judge.py` exists, tests fact matching with known positive/negative pairs |
| `src/models.py` has unit tests | `tests/test_models.py` exists, tests serialization/deserialization including `tags` field |
| `src/data_generator.py` has unit tests | Tests verify 75 cases generated, correct category distribution across all 7 categories |
| `src/search_strategies.py` has unit tests | Tests verify SearchConfig construction for all 3 strategies |
| `src/reporter.py` has unit tests | Tests verify table formatting with sample data |
| `src/sentence_splitter.py` has unit tests | `tests/test_sentence_splitter.py` exists, tests: (1) simple pass-through returns 1 fact, (2) compound sentence returns 2+ facts, (3) typo normalization produces clean output |
| `src/presplit_inserter.py` has unit tests | Tests verify episode decomposition — each atomic fact produces its own `add_episode` call with the same `reference_time` |
| All unit tests pass | `pytest tests/ -v` exits with code 0 |

**Exceptions:** `controlled_inserter.py`, `pipeline_inserter.py`, and `presplit_inserter.py` require Neo4j for integration — they are also covered by functional gates (S4, S5, S9). The unit tests for `presplit_inserter.py` mock the Graphiti client.

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
| `presplit_inserter.py` | Pre-split + Phase 3 insertion only — delegates decomposition to `sentence_splitter.py` |
| `sentence_splitter.py` | Atomic fact decomposition only — no Graphiti calls |
| `reporter.py` | Formatting and display only — no computation |
| `benchmark_runner.py` | Orchestration only — delegates to all above |

**Evidence:** Code review agent confirms each module stays within its responsibility boundary.

### Gate C4: Modular Test Data

| Requirement | Pass Criteria |
|-------------|---------------|
| Each category lives in its own file | `src/test_data/` contains one file per category (e.g., `static_fact.py`, `evolving_compound.py`, `noisy_fact.py`) |
| Combiner uses registry pattern | A central registry (e.g., `src/test_data/__init__.py` or `data_generator.py`) imports and combines categories without hardcoding per-category logic inline |
| Adding a new category requires only: | (1) a new file in `src/test_data/`, (2) one registry entry — no changes to combiner logic |

**Evidence:** Code review confirms the registry pattern is in place and no combiner code duplicates per-category handling.

---

## 5. Ingestion Quality Gates

### Gate I1: Controlled Phase Entity Recall

| Metric | Threshold | Applies To |
|--------|-----------|------------|
| Entity Recall | >= 0.95 | `controlled` phase, all categories |

**Why:** Controlled insertion uses known-good triplets — entity extraction should be near-perfect.

### Gate I2: Pipeline Phase Entity Recall

| Metric | Threshold | Applies To |
|--------|-----------|------------|
| Entity Recall | >= 0.80 | `pipeline` phase, all categories |

**Why:** LLM extraction is lossy. 0.80 is a realistic floor for entity extraction from natural text.

### Gate I3: Edge Recall and Temporal Invalidation

| Metric | Threshold | Applies To |
|--------|-----------|------------|
| Edge Recall | >= 0.75 | `pipeline` phase |
| Temporal Invalidation Accuracy | >= 0.70 | `evolving_fact` + `contradiction_resolution` categories |

**Why:** Relationships are harder to extract than entities. Temporal invalidation is Graphiti's core differentiator — mirrors M3 threshold.

---

## 6. Search Tuning Gates

### Gate T1: Parametric Variation Produces Measurable Differences

| Metric | Threshold | Applies To |
|--------|-----------|------------|
| Score difference between runs | > 0.05 on at least one metric | Any two runs with different params |

**Why:** If parameter changes produce no score difference, the tuning framework is broken or the params don't matter.

### Gate T2: Reproducibility

| Metric | Threshold | Applies To |
|--------|-----------|------------|
| Same params, same results | Within floating-point tolerance | Repeated runs |

### Gate T3: No Confounds

| Requirement | Pass Criteria |
|-------------|---------------|
| Temporal filter only on hybrid | Baselines run raw, no pre-filtering |
| No data leakage | Tuning params not derived from test data |

---

## 7. Framework Modularity Gates

### Gate F1: Backward Compatibility

| Requirement | Pass Criteria |
|-------------|---------------|
| Existing CLI unchanged | `python -m src.benchmark_runner pipeline --stage report` produces identical output |

### Gate F2: Experiment Registration

| Requirement | Pass Criteria |
|-------------|---------------|
| `--list-experiments` | Shows all 3 experiments: retrieval, ingestion, search_tuning |

### Gate F3: Graph Reuse

| Requirement | Pass Criteria |
|-------------|---------------|
| Shared insertion | Multiple experiments on same phase skip duplicate insertion |

### Gate F4: Judge Cache

| Requirement | Pass Criteria |
|-------------|---------------|
| Persistent cache | `results/judge_cache/judgments.json` survives across runs |

### Gate F5: Run Isolation

| Requirement | Pass Criteria |
|-------------|---------------|
| Per-run checkpoints | Each run gets isolated checkpoint file, no overwrites |

---

## 8. Gate Summary Checklist

```
Retrieval Quality:
[ ] M1: Static fact recall@5 >= 0.80 for ALL strategies
[ ] M2: Hybrid recall@5 beats both baselines by >= 10pp on evolving_fact
[ ] M3: Hybrid temporal accuracy >= 70% on evolving_fact + contradiction_resolution
[ ] M4: Hybrid MRR >= 0.50 averaged across all categories
[ ] M5: pipeline_presplit temporal accuracy beats pipeline by >= 20pp on evolving_compound
[ ] M6: pipeline_presplit recall@5 >= 0.60 on noisy_fact

Ingestion Quality:
[ ] I1: Controlled phase entity_recall >= 0.95
[ ] I2: Pipeline phase entity_recall >= 0.80
[ ] I3: Edge recall >= 0.75, temporal_invalidation_accuracy >= 0.70

Search Tuning:
[ ] T1: Different params produce measurable score difference (> 0.05)
[ ] T2: Same params reproduce within tolerance
[ ] T3: No confounds (temporal filter only on hybrid)

Framework:
[ ] F1: Backward compat — existing CLI unchanged
[ ] F2: --list-experiments shows all 3 experiments
[ ] F3: Graph reuse — shared insertion across experiments
[ ] F4: Judge cache persistent across runs
[ ] F5: Run isolation — per-run checkpoints

Quality:
[ ] Q1-Q5: All five benchmark questions answered with numeric evidence
[ ] NW: No workarounds
[ ] C1: Unit tests pass for all modules
[ ] C2: No dead code
[ ] C3: Clear separation of concerns
[ ] C4: Test data modular — registry pattern
```

**The benchmark is DONE only when every box is checked.**
