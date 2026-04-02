# Functional Gates — Integration Test Scenarios

> **These are HARD GATES.** Each scenario is binary pass/fail.
> Evidence must be captured (command output, file contents, or screenshots).

---

## Scenario 1: Spike Test — Neo4j Round-Trip

**What:** Validate that the infrastructure works end-to-end: Neo4j connection, `add_triplet()`, and `search()` produce a round-trip result.

**Steps:**
1. Connect to Neo4j using credentials from `.env`
2. Create a Graphiti client and build indices
3. Insert one triplet via `add_triplet()`
4. Search for the inserted fact
5. Verify the search returns the inserted fact

**Pass criteria:**
- [ ] Neo4j connection succeeds (no `ServiceUnavailable` or auth errors)
- [ ] `add_triplet()` completes without exception
- [ ] `search()` returns at least 1 result containing the inserted fact
- [ ] Spike script exits with code 0

**Evidence:** Terminal output of spike script showing all steps completed.

---

## Scenario 2: Unit Tests Pass

**What:** All unit tests pass in a clean run.

**Steps:**
1. Run `pytest tests/ -v`

**Pass criteria:**
- [ ] Exit code 0
- [ ] All test files discovered: `test_evaluator.py`, `test_judge.py`, `test_models.py`, and any others
- [ ] Zero failures, zero errors
- [ ] No skipped tests (unless explicitly documented with reason)

**Evidence:** Full `pytest` output showing all tests green.

---

## Scenario 3: Data Generator — 55 Cases, 5 Categories

**What:** The data generator produces the correct test dataset.

**Steps:**
1. Run the data generator (or verify `data/test_cases.json` exists)
2. Count total test cases
3. Count cases per category

**Pass criteria:**
- [ ] `data/test_cases.json` exists and is valid JSON
- [ ] Total test cases == 55
- [ ] Category distribution matches spec:
  - `static_fact`: 10
  - `evolving_fact`: 15
  - `multi_entity_evolution`: 10
  - `contradiction_resolution`: 10
  - `entity_disambiguation`: 10
- [ ] Every test case has: `id`, `category`, `episodes` (non-empty), `queries` (non-empty)
- [ ] Every evolving/contradiction case has `expected_not` in at least one query
- [ ] Every case used for controlled insertion (Phase 1) has `triplets`

**Evidence:** `jq` output showing counts per category and schema validation.

---

## Scenario 4: Controlled Insertion (Phase 1) Completes

**What:** Phase 1 inserts all 55 test cases' triplets into Neo4j without errors.

**Steps:**
1. Wipe Neo4j graph
2. Rebuild indices via `build_indices_and_constraints()`
3. Run controlled inserter on all 55 test cases
4. Verify node/edge counts in Neo4j

**Pass criteria:**
- [ ] Graph wipe succeeds
- [ ] All 55 test cases processed (no exceptions, no skipped cases)
- [ ] Neo4j contains nodes and edges (count > 0)
- [ ] Inserter logs or returns a summary showing all cases inserted

**Evidence:** Inserter output showing 55/55 cases processed + Neo4j node/edge count query.

---

## Scenario 5: Pipeline Insertion (Phase 2) Completes

**What:** Phase 2 inserts all episodes via `add_episode()` without errors.

**Steps:**
1. Wipe Neo4j graph
2. Rebuild indices
3. Run pipeline inserter on all 55 test cases, inserting episodes in chronological order
4. Verify graph is populated

**Pass criteria:**
- [ ] Graph wipe succeeds
- [ ] All 55 test cases processed (no exceptions, no skipped cases)
- [ ] Episodes inserted in correct chronological order per test case
- [ ] Neo4j contains nodes and edges (count > 0)

**Evidence:** Inserter output showing 55/55 cases processed + Neo4j node/edge count query.

---

## Scenario 6: All Search Strategies Return Results

**What:** Each of the 3 search strategies (`hybrid`, `bm25_only`, `cosine_only`) returns results for the vast majority of queries.

**Steps:**
1. After insertion (Phase 1 or Phase 2), run all queries through all 3 strategies
2. Count how many queries return at least 1 result per strategy

**Pass criteria:**
- [ ] `hybrid`: returns >= 1 result for >= 90% of queries
- [ ] `bm25_only`: returns >= 1 result for >= 90% of queries
- [ ] `cosine_only`: returns >= 1 result for >= 90% of queries
- [ ] `SearchFilters` with `DateFilter` are applied to respect temporal validity (per plan: Graphiti does NOT auto-filter expired edges)

**Evidence:** Query-level results JSON showing result counts per strategy.

---

## Scenario 7: Results JSON Is Valid and Complete

**What:** The benchmark produces a structured results file with all expected fields.

**Steps:**
1. Run the full benchmark (or inspect `results/` output after a run)
2. Validate the JSON/CSV structure

**Pass criteria:**
- [ ] Results file exists in `results/` directory
- [ ] Each result entry contains: `phase`, `category`, `strategy`, `query_id`, `returned_facts`, `scores`
- [ ] Scores include: `precision_at_5`, `recall_at_5`, `mrr`, `temporal_accuracy`
- [ ] All combinations present: 2 phases × 5 categories × 3 strategies = 30 aggregate rows
- [ ] No `null` or `NaN` values in metric fields

**Evidence:** `jq` or Python validation showing schema compliance and row count.

---

## Scenario 8: Report Table Renders Correctly

**What:** The final report is human-readable and contains all categories and strategies.

**Steps:**
1. Run the reporter on benchmark results
2. Inspect the output table

**Pass criteria:**
- [ ] Report contains tables for both Phase 1 (controlled) and Phase 2 (pipeline)
- [ ] Each table has rows for all 5 categories
- [ ] Each category row shows all 3 strategies
- [ ] Columns include: Strategy, Precision@5, Recall@5, MRR, Temporal Accuracy
- [ ] A summary section shows hybrid-vs-baseline deltas per category
- [ ] Numbers are formatted consistently (2 decimal places for ratios, percentage for temporal accuracy)

**Evidence:** Full report output captured in terminal or saved to `results/`.

---

## Gate Summary Checklist

```
[ ] S1: Spike test — Neo4j connection + add_triplet + search round-trip
[ ] S2: Unit tests — pytest tests/ -v, all green
[ ] S3: Data generator — 55 cases, 5 categories, correct schema
[ ] S4: Controlled insertion (Phase 1) — 55/55 cases, no errors
[ ] S5: Pipeline insertion (Phase 2) — 55/55 cases, no errors
[ ] S6: Search strategies — all 3 return results for >= 90% of queries
[ ] S7: Results JSON — valid, complete, all 30 aggregate rows present
[ ] S8: Report table — renders with all categories, strategies, and metrics
```

**The benchmark is DONE only when every scenario passes.**
