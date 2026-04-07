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
- [ ] All test files discovered: `test_evaluator.py`, `test_judge.py`, `test_models.py`, `test_sentence_splitter.py`, and any others
- [ ] Zero failures, zero errors
- [ ] No skipped tests (unless explicitly documented with reason)

**Evidence:** Full `pytest` output showing all tests green.

---

## Scenario 3: Data Generator — 75 Cases, 7 Categories

**What:** The data generator produces the correct test dataset.

**Steps:**
1. Run the data generator (or verify `data/test_cases.json` exists)
2. Count total test cases
3. Count cases per category

**Pass criteria:**
- [ ] `data/test_cases.json` exists and is valid JSON
- [ ] Total test cases == 75
- [ ] Category distribution matches spec:
  - `static_fact`: 10
  - `evolving_fact`: 15
  - `evolving_compound`: 10
  - `multi_entity_evolution`: 10
  - `contradiction_resolution`: 10
  - `entity_disambiguation`: 10
  - `noisy_fact`: 10
- [ ] Every test case has: `id`, `category`, `tags`, `episodes` (non-empty), `queries` (non-empty)
- [ ] Every evolving/contradiction case has `expected_not` in at least one query
- [ ] Every case used for controlled insertion (Phase 1) has `triplets`
- [ ] `evolving_compound` cases have `tags` containing `"compound"` and `"evolving"`
- [ ] `noisy_fact` cases have `tags` containing `"noisy"` and `"evolving"`

**Evidence:** `jq` output showing counts per category and schema validation.

---

## Scenario 4: Controlled Insertion (Phase 1) Completes

**What:** Phase 1 inserts all 75 test cases' triplets into Neo4j without errors.

**Steps:**
1. Wipe Neo4j graph
2. Rebuild indices via `build_indices_and_constraints()`
3. Run controlled inserter on all 75 test cases
4. Verify node/edge counts in Neo4j

**Pass criteria:**
- [ ] Graph wipe succeeds
- [ ] All 75 test cases processed (no exceptions, no skipped cases)
- [ ] Neo4j contains nodes and edges (count > 0)
- [ ] Inserter logs or returns a summary showing all cases inserted

**Evidence:** Inserter output showing 75/75 cases processed + Neo4j node/edge count query.

---

## Scenario 5: Pipeline Insertion (Phase 2) Completes

**What:** Phase 2 inserts all episodes via `add_episode()` without errors.

**Steps:**
1. Wipe Neo4j graph
2. Rebuild indices
3. Run pipeline inserter on all 75 test cases, inserting episodes in chronological order
4. Verify graph is populated

**Pass criteria:**
- [ ] Graph wipe succeeds
- [ ] All 75 test cases processed (no exceptions, no skipped cases)
- [ ] Episodes inserted in correct chronological order per test case
- [ ] Neo4j contains nodes and edges (count > 0)

**Evidence:** Inserter output showing 75/75 cases processed + Neo4j node/edge count query.

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
- [ ] All combinations present: 3 phases × 7 categories × 3 strategies = 63 aggregate rows
- [ ] No `null` or `NaN` values in metric fields

**Evidence:** `jq` or Python validation showing schema compliance and row count.

---

## Scenario 8: Report Table Renders Correctly

**What:** The final report is human-readable and contains all categories and strategies.

**Steps:**
1. Run the reporter on benchmark results
2. Inspect the output table

**Pass criteria:**
- [ ] Report contains tables for all three phases: Phase 1 (controlled), Phase 2 (pipeline), Phase 3 (pipeline_presplit)
- [ ] Each table has rows for all 7 categories
- [ ] Each category row shows all 3 strategies
- [ ] Columns include: Strategy, Precision@5, Recall@5, MRR, Temporal Accuracy
- [ ] A summary section shows hybrid-vs-baseline deltas per category
- [ ] A summary section shows pipeline vs pipeline_presplit delta for `evolving_compound` and `noisy_fact`
- [ ] Numbers are formatted consistently (2 decimal places for ratios, percentage for temporal accuracy)

**Evidence:** Full report output captured in terminal or saved to `results/`.

---

## Scenario 9: Pre-Split Pipeline (Phase 3) Completes

**What:** Phase 3 (`pipeline_presplit`) processes all 75 test cases through the pre-split inserter without errors.

**Steps:**
1. Wipe Neo4j graph
2. Rebuild indices
3. Run `presplit_inserter.py` on all 75 test cases in chronological episode order
4. Verify each episode is decomposed and each atomic fact produces an `add_episode` call
5. Verify graph is populated

**Pass criteria:**
- [ ] Graph wipe succeeds
- [ ] All 75 test cases processed (no exceptions, no skipped cases)
- [ ] Each episode produces at least 1 `add_episode` call
- [ ] Compound episodes (tagged `"compound"`) produce 2+ `add_episode` calls
- [ ] All `add_episode` calls for the same episode share the same `reference_time`
- [ ] Neo4j contains nodes and edges (count > 0)

**Evidence:** Inserter output showing 75/75 cases processed, per-episode decomposition counts, and Neo4j node/edge count query.

---

## Scenario 10: Sentence Splitter Correctly Decomposes Text

**What:** The `split_into_atomic_facts` function produces correct output across all input types.

**Steps:**
1. Run `tests/test_sentence_splitter.py` (unit)
2. Manually verify a compound case from `evolving_compound` is split correctly

**Pass criteria:**
- [ ] Simple sentence returns exactly 1 atomic fact (pass-through)
- [ ] Compound sentence (e.g., "Amy left Google and joined Facebook.") returns 2 atomic facts
- [ ] Noisy input (e.g., "dave quit msft & joined amzn as sr. dev") returns a normalized clean sentence
- [ ] Output facts contain no conjunctions linking independent clauses
- [ ] All unit tests in `test_sentence_splitter.py` pass

**Evidence:** `pytest tests/test_sentence_splitter.py -v` output (unit), plus one manually verified compound-to-atomic split logged to terminal.

---

## Scenario 11: Ingestion Experiment End-to-End

**What:** The ingestion experiment inspects graph state and produces quality scores.

**Steps:**
1. Ensure controlled phase graph is populated
2. Run `python -m src.benchmark_runner controlled -e ingestion`
3. Verify report shows per-category ingestion metrics

**Pass criteria:**
- [ ] Report includes entity_recall, edge_recall, temporal_invalidation_accuracy, dedup_score
- [ ] Controlled phase entity_recall >= 0.95 for static_fact category
- [ ] Step-by-step results produced for annotated evolving cases
- [ ] Results persisted to `results/runs/` directory

**Evidence:** Report output + results directory listing.

---

## Scenario 12: Search Tuning with Parameterized Runs

**What:** Tuning experiment accepts `--param` and produces results.

**Steps:**
1. Run `python -m src.benchmark_runner controlled -e search_tuning --param mmr_lambda=0.3 --run-id tune_mmr03`
2. Run `python -m src.benchmark_runner controlled -e search_tuning --param mmr_lambda=0.7 --run-id tune_mmr07`
3. Verify both runs complete with different scores

**Pass criteria:**
- [ ] Both runs complete without error
- [ ] Results stored in `results/runs/tune_mmr03/` and `results/runs/tune_mmr07/`
- [ ] Each directory contains `metadata.json`, `results.json`, `report.json`

**Evidence:** Directory listing + metadata.json contents.

---

## Scenario 13: Run Comparison with --compare

**What:** `--compare` flag produces side-by-side table of multiple runs.

**Steps:**
1. Complete two search_tuning runs (from S12)
2. Run `python -m src.benchmark_runner --compare tune_mmr03 tune_mmr07`

**Pass criteria:**
- [ ] Comparison table shows both runs with metrics
- [ ] Parameters section shows different mmr_lambda values
- [ ] No errors or missing data

**Evidence:** Comparison table output.

---

## Scenario 14: Multiple Experiments Sharing Insertion

**What:** Running two experiments in one invocation reuses graph population.

**Steps:**
1. Run `python -m src.benchmark_runner controlled -e retrieval -e ingestion`
2. Verify insertion only happens once

**Pass criteria:**
- [ ] INSERT stage runs once (not twice)
- [ ] Both experiment reports produced
- [ ] Results stored separately per experiment

**Evidence:** Console output showing single INSERT + two experiment reports.

---

## Scenario 15: Experiment Registration and CLI Routing

**What:** `--list-experiments` shows all registered experiments.

**Steps:**
1. Run `python -m src.benchmark_runner --list-experiments`

**Pass criteria:**
- [ ] Output lists: retrieval, ingestion, search_tuning
- [ ] Each experiment shows its default params

**Evidence:** CLI output.

---

## Gate Summary Checklist

```
[ ] S1: Spike test — Neo4j connection + add_triplet + search round-trip
[ ] S2: Unit tests — pytest tests/ -v, all green
[ ] S3: Data generator — 75 cases, 7 categories, correct schema + tags
[ ] S4: Controlled insertion (Phase 1) — 75/75 cases, no errors
[ ] S5: Pipeline insertion (Phase 2) — 75/75 cases, no errors
[ ] S6: Search strategies — all 3 return results for >= 90% of queries
[ ] S7: Results JSON — valid, complete, all 63 aggregate rows present
[ ] S8: Report table — renders with all phases, categories, strategies
[ ] S9: Pre-split pipeline (Phase 3) — 75/75 cases
[ ] S10: Sentence splitter — pass-through, compound splits, noisy normalization
[ ] S11: Ingestion experiment — E2E with graph inspection + step validation
[ ] S12: Search tuning — parameterized runs with --param
[ ] S13: Run comparison — --compare produces side-by-side table
[ ] S14: Multi-experiment — shared insertion, separate reports
[ ] S15: Experiment registration — --list-experiments shows all 3
```

**The benchmark is DONE only when every scenario passes.**
