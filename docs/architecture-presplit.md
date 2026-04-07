# Architecture: Pre-Split Preprocessor

**Date:** 2026-04-03  
**Scope:** Updated component map and data flow for `pipeline_presplit` phase.

---

## Component Diagram

```
src/
├── models.py                  # Shared data contracts (TestCase, Episode, Query, ...)
├── data_generator.py          # Orchestrator: GENERATORS registry + save_test_cases()
├── test_data/                 # Category generators (one file per category)
│   ├── common.py              # Shared helpers (_dt, etc.)
│   ├── static_facts.py        # make_static_facts()
│   ├── evolving_facts.py      # make_evolving_facts()
│   ├── compound.py            # make_evolving_compound()   ← new
│   ├── noisy.py               # make_noisy_facts()         ← new
│   ├── multi_entity.py        # make_multi_entity()
│   ├── contradictions.py      # make_contradictions()
│   └── disambiguation.py      # make_disambiguation()
│
├── sentence_splitter.py       # split_into_atomic_facts(text) → list[str]  ← new
│
├── controlled_inserter.py     # Phase 1: EntityNode/Edge.save() — bypasses LLM
├── pipeline_inserter.py       # Phase 2: add_episode() — full LLM extraction
├── presplit_inserter.py       # Phase 3: sentence_splitter → add_episode()  ← new
│
├── benchmark_runner.py        # Orchestrator: phases × strategies × test_cases
├── search_strategies.py       # hybrid / bm25_only / cosine_only configs
├── evaluator.py               # P@5, R@5, MRR, TemporalAccuracy
├── judge.py                   # LLM-as-judge: facts_match(), find_matches()
└── reporter.py                # Aggregation + console/JSON output
```

---

## Data Flow: `pipeline_presplit` Phase

```
data/test_cases.json
        │ load_test_cases()
        ▼
  list[TestCase]
        │
        │  for each TestCase
        ▼
  presplit_inserter.insert_test_case()
        │
        │  for each Episode (sorted by .order)
        ▼
  sentence_splitter.split_into_atomic_facts(episode.text)
        │
        │  heuristic fast-path (simple ASCII text < 15 words): return [text]
        │  otherwise: LLM call (Qwen/Qwen3.5-397B-A17B via SiliconFlow)
        ▼
  list[str]  — atomic facts
        │
        │  for each fact (i)
        ▼
  graphiti.add_episode(
      name=f"{tc.id}_ep{ep.order}_part{i}",
      episode_body=fact,
      reference_time=episode.reference_time,   ← shared across all parts
      group_id="benchmark_presplit",
  )
        │
        ▼
  Neo4j graph (group: benchmark_presplit)
        │
        │  benchmark_runner.evaluate_query()
        ▼
  graphiti.search_(query, group_ids=["benchmark_presplit"],
                   search_filter=temporal_filter if hybrid else None)
        │
        ▼
  list[str] facts → evaluator → QueryResult → CategoryReport
```

---

## GENERATORS Registry Pattern

`data_generator.py` maintains a single dict that maps category name → generator:

```python
GENERATORS = {
    "static_fact":             make_static_facts,
    "evolving_fact":           make_evolving_facts,
    "evolving_compound":       make_evolving_compound,   # ← new
    "multi_entity_evolution":  make_multi_entity,
    "contradiction_resolution":make_contradictions,
    "entity_disambiguation":   make_disambiguation,
    "noisy_fact":              make_noisy_facts,          # ← new
}
```

`generate_test_cases(categories=None)` iterates the selected keys, calls each function,
and concatenates results. To add a category: (1) create a module under `test_data/`,
(2) implement `make_<category>() -> list[TestCase]`, (3) add one entry to GENERATORS.
No other file needs to change.

---

## Interface Contracts

### `sentence_splitter`
```python
# Public
async def split_into_atomic_facts(text: str) -> list[str]
# Internal helper (also used in tests)
def is_likely_compound(text: str) -> bool
```
- Pure function signature: text in, list of strings out.
- No Graphiti dependency.
- Uses a module-level `AsyncOpenAI` singleton (lazy-initialized from env vars).

### `presplit_inserter`
```python
async def insert_test_case(graphiti: Graphiti, test_case: TestCase) -> None
async def insert_all(graphiti: Graphiti, test_cases: list[TestCase]) -> None
GROUP_ID: str  # = "benchmark_presplit"
```
- Identical surface to `controlled_inserter` and `pipeline_inserter` — all three share
  `(graphiti, test_cases)` signatures, enabling drop-in substitution in `run_phase()`.

### `benchmark_runner` dispatch
```python
GROUP_IDS = {
    "controlled":       controlled_inserter.GROUP_ID,
    "pipeline":         pipeline_inserter.GROUP_ID,
    "pipeline_presplit":presplit_inserter.GROUP_ID,
}
# run_phase() uses GROUP_IDS[phase] for graph isolation and search scoping
```

---

## Architectural Concerns

### 1. `data_generator_extended.py` is dead code
`data_generator.py` imports from `src.test_data.compound` and `src.test_data.noisy`.
`src/data_generator_extended.py` duplicates those generators and is never imported.
**Action:** delete `src/data_generator_extended.py`.

### 2. Noisy text bypasses the LLM fast-path
`split_into_atomic_facts` skips the LLM when `text.isascii() and len(text.split()) < 15`.
Many noisy cases are short ASCII strings (e.g., `"dave → msft as sr. dev"` = 6 words).
The `→` character is not ASCII, so this particular case does trigger the LLM — but other
abbreviation/typo cases under 15 words and all-ASCII will silently pass through unnormalized.
**Impact:** `noisy_fact` cases with short typo text may not get cleaned before `add_episode`,
reducing the expected improvement of `pipeline_presplit` vs `pipeline` on that category.
**Recommendation:** widen the heuristic or remove the ASCII/length fast-path for noisy categories.

### 3. `sentence_splitter` client is not injectable
The `AsyncOpenAI` client is a module-level singleton. The spec's `presplit_inserter` draft
showed `splitter_client` as an explicit parameter; the implementation dropped it.
This is fine for production but makes unit tests depend on env vars being set.
Tests already handle this with `@pytest.mark.skipif(not os.environ.get(...))`.

### 4. No concurrency in inserters
All three inserters process episodes sequentially. For 75 cases × `pipeline_presplit`,
each `add_episode` involves multiple LLM calls — this will be slow. Not a correctness
concern, but worth noting for benchmark runtime planning.
