# Pre-Split Preprocessor & Noisy Test Data — Design Spec

**Date:** 2026-04-03
**Goal:** Add a sentence decomposition preprocessor to improve Graphiti's temporal accuracy on compound/noisy text, and benchmark it against the original pipeline.

---

## 1. Problem

Graphiti's `add_episode` extracts multiple edges from compound text but fails to cross-reference them for contradiction detection. When "Alice left Google and joined Meta" produces edges `LEFT_COMPANY` and `JOINED_COMPANY`, the resolution logic doesn't connect `JOINED_COMPANY Meta` with the existing `WORKS_AT Google` edge (different node pair).

Separate episodes ("Alice left Google" then "Alice joined Meta") work correctly because each edge is resolved independently against the full graph.

Additionally, real-world text contains typos, abbreviations, and informal language that degrade extraction quality.

---

## 2. Solution: Pre-Split Preprocessor

### Module: `src/sentence_splitter.py`

A single async function that decomposes text into atomic facts:

```python
async def split_into_atomic_facts(text: str, client: AsyncOpenAI) -> list[str]
```

- Input: any text (simple, compound, noisy)
- Output: list of atomic fact strings
- Simple sentences pass through unchanged (1 in → 1 out)
- Compound sentences split into independent facts
- Typos/abbreviations normalized into clean sentences
- Uses `Qwen/Qwen3.5-397B-A17B` via SiliconFlow

### Modified Pipeline Inserter: `src/presplit_inserter.py`

New inserter that pre-splits before calling `add_episode`:

```python
async def insert_test_case(graphiti, test_case, splitter_client):
    for episode in sorted(test_case.episodes, key=lambda e: e.order):
        atomic_facts = await split_into_atomic_facts(episode.text, splitter_client)
        for i, fact in enumerate(atomic_facts):
            await graphiti.add_episode(
                name=f"{test_case.id}_ep{episode.order}_part{i}",
                episode_body=fact,
                reference_time=episode.reference_time,
                ...
            )
```

Each atomic fact gets its own `add_episode` call with the same `reference_time`.

---

## 3. New Test Data

### New Category: `evolving_compound` (10 cases)

Compound episodes where two facts are merged into one sentence. Tests whether pre-split helps contradiction detection.

| ID | Episode 1 | Episode 2 (compound) |
|----|-----------|---------------------|
| compound_001 | "Amy works at Google." | "Amy left Google and joined Facebook." |
| compound_002 | "Bob lives in NYC." | "Bob sold his NYC apartment and moved to SF." |
| ... | (single fact) | (compound: departure + arrival) |

Tags: `[compound, evolving]`

### New Category: `noisy_fact` (10 cases)

Episodes with typos, abbreviations, and informal language.

| ID | Noise Type | Episode Example |
|----|-----------|-----------------|
| noisy_001 | typo | "Amy workz at Gogle as enginere." |
| noisy_002 | abbreviation | "Bob → SF, prev. NYC" |
| noisy_003 | informal | "carol's now running team beta lol" |
| noisy_004 | mixed | "dave quit msft & joined amzn as sr. dev" |
| ... | | |

Tags: `[noisy, evolving]`

Each noisy case has a corresponding clean expected fact for evaluation.

### Updated TestCase Model

Add `tags` field:

```python
class TestCase(BaseModel):
    id: str
    category: str
    tags: list[str] = []  # e.g., ["compound", "evolving", "typo"]
    episodes: list[Episode]
    queries: list[Query]
    triplets: list[Triplet]
```

### Updated Totals

| Category | Count | Tags |
|----------|-------|------|
| static_fact | 10 | clean |
| evolving_fact | 15 | clean, evolving |
| evolving_compound | 10 | compound, evolving |
| multi_entity_evolution | 10 | clean, multi |
| contradiction_resolution | 10 | clean, contradiction |
| entity_disambiguation | 10 | clean, disambig |
| noisy_fact | 10 | noisy, evolving |
| **Total** | **75** | |

---

## 4. Benchmark Design

### Phases

| Phase | Inserter | Pre-split | What It Tests |
|-------|----------|-----------|---------------|
| `controlled` | direct save | No | Search + filter quality (baseline) |
| `pipeline` | add_episode | No | Full pipeline with raw text |
| `pipeline_presplit` | presplit_inserter | Yes | Pipeline with decomposed text |

All phases run on all 75 test cases. All use `Qwen/Qwen3.5-397B-A17B`.

### Key Comparisons

1. **`pipeline` vs `pipeline_presplit` on `evolving_compound`** — Does pre-split fix compound sentence contradiction detection?
2. **`pipeline` vs `pipeline_presplit` on `noisy_fact`** — Does pre-split normalization help with typos?
3. **`pipeline` vs `pipeline_presplit` on `evolving_fact`** — Does pre-split hurt already-clean data? (Should be neutral)

### Evaluation

Same metrics: Precision@5, Recall@5, MRR, Temporal Accuracy. Results grouped by `(phase, category, strategy)` with tag-based filtering for analysis.

---

## 5. Project Structure Changes

```
src/
  sentence_splitter.py     # NEW: atomic fact decomposition
  presplit_inserter.py     # NEW: pre-split + add_episode
  data_generator.py        # MODIFIED: add compound + noisy categories
  models.py                # MODIFIED: add tags field
  benchmark_runner.py      # MODIFIED: add pipeline_presplit phase
tests/
  test_sentence_splitter.py  # NEW
  test_data_generator.py     # MODIFIED: update counts
```

---

## 6. LLM Configuration

All LLM calls use `Qwen/Qwen3.5-397B-A17B` via SiliconFlow:
- Sentence splitting (new)
- Entity/edge extraction (Graphiti internal)
- Contradiction resolution (Graphiti internal)
- LLM-as-judge (evaluation)

Embedding: `Qwen/Qwen3-Embedding-0.6B` (unchanged).
