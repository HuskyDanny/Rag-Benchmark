# Pre-Split Preprocessor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sentence decomposition preprocessing to improve Graphiti's temporal accuracy on compound/noisy text, with 20 new test cases and a `pipeline_presplit` benchmark phase.

**Architecture:** New `sentence_splitter.py` decomposes text into atomic facts via LLM. New `presplit_inserter.py` splits before calling `add_episode`. Data generator extended with `evolving_compound` (10) and `noisy_fact` (10) categories. Benchmark runner gets a third pipeline phase.

**Tech Stack:** Python 3.11+, graphiti-core, OpenAI-compatible API (SiliconFlow), Qwen/Qwen3.5-397B-A17B for all LLM calls.

**Spec:** `docs/superpowers/specs/2026-04-03-presplit-preprocessor-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/models.py` | Modify | Add `tags` field to TestCase |
| `src/sentence_splitter.py` | Create | LLM-based atomic fact decomposition |
| `src/presplit_inserter.py` | Create | Pre-split + add_episode pipeline |
| `src/data_generator.py` | Modify | Add `evolving_compound` + `noisy_fact` categories |
| `src/benchmark_runner.py` | Modify | Add `pipeline_presplit` phase |
| `tests/test_sentence_splitter.py` | Create | Unit tests for splitter |
| `tests/test_data_generator.py` | Modify | Update counts for 75 cases |
| `data/test_cases.json` | Regenerate | 75 cases |

## Dependency Graph

```
Task 1 (models + tags)
  → Task 2 (sentence splitter + tests)
  → Task 3 (presplit inserter)
  → Task 4 (compound test data)
  → Task 5 (noisy test data)
       → Task 6 (benchmark runner update)
            → Task 7 (regenerate data + run benchmark)
```

Tasks 2-5 are parallelizable after Task 1.

---

### Task 1: Add `tags` Field to TestCase Model

**Files:**
- Modify: `src/models.py:40-45`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_models.py
def test_test_case_with_tags():
    from src.models import TestCase

    data = {
        "id": "compound_001",
        "category": "evolving_compound",
        "tags": ["compound", "evolving"],
        "episodes": [
            {"text": "Amy works at Google.", "reference_time": "2024-01-01T00:00:00Z", "order": 0}
        ],
        "queries": [
            {
                "query": "Where does Amy work?",
                "expected_facts": ["Amy works at Google"],
                "expected_not": [],
                "query_time": "2024-07-01T00:00:00Z",
            }
        ],
        "triplets": [
            {
                "source": {"name": "Amy", "labels": ["Person"]},
                "target": {"name": "Google", "labels": ["Company"]},
                "edge": {
                    "name": "WORKS_AT",
                    "fact": "Amy works at Google",
                    "valid_at": "2024-01-01T00:00:00Z",
                    "invalid_at": None,
                },
            }
        ],
    }
    tc = TestCase.model_validate(data)
    assert tc.tags == ["compound", "evolving"]


def test_test_case_tags_default_empty():
    from src.models import TestCase

    data = {
        "id": "static_001",
        "category": "static_fact",
        "episodes": [
            {"text": "Paris is the capital of France.", "reference_time": "2024-01-01T00:00:00Z", "order": 0}
        ],
        "queries": [
            {
                "query": "What is the capital of France?",
                "expected_facts": ["Paris is the capital of France"],
                "expected_not": [],
                "query_time": "2024-07-01T00:00:00Z",
            }
        ],
        "triplets": [
            {
                "source": {"name": "Paris", "labels": ["City"]},
                "target": {"name": "France", "labels": ["Country"]},
                "edge": {
                    "name": "CAPITAL_OF",
                    "fact": "Paris is the capital of France",
                    "valid_at": "2024-01-01T00:00:00Z",
                    "invalid_at": None,
                },
            }
        ],
    }
    tc = TestCase.model_validate(data)
    assert tc.tags == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: FAIL on `test_test_case_with_tags` — `tags` field not recognized

- [ ] **Step 3: Add tags field to TestCase**

In `src/models.py`, modify the `TestCase` class:

```python
class TestCase(BaseModel):
    id: str
    category: str
    tags: list[str] = []
    episodes: list[Episode]
    queries: list[Query]
    triplets: list[Triplet]
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: add tags field to TestCase model"
```

---

### Task 2: Sentence Splitter Module

**Files:**
- Create: `src/sentence_splitter.py`
- Create: `tests/test_sentence_splitter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sentence_splitter.py
import os
import pytest
from dotenv import load_dotenv

load_dotenv()

needs_llm = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


def test_simple_sentence_passthrough():
    """Simple sentences should return as-is (no LLM call needed)."""
    from src.sentence_splitter import is_likely_compound

    assert is_likely_compound("Amy works at Google.") is False
    assert is_likely_compound("Paris is the capital of France.") is False


def test_compound_detection():
    """Compound sentences should be detected."""
    from src.sentence_splitter import is_likely_compound

    assert is_likely_compound("Amy left Google and joined Facebook.") is True
    assert is_likely_compound("Bob sold his house, then moved to SF.") is True


@needs_llm
@pytest.mark.asyncio
async def test_split_compound_sentence():
    from src.sentence_splitter import split_into_atomic_facts

    facts = await split_into_atomic_facts("Amy left Google and joined Facebook as a tech lead.")
    assert len(facts) >= 2
    # Should contain both the departure and the new job
    combined = " ".join(facts).lower()
    assert "google" in combined
    assert "facebook" in combined


@needs_llm
@pytest.mark.asyncio
async def test_split_simple_passthrough():
    from src.sentence_splitter import split_into_atomic_facts

    facts = await split_into_atomic_facts("Amy works at Google.")
    assert len(facts) == 1
    assert "google" in facts[0].lower()


@needs_llm
@pytest.mark.asyncio
async def test_split_noisy_text():
    from src.sentence_splitter import split_into_atomic_facts

    facts = await split_into_atomic_facts("amy workz at gogle as enginere")
    assert len(facts) >= 1
    # Should normalize the text
    combined = " ".join(facts).lower()
    assert "google" in combined or "gogle" in combined
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sentence_splitter.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the sentence splitter**

```python
# src/sentence_splitter.py
"""Decompose compound/noisy text into atomic facts via LLM."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_client: AsyncOpenAI | None = None

SPLITTER_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-397B-A17B")


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    return _client


def is_likely_compound(text: str) -> bool:
    """Heuristic check: does the text contain multiple facts?"""
    indicators = [" and ", " then ", " but ", ", then", " & ", "; "]
    text_lower = text.lower()
    return any(ind in text_lower for ind in indicators)


async def split_into_atomic_facts(text: str) -> list[str]:
    """Split text into atomic factual statements.

    Simple sentences pass through unchanged. Compound/noisy text
    is decomposed via LLM into clean, independent facts.
    """
    if not is_likely_compound(text) and text.isascii() and len(text.split()) < 15:
        return [text]

    client = _get_client()
    response = await client.chat.completions.create(
        model=SPLITTER_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You decompose text into simple, independent factual statements. "
                    "Rules:\n"
                    "1. Each output fact must be one complete sentence with one subject-verb-object.\n"
                    "2. Fix typos and expand abbreviations into proper English.\n"
                    "3. If the input is already a simple fact, return it as-is.\n"
                    "4. Return ONLY the facts, one per line. No numbering, no bullets."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    result = response.choices[0].message.content or text
    facts = [line.strip() for line in result.strip().split("\n") if line.strip()]
    return facts if facts else [text]
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_sentence_splitter.py -v`
Expected: All pass (LLM tests need OPENAI_API_KEY)

- [ ] **Step 5: Commit**

```bash
git add src/sentence_splitter.py tests/test_sentence_splitter.py
git commit -m "feat: sentence splitter for atomic fact decomposition"
```

---

### Task 3: Pre-Split Inserter Module

**Files:**
- Create: `src/presplit_inserter.py`

- [ ] **Step 1: Write the presplit inserter**

```python
# src/presplit_inserter.py
"""Phase 3: Pre-split text into atomic facts, then insert via add_episode()."""

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from src.models import TestCase
from src.sentence_splitter import split_into_atomic_facts

GROUP_ID = "benchmark_presplit"


async def insert_test_case(graphiti: Graphiti, test_case: TestCase) -> None:
    """Pre-split episodes into atomic facts, then insert each via add_episode."""
    sorted_episodes = sorted(test_case.episodes, key=lambda e: e.order)
    for episode in sorted_episodes:
        atomic_facts = await split_into_atomic_facts(episode.text)
        for i, fact in enumerate(atomic_facts):
            await graphiti.add_episode(
                name=f"{test_case.id}_ep{episode.order}_part{i}",
                episode_body=fact,
                source=EpisodeType.text,
                source_description="benchmark test data (pre-split)",
                reference_time=episode.reference_time,
                group_id=GROUP_ID,
            )


async def insert_all(graphiti: Graphiti, test_cases: list[TestCase]) -> None:
    """Insert all test cases with pre-splitting."""
    for i, tc in enumerate(test_cases):
        print(f"  Inserting test case {i + 1}/{len(test_cases)}: {tc.id}")
        await insert_test_case(graphiti, tc)
```

- [ ] **Step 2: Commit**

```bash
git add src/presplit_inserter.py
git commit -m "feat: presplit inserter — decompose then add_episode"
```

---

### Task 4: Add `evolving_compound` Test Data (10 cases)

**Files:**
- Modify: `src/data_generator.py`

- [ ] **Step 1: Add `_evolving_compound()` function**

Add to `src/data_generator.py` a new function that generates 10 compound evolving-fact cases. Each case has:
- Episode 0: simple sentence (initial fact)
- Episode 1: compound sentence (departure + new fact merged)
- Queries with expected_facts and expected_not
- Triplets for controlled insertion
- Tags: `["compound", "evolving"]`

Cases should cover: job changes, location moves, role changes, relationship changes — all expressed as compound sentences like "Amy left Google and joined Facebook" or "Bob sold his NYC apartment and moved to San Francisco."

Follow the exact same TestCase schema as existing evolving facts. Use `_dt(year, month, day)` helper for datetimes. IDs: `compound_001` through `compound_010`.

- [ ] **Step 2: Register in `generate_test_cases()`**

Add `cases.extend(_evolving_compound())` to the `generate_test_cases()` function.

- [ ] **Step 3: Commit**

```bash
git add src/data_generator.py
git commit -m "feat: add evolving_compound test cases (10 compound episodes)"
```

---

### Task 5: Add `noisy_fact` Test Data (10 cases)

**Files:**
- Modify: `src/data_generator.py`

- [ ] **Step 1: Add `_noisy_facts()` function**

Add 10 noisy evolving-fact cases. Each case has:
- Episode 0: noisy text (typos, abbreviations, informal)
- Episode 1: noisy text (updated fact, also noisy)
- Clean expected_facts in queries (the judge compares against clean text)
- Triplets with clean data for controlled insertion
- Tags include noise type: `["noisy", "typo"]` or `["noisy", "abbreviation"]` etc.

Noise types to cover:
- 3 typo cases: misspelled names/verbs ("workz", "enginere", "Gogle")
- 3 abbreviation cases: ("sr. dev", "prev.", "mgr", "→", "&")
- 2 informal cases: ("lol", "ngl", lowercase, no punctuation)
- 2 mixed cases: combination of above

IDs: `noisy_001` through `noisy_010`.

- [ ] **Step 2: Register in `generate_test_cases()`**

Add `cases.extend(_noisy_facts())` to `generate_test_cases()`.

- [ ] **Step 3: Commit**

```bash
git add src/data_generator.py
git commit -m "feat: add noisy_fact test cases (10 noisy episodes)"
```

---

### Task 6: Update Benchmark Runner + Data Generator Tests

**Files:**
- Modify: `src/benchmark_runner.py:35-38`
- Modify: `tests/test_data_generator.py`

- [ ] **Step 1: Update benchmark runner to support `pipeline_presplit` phase**

In `src/benchmark_runner.py`, add the presplit inserter to imports and GROUP_IDS:

```python
from src import controlled_inserter, pipeline_inserter, presplit_inserter

GROUP_IDS = {
    "controlled": controlled_inserter.GROUP_ID,
    "pipeline": pipeline_inserter.GROUP_ID,
    "pipeline_presplit": presplit_inserter.GROUP_ID,
}
```

In `run_phase()`, add the presplit branch:

```python
if phase == "controlled":
    await controlled_inserter.insert_all(graphiti, test_cases)
elif phase == "pipeline_presplit":
    await presplit_inserter.insert_all(graphiti, test_cases)
else:
    await pipeline_inserter.insert_all(graphiti, test_cases)
```

Update default phases in `run_benchmark()`:

```python
if phases is None:
    phases = ["controlled", "pipeline", "pipeline_presplit"]
```

- [ ] **Step 2: Update data generator tests for 75 cases**

In `tests/test_data_generator.py`, update:

```python
def test_generates_all_categories():
    from src.data_generator import generate_test_cases

    cases = generate_test_cases()
    categories = {c.category for c in cases}
    assert categories == {
        "static_fact",
        "evolving_fact",
        "evolving_compound",
        "multi_entity_evolution",
        "contradiction_resolution",
        "entity_disambiguation",
        "noisy_fact",
    }


def test_generates_at_least_75_cases():
    from src.data_generator import generate_test_cases

    cases = generate_test_cases()
    assert len(cases) >= 75


def test_compound_has_compound_tag():
    from src.data_generator import generate_test_cases

    cases = generate_test_cases()
    compound = [c for c in cases if c.category == "evolving_compound"]
    for case in compound:
        assert "compound" in case.tags, f"{case.id} missing compound tag"


def test_noisy_has_noisy_tag():
    from src.data_generator import generate_test_cases

    cases = generate_test_cases()
    noisy = [c for c in cases if c.category == "noisy_fact"]
    for case in noisy:
        assert "noisy" in case.tags, f"{case.id} missing noisy tag"
```

- [ ] **Step 3: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 4: Regenerate test data**

Run: `.venv/bin/python -m src.data_generator`
Expected: `Saved 75 test cases to data/test_cases.json`

- [ ] **Step 5: Commit**

```bash
git add src/benchmark_runner.py tests/test_data_generator.py data/test_cases.json
git commit -m "feat: pipeline_presplit phase + updated tests for 75 cases"
```

---

### Task 7: Run Full Benchmark (3 phases × 75 cases)

- [ ] **Step 1: Run all unit tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 2: Run controlled phase**

Run: `.venv/bin/python -m src.benchmark_runner controlled`
Expected: Results table for controlled phase, saved to `results/`

- [ ] **Step 3: Run pipeline phase**

Run: `.venv/bin/python -m src.benchmark_runner pipeline`
Expected: Results table for pipeline phase

- [ ] **Step 4: Run pipeline_presplit phase**

Run: `.venv/bin/python -m src.benchmark_runner pipeline_presplit`
Expected: Results table for presplit phase

- [ ] **Step 5: Compare results**

Key comparisons:
- `pipeline` vs `pipeline_presplit` on `evolving_compound` — TempAcc improvement?
- `pipeline` vs `pipeline_presplit` on `noisy_fact` — R@5 improvement?
- `pipeline` vs `pipeline_presplit` on `evolving_fact` — no regression?

- [ ] **Step 6: Commit results**

```bash
git add -f results/
git commit -m "feat: benchmark results — controlled + pipeline + presplit"
```
