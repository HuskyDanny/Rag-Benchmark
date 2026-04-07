# Benchmark Plan: Tasks 5-8

Parent: [[benchmark-plan-index]]

---

### Task 5: LLM-as-Judge Fact Matcher

**Files:**
- Create: `src/judge.py`
- Create: `tests/test_judge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_judge.py
import pytest


@pytest.mark.asyncio
async def test_exact_match():
    from src.judge import facts_match

    assert await facts_match("Amy works at Facebook", "Amy works at Facebook") is True


@pytest.mark.asyncio
async def test_semantic_match():
    from src.judge import facts_match

    assert await facts_match(
        "Amy is employed by Facebook",
        "Amy works at Facebook",
    ) is True


@pytest.mark.asyncio
async def test_no_match():
    from src.judge import facts_match

    assert await facts_match(
        "Amy works at Google",
        "Amy works at Facebook",
    ) is False


@pytest.mark.asyncio
async def test_batch_match():
    from src.judge import find_matches

    returned = ["Amy works at Facebook", "Bob lives in NYC", "Carol is a CEO"]
    expected = ["Amy is employed by Facebook"]
    matches = await find_matches(returned, expected)
    assert len(matches) == 1
    assert matches[0][0] == "Amy works at Facebook"
    assert matches[0][1] == "Amy is employed by Facebook"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_judge.py -v`
Expected: FAIL

- [ ] **Step 3: Write the judge module**

```python
# src/judge.py
"""LLM-as-judge for semantic fact matching."""

import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


async def facts_match(returned_fact: str, expected_fact: str) -> bool:
    """Ask GPT-4o-mini if two facts express the same factual claim."""
    if returned_fact.strip().lower() == expected_fact.strip().lower():
        return True

    client = _get_client()
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Do these two statements express the same core factual claim? "
                    "Ignore minor wording differences. Focus on whether the key entities "
                    "and relationship are the same.\n\n"
                    f"Statement A: {returned_fact}\n"
                    f"Statement B: {expected_fact}\n\n"
                    "Answer YES or NO only."
                ),
            }
        ],
        max_tokens=3,
        temperature=0,
    )
    answer = response.choices[0].message.content or ""
    return "YES" in answer.upper()


async def find_matches(
    returned_facts: list[str], expected_facts: list[str]
) -> list[tuple[str, str]]:
    """Find which returned facts match which expected facts."""
    matches = []
    for expected in expected_facts:
        for returned in returned_facts:
            if await facts_match(returned, expected):
                matches.append((returned, expected))
                break
    return matches


async def any_match(
    returned_facts: list[str], unwanted_facts: list[str]
) -> list[str]:
    """Check if any returned facts match the unwanted facts."""
    matched_unwanted = []
    for unwanted in unwanted_facts:
        for returned in returned_facts:
            if await facts_match(returned, unwanted):
                matched_unwanted.append(returned)
                break
    return matched_unwanted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_judge.py -v`
Expected: 4 passed (requires OPENAI_API_KEY in .env)

- [ ] **Step 5: Commit**

```bash
git add src/judge.py tests/test_judge.py
git commit -m "feat: LLM-as-judge for semantic fact matching"
```

---

### Task 6: Evaluator (Precision, Recall, MRR, Temporal Accuracy)

**Files:**
- Create: `src/evaluator.py`
- Create: `tests/test_evaluator.py`

**Depends on:** Task 5 (judge)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluator.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_precision_at_k_perfect():
    from src.evaluator import compute_precision_at_k

    with patch("src.evaluator.find_matches", new_callable=AsyncMock) as mock:
        mock.return_value = [("A", "A"), ("B", "B")]
        result = await compute_precision_at_k(
            returned=["A", "B", "C", "D", "E"],
            expected=["A", "B"],
            k=5,
        )
        assert result == 2 / 5


@pytest.mark.asyncio
async def test_recall_at_k_perfect():
    from src.evaluator import compute_recall_at_k

    with patch("src.evaluator.find_matches", new_callable=AsyncMock) as mock:
        mock.return_value = [("A", "A"), ("B", "B")]
        result = await compute_recall_at_k(
            returned=["A", "B", "C"],
            expected=["A", "B"],
            k=5,
        )
        assert result == 1.0


@pytest.mark.asyncio
async def test_recall_at_k_partial():
    from src.evaluator import compute_recall_at_k

    with patch("src.evaluator.find_matches", new_callable=AsyncMock) as mock:
        mock.return_value = [("A", "A")]
        result = await compute_recall_at_k(
            returned=["A", "C"],
            expected=["A", "B"],
            k=5,
        )
        assert result == 0.5


@pytest.mark.asyncio
async def test_mrr_first_result():
    from src.evaluator import compute_mrr

    with patch("src.evaluator.facts_match", new_callable=AsyncMock) as mock:
        mock.side_effect = lambda r, e: r == "A"
        result = await compute_mrr(returned=["A", "B", "C"], expected=["A"])
        assert result == 1.0


@pytest.mark.asyncio
async def test_mrr_second_result():
    from src.evaluator import compute_mrr

    async def mock_match(r, e):
        return r == "B" and e == "A"

    with patch("src.evaluator.facts_match", side_effect=mock_match):
        result = await compute_mrr(returned=["X", "B", "C"], expected=["A"])
        assert result == 0.5


@pytest.mark.asyncio
async def test_temporal_accuracy_pass():
    from src.evaluator import compute_temporal_accuracy

    with patch("src.evaluator.any_match", new_callable=AsyncMock) as mock:
        mock.return_value = []
        result = await compute_temporal_accuracy(
            returned=["Amy works at Facebook"],
            expected_not=["Amy works at Google"],
        )
        assert result is True


@pytest.mark.asyncio
async def test_temporal_accuracy_fail():
    from src.evaluator import compute_temporal_accuracy

    with patch("src.evaluator.any_match", new_callable=AsyncMock) as mock:
        mock.return_value = ["Amy works at Google"]
        result = await compute_temporal_accuracy(
            returned=["Amy works at Facebook", "Amy works at Google"],
            expected_not=["Amy works at Google"],
        )
        assert result is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluator.py -v`
Expected: FAIL

- [ ] **Step 3: Write the evaluator**

```python
# src/evaluator.py
"""Evaluation metrics: precision@k, recall@k, MRR, temporal accuracy."""

from src.judge import facts_match, find_matches, any_match


async def compute_precision_at_k(
    returned: list[str], expected: list[str], k: int = 5
) -> float:
    """Precision@k: fraction of top-k returned that match expected."""
    top_k = returned[:k]
    if not top_k:
        return 0.0
    matches = await find_matches(top_k, expected)
    return len(matches) / len(top_k)


async def compute_recall_at_k(
    returned: list[str], expected: list[str], k: int = 5
) -> float:
    """Recall@k: fraction of expected facts found in top-k returned."""
    if not expected:
        return 1.0
    top_k = returned[:k]
    matches = await find_matches(top_k, expected)
    return len(matches) / len(expected)


async def compute_mrr(returned: list[str], expected: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first correct result."""
    for i, r in enumerate(returned):
        for e in expected:
            if await facts_match(r, e):
                return 1.0 / (i + 1)
    return 0.0


async def compute_temporal_accuracy(
    returned: list[str], expected_not: list[str]
) -> bool:
    """True if no unwanted (outdated) facts appear in results."""
    if not expected_not:
        return True
    matched_unwanted = await any_match(returned, expected_not)
    return len(matched_unwanted) == 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluator.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/evaluator.py tests/test_evaluator.py
git commit -m "feat: evaluation metrics (precision, recall, MRR, temporal accuracy)"
```

---

### Task 7: Controlled Inserter (Phase 1: add_triplet)

**Files:**
- Create: `src/controlled_inserter.py`

- [ ] **Step 1: Write the controlled inserter**

No unit test — requires Neo4j. Tested in E2E (Task 11 spike).

```python
# src/controlled_inserter.py
"""Phase 1: Insert known-good triplets via add_triplet()."""

from graphiti_core import Graphiti
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from src.models import TestCase

GROUP_ID = "benchmark_controlled"


async def insert_test_case(graphiti: Graphiti, test_case: TestCase) -> None:
    """Insert all triplets from a test case into the graph."""
    node_cache: dict[str, EntityNode] = {}

    for triplet in test_case.triplets:
        source_node = _get_or_create_node(
            node_cache, triplet.source.name, triplet.source.labels
        )
        target_node = _get_or_create_node(
            node_cache, triplet.target.name, triplet.target.labels
        )

        edge = EntityEdge(
            group_id=GROUP_ID,
            source_node_uuid=source_node.uuid,
            target_node_uuid=target_node.uuid,
            created_at=triplet.edge.valid_at,
            name=triplet.edge.name,
            fact=triplet.edge.fact,
            valid_at=triplet.edge.valid_at,
            invalid_at=triplet.edge.invalid_at,
        )

        await graphiti.add_triplet(
            source_node=source_node, edge=edge, target_node=target_node
        )


def _get_or_create_node(
    cache: dict[str, EntityNode], name: str, labels: list[str]
) -> EntityNode:
    """Get existing node from cache or create a new one."""
    if name in cache:
        return cache[name]
    node = EntityNode(name=name, group_id=GROUP_ID, labels=labels)
    cache[name] = node
    return node


async def insert_all(graphiti: Graphiti, test_cases: list[TestCase]) -> None:
    """Insert all test cases into the graph."""
    for i, tc in enumerate(test_cases):
        print(f"  Inserting test case {i+1}/{len(test_cases)}: {tc.id}")
        await insert_test_case(graphiti, tc)
```

- [ ] **Step 2: Commit**

```bash
git add src/controlled_inserter.py
git commit -m "feat: controlled inserter for Phase 1 (add_triplet)"
```

---

### Task 8: Pipeline Inserter (Phase 2: add_episode)

**Files:**
- Create: `src/pipeline_inserter.py`

- [ ] **Step 1: Write the pipeline inserter**

```python
# src/pipeline_inserter.py
"""Phase 2: Insert raw text episodes via add_episode()."""

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from src.models import TestCase

GROUP_ID = "benchmark_pipeline"


async def insert_test_case(graphiti: Graphiti, test_case: TestCase) -> None:
    """Insert all episodes from a test case via add_episode()."""
    sorted_episodes = sorted(test_case.episodes, key=lambda e: e.order)
    for episode in sorted_episodes:
        await graphiti.add_episode(
            name=f"{test_case.id}_ep{episode.order}",
            episode_body=episode.text,
            source=EpisodeType.text,
            source_description="benchmark test data",
            reference_time=episode.reference_time,
            group_id=GROUP_ID,
        )


async def insert_all(graphiti: Graphiti, test_cases: list[TestCase]) -> None:
    """Insert all test cases via the full LLM extraction pipeline."""
    for i, tc in enumerate(test_cases):
        print(f"  Inserting test case {i+1}/{len(test_cases)}: {tc.id}")
        await insert_test_case(graphiti, tc)
```

- [ ] **Step 2: Commit**

```bash
git add src/pipeline_inserter.py
git commit -m "feat: pipeline inserter for Phase 2 (add_episode)"
```
