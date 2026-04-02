# Benchmark Plan: Tasks 1-4

Parent: [[benchmark-plan-index]]

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "graphiti-temporal-benchmark"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "graphiti-core[neo4j]",
    "openai>=1.0.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "tabulate>=0.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create .env.example**

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
OPENAI_API_KEY=sk-...
```

- [ ] **Step 3: Create package directories**

```bash
mkdir -p src tests data results
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 4: Create venv and install deps**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

- [ ] **Step 5: Verify Graphiti imports work**

```bash
python3 -c "from graphiti_core import Graphiti; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml .env.example src/__init__.py tests/__init__.py
git commit -m "feat: project scaffolding for graphiti temporal benchmark"
```

---

### Task 2: Pydantic Models for Test Cases and Results

**Files:**
- Create: `src/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
import pytest


def test_test_case_model_parses_json():
    from src.models import TestCase

    data = {
        "id": "static_001",
        "category": "static_fact",
        "episodes": [
            {
                "text": "Paris is the capital of France.",
                "reference_time": "2024-01-01T00:00:00Z",
                "order": 1,
            }
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
    assert tc.id == "static_001"
    assert tc.category == "static_fact"
    assert len(tc.episodes) == 1
    assert len(tc.queries) == 1
    assert len(tc.triplets) == 1
    assert tc.triplets[0].edge.invalid_at is None


def test_query_result_model():
    from src.models import QueryResult

    qr = QueryResult(
        test_case_id="static_001",
        query="What is the capital of France?",
        strategy="hybrid",
        returned_facts=["Paris is the capital of France"],
        expected_facts=["Paris is the capital of France"],
        expected_not=[],
        precision_at_5=1.0,
        recall_at_5=1.0,
        mrr=1.0,
        temporal_accuracy=True,
    )
    assert qr.precision_at_5 == 1.0
    assert qr.temporal_accuracy is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models'`

- [ ] **Step 3: Write the models**

```python
# src/models.py
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel


class Episode(BaseModel):
    text: str
    reference_time: datetime
    order: int


class Query(BaseModel):
    query: str
    expected_facts: list[str]
    expected_not: list[str]
    query_time: datetime


class TripletNode(BaseModel):
    name: str
    labels: list[str]


class TripletEdge(BaseModel):
    name: str
    fact: str
    valid_at: datetime
    invalid_at: datetime | None = None


class Triplet(BaseModel):
    source: TripletNode
    target: TripletNode
    edge: TripletEdge


class TestCase(BaseModel):
    id: str
    category: str
    episodes: list[Episode]
    queries: list[Query]
    triplets: list[Triplet]


class QueryResult(BaseModel):
    test_case_id: str
    query: str
    strategy: str
    returned_facts: list[str]
    expected_facts: list[str]
    expected_not: list[str]
    precision_at_5: float
    recall_at_5: float
    mrr: float
    temporal_accuracy: bool


class CategoryReport(BaseModel):
    phase: str
    category: str
    strategy: str
    avg_precision_at_5: float
    avg_recall_at_5: float
    avg_mrr: float
    temporal_accuracy_pct: float
    num_queries: int
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: pydantic models for test cases and benchmark results"
```

---

### Task 3: Test Data Generator

**Files:**
- Create: `src/data_generator.py`
- Create: `tests/test_data_generator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_generator.py
import pytest
import json


def test_generates_all_categories():
    from src.data_generator import generate_test_cases

    cases = generate_test_cases()
    categories = {c.category for c in cases}
    assert categories == {
        "static_fact",
        "evolving_fact",
        "multi_entity_evolution",
        "contradiction_resolution",
        "entity_disambiguation",
    }


def test_generates_at_least_55_cases():
    from src.data_generator import generate_test_cases

    cases = generate_test_cases()
    assert len(cases) >= 55


def test_evolving_fact_has_multiple_episodes():
    from src.data_generator import generate_test_cases

    cases = generate_test_cases()
    evolving = [c for c in cases if c.category == "evolving_fact"]
    for case in evolving:
        assert len(case.episodes) >= 2, f"{case.id} must have >=2 episodes"
        assert len(case.triplets) >= 2, f"{case.id} must have >=2 triplets"


def test_each_case_has_queries():
    from src.data_generator import generate_test_cases

    cases = generate_test_cases()
    for case in cases:
        assert len(case.queries) >= 1, f"{case.id} must have >=1 query"


def test_evolving_fact_has_expected_not():
    from src.data_generator import generate_test_cases

    cases = generate_test_cases()
    evolving = [c for c in cases if c.category == "evolving_fact"]
    has_expected_not = any(
        q.expected_not for case in evolving for q in case.queries
    )
    assert has_expected_not, "At least one evolving query must have expected_not"


def test_serializes_to_json():
    from src.data_generator import generate_test_cases

    cases = generate_test_cases()
    json_str = json.dumps([c.model_dump(mode="json") for c in cases], indent=2)
    assert len(json_str) > 0
    parsed = json.loads(json_str)
    assert len(parsed) >= 55
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_generator.py -v`
Expected: FAIL

- [ ] **Step 3: Write the data generator**

Create `src/data_generator.py` with these functions:
- `_dt(year, month, day)` — helper to create UTC datetime
- `_static_facts()` — 10 static facts (Paris/France, Python/Guido, etc.)
- `_evolving_facts()` — 15 evolving facts (Amy Google->Facebook, Bob Microsoft->Apple, etc.)
- `_multi_entity_evolution()` — 10 cases with 2 entities evolving per case
- `_contradiction_resolution()` — 10 cases (Acme revenue $10M->$12M, etc.)
- `_entity_disambiguation()` — 10 cases (John Smith engineer vs John Smith doctor, etc.)
- `generate_test_cases()` — combines all 55 cases
- `save_test_cases(path)` — serializes to JSON

Each category follows the TestCase schema from `src/models.py`. Evolving/contradiction cases must have:
- 2+ episodes with different `reference_time` and `order`
- 2+ triplets where the older one has `invalid_at` set
- Queries with `expected_not` containing the outdated fact

Entity disambiguation cases use `TripletNode(name="John Smith (Engineer)")` to differentiate in controlled insertion.

See the spec at `docs/superpowers/specs/2026-04-01-graphiti-temporal-retrieval-benchmark-design.md` Section 3 for category details.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data_generator.py -v`
Expected: 6 passed

- [ ] **Step 5: Generate test_cases.json**

```bash
python3 -m src.data_generator
```

Expected: `Saved 55 test cases to data/test_cases.json`

- [ ] **Step 6: Commit**

```bash
git add src/data_generator.py tests/test_data_generator.py data/test_cases.json
git commit -m "feat: test data generator with 55 cases across 5 categories"
```

---

### Task 4: Search Strategy Configurations

**Files:**
- Create: `src/search_strategies.py`
- Create: `tests/test_search_strategies.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_search_strategies.py
import pytest


def test_get_strategies_returns_three():
    from src.search_strategies import get_search_strategies

    strategies = get_search_strategies()
    assert set(strategies.keys()) == {"hybrid", "bm25_only", "cosine_only"}


def test_hybrid_uses_multiple_methods():
    from src.search_strategies import get_search_strategies
    from graphiti_core.search.search_config import EdgeSearchMethod

    strategies = get_search_strategies()
    hybrid = strategies["hybrid"]
    assert hybrid.edge_config is not None
    assert len(hybrid.edge_config.search_methods) >= 2


def test_bm25_only_uses_bm25():
    from src.search_strategies import get_search_strategies
    from graphiti_core.search.search_config import EdgeSearchMethod

    strategies = get_search_strategies()
    bm25 = strategies["bm25_only"]
    assert bm25.edge_config is not None
    assert bm25.edge_config.search_methods == [EdgeSearchMethod.bm25]


def test_cosine_only_uses_cosine():
    from src.search_strategies import get_search_strategies
    from graphiti_core.search.search_config import EdgeSearchMethod

    strategies = get_search_strategies()
    cosine = strategies["cosine_only"]
    assert cosine.edge_config is not None
    assert cosine.edge_config.search_methods == [EdgeSearchMethod.cosine_similarity]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_search_strategies.py -v`
Expected: FAIL

- [ ] **Step 3: Write the search strategies module**

```python
# src/search_strategies.py
"""Search strategy configurations for the benchmark."""

from graphiti_core.search.search_config import (
    SearchConfig,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EdgeReranker,
)
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF


def get_search_strategies() -> dict[str, SearchConfig]:
    """Return the three search strategy configs for benchmarking."""
    hybrid = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    hybrid.limit = 10

    bm25_only = SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25],
            reranker=EdgeReranker.rrf,
        ),
        limit=10,
    )

    cosine_only = SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.rrf,
        ),
        limit=10,
    )

    return {
        "hybrid": hybrid,
        "bm25_only": bm25_only,
        "cosine_only": cosine_only,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_search_strategies.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/search_strategies.py tests/test_search_strategies.py
git commit -m "feat: search strategy configurations (hybrid, bm25, cosine)"
```
