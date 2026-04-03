"""Test case generator — combines all categories."""

from __future__ import annotations

import json
from pathlib import Path

from src.models import TestCase
from src.test_data.compound import make_evolving_compound
from src.test_data.contradictions import make_contradictions
from src.test_data.disambiguation import make_disambiguation
from src.test_data.evolving_facts import make_evolving_facts
from src.test_data.multi_entity import make_multi_entity
from src.test_data.noisy import make_noisy_facts
from src.test_data.static_facts import make_static_facts

# Registry of all generators — add/remove categories here
GENERATORS = {
    "static_fact": make_static_facts,
    "evolving_fact": make_evolving_facts,
    "evolving_compound": make_evolving_compound,
    "multi_entity_evolution": make_multi_entity,
    "contradiction_resolution": make_contradictions,
    "entity_disambiguation": make_disambiguation,
    "noisy_fact": make_noisy_facts,
}


def generate_test_cases(categories: list[str] | None = None) -> list[TestCase]:
    """Generate test cases for selected (or all) categories."""
    selected = categories or list(GENERATORS.keys())
    cases: list[TestCase] = []
    for cat in selected:
        cases.extend(GENERATORS[cat]())
    return cases


def save_test_cases(path: str = "data/test_cases.json") -> None:
    """Serialize test cases to JSON and write to file."""
    cases = generate_test_cases()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump(mode="json") for c in cases]
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Saved {len(cases)} test cases to {out_path}")


if __name__ == "__main__":
    save_test_cases("data/test_cases.json")
