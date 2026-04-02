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
    has_expected_not = any(q.expected_not for case in evolving for q in case.queries)
    assert has_expected_not, "At least one evolving query must have expected_not"


def test_serializes_to_json():
    from src.data_generator import generate_test_cases

    cases = generate_test_cases()
    json_str = json.dumps([c.model_dump(mode="json") for c in cases], indent=2)
    assert len(json_str) > 0
    parsed = json.loads(json_str)
    assert len(parsed) >= 55
