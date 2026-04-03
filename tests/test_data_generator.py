"""Tests for the test case generator and its category modules."""

import json

from src.data_generator import GENERATORS, generate_test_cases


def test_generates_all_7_categories():
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


def test_generates_75_cases():
    cases = generate_test_cases()
    assert len(cases) == 75


def test_generator_registry_has_7_entries():
    assert len(GENERATORS) == 7


def test_selective_generation():
    cases = generate_test_cases(categories=["static_fact", "noisy_fact"])
    categories = {c.category for c in cases}
    assert categories == {"static_fact", "noisy_fact"}
    assert len(cases) == 20  # 10 static + 10 noisy


def test_evolving_fact_has_multiple_episodes():
    cases = generate_test_cases(categories=["evolving_fact"])
    for case in cases:
        assert len(case.episodes) >= 2, f"{case.id} must have >=2 episodes"
        assert len(case.triplets) >= 2, f"{case.id} must have >=2 triplets"


def test_each_case_has_queries():
    cases = generate_test_cases()
    for case in cases:
        assert len(case.queries) >= 1, f"{case.id} must have >=1 query"


def test_evolving_fact_has_expected_not():
    cases = generate_test_cases(categories=["evolving_fact"])
    has_expected_not = any(q.expected_not for case in cases for q in case.queries)
    assert has_expected_not, "At least one evolving query must have expected_not"


def test_compound_cases_have_tags():
    cases = generate_test_cases(categories=["evolving_compound"])
    for case in cases:
        assert "compound" in case.tags, f"{case.id} must have 'compound' tag"
        assert "evolving" in case.tags, f"{case.id} must have 'evolving' tag"


def test_noisy_cases_have_tags():
    cases = generate_test_cases(categories=["noisy_fact"])
    for case in cases:
        assert "noisy" in case.tags, f"{case.id} must have 'noisy' tag"
        assert "evolving" in case.tags, f"{case.id} must have 'evolving' tag"


def test_serializes_to_json():
    cases = generate_test_cases()
    json_str = json.dumps([c.model_dump(mode="json") for c in cases], indent=2)
    assert len(json_str) > 0
    parsed = json.loads(json_str)
    assert len(parsed) == 75
