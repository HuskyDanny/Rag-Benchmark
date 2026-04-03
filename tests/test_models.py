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
    assert tc.tags == []  # default empty


def test_test_case_with_tags():
    from src.models import TestCase

    data = {
        "id": "compound_001",
        "category": "evolving_compound",
        "tags": ["compound", "evolving"],
        "episodes": [
            {
                "text": "Amy works at Google.",
                "reference_time": "2024-01-01T00:00:00Z",
                "order": 0,
            }
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
